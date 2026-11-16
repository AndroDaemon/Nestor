"""Tests for the 1.1.0 workflow items:

* add-only uploads (status "added", job not started);
* POST /api/recordings/{id}/process: modes, guards, re-run protection;
* speakers-only reassignment keeps text, bookmarks and manual names;
* job_settings survive a reload and prefill job_config;
* the upload endpoint must not block the event loop while stashing a file.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nestor.web.app as app_module  # noqa: E402
from nestor import cpuinfo  # noqa: E402
from nestor.library import Library, Recording, Segment  # noqa: E402
from nestor.types import Turn  # noqa: E402
from nestor.web.app import ProcessBody, ProcessOptions, build_export  # noqa: E402


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    """Poll a predicate; the worker may need a moment."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(0.02)
    return False


def _endpoint(app, method: str, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", ()):
            return route.endpoint
    raise AssertionError(f"route {method} {path} not found")


class FakeDiarizer:
    """First half SPEAKER_00, second half SPEAKER_01 (4 s of audio)."""

    model_id = "fake-diarizer"

    def __init__(self, *args, **kwargs):
        pass

    def diarize(self, waveform, sample_rate):
        total = max(1.0, waveform.shape[0] / sample_rate)
        half = total / 2
        return [Turn(start=0.0, end=half, speaker="SPEAKER_00"), Turn(start=half, end=total, speaker="SPEAKER_01")]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "decode",
        lambda p, on_progress=None: np.zeros(16000 * 4, dtype=np.float32),
    )
    return app_module.create_app(work_dir=tmp_path)


# --------------------------------------------------------------- add-only flow
def test_added_status_roundtrip(tmp_path):
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="added")
    library.save(recording)
    assert library.load("rec1").status == "added"


# -------------------------------------------------------------- import route
def test_import_registers_local_file_without_copy(tmp_path, monkeypatch):
    """The native Open dialog hands us an absolute path: the record must
    reference it in place (no copy into uploads/) and pass validation."""
    library = Library(tmp_path)
    media = tmp_path / "external" / "talk.mp3"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"fake mp3")
    monkeypatch.setattr(
        app_module,
        "decode",
        lambda p, on_progress=None: np.zeros(16000, dtype=np.float32),
    )
    app = app_module.create_app(work_dir=tmp_path)
    import_ep = _endpoint(app, "POST", "/api/recordings/import")

    result = import_ep(path=str(media), auto_start=False)
    assert result["status"] == "added"
    assert result["filename"] == "talk.mp3"

    stored = library.load(result["id"])
    assert stored.source_path == str(media)
    # the file was NOT copied into the library
    assert not (library.uploads_dir / result["id"]).exists()

    # the library resolves the media back to the original location
    resolved = library.recording_source(stored)
    assert resolved == media


def test_import_seeds_job_settings_from_server_settings(tmp_path, monkeypatch):
    """An imported file must honour the Settings page - previously the import
    route hardcoded ``num_speakers: None`` and ignored everything the user
    had chosen (the "hundred speakers" complaint)."""
    library = Library(tmp_path)
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"fake mp3")
    monkeypatch.setattr(
        app_module,
        "decode",
        lambda p, on_progress=None: np.zeros(16000, dtype=np.float32),
    )
    app = app_module.create_app(work_dir=tmp_path)
    app.state.settings.update(
        {
            "num_speakers": 3,
            "vp_match": True,
            "merge_speakers": False,
            "merge_min_weight": 3.0,
        }
    )
    import_ep = _endpoint(app, "POST", "/api/recordings/import")

    result = import_ep(path=str(media), auto_start=False)
    stored = library.load(result["id"])
    job = stored.job_settings
    assert job["model"] == "gigaam-v3-punct"
    assert job["num_speakers"] == 3
    assert job["match_voiceprints"] is True
    assert job["merge_speakers"] is False
    assert job["merge_min_weight"] == 3.0

    from nestor.web.app import job_config

    config = job_config(library, stored, {})
    assert config.num_speakers == 3
    assert config.merge_speakers is False
    assert config.merge_min_weight == 3.0


def test_import_rejects_unknown_extension(tmp_path):
    app = app_module.create_app(work_dir=tmp_path)
    import_ep = _endpoint(app, "POST", "/api/recordings/import")
    alien = tmp_path / "doc.pdf"
    alien.write_bytes(b"not media")
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        import_ep(path=str(alien), auto_start=False)
    assert excinfo.value.status_code == 400


def test_import_missing_file(tmp_path):
    app = app_module.create_app(work_dir=tmp_path)
    import_ep = _endpoint(app, "POST", "/api/recordings/import")
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        import_ep(path=str(tmp_path / "nope.mp3"), auto_start=False)
    assert excinfo.value.status_code == 404


# ------------------------------------------------------------- speakers merge
def test_merge_speakers_moves_segments_and_drops_name(tmp_path):
    """All segments of one label join the target; the merged label's manual
    name is dropped so duplicate names cannot linger."""
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [
        Segment("s1", 0.0, 1.0, "SPEAKER_00", "первый"),
        Segment("s2", 1.0, 2.0, "SPEAKER_01", "второй"),
        Segment("s3", 2.0, 3.0, "SPEAKER_00", "третий"),
    ]
    recording.speaker_names = {"SPEAKER_00": "Иван", "SPEAKER_01": "Иван"}
    library.save(recording)
    app = app_module.create_app(work_dir=tmp_path)
    merge = _endpoint(app, "POST", "/api/recordings/{record_id}/speakers/merge")

    result = merge(record_id="rec1", from_label="SPEAKER_00", into="SPEAKER_01")
    assert [s["speaker"] for s in result["segments"]] == [
        "SPEAKER_01", "SPEAKER_01", "SPEAKER_01",
    ]

    stored = library.load("rec1")
    assert stored.speakers == ["SPEAKER_01"]
    # the merged label lost its name; the target keeps the user's entry
    assert stored.speaker_names == {"SPEAKER_01": "Иван"}


def test_merge_speakers_guards(tmp_path):
    """Same-label merge and unknown labels are rejected."""
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [Segment("s1", 0.0, 1.0, "SPEAKER_00", "текст")]
    library.save(recording)
    app = app_module.create_app(work_dir=tmp_path)
    merge = _endpoint(app, "POST", "/api/recordings/{record_id}/speakers/merge")
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        merge(record_id="rec1", from_label="SPEAKER_00", into="SPEAKER_00")
    assert excinfo.value.status_code == 400

    with pytest.raises(HTTPException) as excinfo:
        merge(record_id="rec1", from_label="SPEAKER_09", into="SPEAKER_00")
    assert excinfo.value.status_code == 404

    # missing record
    with pytest.raises(HTTPException) as excinfo:
        merge(record_id="ghost", from_label="SPEAKER_00", into="SPEAKER_01")
    assert excinfo.value.status_code == 404


def test_merge_into_unnamed_target_moves_name(tmp_path):
    """Gluing «Иван» into a still-unnamed chip keeps the name on the target."""
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [
        Segment("s1", 0.0, 1.0, "SPEAKER_00", "один"),
        Segment("s2", 1.0, 2.0, "SPEAKER_01", "два"),
        Segment("s3", 2.0, 3.0, "SPEAKER_00", "три"),
    ]
    recording.speaker_names = {"SPEAKER_00": "Иван"}
    library.save(recording)
    app = app_module.create_app(work_dir=tmp_path)
    merge = _endpoint(app, "POST", "/api/recordings/{record_id}/speakers/merge")

    merge(record_id="rec1", from_label="SPEAKER_00", into="SPEAKER_01")
    stored = library.load("rec1")
    assert stored.speaker_names == {"SPEAKER_01": "Иван"}


def test_merge_drops_stale_embeddings_cache(tmp_path):
    """After a merge the npz still holds the OLD labels' means - drop it so
    the next "remember voice" rebuilds the target from all its segments."""
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [
        Segment("s1", 0.0, 2.0, "SPEAKER_00", "один"),
        Segment("s2", 2.0, 4.0, "SPEAKER_01", "два"),
    ]
    library.save(recording)
    npz = library.root / "embeddings" / "rec1.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz.write_bytes(b"stub")
    app = app_module.create_app(work_dir=tmp_path)
    merge = _endpoint(app, "POST", "/api/recordings/{record_id}/speakers/merge")

    merge(record_id="rec1", from_label="SPEAKER_00", into="SPEAKER_01")
    assert not npz.exists()


# ------------------------------------------------------------ remember voice
def test_voiceprint_uses_embeddings_cache(tmp_path, monkeypatch):
    """The cache lookup must point at <root>/embeddings (not the data root)
    and accept the cached vector without re-embedding."""
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [Segment("s1", 0.0, 2.0, "SPEAKER_00", "привет")]
    library.save(recording)
    # a valid npz cache with one label
    import json as _json

    import numpy as np2

    from nestor.speaker_embed import save_record_embeddings

    save_record_embeddings(
        library.embeddings_dir,
        "rec1",
        {"SPEAKER_00": np2.ones(4, dtype=np2.float32) / 2},
        {"SPEAKER_00": [np2.ones(4, dtype=np2.float32) / 2]},
    )
    app = app_module.create_app(work_dir=tmp_path)

    called = {"n": 0}

    def failing_decode(path, on_progress=None):
        called["n"] += 1
        raise AssertionError("decode must not run when the cache is valid")

    monkeypatch.setattr(app_module, "decode", failing_decode)
    add = _endpoint(app, "POST", "/api/voiceprints")

    class Body:
        async def json(self):
            return {
                "name": "Иван",
                "recording_id": "rec1",
                "speaker": "SPEAKER_00",
            }

    import asyncio

    result = asyncio.run(add(request=Body()))
    assert result["saved"] == "Иван"
    assert called["n"] == 0


def test_voiceprint_too_short_speaker(tmp_path):
    """A speaker whose every segment is shorter than the embedding minimum
    gets a clear "no voice to remember" message, not a 404."""
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [Segment("s1", 0.0, 0.3, "SPEAKER_00", "а")]
    recording.speaker_names = {"SPEAKER_00": "Иван"}
    library.save(recording)
    app = app_module.create_app(work_dir=tmp_path)
    add = _endpoint(app, "POST", "/api/voiceprints")
    from fastapi import HTTPException

    class Body:
        async def json(self):
            return {
                "name": "Иван",
                "recording_id": "rec1",
                "speaker": "SPEAKER_00",
            }

    import asyncio

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(add(request=Body()))
    assert excinfo.value.status_code == 409
    assert "Иван" in excinfo.value.detail
    assert "длиннее" in excinfo.value.detail


def test_voiceprint_stale_cache_is_rebuilt(tmp_path, monkeypatch):
    """A cache whose labels no longer cover the current segments (e.g. after
    a speakers re-run) must not be trusted: the vector is rebuilt from the
    audio, embedding only the requested speaker's own segments (a full
    re-embed of a long recording is minutes; per-speaker stays quick), and
    no partial cache is written back."""
    from nestor.models import has_local_diarization

    if not has_local_diarization():
        pytest.skip("diarization/embedding models are not downloaded")
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [
        Segment("s1", 0.0, 2.0, "SPEAKER_00", "а"),
        Segment("s2", 2.0, 4.0, "SPEAKER_01", "б"),
        Segment("s3", 4.0, 6.0, "SPEAKER_01", "в"),
    ]
    library.save(recording)
    (library.uploads_dir / "rec1").mkdir(parents=True, exist_ok=True)
    (library.uploads_dir / "rec1" / "a.wav").write_bytes(b"fake audio")
    # stale cache: has SPEAKER_00 but knows nothing about SPEAKER_01
    import numpy as np2

    from nestor.speaker_embed import save_record_embeddings

    save_record_embeddings(
        library.embeddings_dir,
        "rec1",
        {"SPEAKER_00": np2.ones(4, dtype=np2.float32) / 2},
        {"SPEAKER_00": [np2.ones(4, dtype=np2.float32) / 2]},
    )

    import nestor.speaker_embed as speaker_embed

    seen: list = []

    real_cluster = speaker_embed.cluster_vectors

    def spying_cluster(waveform, sample_rate, segments, min_segment_sec=0.5):
        seen.append([seg.speaker for seg in segments])
        return real_cluster(waveform, sample_rate, segments, min_segment_sec)

    monkeypatch.setattr(speaker_embed, "cluster_vectors", spying_cluster)
    app = app_module.create_app(work_dir=tmp_path)

    decoded = {"n": 0}

    def fake_decode(path, on_progress=None):
        decoded["n"] += 1
        return np.zeros(16000 * 4, dtype=np.float32)

    monkeypatch.setattr(app_module, "decode", fake_decode)
    add = _endpoint(app, "POST", "/api/voiceprints")

    class Body:
        async def json(self):
            return {
                "name": "Иван",
                "recording_id": "rec1",
                "speaker": "SPEAKER_01",
            }

    import asyncio

    result = asyncio.run(add(request=Body()))
    assert result["saved"] == "Иван"
    assert decoded["n"] == 1  # the stale cache forced a rebuild from audio
    # only the requested speaker's segments were embedded - the others are
    # not needed for his vector and would waste tens of seconds
    assert seen == [["SPEAKER_01", "SPEAKER_01"]]
    # the cache keeps its original form: a partial rebuild must not be
    # written back (it would look like a valid cache to later calls)
    cache, per_segment = speaker_embed.load_record_embeddings(
        library.embeddings_dir, "rec1"
    )
    assert list(cache) == ["SPEAKER_00"]


def test_voiceprint_rejected_while_running(tmp_path):
    """No voice saving during processing (the worker owns the record)."""
    app = app_module.create_app(work_dir=tmp_path)
    # saved after create_app: its startup reset turns a stored "running"
    # (crash leftover) back to "cancelled", so a live job is simulated here
    library = app.state.library
    recording = Recording(id="rec1", filename="a.wav", status="running")
    recording.segments = [Segment("s1", 0.0, 2.0, "SPEAKER_00", "привет")]
    library.save(recording)
    add = _endpoint(app, "POST", "/api/voiceprints")
    from fastapi import HTTPException

    class Body:
        async def json(self):
            return {
                "name": "Иван",
                "recording_id": "rec1",
                "speaker": "SPEAKER_00",
            }

    import asyncio

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(add(request=Body()))
    assert excinfo.value.status_code == 409
    assert "Идёт обработка" in excinfo.value.detail
    assert "SPEAKER_00" not in excinfo.value.detail  # human message, no labels


def test_startup_reset_leaves_added_alone(tmp_path):
    """'added' is not a job - a crash reset must not touch it."""
    library = Library(tmp_path)
    library.save(Recording(id="added1", filename="n.wav", status="added"))

    app_module.create_app(work_dir=tmp_path)

    assert library.load("added1").status == "added"
    assert library.load("added1").error is None


# ------------------------------------------------------------------ job_config
def test_job_settings_roundtrip(tmp_path):
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav")
    recording.job_settings = {"num_speakers": 2}
    library.save(recording)
    assert library.load("rec1").job_settings == {"num_speakers": 2}


def test_job_config_uses_stored_settings(tmp_path):
    from nestor.web.app import job_config

    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav")
    recording.job_settings = {
        "backend": "local",
        "num_speakers": 3,
        "match_voiceprints": True,
    }
    config = job_config(library, recording, {})
    assert config.model_name == "gigaam-v3-punct"
    assert config.diarization_backend == "local"
    assert config.num_speakers == 3
    assert config.match_voiceprints is True
    assert config.diarize is True
    assert config.record_id == "rec1"
    assert config.embedding_dir == library.root / "embeddings"

    # a legacy stored model name is ignored: only one model is shipped
    legacy = job_config(library, recording, {"model": "v3_e2e_ctc"})
    assert legacy.model_name == "gigaam-v3-punct"


def test_job_config_transcribe_disables_diarization(tmp_path):
    from nestor.web.app import job_config

    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav")
    config = job_config(library, recording, {}, mode="transcribe")
    assert config.diarize is False


# ------------------------------------------------------------ /process endpoint
def test_process_full_mode_and_guards(app, monkeypatch):
    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    processed = []

    def fake_process_file(config, source, **kwargs):
        processed.append(source.name)
        from nestor.types import Segment, Transcript

        return Transcript(
            source=str(source),
            duration=4.0,
            model=config.model_name,
            segments=[Segment(start=0.0, end=4.0, speaker="SPEAKER_00", text="текст")],
        )

    monkeypatch.setattr(app_module, "process_file", fake_process_file)

    recording = Recording(id="rec1", filename="a.wav", status="added")
    library.save(recording)
    # /process refuses to run without the source file - create a fake one
    (library.uploads_dir / "rec1").mkdir(parents=True, exist_ok=True)
    (library.uploads_dir / "rec1" / "a.wav").write_bytes(b"fake audio")

    # "speakers" without a transcript is rejected
    with pytest.raises(Exception):
        process(record_id="rec1", body=ProcessBody(mode="speakers"))

    # unknown mode is rejected
    with pytest.raises(Exception):
        process(record_id="rec1", body=ProcessBody(mode="what"))

    # full mode runs and produces a transcript; the worker may already be
    # running by the time the call returns, so only the final state is strict
    result = process(record_id="rec1", body=ProcessBody(mode="full"))
    assert result["status"] in ("queued", "running")
    assert _wait_for(lambda: library.load("rec1").status == "done")
    assert processed == ["a.wav"]
    loaded = library.load("rec1")
    assert loaded.segments[0].text == "текст"
    # job_settings survived (prefill for the next run)
    assert isinstance(loaded.job_settings, dict)

    # chosen options are remembered for the next run's pre-filled dialog
    process(record_id="rec1", body=ProcessBody(mode="full", options=ProcessOptions(num_speakers=2)))
    assert _wait_for(lambda: library.load("rec1").status == "done")
    assert library.load("rec1").job_settings["num_speakers"] == 2

    # a running job rejects a second /process call
    def slow_process_file(config, source, **kwargs):
        time.sleep(1.0)
        return fake_process_file(config, source, **kwargs)

    monkeypatch.setattr(app_module, "process_file", slow_process_file)
    process(record_id="rec1", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec1").status in ("queued", "running"))
    with pytest.raises(Exception):
        process(record_id="rec1", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec1").status == "done")


def test_process_missing_source_rejected(app):
    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    library.save(Recording(id="rec9", filename="gone.wav", status="added"))
    with pytest.raises(Exception):
        process(record_id="rec9", body=ProcessBody(mode="full"))


# ------------------------------------------------------------- speakers mode
def test_speakers_mode_reassigns_and_keeps_text(app, monkeypatch):
    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    monkeypatch.setattr(app_module, "process_file", lambda *a, **k: None)

    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [
        Segment("s1", 0.0, 1.5, "SPEAKER_00", "привет"),
        Segment("s2", 2.0, 3.5, "SPEAKER_01", "пока"),
    ]
    recording.segments[0].bookmark = True
    recording.speaker_names = {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"}
    library.save(recording)
    (library.uploads_dir / "rec1").mkdir(parents=True, exist_ok=True)
    (library.uploads_dir / "rec1" / "a.wav").write_bytes(b"fake audio")

    # the diarizer cache resolves through nestor.pipeline - seed it directly
    import nestor.pipeline as pipeline_module

    key = (True, "auto", None, cpuinfo.resolve_threads(0))
    monkeypatch.setitem(
        pipeline_module._DIARIZER_CACHE, key, (FakeDiarizer(), threading.Lock())
    )

    result = process(record_id="rec1", body=ProcessBody(mode="speakers"))
    assert result["status"] in ("queued", "running")
    assert _wait_for(lambda: library.load("rec1").status == "done")

    loaded = library.load("rec1")
    assert [seg.text for seg in loaded.segments] == ["привет", "пока"]
    assert loaded.segments[0].bookmark is True, "bookmarks survive"
    # segment 0..1.5 overlaps the first half; segment 2..3.5 the second half
    assert loaded.segments[0].speaker == "SPEAKER_00"
    assert loaded.segments[1].speaker == "SPEAKER_01"
    # manual names follow their clusters
    assert loaded.speaker_names.get("SPEAKER_00") == "Иван"
    assert loaded.speaker_names.get("SPEAKER_01") == "Мария"
    assert loaded.status == "done"


class OneSpeakerDiarizer:
    """Every utterance belongs to a single speaker (the split changed)."""

    model_id = "fake-diarizer"

    def __init__(self, *args, **kwargs):
        pass

    def diarize(self, waveform, sample_rate):
        return [Turn(start=0.0, end=10.0, speaker="SPEAKER_00")]


def test_speakers_mode_drops_names_when_split_changes(app, monkeypatch):
    """QA: a re-run that folds two people into one cluster must not keep their
    old names on the merged voice — the names are dropped and reported."""
    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    monkeypatch.setattr(app_module, "process_file", lambda *a, **k: None)

    recording = Recording(id="rec2", filename="b.wav", status="done")
    recording.segments = [
        Segment("s1", 0.0, 1.5, "SPEAKER_00", "привет"),
        Segment("s2", 2.0, 3.5, "SPEAKER_01", "пока"),
    ]
    recording.speaker_names = {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"}
    library.save(recording)
    (library.uploads_dir / "rec2").mkdir(parents=True, exist_ok=True)
    (library.uploads_dir / "rec2" / "b.wav").write_bytes(b"fake audio")

    import nestor.pipeline as pipeline_module

    key = (True, "auto", None, cpuinfo.resolve_threads(0))
    monkeypatch.setitem(
        pipeline_module._DIARIZER_CACHE, key, (OneSpeakerDiarizer(), threading.Lock())
    )

    result = process(record_id="rec2", body=ProcessBody(mode="speakers"))
    assert result["status"] in ("queued", "running")
    assert _wait_for(lambda: library.load("rec2").status == "done")

    loaded = library.load("rec2")
    # both utterances now belong to one voice; neither name describes it any more
    assert loaded.speaker_names == {}
    assert loaded.speakers == ["SPEAKER_00"]
    assert "сброшены имена" in (loaded.speakers_diff or "")
    assert "Иван" in loaded.speakers_diff and "Мария" in loaded.speakers_diff


# ------------------------------------------------- cancellation & delete races
class CancellableDiarizer:
    """Accepts the progress callback and loops until the job asks to stop."""

    model_id = "fake-diarizer"

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def diarize(self, waveform, sample_rate, progress=None):
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            self.calls += 1
            if progress is not None and progress(0.5):
                return []  # aborted mid-run, as sherpa-onnx does
            time.sleep(0.01)
        return [Turn(start=0.0, end=4.0, speaker="SPEAKER_00")]


def test_speakers_job_stops_mid_diarization(app, monkeypatch):
    """The progress callback must be able to abort a long diarization run."""
    import nestor.pipeline as pipeline_module

    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    cancel = _endpoint(app, "POST", "/api/recordings/{record_id}/cancel")
    monkeypatch.setattr(app_module, "process_file", lambda *a, **k: None)

    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [Segment("s1", 0.0, 1.0, "SPEAKER_00", "текст")]
    library.save(recording)
    (library.uploads_dir / "rec1").mkdir(parents=True, exist_ok=True)
    (library.uploads_dir / "rec1" / "a.wav").write_bytes(b"fake audio")

    key = (True, "auto", None, cpuinfo.resolve_threads(0))
    monkeypatch.setitem(
        pipeline_module._DIARIZER_CACHE, key, (CancellableDiarizer(), threading.Lock())
    )

    result = process(record_id="rec1", body=ProcessBody(mode="speakers"))
    assert result["status"] in ("queued", "running")
    assert _wait_for(lambda: library.load("rec1").status == "running")
    assert cancel(record_id="rec1")["status"] == "stopping"
    assert _wait_for(lambda: library.load("rec1").status == "cancelled", timeout=15.0)


def test_delete_during_speakers_job_succeeds(app, monkeypatch):
    """Deleting while a speakers-only run is stuck in diarization must work:
    the cancel event aborts the run and the record is removed."""
    import nestor.pipeline as pipeline_module

    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    delete = _endpoint(app, "DELETE", "/api/recordings/{record_id}")
    monkeypatch.setattr(app_module, "process_file", lambda *a, **k: None)

    recording = Recording(id="rec2", filename="a.wav", status="done")
    recording.segments = [Segment("s1", 0.0, 1.0, "SPEAKER_00", "текст")]
    library.save(recording)
    (library.uploads_dir / "rec2").mkdir(parents=True, exist_ok=True)
    (library.uploads_dir / "rec2" / "a.wav").write_bytes(b"fake audio")

    key = (True, "auto", None, cpuinfo.resolve_threads(0))
    monkeypatch.setitem(
        pipeline_module._DIARIZER_CACHE, key, (CancellableDiarizer(), threading.Lock())
    )

    process(record_id="rec2", body=ProcessBody(mode="speakers"))
    assert _wait_for(lambda: library.load("rec2").status == "running", timeout=10.0)

    started = time.monotonic()
    assert delete(record_id="rec2") == {"deleted": "rec2"}
    elapsed = time.monotonic() - started
    assert library.load("rec2") is None
    # the abort must be near-instant, not the full 20 s timeout
    assert elapsed < 10.0, f"delete took {elapsed:.1f}s — diarization not abortable"


def test_save_put_rejected_while_job_running(app, monkeypatch):
    """A PUT from the editor must not race the worker's own record writes."""
    import asyncio

    library = app.state.library
    save = _endpoint(app, "PUT", "/api/recordings/{record_id}")
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    cancel = _endpoint(app, "POST", "/api/recordings/{record_id}/cancel")
    monkeypatch.setattr(app_module, "process_file", _blocking_process_file())

    recording = Recording(id="rec3", filename="a.wav", status="added")
    library.save(recording)
    (library.uploads_dir / "rec3").mkdir(parents=True, exist_ok=True)
    (library.uploads_dir / "rec3" / "a.wav").write_bytes(b"fake audio")

    class Body:
        async def json(self):
            return {"segments": [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "x"}]}

    process(record_id="rec3", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec3").status in ("queued", "running"))
    with pytest.raises(Exception):
        asyncio.run(save(record_id="rec3", request=Body()))

    # stop the blocking job, then editing works again
    assert cancel(record_id="rec3")["status"] in ("stopping", "cancelled")
    assert _wait_for(lambda: library.load("rec3").status == "cancelled")
    result = asyncio.run(save(record_id="rec3", request=Body()))
    assert result["segments"][0]["text"] == "x"


def _blocking_process_file():
    """Stands in for the pipeline: runs until the job is cancelled."""

    def blocking(config, source, on_stage=None, waveform=None, cancel=None, progress=None, **kwargs):
        if on_stage is not None:
            on_stage("transcription", 0.3)
        while not (cancel and cancel()):
            time.sleep(0.01)
        from nestor.pipeline import JobCancelled

        raise JobCancelled("stopped")

    return blocking


def test_delete_removes_embeddings_cache(tmp_path):
    """The per-record npz must not survive the recording (orphaned garbage)."""
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="done")
    library.save(recording)
    npz = library.root / "embeddings" / "rec1.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz.write_bytes(b"stub")

    assert library.delete("rec1") is True
    assert not npz.exists()


def test_delete_is_idempotent_after_partial_removal(tmp_path):
    """A record.json-less leftover (earlier failed delete) cleans up quietly."""
    library = Library(tmp_path)
    upload = library.uploads_dir / "ghost"
    upload.mkdir(parents=True)
    (upload / "a.wav").write_bytes(b"stub")

    assert library.delete("ghost") is True
    assert not upload.exists()


# --------------------------------------------------------------- upload stall
def test_upload_does_not_block_event_loop(app):
    """Copying a big upload to disk runs in a worker thread: while it is in
    progress the event loop must keep serving other requests (the UI polls
    statuses and the stop button go through the same loop).

    The fake spool holds the "disk write" until a probe coroutine has run on
    the event loop. If the endpoint ever blocked the loop with an inline copy
    (the pre-1.1.1 bug), the probe cannot run before the hold times out and
    the timing assertion fails."""
    from starlette.datastructures import Headers, UploadFile

    library = app.state.library
    create = _endpoint(app, "POST", "/api/recordings")
    peaks = _endpoint(app, "GET", "/api/recordings/{record_id}/peaks")

    times: dict = {}
    copy_started = threading.Event()
    probe_ran = threading.Event()
    hold_seconds = 5.0  # how long the fake write may stall waiting for the probe

    class BlockingSpool:
        """Read blocks like a slow disk write until the probe has run."""

        def read(self, *args):
            times["copy_started"] = time.monotonic()
            copy_started.set()
            deadline = times["copy_started"] + hold_seconds
            while not probe_ran.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
            return b""  # empty: the copy finishes once unheld

        def close(self):
            pass

    body = UploadFile(
        filename="big.wav",
        file=BlockingSpool(),
        headers=Headers({"content-type": "audio/wav"}),
    )

    async def scenario():
        loop = asyncio.get_running_loop()

        async def probe():
            # the probe may only confirm liveness once the copy is in its
            # blocking read - and it can only run if the loop is free
            while not copy_started.is_set():
                await asyncio.sleep(0.02)
            times["probe"] = time.monotonic()
            probe_ran.set()

        probe_task = loop.create_task(probe())
        upload_task = loop.create_task(
            create(
                file=body,
                diarize=True,
                num_speakers=None,
                backend="auto",
                match_voiceprints=False,
                vp_abs_threshold=None,
                vp_margin=None,
                vp_segment_agreement=None,
                auto_start=False,
            )
        )
        for _ in range(100):
            await asyncio.sleep(0.05)
            if copy_started.is_set() and probe_ran.is_set():
                break
        assert copy_started.is_set(), "upload copy never started"
        assert probe_ran.is_set(), "event loop was blocked during the upload copy"
        assert times["probe"] - times["copy_started"] < hold_seconds - 1.0, (
            "the probe ran only after the copy gave up - the loop was blocked"
        )
        # sanity: another handler answers while nothing is blocked anymore
        assert peaks(record_id="zzz") == {"peaks": []}
        return await asyncio.wait_for(upload_task, timeout=10.0)

    result = asyncio.run(scenario())
    assert result["status"] == "added"
    assert library.load(result["id"]) is not None


# ---------------------------------------------------------- export & progress
def test_json_export_matches_other_formats_and_is_structured(tmp_path):
    """QA М8: the JSON export used to hand back the internal record.json
    (named `record.json`, editable-only fields) while every other format
    produced `transcript.<fmt>` with the public structure."""
    library = Library(tmp_path)
    recording = Recording(id="rec1", filename="talk.wav", status="done")
    recording.duration = 12.0
    recording.model = "gigaam-v3-punct"
    recording.segments = [Segment("s1", 0.0, 1.0, "SPEAKER_00", "привет")]
    library.save(recording)

    exported = build_export(library, "rec1", "json")
    assert exported.name == "transcript.json"
    import json as _json

    payload = _json.loads(exported.read_text(encoding="utf-8"))
    assert payload["source"] == "talk.wav"
    assert payload["asr_model"] == "gigaam-v3-punct"
    assert payload["segments"][0]["text"] == "привет"
    # the internal record keeps editable bookkeeping, the export must not leak it
    assert "job_settings" not in payload
    assert "progress" not in payload


def test_failed_job_resets_progress(tmp_path, monkeypatch):
    """QA М11: a crashed run kept its last percentage, so the UI showed a
    frozen progress bar next to an error."""
    monkeypatch.setattr(
        app_module,
        "decode",
        lambda p, on_progress=None: np.zeros(16000 * 4, dtype=np.float32),
    )

    def exploding(config, source, on_stage=None, waveform=None, cancel=None, progress=None, **kwargs):
        if on_stage is not None:
            on_stage("transcription", 0.4)
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "process_file", exploding)
    library = Library(tmp_path)
    app = app_module.create_app(work_dir=tmp_path)
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")

    library.save(Recording(id="rec1", filename="a.wav", status="added"))
    (library.uploads_dir / "rec1").mkdir(parents=True, exist_ok=True)
    (library.uploads_dir / "rec1" / "a.wav").write_bytes(b"fake")

    process(record_id="rec1", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec1").status == "error")
    assert library.load("rec1").progress == 0.0
