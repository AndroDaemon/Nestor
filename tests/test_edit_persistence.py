"""Server-side correctness of the editor's PUT and of the list cache.

* PUT must drop the per-record embedding cache when the segment -> speaker
  composition changes: the npz holds the OLD labels' means, and a trusting
  cache would hand out a voice built from the wrong segments. A text-only edit
  must NOT pay that rebuild.
* A label that loses its last segment must lose its name and match entry too,
  otherwise an orphan chip entry lingers in the manifest.
* ``Library.list()`` parses each manifest once and reuses it while mtime/size
  is unchanged - the UI polls the list every ~1.2 s while a job runs.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nestor.web.app as app_module  # noqa: E402
from nestor.library import Library, Recording, Segment  # noqa: E402


def _endpoint(app, method: str, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", ()):
            return route.endpoint
    raise AssertionError(f"route {method} {path} not found")


def _body(payload: dict):
    class Body:
        async def json(self):
            return payload

    return Body()


def _put(app, record_id: str, payload: dict):
    save = _endpoint(app, "PUT", "/api/recordings/{record_id}")
    return asyncio.run(save(record_id=record_id, request=_body(payload)))


def _recording_with_cache(library: Library) -> Path:
    """A two-speaker record plus a stub npz in the embeddings directory."""
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [
        Segment("s1", 0.0, 2.0, "SPEAKER_00", "привет"),
        Segment("s2", 2.0, 4.0, "SPEAKER_01", "пока"),
    ]
    recording.speaker_names = {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"}
    recording.speaker_match = {
        "SPEAKER_01": {"assigned_name": "Мария", "confidence": 0.9}
    }
    library.save(recording)
    npz = library.embeddings_dir / "rec1.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz.write_bytes(b"stub")
    return npz


# ------------------------------------------------------- PUT drops the cache
def test_put_reassign_drops_embeddings_cache(tmp_path):
    library = Library(tmp_path)
    npz = _recording_with_cache(library)
    app = app_module.create_app(work_dir=tmp_path)

    # segment 2 moves from SPEAKER_01 to SPEAKER_00: same label set minus one
    result = _put(
        app,
        "rec1",
        {
            "segments": [
                {"id": "s1", "start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "привет"},
                {"id": "s2", "start": 2, "end": 4, "speaker": "SPEAKER_00", "text": "пока"},
            ],
            "speaker_names": {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"},
        },
    )
    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_00"]
    assert not npz.exists(), "the stale per-label means must not survive a reassignment"


def test_put_split_drops_embeddings_cache(tmp_path):
    """A split adds a new segment id - the composition changed, so is the cache."""
    library = Library(tmp_path)
    npz = _recording_with_cache(library)
    app = app_module.create_app(work_dir=tmp_path)

    _put(
        app,
        "rec1",
        {
            "segments": [
                {"id": "s1", "start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "при"},
                {"id": "s1b", "start": 1, "end": 2, "speaker": "SPEAKER_00", "text": "вет"},
                {"id": "s2", "start": 2, "end": 4, "speaker": "SPEAKER_01", "text": "пока"},
            ]
        },
    )
    assert not npz.exists()


def test_put_text_only_edit_keeps_cache(tmp_path):
    """Correcting a word must not force a voice rebuild from the audio."""
    library = Library(tmp_path)
    npz = _recording_with_cache(library)
    app = app_module.create_app(work_dir=tmp_path)

    _put(
        app,
        "rec1",
        {
            "segments": [
                {"id": "s1", "start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "исправлено"},
                {"id": "s2", "start": 2, "end": 4, "speaker": "SPEAKER_01", "text": "пока"},
            ]
        },
    )
    assert npz.exists(), "a text-only edit must not invalidate the embedding cache"


def test_put_prunes_orphan_name_and_match(tmp_path):
    """The label that lost its last segment keeps no name/match in the manifest."""
    library = Library(tmp_path)
    _recording_with_cache(library)
    app = app_module.create_app(work_dir=tmp_path)

    result = _put(
        app,
        "rec1",
        {
            "segments": [
                {"id": "s1", "start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "привет"},
                {"id": "s2", "start": 2, "end": 4, "speaker": "SPEAKER_00", "text": "пока"},
            ],
            "speaker_names": {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"},
        },
    )
    assert result["speaker_names"] == {"SPEAKER_00": "Иван"}
    assert "SPEAKER_01" not in result["speaker_match"]
    stored = library.load("rec1")
    assert stored.speaker_names == {"SPEAKER_00": "Иван"}
    assert stored.speaker_match == {}


def test_put_keeps_names_when_composition_unchanged(tmp_path):
    """Renaming a chip alone must not touch names or the cache."""
    library = Library(tmp_path)
    npz = _recording_with_cache(library)
    app = app_module.create_app(work_dir=tmp_path)

    result = _put(
        app,
        "rec1",
        {
            "segments": [
                {"id": "s1", "start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "привет"},
                {"id": "s2", "start": 2, "end": 4, "speaker": "SPEAKER_01", "text": "пока"},
            ],
            "speaker_names": {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"},
        },
    )
    assert result["speaker_names"] == {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"}
    assert npz.exists()


# ------------------------------------------- "return from Фон" is remembered
def test_put_stores_and_returns_speaker_keep(tmp_path):
    """The list of labels pulled out of the "Фон" group survives save/load."""
    library = Library(tmp_path)
    _recording_with_cache(library)
    app = app_module.create_app(work_dir=tmp_path)

    result = _put(
        app,
        "rec1",
        {
            "segments": [
                {"id": "s1", "start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "привет"},
                {"id": "s2", "start": 2, "end": 4, "speaker": "SPEAKER_01", "text": "пока"},
            ],
            "speaker_names": {"SPEAKER_00": "Иван"},
            "speaker_keep": ["SPEAKER_01"],
        },
    )
    assert result["speaker_keep"] == ["SPEAKER_01"]
    assert library.load("rec1").speaker_keep == ["SPEAKER_01"]

    # a later save without the field leaves the stored decision alone
    _put(app, "rec1", {"speaker_names": {"SPEAKER_00": "Иван"}})
    assert library.load("rec1").speaker_keep == ["SPEAKER_01"]


def test_put_rejects_malformed_speaker_keep(tmp_path):
    library = Library(tmp_path)
    _recording_with_cache(library)
    app = app_module.create_app(work_dir=tmp_path)

    with pytest.raises(HTTPException) as exc:
        _put(app, "rec1", {"speaker_keep": "SPEAKER_01"})
    assert exc.value.status_code == 400


# --------------------------------------------------------------- list cache
def test_list_parses_each_manifest_once(tmp_path, monkeypatch):
    library = Library(tmp_path)
    for i in (1, 2, 3):
        library.save(
            Recording(
                id=f"rec{i}",
                filename=f"{i}.wav",
                status="done",
                segments=[Segment(f"s{i}", 0.0, 1.0, "SPEAKER_00", "текст")],
            )
        )

    calls = {"n": 0}
    real_read = library._read_record_data

    def counting_read(path):
        calls["n"] += 1
        return real_read(path)

    monkeypatch.setattr(library, "_read_record_data", counting_read)

    first = library.list()
    assert len(first) == 3
    # the manifests were seeded by save(); polling must not re-read them
    assert calls["n"] == 0
    library.list()
    library.list()
    assert calls["n"] == 0, "unchanged manifests must be served from memory"


def test_list_notices_an_external_edit(tmp_path):
    """A manifest changed outside save() (e.g. by the worker) is re-parsed."""
    library = Library(tmp_path)
    library.save(
        Recording(id="rec1", filename="a.wav", status="running", progress=0.1)
    )
    assert library.list()[0].progress == 0.1

    # the worker writes directly to disk in other code paths / another process
    time.sleep(0.02)  # let the filesystem timestamp tick
    path = library.record_path("rec1")
    data = path.read_text(encoding="utf-8").replace('"progress":0.1', '"progress":0.9')
    path.write_text(data, encoding="utf-8")

    assert library.list()[0].progress == 0.9


def test_list_forgets_a_deleted_manifest(tmp_path):
    library = Library(tmp_path)
    library.save(Recording(id="rec1", filename="a.wav", status="done"))
    assert len(library.list()) == 1

    (library.output_dir / "rec1" / "record.json").unlink()
    assert library.list() == []
    assert "rec1" not in library._record_cache


def test_save_seeds_cache_so_poll_does_not_hit_disk(tmp_path, monkeypatch):
    """The worker's own frequent saves keep the poll served from memory."""
    library = Library(tmp_path)
    library.save(Recording(id="rec1", filename="a.wav", status="running"))

    def boom(path):
        raise AssertionError("list() re-read a manifest it had just written")

    monkeypatch.setattr(library, "_read_record_data", boom)
    assert library.list()[0].status == "running"
