"""Cancellation: stop a job between chunks, keep the text recognised so far,
and make sure a stopped or deleted job never comes back to life."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import nestor.pipeline as pipeline_module  # noqa: E402
from nestor.config import Config  # noqa: E402
from nestor.library import Library, Recording  # noqa: E402
from nestor.pipeline import JobCancelled  # noqa: E402
from nestor.types import Turn, Word  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "example.wav"


class FakeDiarizer:
    """Two speakers talking in turn."""

    model_id = "fake-diarizer"

    def __init__(self, *args, **kwargs):
        pass

    def diarize(self, waveform, sample_rate):
        total = waveform.shape[0] / sample_rate
        half = total / 2
        return [Turn(0.0, half, "SPEAKER_00"), Turn(half, total, "SPEAKER_01")]


class FakeAsr:
    """Emits one word per second of the requested chunk."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe_chunk(self, waveform, start, end, word_timestamps=True):
        words = []
        position = start
        while position < end:
            words.append(Word("слово", position, min(position + 0.8, end)))
            position += 1.0
        return words

    def close(self):
        pass


def _skip_without_sample() -> None:
    if not SAMPLE.exists():
        pytest.skip("sample audio is missing")


def _fake_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "build_diarizer", lambda *a, **k: FakeDiarizer())
    monkeypatch.setattr(pipeline_module, "SherpaOnnxAsr", FakeAsr)
    monkeypatch.setattr(pipeline_module, "ensure_gigaam_punct", lambda *a, **k: {})


def test_cancel_before_the_first_chunk_raises(tmp_path, monkeypatch):
    """Decode/diarization checkpoints abort with nothing to keep."""
    _skip_without_sample()
    _fake_pipeline(monkeypatch)

    config = Config(diarize=True, output_dir=tmp_path, formats=["txt"])
    with pytest.raises(JobCancelled):
        pipeline_module.process_file(config, SAMPLE, cancel=lambda: True)


def test_cancel_during_transcription_keeps_partial_text(tmp_path, monkeypatch):
    """The chunks done so far are transcribed and returned, progress stays."""
    _skip_without_sample()
    _fake_pipeline(monkeypatch)

    config = Config(diarize=True, chunk_seconds=3.0, output_dir=tmp_path, formats=["txt"])
    full = pipeline_module.process_file(config, SAMPLE, progress=None)
    assert len(full.segments) > 1, "the sample must split into several chunks"

    # checkpoints: after decode, after diarization, then before every chunk
    calls = {"n": 0}
    seen = []

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3   # let exactly one chunk through

    partial = pipeline_module.process_file(
        config,
        SAMPLE,
        cancel=cancel,
        on_stage=lambda stage, fraction: seen.append((stage, fraction)),
    )

    assert partial.segments, "the transcribed chunk must be kept"
    assert len(partial.segments) < len(full.segments)
    assert partial.duration == full.duration
    # a stopped job must not report completion
    assert seen[-1][0] != "done" and seen[-1][1] < 1.0


# --------------------------------------------------------------------- processor
def _blocking_process_file(calls):
    """Stands in for the pipeline: runs until the job is cancelled."""

    def blocking(config, source, on_stage=None, waveform=None, cancel=None, progress=None, **kwargs):
        calls.append(Path(source).name)
        if on_stage is not None:
            on_stage("transcription", 0.3)
        while not (cancel and cancel()):
            time.sleep(0.01)
        raise JobCancelled("stopped")

    return blocking


def _wait_for(predicate, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:  # status may flip between load() and attribute access
            pass
        time.sleep(0.02)
    try:
        return predicate()
    except Exception:
        return False


def _endpoint(app, method: str, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", ()):
            return route.endpoint
    raise AssertionError(f"route {method} {path} not found")


def test_cancel_endpoint_stops_running_and_queued_jobs(tmp_path, monkeypatch):
    import nestor.web.app as app_module

    calls = []
    monkeypatch.setattr(
        app_module,
        "decode",
        lambda p, on_progress=None: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(app_module, "process_file", _blocking_process_file(calls))

    app = app_module.create_app(work_dir=tmp_path)
    library = app.state.library
    processor = app.state.processor
    cancel_endpoint = _endpoint(app, "POST", "/api/recordings/{record_id}/cancel")

    running = Recording(id="job1", filename="first.wav")
    library.save(running)
    processor.submit(running, Config())
    assert _wait_for(lambda: library.load("job1").status == "running")

    queued = Recording(id="job2", filename="second.wav")
    library.save(queued)
    processor.submit(queued, Config())

    # still waiting in the queue -> stopped at once, status persisted by the API
    assert cancel_endpoint(record_id="job2")["status"] == "cancelled"
    assert library.load("job2").status == "cancelled"

    # running -> the worker stops at its next checkpoint
    assert cancel_endpoint(record_id="job1")["status"] == "stopping"
    assert _wait_for(lambda: library.load("job1").status == "cancelled")

    assert library.load("job1").progress == pytest.approx(0.3, abs=0.01)
    assert calls == ["first.wav"], "the queued job must never start"

    # a record the process no longer owns (e.g. leftover) is fixed inline
    assert cancel_endpoint(record_id="job2") == {"cancel": "noop", "status": "cancelled"}


def test_delete_endpoint_does_not_resurrect_a_running_job(tmp_path, monkeypatch):
    """Deleting while running used to leave a record without its audio."""
    import nestor.web.app as app_module

    calls = []
    monkeypatch.setattr(
        app_module,
        "decode",
        lambda p, on_progress=None: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(app_module, "process_file", _blocking_process_file(calls))

    app = app_module.create_app(work_dir=tmp_path)
    library = app.state.library
    processor = app.state.processor

    recording = Recording(id="job1", filename="first.wav")
    library.save(recording)
    processor.submit(recording, Config())
    assert _wait_for(lambda: library.load("job1").status == "running")

    delete_endpoint = _endpoint(app, "DELETE", "/api/recordings/{record_id}")
    assert delete_endpoint(record_id="job1") == {"deleted": "job1"}
    assert library.load("job1") is None

    # the worker is gone, so nothing may write the record back
    time.sleep(0.3)
    assert library.load("job1") is None
    assert not (library.output_dir / "job1").exists()
    assert not (library.peaks_dir / "job1.json").exists()


def test_startup_resets_interrupted_records(tmp_path):
    """A job killed with the app would stay 'running' forever."""
    import nestor.web.app as app_module

    library = Library(tmp_path)
    stale = Recording(id="stale1", filename="old.wav", status="running", progress=0.42)
    library.save(stale)
    queued = Recording(id="stale2", filename="queued.wav", status="queued")
    library.save(queued)
    done = Recording(id="stale3", filename="ready.wav", status="done")
    library.save(done)

    app_module.create_app(work_dir=tmp_path)

    for record_id in ("stale1", "stale2"):
        reset = library.load(record_id)
        assert reset.status == "cancelled"
        assert reset.stage == "cancelled"
        assert "прервана" in (reset.error or "")
    untouched = library.load("stale3")
    assert untouched.status == "done"
    assert untouched.error is None
