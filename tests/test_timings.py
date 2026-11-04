"""Tests for the 1.1.4 timings:

* 5.1 - a running job exposes a backend-computed ``elapsed_sec`` (no
  client-side timers); a done record exposes its ``processing_seconds``;
* 5.2 - after a job the output/<id>/timings.json holds the per-stage
  breakdown (schema_version 1, ISO timestamps, stage seconds).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nestor.web.app as app_module  # noqa: E402
from nestor.library import Library, Recording  # noqa: E402
from nestor.types import Segment, Transcript  # noqa: E402


def _wait_for(predicate, timeout: float = 10.0) -> bool:
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


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "decode",
        lambda p, on_progress=None: np.zeros(16000 * 4, dtype=np.float32),
    )
    return app_module.create_app(work_dir=tmp_path)


def _transcript(config, source, **kwargs):
    # a job that has actually entered the transcription stage
    on_stage = kwargs.get("on_stage")
    if on_stage is not None:
        on_stage("transcription", 0.5)
    return Transcript(
        source=str(source),
        duration=4.0,
        model=config.model_name,
        processing_seconds=0.42,
        segments=[Segment(start=0.0, end=4.0, speaker="SPEAKER_00", text="текст")],
    )


def _setup_recording(library: Library) -> None:
    recording = Recording(id="rec1", filename="a.wav", status="added")
    library.save(recording)
    (library.uploads_dir / "rec1").mkdir(parents=True, exist_ok=True)
    (library.uploads_dir / "rec1" / "a.wav").write_bytes(b"fake audio")


def _read_timings(library: Library, record_id: str = "rec1") -> dict:
    path = library.output_dir / record_id / "timings.json"
    assert path.exists(), "timings.json must be written after the job"
    return json.loads(path.read_text(encoding="utf-8"))


def test_running_job_reports_backend_elapsed(app, monkeypatch):
    """While a job runs, /api/recordings carries elapsed_sec computed on the
    server (a live counter for the UI, not a client-side timer)."""
    import threading as _threading

    library = app.state.library
    processor = app.state.processor
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    gate = _threading.Event()

    def slow_process_file(config, source, **kwargs):
        gate.wait(timeout=5.0)
        return _transcript(config, source, **kwargs)

    monkeypatch.setattr(app_module, "process_file", slow_process_file)

    _setup_recording(library)

    from nestor.web.app import ProcessBody

    result = process(record_id="rec1", body=ProcessBody(mode="full"))
    assert result["status"] in ("queued", "running")

    assert _wait_for(
        lambda: library.load("rec1").status == "running"
        and processor.elapsed_sec("rec1") is not None
    )
    # elapsed grows over time and is anchored in the backend
    first = processor.elapsed_sec("rec1")
    time.sleep(0.25)
    second = processor.elapsed_sec("rec1")
    assert second > first
    # the list endpoint exposes it
    listing = _endpoint(app, "GET", "/api/recordings")()
    row = next(r for r in listing["recordings"] if r["id"] == "rec1")
    assert row["elapsed_sec"] >= 0.0

    gate.set()
    assert _wait_for(lambda: library.load("rec1").status == "done")
    done_row = next(
        r for r in _endpoint(app, "GET", "/api/recordings")()["recordings"]
        if r["id"] == "rec1"
    )
    # after completion the live counter is replaced by the total spent time
    assert done_row["elapsed_sec"] >= 0.0


def test_timings_written_after_job(app, monkeypatch):
    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    monkeypatch.setattr(app_module, "process_file", _transcript)

    _setup_recording(library)
    from nestor.web.app import ProcessBody

    process(record_id="rec1", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec1").status == "done")

    payload = _read_timings(library)
    assert payload["schema_version"] == 1
    assert payload["status"] == "done"
    assert payload["total_sec"] >= 0.0
    assert "started_at" in payload and "finished_at" in payload
    # the fake job entered "transcription"; decode ran for real (fake decode)
    assert "transcription" in payload["stages"]
    assert payload["stages"]["transcription"] >= 0.0
    # the same breakdown is persisted in record.json and exposed by the API
    assert library.load("rec1").stage_seconds.get("transcription") is not None
    listing = _endpoint(app, "GET", "/api/recordings")()
    row = next(r for r in listing["recordings"] if r["id"] == "rec1")
    assert row["stage_seconds"] == library.load("rec1").stage_seconds


def test_stage_seconds_accumulate_across_ticks(app, monkeypatch):
    """Many ticks in the same stage must add up.

    Regression: time used to be attributed only when the stage *changed*, so a
    long transcription was recorded as the last inter-tick fragment (1.8 s for
    a 17-minute run on a real recording).
    """
    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")

    def ticking_process_file(config, source, **kwargs):
        on_stage = kwargs.get("on_stage")
        for index in range(10):
            on_stage("transcription", 0.2 + 0.05 * index)
            time.sleep(0.05)
        return _transcript(config, source, **kwargs)

    monkeypatch.setattr(app_module, "process_file", ticking_process_file)
    _setup_recording(library)

    from nestor.web.app import ProcessBody

    process(record_id="rec1", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec1").status == "done")

    stages = _read_timings(library)["stages"]
    assert stages["transcription"] >= 0.35, stages
    assert library.load("rec1").stage_seconds["transcription"] >= 0.35


def test_timings_not_written_while_running(app, monkeypatch):
    """timings.json appears only when the job is finished (any outcome)."""
    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    import threading as _threading

    release = _threading.Event()

    def gated_process_file(config, source, **kwargs):
        release.wait(timeout=5.0)
        return _transcript(config, source, **kwargs)

    monkeypatch.setattr(app_module, "process_file", gated_process_file)
    _setup_recording(library)

    from nestor.web.app import ProcessBody

    process(record_id="rec1", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec1").status == "running")
    assert not (library.output_dir / "rec1" / "timings.json").exists()

    release.set()
    assert _wait_for(lambda: library.load("rec1").status == "done")
    assert (library.output_dir / "rec1" / "timings.json").exists()


def test_timings_overwritten_on_rerun(app, monkeypatch):
    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    monkeypatch.setattr(app_module, "process_file", _transcript)
    _setup_recording(library)

    from nestor.web.app import ProcessBody

    process(record_id="rec1", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec1").status == "done")
    first = _read_timings(library)["finished_at"]

    time.sleep(1.1)  # ensure the ISO seconds tick differs
    process(record_id="rec1", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec1").status == "done")
    second = _read_timings(library)["finished_at"]
    assert second >= first


def test_run_meta_written_and_exposed(app, monkeypatch):
    """1.3.2: every run stores its own diagnostics (hardware, threads, pinning,
    settings) so a benchmark row can be read from the UI alone."""
    library = app.state.library
    process = _endpoint(app, "POST", "/api/recordings/{record_id}/process")
    monkeypatch.setattr(app_module, "process_file", _transcript)
    _setup_recording(library)

    from nestor.web.app import ProcessBody

    process(record_id="rec1", body=ProcessBody(mode="full"))
    assert _wait_for(lambda: library.load("rec1").status == "done")

    meta = library.load("rec1").run_meta
    assert meta["app_version"]
    assert meta["mode"] == "full"
    assert isinstance(meta["threads_used"], int) and meta["threads_used"] >= 1
    assert meta["cpu_physical_cores"] >= 1
    assert meta["cpu_logical_processors"] >= 1
    assert isinstance(meta["pin_applied"], bool)
    assert meta["pin_message"]
    assert meta["model"]
    assert meta["settings"]["diarize"] is True

    # the same diagnostics travel in timings.json and in the list endpoint
    payload = _read_timings(library)
    assert payload["run_meta"]["app_version"] == meta["app_version"]
    assert payload["run_meta"]["settings"] == meta["settings"]
    listing = _endpoint(app, "GET", "/api/recordings")()
    row = next(r for r in listing["recordings"] if r["id"] == "rec1")
    assert row["run_meta"] == meta


def test_run_meta_survives_record_roundtrip(tmp_path):
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.run_meta = {"threads_used": 4, "settings": {"diarize": True}}
    library = Library(tmp_path)
    library.save(recording)

    restored = Library(tmp_path).load("rec1")
    assert restored.run_meta == {"threads_used": 4, "settings": {"diarize": True}}
    assert restored.to_dict()["run_meta"]["threads_used"] == 4


def test_missing_run_meta_defaults_to_empty(tmp_path):
    """Records made before 1.3.2 simply have no diagnostics, not an error."""
    library = Library(tmp_path)
    library.save(Recording(id="old", filename="old.wav", status="done"))
    assert library.load("old").run_meta == {}
