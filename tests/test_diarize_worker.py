"""Diarization in a child process (the desktop fast path).

sherpa-onnx holds the GIL for long stretches, so diarizing in the same process
as the HTTP server freezes the window. The worker reads the cached wav, repeats
the speech enhancement, and writes a small JSON of speaker turns that the
parent reads back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor import diarize_worker  # noqa: E402
from nestor.diarize_worker import diarize_cache_to_turns  # noqa: E402
from nestor.types import Turn  # noqa: E402

SR = 16000


def _write_cache(path: Path, seconds: float = 2.0) -> Path:
    time_axis = np.arange(int(SR * seconds), dtype=np.float32) / SR
    data = (0.3 * np.sin(2 * np.pi * 440.0 * time_axis)).astype(np.float32)
    sf.write(str(path), data, SR, subtype="PCM_16")
    return path


class _FakeDiarizer:
    """Stands in for LocalDiarizer: records the call, returns fixed turns."""

    model_id = "fake"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.diarized: list = []

    def diarize(self, waveform, sample_rate, progress=None, block_seconds=None,
                block_overlap=60.0):
        self.diarized.append(
            {
                "size": int(waveform.size),
                "rate": int(sample_rate),
                "block_seconds": block_seconds,
                "block_overlap": block_overlap,
            }
        )
        if progress is not None:
            progress(0.5)
        return [
            Turn(0.0, 1.0, "SPEAKER_00"),
            Turn(1.0, 2.0, "SPEAKER_01"),
        ]


def test_worker_writes_turns_and_finishes_progress(tmp_path, monkeypatch):
    """What the child process runs: a JSON of turns plus a progress file."""
    cache = _write_cache(tmp_path / "audio16k.wav")
    turns_out = tmp_path / "diarize.turns.json"
    progress = tmp_path / "diarize.turns.json.progress"

    monkeypatch.setattr(diarize_worker, "_build_diarizer", _FakeDiarizer)

    count = diarize_cache_to_turns(
        str(cache), str(turns_out), str(progress), enhance=False
    )

    assert count == 2
    payload = json.loads(turns_out.read_text(encoding="utf-8"))
    assert payload == [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
    ]
    assert progress.read_text(encoding="ascii").strip() == "1.000"


def test_worker_passes_settings_through(tmp_path, monkeypatch):
    cache = _write_cache(tmp_path / "audio16k.wav")
    captured: dict = {}

    def build(**kwargs):
        captured.update(kwargs)
        return _FakeDiarizer(**kwargs)

    monkeypatch.setattr(diarize_worker, "_build_diarizer", build)

    diarize_cache_to_turns(
        str(cache),
        str(tmp_path / "t.json"),
        "",
        enhance=False,
        num_speakers=0,
        num_threads=4,
        threshold=0.6,
        min_duration_on=0.7,
        min_duration_off=0.3,
        block_seconds=1200.0,
        block_overlap=60.0,
    )

    assert captured["num_threads"] == 4
    assert captured["num_speakers"] == 0  # 0 = "let the model decide" (see below)
    assert captured["threshold"] == pytest.approx(0.6)
    assert captured["min_duration_on"] == pytest.approx(0.7)
    assert captured["min_duration_off"] == pytest.approx(0.3)


def test_worker_enhances_when_asked(tmp_path, monkeypatch):
    """enhance=True must lift the level; enhance=False must leave it alone."""
    cache = _write_cache(tmp_path / "audio16k.wav")
    seen: dict = {}

    def spy(waveform, sample_rate):
        seen["call"] = True
        return waveform

    monkeypatch.setattr(diarize_worker, "_enhance_speech", spy)
    monkeypatch.setattr(diarize_worker, "_build_diarizer", _FakeDiarizer)

    diarize_cache_to_turns(str(cache), str(tmp_path / "a.json"), "", enhance=True)
    assert seen.get("call") is True

    seen.clear()
    diarize_cache_to_turns(str(cache), str(tmp_path / "b.json"), "", enhance=False)
    assert seen == {}


def test_worker_rejects_foreign_wav(tmp_path, monkeypatch):
    """A wav that is not 16 kHz mono is not our cache: no turns file."""
    rate = 8000
    data = (0.3 * np.sin(2 * np.pi * 440.0 * np.arange(rate) / rate)).astype(np.float32)
    path = tmp_path / "other.wav"
    sf.write(str(path), data, rate, subtype="PCM_16")
    turns_out = tmp_path / "turns.json"

    monkeypatch.setattr(diarize_worker, "_build_diarizer", _FakeDiarizer)

    assert diarize_cache_to_turns(str(path), str(turns_out)) == 0
    assert not turns_out.exists()


def test_worker_handles_empty_cache(tmp_path, monkeypatch):
    path = tmp_path / "empty.wav"
    sf.write(str(path), np.zeros(0, dtype=np.float32), SR, subtype="PCM_16")
    turns_out = tmp_path / "turns.json"

    monkeypatch.setattr(diarize_worker, "_build_diarizer", _FakeDiarizer)

    assert diarize_cache_to_turns(str(path), str(turns_out)) == 0
    assert json.loads(turns_out.read_text(encoding="utf-8")) == []


def test_subprocess_switch(monkeypatch):
    from nestor.web import app as web_app

    monkeypatch.delenv(web_app.DIARIZE_SUBPROCESS_ENV, raising=False)
    assert web_app._diarize_in_subprocess_enabled()
    for value in ("0", "false", "NO", " off "):
        monkeypatch.setenv(web_app.DIARIZE_SUBPROCESS_ENV, value)
        assert not web_app._diarize_in_subprocess_enabled()


def test_read_turns_roundtrip(tmp_path):
    from nestor.web import app as web_app

    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            [
                {"start": 0.5, "end": 1.5, "speaker": "SPEAKER_00"},
                {"start": "bad"},  # skipped, not fatal
            ]
        ),
        encoding="utf-8",
    )

    turns = web_app._read_turns(path)

    assert turns == [Turn(0.5, 1.5, "SPEAKER_00")]
    assert web_app._read_turns(tmp_path / "missing.json") is None


def test_child_unavailable_returns_none(tmp_path, monkeypatch):
    """No spawn available: the caller must silently diarize in-process."""
    import multiprocessing

    import nestor.models as models

    from nestor.web import app as web_app
    from nestor.library import Library, Recording

    monkeypatch.setenv(web_app.DIARIZE_SUBPROCESS_ENV, "1")
    monkeypatch.setattr(models, "has_local_diarization", lambda: True)
    monkeypatch.setattr(web_app, "waveform_cache_path",
                        lambda library, record_id: tmp_path / "audio16k.wav")
    (tmp_path / "audio16k.wav").write_bytes(b"RIFF" + b"\0" * 100)

    def boom(*args, **kwargs):
        raise OSError("spawn is unavailable here")

    monkeypatch.setattr(multiprocessing, "get_context", boom)

    library = Library(tmp_path / "lib")
    recording = Recording(id="rec1", filename="src.wav", source_path="src.wav")

    assert web_app.diarize_via_child(library, recording, _config()) is None


def test_child_skipped_without_cache(tmp_path, monkeypatch):
    import nestor.models as models

    from nestor.web import app as web_app
    from nestor.library import Library, Recording

    monkeypatch.setenv(web_app.DIARIZE_SUBPROCESS_ENV, "1")
    monkeypatch.setattr(models, "has_local_diarization", lambda: True)

    library = Library(tmp_path / "lib")
    recording = Recording(id="rec1", filename="src.wav", source_path="src.wav")

    assert web_app.diarize_via_child(library, recording, _config()) is None


def _config():
    from nestor.config import Config

    return Config()
