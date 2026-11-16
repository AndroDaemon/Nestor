"""Decoding straight into the waveform cache (the desktop fast path)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.audio_io import (  # noqa: E402
    AudioDecodeError,
    decode_to_mono_16k,
    decode_to_wav_16k,
)
from nestor.decode_worker import decode_source_to_cache  # noqa: E402

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
SOURCE = SAMPLES_DIR / "dialogue.webm"


@pytest.fixture(scope="module")
def source() -> Path:
    if not SOURCE.exists():
        pytest.skip("dialogue.webm not found")
    return SOURCE


def test_wav_matches_memory_decode(tmp_path, source):
    """The streamed wav carries the same signal as the in-memory decoder."""
    dest = tmp_path / "audio16k.wav"
    total = decode_to_wav_16k(source, dest)

    waveform, rate = sf.read(str(dest), dtype="float32", always_2d=False)
    reference = decode_to_mono_16k(source)

    assert rate == 16000
    assert waveform.ndim == 1
    assert waveform.size == total == reference.size
    # PCM 16 stores one step as 1/32768; allow a couple of steps of rounding.
    clip = np.clip(reference, -1.0, 1.0)
    assert float(np.max(np.abs(waveform - clip))) <= 4 / 32768


def test_progress_only_moves_forward(tmp_path, source):
    seen: list[float] = []
    decode_to_wav_16k(source, tmp_path / "a.wav", on_progress=seen.append)

    assert seen, "decoder reported no progress"
    assert seen == sorted(seen)
    assert 0.0 <= seen[0] <= seen[-1] <= 0.999


def test_missing_source_leaves_no_file(tmp_path):
    dest = tmp_path / "out.wav"
    with pytest.raises(AudioDecodeError):
        decode_to_wav_16k(tmp_path / "nope.wav", dest)
    assert not dest.exists(), "a failed decode must not leave a usable cache"


def test_worker_writes_file_and_finishes_progress(tmp_path, source):
    """What the child process runs: wav plus a progress file for the parent."""
    dest = tmp_path / "cache.wav"
    progress = tmp_path / "cache.progress"

    total = decode_source_to_cache(str(source), str(dest), str(progress))

    assert dest.exists() and dest.stat().st_size > 44
    assert total > 0
    assert progress.read_text(encoding="ascii").strip() == "1.000"


def test_subprocess_switch(monkeypatch):
    from nestor.web import app as web_app

    monkeypatch.delenv(web_app.DECODE_SUBPROCESS_ENV, raising=False)
    assert web_app._decode_in_subprocess_enabled()
    for value in ("0", "false", "NO", " off "):
        monkeypatch.setenv(web_app.DECODE_SUBPROCESS_ENV, value)
        assert not web_app._decode_in_subprocess_enabled()


def test_child_failure_returns_false(tmp_path, source, monkeypatch):
    """No spawn available: the caller must silently decode in-process."""
    import multiprocessing

    from nestor.web import app as web_app

    def boom(*args, **kwargs):
        raise OSError("spawn is unavailable here")

    monkeypatch.setattr(multiprocessing, "get_context", boom)
    dest = tmp_path / "x.wav"

    assert web_app._decode_via_child(source, dest) is False
    assert not dest.exists()


def _write_source(path: Path, seconds: float = 1.0) -> Path:
    """A short 16 kHz wav standing in for a recording the library knows about."""
    rate = 16000
    time_axis = np.arange(int(rate * seconds), dtype=np.float32) / rate
    data = (0.3 * np.sin(2 * np.pi * 440.0 * time_axis)).astype(np.float32)
    sf.write(str(path), data, rate, subtype="PCM_16")
    return path


def _library_with(tmp_path, monkeypatch, *, subprocess: bool):
    from nestor.library import Library, Recording
    from nestor.web import app as web_app

    monkeypatch.setenv(web_app.DECODE_SUBPROCESS_ENV, "1" if subprocess else "0")
    library = Library(tmp_path / "lib")
    source_file = _write_source(tmp_path / "src.wav")
    recording = Recording(id="rec1", filename="src.wav", source_path=str(source_file))
    return web_app, library, recording


def test_load_or_decode_builds_cache_and_reuses_it(tmp_path, monkeypatch):
    """First run prepares the 16 kHz cache, the second reads it back."""
    web_app, library, recording = _library_with(tmp_path, monkeypatch, subprocess=False)

    first, from_cache = web_app.load_or_decode(library, recording)
    cache = web_app.waveform_cache_path(library, "rec1")

    assert from_cache is False
    assert cache.exists() and cache.stat().st_size > 44
    assert not cache.with_name(cache.name + ".part").exists()

    second, from_cache = web_app.load_or_decode(library, recording)

    assert from_cache is True
    assert second.shape == first.shape
    assert np.array_equal(second, first)


def test_load_or_decode_uses_the_child_process(tmp_path, monkeypatch):
    """The spawned worker fills the same cache; the parent only reads it."""
    web_app, library, recording = _library_with(tmp_path, monkeypatch, subprocess=True)

    waveform, from_cache = web_app.load_or_decode(library, recording)
    cache = web_app.waveform_cache_path(library, "rec1")
    reference = decode_to_mono_16k(tmp_path / "src.wav")

    assert from_cache is False
    assert cache.exists()
    assert waveform.size == reference.size
    clip = np.clip(reference, -1.0, 1.0)
    assert float(np.max(np.abs(waveform - clip))) <= 4 / 32768
