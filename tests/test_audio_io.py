"""Tests for audio_io.py — PyAV-based decoding."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.audio_io import (  # noqa: E402
    decode_to_mono_16k,
    enhance_speech,
    get_duration,
    highpass_filter,
    normalize_rms,
    speech_level_db,
)

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"

TEST_FILES = [
    "example.wav",
    "dialogue.mp3",
    "dialogue.webm",
    "long_example.wav",
]


@pytest.mark.parametrize("fname", TEST_FILES)
def test_decode_invariants(fname: str):
    path = SAMPLES_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not found")

    waveform = decode_to_mono_16k(path)

    # Invariants
    assert waveform.ndim == 1, f"Expected mono (1D), got {waveform.ndim}D"
    assert waveform.dtype == np.float32, f"Expected float32, got {waveform.dtype}"
    assert waveform.size > 0, "Empty waveform"

    # Duration from metadata ≈ length / sr (±0.1 s)
    dur_meta = get_duration(path)
    dur_samples = len(waveform) / 16000
    assert abs(dur_meta - dur_samples) < 0.1, (
        f"Duration mismatch: meta={dur_meta:.3f}s samples={dur_samples:.3f}s"
    )


def test_highpass_filter_removes_rumble():
    """The 70 Hz highpass keeps the speech band and kills sub-bass rumble."""
    sr = 16000
    t = np.arange(sr * 2) / sr  # 2 seconds
    low = 0.5 * np.sin(2 * np.pi * 20 * t)  # rumble, below cutoff
    high = 0.5 * np.sin(2 * np.pi * 1000 * t)  # speech-band tone
    x = (low + high).astype(np.float32)

    y = highpass_filter(x, sr)
    assert y.dtype == np.float32
    assert y.size == x.size
    assert np.all(np.isfinite(y))

    def tone_level(sig, freq):
        # project the interior onto a complex reference tone (edges carry
        # the filter transient); amplitude = 2·|mean(x·e^{-j2πfn})|
        core = sig[sr // 4: sig.size - sr // 4]
        ref = np.exp(-2j * np.pi * freq * np.arange(core.size) / sr)
        return float(2.0 * np.abs(np.mean(core * ref)))

    low_level = tone_level(y, 20)
    high_level = tone_level(y, 1000)
    # 20 Hz must be strongly attenuated, 1 kHz must survive
    assert low_level < 0.05, f"20 Hz not attenuated: {low_level:.4f}"
    assert high_level > 0.3, f"1 kHz lost: {high_level:.4f}"


def test_enhance_speech_boosts_quiet_signal():
    """Quiet input gets the 'loud' treatment and stays finite."""
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.005 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # very quiet
    y, level_db, mode = enhance_speech(x, sr)
    assert mode == "loud"
    assert level_db < -38.0
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))
    assert np.sqrt(np.mean(y**2)) > np.sqrt(np.mean(x**2))


def test_normalize_rms():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 0.1, size=16000).astype(np.float32)  # ~ -20 dBFS
    y = normalize_rms(x, target_dbfs=-20.0)
    assert y.dtype == np.float32
    assert np.all(y >= -1.0) and np.all(y <= 1.0)
    # RMS should be close to target
    rms = np.sqrt(np.mean(y**2))
    target_rms = 10 ** (-20.0 / 20.0)
    assert abs(rms - target_rms) < 0.01


def test_decode_synthetic_sine(tmp_path):
    """
    Generate a 1-second 440 Hz sine at 44.1 kHz stereo via soundfile,
    decode via PyAV, and verify invariants.

    This test does NOT require ffmpeg.
    """
    sr_in = 44100
    duration = 1.0
    t = np.arange(int(sr_in * duration)) / sr_in
    # Stereo: left channel sine, right channel cosine (to verify mono mixing)
    # Amplitude 0.5 to avoid clipping after mono downmix + resampling overshoot
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.cos(2 * np.pi * 440 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)  # (N, 2)

    wav_path = tmp_path / "sine_440hz_stereo_44k.wav"
    sf.write(str(wav_path), stereo, sr_in, subtype="FLOAT")

    # Decode via PyAV
    waveform = decode_to_mono_16k(wav_path)

    # Invariants
    assert waveform.ndim == 1, f"Expected mono (1D), got {waveform.ndim}D"
    assert waveform.dtype == np.float32, f"Expected float32, got {waveform.dtype}"
    assert waveform.size > 0, "Empty waveform"

    # Duration: 1 second ± 20 ms (resampling filter delay)
    dur = len(waveform) / 16000
    assert abs(dur - 1.0) < 0.02, f"Duration {dur:.4f}s not within 1.0±0.02s"

    # No clipping: values should be in [-1, 1]
    assert np.all(waveform >= -1.0) and np.all(waveform <= 1.0), "Clipping detected"

    # No NaN/Inf
    assert np.all(np.isfinite(waveform)), "NaN/Inf in decoded waveform"

    # RMS check: signal should not be silence (RMS > 0.01)
    rms = np.sqrt(np.mean(waveform**2))
    assert rms > 0.01, f"RMS too low ({rms:.6f}), likely silence/zero"

    # Frequency content check: peak in FFT should be near 440 Hz
    # (basic sanity that resampling didn't destroy the signal)
    fft = np.fft.rfft(waveform)
    freqs = np.fft.rfftfreq(len(waveform), 1 / 16000)
    peak_idx = np.argmax(np.abs(fft))
    peak_freq = freqs[peak_idx]
    assert abs(peak_freq - 440) < 10, f"Peak frequency {peak_freq:.1f} Hz, expected ~440 Hz"


# Optional: compare with ffmpeg if available
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not in PATH")
def test_decode_matches_ffmpeg():
    from nestor.audio import decode as ffmpeg_decode

    for fname in TEST_FILES:
        path = SAMPLES_DIR / fname
        if not path.exists():
            continue

        wav_ffmpeg = ffmpeg_decode(path)
        wav_pyav = decode_to_mono_16k(path)

        assert wav_ffmpeg.ndim == 1
        assert wav_pyav.ndim == 1
        assert wav_ffmpeg.dtype == np.float32
        assert wav_pyav.dtype == np.float32

        # Duration match within 50 ms
        dur_ffmpeg = len(wav_ffmpeg) / 16000
        dur_pyav = len(wav_pyav) / 16000
        assert abs(dur_ffmpeg - dur_pyav) < 0.05, (
            f"{fname}: ffmpeg={dur_ffmpeg:.3f}s pyav={dur_pyav:.3f}s"
        )