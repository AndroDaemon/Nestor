"""Audio I/O via PyAV (no external ffmpeg binary)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Union

import av
import numpy as np

from .constants import SAMPLE_RATE

logger = logging.getLogger(__name__)


class AudioDecodeError(RuntimeError):
    """Raised when audio cannot be decoded via PyAV."""


def decode_to_mono_16k(
    path: Union[str, Path],
    on_progress: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    """
    Decode any supported file (audio or video container) to mono float32 @ 16 kHz.

    Parameters
    ----------
    path:
        Path to the source file (audio or video container).
    on_progress:
        Optional callback receiving the decoded fraction (0..1), estimated
        from the container's reported stream duration. Only called when the
        duration is known; receives 1.0 right before returning otherwise is
        skipped entirely.

    Returns
    -------
    np.ndarray
        1-D float32 waveform, mono, 16 kHz, normalized to [-1.0, 1.0].

    Raises
    ------
    AudioDecodeError
        If the file cannot be opened, has no audio stream, or decoding fails.

    Notes
    -----
    TODO: On real-world stereo files with correlated loud channels (e.g. phone
    recordings, dictaphone), mono downmix may exceed ±1.0 due to libswresample
    coefficients + sinc-filter overshoot. If observed, add optional `clip=True`
    parameter to apply `np.clip(waveform, -1.0, 1.0)` at the end (default off).
    """
    path = Path(path)
    if not path.exists():
        raise AudioDecodeError(f"Input file does not exist: {path}")

    t0 = time.perf_counter()
    frame_count = 0

    try:
        container = av.open(str(path))
    except (av.AVError, FileNotFoundError, OSError) as exc:
        raise AudioDecodeError(f"Failed to open container {path}: {exc}") from exc

    try:
        # Select first audio stream
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise AudioDecodeError(f"No audio stream found in {path}")

        # Rough decoded-fraction estimation: the container usually reports the
        # stream duration in seconds (may be None for some raw formats).
        total_samples: Optional[float] = None
        try:
            stream_duration = audio_stream.duration
            if stream_duration is None:
                stream_duration = container.duration
            if stream_duration is not None:
                # both are in AV_TIME_BASE (1e-6) units for container.duration;
                # stream.duration uses the stream time_base - normalize both
                if audio_stream.duration is not None and audio_stream.time_base:
                    total_seconds = float(audio_stream.duration * audio_stream.time_base)
                else:
                    total_seconds = float(stream_duration) / 1_000_000
                total_samples = total_seconds * SAMPLE_RATE
        except (TypeError, ValueError, ZeroDivisionError):
            total_samples = None

        # Resampler: input format/layout/rate -> output float32 mono 16000
        resampler = av.AudioResampler(
            format="flt",      # float32
            layout="mono",     # 1 channel
            rate=SAMPLE_RATE,
        )

        chunks = []
        decoded_samples = 0
        next_report = 0.0
        try:
            for frame in container.decode(audio_stream):
                frame_count += 1
                resampled_frames = resampler.resample(frame)
                for rframe in resampled_frames:
                    arr = rframe.to_ndarray(format="flt")  # shape: (channels, samples)
                    # PyAV returns (channels, samples); mono -> (1, n)
                    if arr.shape[0] != 1:
                        raise AudioDecodeError(
                            f"Expected mono after resampling, got shape {arr.shape}"
                        )
                    chunks.append(arr[0])  # take channel 0
                    decoded_samples += arr.shape[1]
                    if on_progress is not None and total_samples:
                        fraction = decoded_samples / total_samples
                        if fraction >= next_report:
                            next_report = fraction + 0.01
                            on_progress(min(0.999, max(0.0, fraction)))
        except av.AVError as exc:
            raise AudioDecodeError(f"Decoding error in {path}: {exc}") from exc

        # Flush resampler (PyAV 12+: resample(None) may raise; guard it)
        try:
            for rframe in resampler.resample(None):
                arr = rframe.to_ndarray(format="flt")
                if arr.shape[0] != 1:
                    raise AudioDecodeError(
                        f"Expected mono after flush, got shape {arr.shape}"
                    )
                chunks.append(arr[0])
        except (AttributeError, TypeError, av.AVError):
            # Flush not supported or no remaining data
            pass

        if not chunks:
            raise AudioDecodeError(f"Decoded audio is empty: {path}")

        waveform = np.concatenate(chunks, dtype=np.float32)

        elapsed = time.perf_counter() - t0
        duration_sec = waveform.shape[0] / SAMPLE_RATE
        logger.debug(
            "Decoded %s: %d frames -> %.3fs audio in %.3fs (%.1fx realtime)",
            path.name, frame_count, duration_sec, elapsed,
            duration_sec / elapsed if elapsed > 0 else float("inf"),
        )

        return waveform

    finally:
        container.close()


def get_duration(path: Union[str, Path]) -> float:
    """
    Return duration in seconds without full decoding.

    Parameters
    ----------
    path:
        Path to the source file.

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    AudioDecodeError
        If the file cannot be opened, has no audio stream, or duration is unknown.
    """
    path = Path(path)

    try:
        container = av.open(str(path))
    except (av.AVError, FileNotFoundError, OSError) as exc:
        raise AudioDecodeError(f"Failed to open container {path}: {exc}") from exc

    try:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise AudioDecodeError(f"No audio stream in {path}")

        # Prefer stream duration (in stream time_base)
        if audio_stream.duration is not None and audio_stream.duration > 0:
            return float(audio_stream.duration * audio_stream.time_base)

        # Fallback: container duration (in AV_TIME_BASE = 1_000_000)
        if container.duration is not None and container.duration > 0:
            return float(container.duration / av.time_base)

        raise AudioDecodeError(f"Cannot determine duration of {path}")
    except av.AVError as exc:
        raise AudioDecodeError(f"Error reading duration from {path}: {exc}") from exc
    finally:
        container.close()


def normalize_rms(waveform: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """
    Optional: light RMS normalization (NOT called automatically in decode_to_mono_16k).

    Parameters
    ----------
    waveform:
        1-D float32 array.
    target_dbfs:
        Target RMS level in dBFS (default -20 dBFS, i.e. 0.1 amplitude).

    Returns
    -------
    np.ndarray
        Normalized waveform, clipped to [-1.0, 1.0], float32.
    """
    rms = np.sqrt(np.mean(waveform**2))
    if rms > 0:
        target_rms = 10 ** (target_dbfs / 20.0)
        waveform = waveform * (target_rms / rms)
        waveform = np.clip(waveform, -1.0, 1.0)
    return waveform.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio enhancement (Sloveya-style pre-processing before ASR)
# ---------------------------------------------------------------------------

HIGHPASS_HZ = 70.0
NORM_TRIGGER_DB = -38.0  # speech quieter than this gets the loud treatment
LOUD_TARGET_DBFS = -18.0
BOOST_TARGET_DBFS = -20.0


# Zero-phase highpass as a linear-phase FIR. Long enough to make the
# transition band narrow (~20 Hz at 16 kHz) without pulling in scipy.
_HIGHPASS_NUMTAPS = 4097
_HIGHPASS_BLOCK = 1 << 16  # overlap-add block size


def _highpass_taps(cutoff: float, sample_rate: int) -> np.ndarray:
    """Linear-phase highpass FIR taps (delta minus a windowed-sinc lowpass).

    The filter is symmetric, so after compensating its group delay the
    result is zero phase — the same effect ``sosfiltfilt`` used to give.
    """
    numtaps = _HIGHPASS_NUMTAPS
    n = np.arange(numtaps, dtype=np.float64)
    centre = (numtaps - 1) / 2.0
    fc = cutoff / float(sample_rate)  # cutoff in cycles/sample
    lowpass = 2.0 * fc * np.sinc(2.0 * fc * (n - centre))
    taps = -lowpass
    taps[int(centre)] += 1.0
    return taps * np.blackman(numtaps)


def _fft_convolve(taps: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Full linear convolution of ``taps`` with ``signal`` via overlap-add.

    Chunked so the FFT workspace stays bounded no matter how long the
    recording is.
    """
    n = int(signal.size)
    m = int(taps.size)
    out_len = n + m - 1
    nfft = 1 << int(np.ceil(np.log2(_HIGHPASS_BLOCK + m - 1)))
    spectrum = np.fft.rfft(taps, nfft)
    out = np.zeros(out_len, dtype=np.float64)
    for start in range(0, n, _HIGHPASS_BLOCK):
        chunk = np.asarray(signal[start:start + _HIGHPASS_BLOCK], dtype=np.float64)
        block = np.fft.irfft(np.fft.rfft(chunk, nfft) * spectrum, nfft)
        seg = block[: max(0, out_len - start)]
        out[start:start + seg.size] += seg
    return out


def highpass_filter(waveform: np.ndarray, sample_rate: int, cutoff: float = HIGHPASS_HZ) -> np.ndarray:
    """Remove rumble below ``cutoff`` with a zero-phase linear-phase FIR."""
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    taps = _highpass_taps(cutoff, sample_rate)
    filtered = _fft_convolve(taps, waveform)
    delay = (taps.size - 1) // 2
    aligned = filtered[delay:delay + waveform.size]
    return aligned.astype(np.float32)


def speech_level_db(waveform: np.ndarray, sample_rate: int) -> float:
    """Estimate the speech level as a percentile of frame RMS (dBFS).

    The global RMS is dominated by long silences, so we take a high
    percentile of 20 ms frame RMS — it tracks the level of actual speech.
    """
    frame = max(1, int(0.02 * sample_rate))
    n_frames = len(waveform) // frame
    if n_frames == 0:
        return -120.0
    frames = waveform[: n_frames * frame].reshape(n_frames, frame)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    voiced = frame_rms[frame_rms > 0]
    if voiced.size == 0:
        return -120.0
    level = float(np.percentile(voiced, 70))
    if level <= 0:
        return -120.0
    return 20.0 * float(np.log10(level))


def enhance_speech(waveform: np.ndarray, sample_rate: int) -> tuple:
    """Highpass + level normalization, the two filters Sloveya applies.

    Returns ``(waveform, level_db, mode)`` where mode is ``"loud"``, ``"boost"``
    or ``"skip"`` (already loud enough).
    """
    if waveform.size == 0:
        return waveform, -120.0, "skip"

    filtered = highpass_filter(waveform, sample_rate)
    level = speech_level_db(filtered, sample_rate)

    if level < NORM_TRIGGER_DB:
        # Quiet recording (phone, dictaphone): lift speech to -18 dBFS.
        return normalize_rms(filtered, target_dbfs=LOUD_TARGET_DBFS), level, "loud"
    if level < -12.0:
        # Reasonable but quiet-ish: a gentle lift, nothing forced.
        return normalize_rms(filtered, target_dbfs=BOOST_TARGET_DBFS), level, "boost"
    # Already healthy: keep the filtered waveform untouched.
    return filtered, level, "skip"