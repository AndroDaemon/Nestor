"""Audio IO: everything is normalised to 16 kHz mono float32.

Decoding goes through PyAV so that mp3/m4a/ogg/opus/... work out of the box
without requiring an external ffmpeg binary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from .audio_io import decode_to_mono_16k
from .constants import SAMPLE_RATE


def decode(
    path: Union[str, Path],
    sample_rate: int = SAMPLE_RATE,
    on_progress: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    """Decode any audio file into a mono float32 waveform via PyAV."""
    if sample_rate != SAMPLE_RATE:
        # For compatibility: if caller requests different rate, we still decode at 16k
        # and let the caller resample if needed (not expected in current pipeline).
        pass

    return decode_to_mono_16k(path, on_progress=on_progress)


def write_wav(
    path: Union[str, Path],
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    block: int = 1 << 22,
) -> Path:
    """Write a mono float32 waveform to a 16-bit WAV file.

    The signal is written in blocks, so clamping into ``[-1, 1]`` never needs a
    second full-length copy: on a five-hour recording that copy alone is 1.2 GB,
    which is the difference between fitting in memory and ``bad allocation``.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.ascontiguousarray(waveform)
    total = int(samples.shape[0])
    with sf.SoundFile(
        str(path), mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16"
    ) as handle:
        for first in range(0, total, block):
            piece = samples[first : first + block]
            if piece.size == 0:
                continue
            if float(piece.min()) < -1.0 or float(piece.max()) > 1.0:
                piece = np.clip(piece, -1.0, 1.0)
            handle.write(piece.astype(np.float32, copy=False))
    return path


def slice_waveform(
    waveform: np.ndarray, start: float, end: float, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Return the ``[start, end)`` slice of a waveform (seconds)."""
    first = max(0, int(round(start * sample_rate)))
    last = min(waveform.shape[0], int(round(end * sample_rate)))
    if last <= first:
        return np.zeros(0, dtype=np.float32)
    return waveform[first:last]


def duration(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Duration of a waveform in seconds."""
    return waveform.shape[0] / float(sample_rate)