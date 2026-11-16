"""Decoding in a child process: keeps the window responsive.

The desktop app serves its UI from an HTTP server that lives in the same
process as the decoding thread. PyAV releases the GIL inside libav, but the
per-frame Python work (resampling, ``to_ndarray``, progress callbacks) keeps
taking it back: on a long recording the server answers late and the window
looks frozen for the first tens of seconds of every job.

A separate process has its own GIL, so the UI thread is never waiting for the
decoder. It also runs below normal priority: the window keeps the CPU it needs
even while the machine is busy converting a multi-hour record.

The child never sends the waveform back — that would mean a pipe write of
hundreds of megabytes. It writes the 16 kHz wav cache itself, and the parent
reads the finished file with ``soundfile`` (one C call, no Python loop).
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from .audio_io import decode_to_wav_16k

# Windows process priority classes (winbase.h).
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def _lower_priority() -> None:
    """Run this process below normal so the UI thread wins the CPU race."""
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(), _BELOW_NORMAL_PRIORITY_CLASS
        )
    except Exception:  # noqa: BLE001 - a refused priority change must not stop decoding
        pass


def _write_progress(path: Path, fraction: float) -> None:
    try:
        path.write_text(f"{fraction:.3f}", encoding="ascii")
    except OSError:
        pass


def decode_source_to_cache(source: str, dest: str, progress: str = "") -> int:
    """Entry point of the decoder child process.

    ``source`` is decoded into a 16 kHz mono PCM_16 wav at ``dest``. When
    ``progress`` names a file, the decoded fraction is mirrored into it, so the
    parent can keep the stage bar moving without any inter-process messaging.

    Returns the number of written samples.
    """
    _lower_priority()
    progress_path = Path(progress) if progress else None

    def on_progress(fraction: float) -> None:
        if progress_path is not None:
            _write_progress(progress_path, fraction)

    total = decode_to_wav_16k(source, dest, on_progress=on_progress)
    if progress_path is not None:
        _write_progress(progress_path, 1.0)
    return total
