"""Diarization in a child process: keeps the window responsive.

``sherpa_onnx.OfflineSpeakerDiarization.process`` holds the GIL for long
stretches. Measured on a 73-minute meeting, a thread in the same process saw
pauses of up to 106 seconds while the server (which the window talks to) waited
for the GIL — the interface looked frozen for the whole diarization stage, and
progress callbacks did not help because they run on the very thread that holds
the lock.

A separate process has its own GIL, so the UI thread is never blocked. This is
the same remedy the decoder already got (``decode_worker``), applied to the
second long CPU stage of a run. The child also lowers its priority, so the
window keeps the CPU it needs while the machine is busy.

The child reads the cached 16 kHz wav itself and re-applies the same speech
enhancement the pipeline does (measured at ~7 s per 73 minutes, so the extra
pass is cheap). It never sends audio back: the result is a small JSON array of
speaker turns, written to a file the parent reads with one ``read_text``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .constants import SAMPLE_RATE
from .decode_worker import _lower_priority, _write_progress


def _build_diarizer(
    *,
    num_threads: int,
    num_speakers: int,
    threshold: float,
    min_duration_on: float,
    min_duration_off: float,
):
    """Build the local diarizer. Split out so tests can stand in a fake one."""
    from .diarize_local import LocalDiarizer

    return LocalDiarizer(
        num_threads=int(num_threads),
        num_speakers=int(num_speakers) or None,
        threshold=float(threshold),
        min_duration_on=float(min_duration_on),
        min_duration_off=float(min_duration_off),
    )


def _enhance_speech(waveform, sample_rate: int):
    """Apply the same lift the pipeline does, in place. Seam for tests."""
    from .audio_io import enhance_speech

    enhanced, _level_db, _mode = enhance_speech(waveform, sample_rate, inplace=True)
    return enhanced


def diarize_cache_to_turns(
    cache: str,
    turns_out: str,
    progress: str = "",
    *,
    enhance: bool = True,
    num_speakers: int = 0,
    num_threads: int = 2,
    threshold: float = 0.5,
    min_duration_on: float = 0.5,
    min_duration_off: float = 0.0,
    block_seconds: float = 0.0,
    block_overlap: float = 60.0,
) -> int:
    """Entry point of the diarization child process.

    Reads the 16 kHz mono wav at ``cache``, optionally applies the speech
    enhancement, diarizes it in overlapping blocks and writes the speaker turns
    as JSON to ``turns_out``. When ``progress`` names a file, the processed
    fraction is mirrored into it so the parent can keep the stage bar moving
    without any inter-process messaging.

    Returns the number of turns. A non-zero exit code (or a missing output
    file) tells the parent to fall back to the in-process path.
    """
    _lower_priority()
    progress_path = Path(progress) if progress else None

    import soundfile as sf

    waveform, sample_rate = sf.read(cache, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or getattr(waveform, "ndim", 1) != 1:
        # The cache is always 16 kHz mono; anything else means we read the
        # wrong file and must not pretend to have diarized it.
        return 0
    if waveform.size == 0:
        Path(turns_out).write_text("[]", encoding="utf-8")
        if progress_path is not None:
            _write_progress(progress_path, 1.0)
        return 0

    if enhance:
        waveform = _enhance_speech(waveform, SAMPLE_RATE)

    diarizer = _build_diarizer(
        num_threads=int(num_threads),
        num_speakers=int(num_speakers),
        threshold=float(threshold),
        min_duration_on=float(min_duration_on),
        min_duration_off=float(min_duration_off),
    )

    def on_progress(fraction: float) -> int:
        if progress_path is not None:
            _write_progress(progress_path, max(0.0, min(1.0, fraction)))
        # Never abort from here: the parent stops the job by terminating this
        # process, exactly like the decoder child.
        return 0

    turns = diarizer.diarize(
        waveform,
        SAMPLE_RATE,
        progress=on_progress,
        block_seconds=float(block_seconds) or None,
        block_overlap=float(block_overlap),
    )

    payload = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": turn.speaker}
        for turn in turns
    ]
    Path(turns_out).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    if progress_path is not None:
        _write_progress(progress_path, 1.0)
    return len(payload)
