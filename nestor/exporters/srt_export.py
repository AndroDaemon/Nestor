"""SRT export: subtitles with speaker names."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..types import Transcript


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm``."""
    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export(transcript: Transcript, path: Path) -> Path:
    """Write the transcript as an SRT file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks: List[str] = []
    for index, segment in enumerate(transcript.segments, start=1):
        if not segment.text.strip():
            continue
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            f"{segment.speaker}: {segment.text}\n"
        )

    path.write_text("\n".join(blocks), encoding="utf-8")
    return path
