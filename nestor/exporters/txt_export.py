"""Plain text export: one line per segment."""

from __future__ import annotations

from pathlib import Path

from ..types import Transcript
from .srt_export import format_timestamp


def export(transcript: Transcript, path: Path) -> Path:
    """Write the transcript as a readable plain text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"[{format_timestamp(segment.start)}] {segment.speaker}: {segment.text}"
        for segment in transcript.segments
        if segment.text.strip()
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
