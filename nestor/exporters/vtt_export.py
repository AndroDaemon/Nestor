"""WebVTT export (subtitle format understood by browsers and video editors)."""

from __future__ import annotations

from pathlib import Path

from ..types import Transcript


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm`` (WebVTT uses a dot)."""
    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export(transcript: Transcript, path: Path, speaker_names=None) -> Path:
    """Write the transcript as a WebVTT file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = speaker_names or {}
    blocks = ["WEBVTT", ""]
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        who = names.get(segment.speaker, segment.speaker)
        blocks.append(
            f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            f"{who}: {segment.text}\n"
        )

    path.write_text("\n".join(blocks), encoding="utf-8")
    return path
