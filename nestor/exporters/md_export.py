"""Markdown export."""

from __future__ import annotations

from pathlib import Path

from ..types import Transcript
from .srt_export import format_timestamp


def export(transcript: Transcript, path: Path, speaker_names=None) -> Path:
    """Write the transcript as Markdown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = speaker_names or {}

    lines = [
        f"# {Path(transcript.source).name}",
        "",
        f"* Длительность: {transcript.duration:.1f} с",
        f"* Модель: {transcript.model}",
    ]
    if transcript.diarization_model:
        lines.append(f"* Диаризация: {transcript.diarization_model}")
    lines += ["", "---", ""]

    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        who = names.get(segment.speaker, segment.speaker)
        lines.append(f"**{who}** · `{format_timestamp(segment.start)}`")
        lines.append("")
        lines.append(segment.text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
