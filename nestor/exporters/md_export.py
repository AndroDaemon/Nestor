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
    from . import format_talk, talk_stats

    stats_total, stats_rows = talk_stats(transcript)
    if stats_rows:
        lines += ["", "## Речь по участникам", ""]
        lines.append(f"Всего речи: {format_talk(stats_total)}")
        lines.append("")
        for row in stats_rows:
            lines.append(
                f"* **{row['speaker']}** — {format_talk(row['seconds'])} "
                f"({round(row['share'] * 100)}%), {row['turns']} реплик, "
                f"средняя {format_talk(row['avg'])}, макс {format_talk(row['longest'])}"
            )
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
