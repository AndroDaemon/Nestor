"""DOCX export: a readable document with speaker labels."""

from __future__ import annotations

from pathlib import Path

from ..types import Transcript
from .srt_export import format_timestamp


def export(transcript: Transcript, path: Path) -> Path:
    """Write the transcript as a .docx document."""
    from docx import Document
    from docx.shared import Pt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    document.add_heading(Path(transcript.source).name, level=0)

    meta = document.add_paragraph()
    meta.add_run("Duration: ").bold = True
    meta.add_run(f"{transcript.duration:.1f}s\n")
    meta.add_run("ASR model: ").bold = True
    meta.add_run(f"{transcript.model}\n")
    if transcript.diarization_model:
        meta.add_run("Diarization: ").bold = True
        meta.add_run(f"{transcript.diarization_model}\n")
    meta.add_run("Created: ").bold = True
    meta.add_run(f"{transcript.created_at}")

    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        paragraph = document.add_paragraph()
        run = paragraph.add_run(f"{segment.speaker} ")
        run.bold = True
        run.font.size = Pt(11)
        stamp = paragraph.add_run(f"({format_timestamp(segment.start)}) ")
        stamp.italic = True
        paragraph.add_run(segment.text)

    document.save(str(path))
    return path
