"""Transcript exporters: JSON, SRT, TXT, Markdown and DOCX."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..config import Config
from ..types import Transcript

Exporter = Callable[..., Path]

FORMATS = ("json", "srt", "vtt", "txt", "md", "docx")


def _registry() -> Dict[str, Exporter]:
    """Resolve exporters lazily so that optional deps stay optional."""
    from . import docx_export, json_export, md_export, srt_export, txt_export, vtt_export

    return {
        "json": json_export.export,
        "srt": srt_export.export,
        "txt": txt_export.export,
        "md": md_export.export,
        "docx": docx_export.export,
        "vtt": vtt_export.export,
    }


def export_all(transcript: Transcript, config: Config) -> List[Path]:
    """Write the transcript to every requested format.

    Returns the list of written files.
    """
    registry = _registry()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(transcript.source).stem

    written: List[Path] = []
    for fmt in config.formats:
        exporter = registry.get(fmt.lower())
        if exporter is None:
            raise ValueError(
                f"Unknown export format '{fmt}'. Available: {sorted(registry)}"
            )
        written.append(exporter(transcript, config.output_dir / f"{stem}.{fmt}"))
    return written


def transcript_from_recording(recording, speaker_names: Optional[Dict[str, str]] = None) -> Transcript:
    """Build a :class:`Transcript` from a library recording."""
    from ..library import Recording
    from ..types import Segment

    assert isinstance(recording, Recording)
    names = speaker_names if speaker_names is not None else recording.speaker_names

    segments: List[Segment] = []
    for segment in recording.segments:
        segments.append(
            Segment(
                start=segment.start,
                end=segment.end,
                speaker=names.get(segment.speaker, segment.speaker),
                text=segment.text,
                words=[],
            )
        )

    return Transcript(
        source=recording.filename,
        duration=recording.duration,
        model=recording.model,
        diarization_model=recording.diarization_model,
        created_at=recording.created_at,
        segments=segments,
        rtf=recording.rtf,
        processing_seconds=recording.processing_seconds,
    )


__all__ = ["export_all", "Exporter", "FORMATS", "transcript_from_recording"]
