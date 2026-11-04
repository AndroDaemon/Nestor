"""JSON export: the full structured transcript."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..types import Transcript


def export(transcript: Transcript, path: Path) -> Path:
    """Write the transcript as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "source": transcript.source,
        "duration": transcript.duration,
        "created_at": transcript.created_at,
        "language": transcript.language,
        "asr_model": transcript.model,
        "diarization_model": transcript.diarization_model,
        "rtf": transcript.rtf,
        "processing_seconds": transcript.processing_seconds,
        "speakers": transcript.speakers,
        "segments": [
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "speaker": segment.speaker,
                "text": segment.text,
                "words": [
                    {
                        "text": word.text,
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "speaker": word.speaker,
                    }
                    for word in segment.words
                ],
            }
            for segment in transcript.segments
        ],
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
