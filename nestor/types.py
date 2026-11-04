"""Shared data structures used across the Nestor pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_SPEAKER = "SPEAKER_00"


@dataclass
class Word:
    """A single recognised word with absolute timestamps (seconds from file start)."""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None


@dataclass
class Turn:
    """A speaker turn produced by the diarizer."""

    start: float
    end: float
    speaker: str


@dataclass
class Segment:
    """A contiguous piece of speech attributed to one speaker."""

    start: float
    end: float
    speaker: str
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """The full result of processing one audio file."""

    source: str
    duration: float
    model: str
    diarization_model: Optional[str] = None
    language: str = "ru"
    created_at: str = ""
    segments: List[Segment] = field(default_factory=list)
    rtf: Optional[float] = None
    processing_seconds: Optional[float] = None
    # label -> {assigned_name, confidence, candidates, reason}; None names mean
    # "not recognised". Exports do not serialise this - names are applied via
    # the speaker_names layer.
    speaker_match: Dict[str, dict] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Plain text of the whole transcript, one line per segment."""
        return "\n".join(f"{seg.speaker}: {seg.text}" for seg in self.segments)

    @property
    def speakers(self) -> List[str]:
        """Speaker labels in order of first appearance."""
        seen: List[str] = []
        for seg in self.segments:
            if seg.speaker not in seen:
                seen.append(seg.speaker)
        return seen
