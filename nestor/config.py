"""Runtime configuration for a Nestor run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DEFAULT_ASR_MODEL = "gigaam-v3-punct"
DEFAULT_FORMATS = ("json", "srt", "txt", "md", "docx")
DIARIZATION_BACKENDS = ("auto", "local")


@dataclass
class Config:
    """Everything that controls one transcription run."""

    # ASR
    model_name: str = DEFAULT_ASR_MODEL
    threads: int = 0  # 0 -> auto (sherpa-onnx picks num_threads)
    word_timestamps: bool = True

    # Audio enhancement (highpass + level normalization before ASR)
    audio_enhance: bool = True

    # Rescue pass: re-transcribe silent holes between recognised words
    rescue_holes: bool = True
    rescue_min_gap: float = 2.0  # shorter gaps are normal pauses
    rescue_max_hole: float = 20.0  # longer stretches are true silence
    rescue_context: float = 0.7  # seconds of context on each side

    # Chunking (GigaAM refuses audio longer than 25 s, we stay well below)
    chunk_seconds: float = 20.0
    chunk_max_gap: float = 0.5

    # Diarization
    diarize: bool = True
    diarization_backend: str = "auto"  # auto | local
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    # Clustering distance for diarization (sherpa-onnx): the number is a
    # DISTANCE, so a lower value splits voices more eagerly. 0.5 matches the
    # sherpa default and gives a stable cluster count on real meetings; the
    # old 0.3 over-split long recordings into 100+ tiny clusters.
    diar_threshold: float = 0.5
    diar_min_duration_on: float = 0.5
    diar_min_duration_off: float = 0.0

    # Voiceprints (speaker identification against a local library)
    match_voiceprints: bool = False
    save_embeddings: bool = False  # cache per-record embeddings for later matching
    vp_abs_threshold: float = 0.75
    vp_margin: float = 0.05
    vp_segment_agreement: float = 0.60
    vp_min_segment_sec: float = 0.5
    vp_library_root: Optional[Path] = None  # override for tests / custom storage
    record_id: str = ""  # used to key the embedding cache (web sets it)
    embedding_dir: Optional[Path] = None  # where <record_id>.npz is written

    # Cluster merge: fold clusters that describe the same voice after the
    # diarizer has finished (long recordings otherwise end up with dozens of
    # SPEAKER_NN labels for a handful of people).
    merge_speakers: bool = False
    merge_threshold: float = 0.45
    merge_min_weight: float = 1.5
    # Segment-agreement floor for the merge pass: a small cluster is only
    # folded into a bigger one when the individual utterances really are that
    # close (their mean cosine to the target's centroid). Without it a real
    # meeting can collapse into a single speaker.
    merge_segment_floor: float = 0.35

    # Output
    output_dir: Path = field(default_factory=lambda: Path("out"))
    formats: List[str] = field(default_factory=lambda: list(DEFAULT_FORMATS))
    speaker_labels: List[str] = field(default_factory=list)
    language: str = "ru"

    # Misc
    keep_temp: bool = False
    quiet: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    def speaker_name(self, label: str) -> str:
        """Map ``SPEAKER_00`` to a human friendly name."""
        if not self.speaker_labels:
            return label
        try:
            index = int(label.rsplit("_", 1)[-1])
        except ValueError:
            return label
        if 0 <= index < len(self.speaker_labels):
            return self.speaker_labels[index]
        return label