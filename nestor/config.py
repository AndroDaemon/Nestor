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
    # Diarize long recordings in overlapping blocks. sherpa-onnx copies the
    # whole waveform into its own buffer, so a single call on a five-hour file
    # costs a second full-length copy (~1.2 GB) on top of the Python one, which
    # is what killed such files with `bad allocation`. Block length is in
    # seconds; 0 disables blocking (single call, higher memory).
    diar_block_seconds: float = 1200.0
    diar_block_overlap: float = 60.0

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
    # meeting can collapse into a single speaker. Measured on real recordings:
    # 0.35 let a two-person talk collapse into one label holding 71% of the
    # words; 0.5 keeps the voices apart (43%) and leaves a plain "star" meeting
    # untouched. The price - more small labels - is covered by the "Фон" group.
    merge_segment_floor: float = 0.5
    # Strict-cohesion guard (opt-in, 0.0 = off): refuse any merge whose group
    # ends up internally inhomogeneous (mean cosine of its own segments to the
    # merged centroid below this). Trades a giant chained "speaker" for more,
    # smaller ones - off by default because it can over-split one voice.
    merge_min_coh: float = 0.0
    # Auto-split: before average linkage folds a big cluster together, try to
    # cut it in two when it is clearly two voices (see speaker_merge). Clusters
    # with fewer than this many segment embeddings are left alone; 0 = off.
    merge_split_min_segments: int = 50

    # Glossary: rewrite recognised terms ("докер", "днс") to canonical form
    # ("Docker", "DNS"). Fully offline, deterministic, no speed impact.
    glossary: bool = False
    # Optional path to a user glossary; when None the user file at the data
    # root is preferred, then the glossary bundled with the package.
    glossary_path: Optional[Path] = None

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