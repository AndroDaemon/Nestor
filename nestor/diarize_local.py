"""Token-free speaker diarization with sherpa-onnx.

Uses two public ONNX models:

* ``sherpa-onnx-pyannote-segmentation-3-0`` — who speaks when
* ``nemo_en_titanet_small.onnx`` — voice embeddings

Everything is bundled locally, so no Hugging Face token and no account are
needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .models import ensure_local_diarization, local_diarization_models
from .types import Turn

# A block below this RMS level holds nothing to diarize. Long recordings made
# with a paused dictaphone or a muted call contain hours of digital silence; at
# -70 dBFS there is no speech at all, so skipping the block saves minutes of
# work and keeps pyannote from reading the noise floor as a speaker.
SILENT_BLOCK_DB = -70.0


def block_level_db(samples: np.ndarray) -> float:
    """RMS level of a block in dBFS (``-120`` for silence or an empty block)."""
    if samples.size == 0:
        return -120.0
    energy = float(np.dot(samples, samples)) / float(samples.size)
    return 10.0 * float(np.log10(energy)) if energy > 0 else -120.0


def is_silent_block(samples: np.ndarray) -> bool:
    """True when a block is too quiet to contain speech."""
    return block_level_db(samples) < SILENT_BLOCK_DB


class LocalDiarizationError(RuntimeError):
    """Raised when the local diarization stack cannot be used."""


# Kept as a module constant so callers can name the model without building the
# diarizer (the desktop app runs diarization in a child process and only needs
# the label for the transcript).
LOCAL_DIARIZATION_MODEL_ID = "sherpa-onnx: pyannote-segmentation-3.0 + TitaNet-small"


class LocalDiarizer:
    """Diarizer that works offline, without any model hub access."""

    model_id = LOCAL_DIARIZATION_MODEL_ID

    def __init__(
        self,
        model_root: Optional[Path] = None,
        num_threads: int = 2,
        num_speakers: Optional[int] = None,
        threshold: float = 0.5,
        min_duration_on: float = 0.25,
        min_duration_off: float = 0.4,
        provider: str = "cpu",
    ) -> None:
        models = local_diarization_models(model_root)
        if models is None:
            try:
                models = ensure_local_diarization()
            except Exception as exc:  # noqa: BLE001
                raise LocalDiarizationError(
                    "The local diarization models are missing and could not be "
                    f"downloaded: {exc}"
                ) from exc

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise LocalDiarizationError(
                "sherpa-onnx is not installed. Run: pip install sherpa-onnx"
            ) from exc

        pyannote_kwargs = {"model": str(models["segmentation"])}
        # window_shift_ratio appeared in sherpa-onnx 2.x; 1.13.x does not have it
        try:
            sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(models["segmentation"]), window_shift_ratio=0.25
            )
            pyannote_kwargs["window_shift_ratio"] = 0.25
        except TypeError:
            pass
        pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            **pyannote_kwargs
        )
        if not pyannote.validate():
            raise LocalDiarizationError(
                f"sherpa-onnx rejected the segmentation model: {models['segmentation']}"
            )

        segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig()
        segmentation.pyannote = pyannote
        segmentation.num_threads = num_threads
        segmentation.provider = provider

        embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(models["embedding"]),
            num_threads=num_threads,
            provider=provider,
        )
        if not embedding.validate():
            raise LocalDiarizationError(
                f"sherpa-onnx rejected the embedding model: {models['embedding']}"
            )

        clustering = sherpa_onnx.FastClusteringConfig(
            num_clusters=-1 if num_speakers is None else int(num_speakers),
            threshold=threshold,
        )

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation,
            embedding,
            clustering,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

        if not config.validate():
            raise LocalDiarizationError(
                "sherpa-onnx rejected the diarization configuration"
            )

        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)

    def diarize(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        progress: Optional[Callable[[float], int]] = None,
        block_seconds: Optional[float] = None,
        block_overlap: float = 60.0,
    ) -> List[Turn]:
        """Return speaker turns for a 16 kHz mono waveform.

        ``progress`` receives the fraction of processed chunks (``0..1``) as
        reported by sherpa-onnx, so long jobs can update the UI. Returning a
        truthy value from ``progress`` aborts the run (sherpa-onnx stops when
        the callback returns non-zero); the result is then empty/partial and
        the caller must check its own stop flag.

        ``block_seconds`` diarizes the recording in overlapping blocks instead
        of one call. sherpa-onnx copies the whole waveform into its own buffer,
        so a single call on a multi-hour file costs a second full-length copy —
        the difference between fitting in a low-RAM machine and dying with
        ``bad allocation``. Per-block memory is bounded by the block length; the
        labels of neighbouring blocks are stitched back together by comparing
        per-cluster voice embeddings.
        """
        samples = np.ascontiguousarray(waveform, dtype=np.float32)
        if samples.size == 0:
            return []

        total_seconds = samples.shape[0] / float(sample_rate)
        if (
            block_seconds is None
            or block_seconds <= 0
            or total_seconds <= block_seconds * 1.5
        ):
            return self._diarize_call(samples, sample_rate, 0.0, progress)

        return self._diarize_blocks(
            samples, sample_rate, float(block_seconds), float(block_overlap), progress
        )

    # ------------------------------------------------------------------ blocks
    def _diarize_call(
        self,
        samples: np.ndarray,
        sample_rate: int,
        offset: float,
        progress: Optional[Callable[[float], int]],
        span: float = 1.0,
        base: float = 0.0,
    ) -> List[Turn]:
        """One sherpa-onnx call over ``samples``; times shifted by ``offset``."""
        callback = None
        if progress is not None:
            def callback(done: int, total: int) -> int:  # noqa: F811
                fraction = done / total if total > 0 else 0.0
                try:
                    stop = progress(base + span * min(1.0, max(0.0, fraction)))
                except Exception:  # noqa: BLE001 - never abort the job from UI
                    stop = None
                return 1 if stop else 0  # non-zero aborts diarization

        result = self._diarizer.process(samples, callback)
        turns: List[Turn] = []
        for segment in result.sort_by_start_time():
            start = float(segment.start)
            end = float(segment.end)
            if end <= start:
                continue
            turns.append(
                Turn(
                    start=round(start + offset, 3),
                    end=round(end + offset, 3),
                    speaker=f"SPEAKER_{int(segment.speaker):02d}",
                )
            )
        return turns

    def _cluster_centroids(
        self,
        samples: np.ndarray,
        sample_rate: int,
        turns: List[Turn],
        offset: float = 0.0,
    ) -> dict:
        """L2-normalised mean voice embedding per local label (may be empty).

        Used only to match labels between blocks; a cluster whose segments are
        all too short yields no centroid and always gets a fresh global label.

        ``turns`` carry absolute times while ``samples`` may be a single block,
        so ``offset`` (where the block starts) shifts the regions back into the
        block's own timeline. Without it every block but the first embedded
        unrelated audio, which is what scattered the label list.
        """
        from .speaker_embed import embed_segments, l2_normalise

        regions = [(turn.start - offset, turn.end - offset) for turn in turns]
        vectors = embed_segments(samples, sample_rate, regions, min_duration=0.5)
        sums: dict = {}
        counts: dict = {}
        for turn, vector in zip(turns, vectors):
            if vector is None:
                continue
            sums[turn.speaker] = sums.get(turn.speaker, 0.0) + vector
            counts[turn.speaker] = counts.get(turn.speaker, 0) + 1
        centroids = {}
        for label, total in sums.items():
            centroids[label] = l2_normalise(total / counts[label])
        return centroids

    def _diarize_blocks(
        self,
        samples: np.ndarray,
        sample_rate: int,
        block_seconds: float,
        overlap: float,
        progress: Optional[Callable[[float], int]],
    ) -> List[Turn]:
        """Diarize in overlapping blocks and stitch labels across the seams."""
        total_seconds = samples.shape[0] / float(sample_rate)
        overlap = max(0.0, min(overlap, block_seconds * 0.5))
        step = block_seconds - overlap
        if step <= 0:
            step = block_seconds

        starts: List[float] = []
        cursor = 0.0
        while cursor < total_seconds:
            starts.append(cursor)
            cursor += step
        block_count = len(starts)

        prototypes: List["np.ndarray"] = []
        names: List[str] = []
        collected: List[Turn] = []
        seam = overlap * 0.5

        for index, start_sec in enumerate(starts):
            first = int(start_sec * sample_rate)
            last = min(samples.shape[0], int((start_sec + block_seconds) * sample_rate))
            if last <= first:
                break
            piece = samples[first:last]
            if is_silent_block(piece):
                # Nothing but the noise floor: no turns, and no embeddings to
                # compute. The seam trimming below is a no-op then. Still tick
                # the progress bar over the skipped block: otherwise the bar
                # freezes and then jumps when a silent stretch is followed by a
                # busy one (a dictaphone recording is mostly silence).
                if progress is not None:
                    progress((index + 1) / float(block_count))
                continue
            base = index / float(block_count)
            span = 1.0 / float(block_count)
            turns = self._diarize_call(
                piece, sample_rate, start_sec, progress, span=span, base=base
            )
            if turns:
                centroids = self._cluster_centroids(
                    piece, sample_rate, turns, offset=start_sec
                )
                mapping = self._match_labels(centroids, prototypes, names)
                for turn in turns:
                    turn.speaker = mapping.get(turn.speaker, turn.speaker)

            is_first = index == 0
            is_last = index == block_count - 1
            block_end = start_sec + block_seconds
            for turn in turns:
                # Trim the overlap halves so the same speech is not emitted by
                # both neighbours: keep the second half of a leading overlap,
                # and the first half of a trailing one.
                if not is_first and turn.start < start_sec + seam:
                    continue
                if not is_last and turn.start >= block_end - seam:
                    continue
                collected.append(turn)

        collected.sort(key=lambda turn: (turn.start, turn.end))
        if progress is not None:
            # The last block may have been skipped as silence, so the loop above
            # never reached fraction 1.0 by itself.
            progress(1.0)
        return collected

    def _match_labels(
        self,
        centroids: dict,
        prototypes: List["np.ndarray"],
        names: List[str],
        threshold: float = 0.6,
    ) -> dict:
        """Map this block's local labels onto the global ones from earlier blocks.

        A label with no centroid, or one whose best cosine falls below
        ``threshold``, becomes a new global label — a wrong stitch merges two
        people for the whole recording, so the bar is deliberately high.
        """
        mapping: dict = {}
        for label, vector in centroids.items():
            best_index = -1
            best_score = threshold
            for index, prototype in enumerate(prototypes):
                score = float(np.dot(vector, prototype))
                if score >= best_score:
                    best_score = score
                    best_index = index
            if best_index < 0:
                prototypes.append(vector)
                names.append(f"SPEAKER_{len(names):02d}")
                best_index = len(names) - 1
            else:
                # keep the prototype running so later blocks compare against a
                # stable estimate of the whole voice, not a single block
                prototype = prototypes[best_index]
                prototypes[best_index] = prototype * 0.7 + vector * 0.3
                norm = float(np.linalg.norm(prototypes[best_index]))
                if norm > 0:
                    prototypes[best_index] = prototypes[best_index] / norm
            mapping[label] = names[best_index]
        return mapping
