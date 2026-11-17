"""Speaker embeddings via sherpa-onnx (the bundled TitaNet-small model).

The diarizer (``LocalDiarizer``) computes embeddings internally but never
exposes them, so the voiceprint matcher needs its own pass over the waveform.
This module wraps ``sherpa_onnx.SpeakerEmbeddingExtractor`` and provides:

* single embedding for an arbitrary audio region,
* per-segment embeddings for the transcript segments,
* per-recording caches stored as ``embeddings/<record_id>.npz``.

Long regions are cut into windows (see :data:`MAX_EMBED_SECONDS`): the model
rejects anything over 12288 frames, and on long recordings a single speaker
turn can be minutes long.

All CPU, no network: the model is the same ``nemo_en_titanet_small.onnx`` that
the diarization stack already ships.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cpuinfo import resolve_threads
from .models import local_diarization_models
from .types import Segment

_EMBEDDING_CACHE_LOCK = threading.Lock()
_EXTRACTOR = None
_EXTRACTOR_THREADS: Optional[int] = None

# TitaNet-small has a hard input ceiling: its encoder masks the activations
# with a fixed 12288-frame table, and anything longer kills the ONNX Runtime
# session with
#   Where node ... Attempting to broadcast an axis by a dimension other than 1.
#   12288 by <frames>
# A speaker turn on a long recording is easily minutes long (a quiet stretch
# gets read as one turn), so every region is cut into windows of this length
# and the window vectors are averaged.
MAX_EMBED_SECONDS = 20.0
# A tail shorter than this carries more noise than voice; drop it when the
# region already produced at least one full window.
_MIN_WINDOW_SECONDS = 0.5


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """Return the L2-normalised float32 copy of ``vector`` (zero stays zero)."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class SpeakerEmbeddingError(RuntimeError):
    """Raised when the embedding model is missing or sherpa-onnx refuses it."""


def _get_extractor():
    """Create (once) the sherpa-onnx embedding extractor; thread-safe."""
    global _EXTRACTOR
    with _EMBEDDING_CACHE_LOCK:
        if _EXTRACTOR is not None:
            return _EXTRACTOR
        import sherpa_onnx

        models = local_diarization_models()
        if models is None:
            raise SpeakerEmbeddingError(
                "Модель голосовых отпечатков не найдена. Выполните: nestor models"
            )
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(models["embedding"]),
            num_threads=max(1, int(_EXTRACTOR_THREADS or 2)),
            provider="cpu",
        )
        if not config.validate():
            raise SpeakerEmbeddingError(
                f"sherpa-onnx rejected the embedding model: {models['embedding']}"
            )
        _EXTRACTOR = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        return _EXTRACTOR


def configure(threads: int) -> None:
    """Set the thread count for the embedding extractor (0 = auto).

    Drops the cached extractor when the value changes so the next call builds
    it with the requested number of threads.
    """
    global _EXTRACTOR, _EXTRACTOR_THREADS
    wanted = resolve_threads(threads)
    with _EMBEDDING_CACHE_LOCK:
        if _EXTRACTOR is not None and wanted != _EXTRACTOR_THREADS:
            _EXTRACTOR = None
        _EXTRACTOR_THREADS = wanted


def reset_cache() -> None:
    """Forget the loaded extractor (used by tests)."""
    global _EXTRACTOR
    with _EMBEDDING_CACHE_LOCK:
        _EXTRACTOR = None


def _embed_samples(extractor, samples: np.ndarray, sample_rate: int) -> np.ndarray:
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()
    return np.asarray(extractor.compute(stream), dtype=np.float32)


def _embed_region(extractor, samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """L2-normalised embedding of a region of *any* length.

    Regions up to :data:`MAX_EMBED_SECONDS` go in as a single window — the
    model's own pooling is better than anything done here. Longer ones (a
    multi-minute speaker turn, a whole file) are cut into consecutive windows,
    embedded one by one and averaged, because one pass over them would exceed
    the model's 12288-frame limit and raise
    ``Where node ... 12288 by N``.
    """
    window = max(1, int(MAX_EMBED_SECONDS * sample_rate))
    if samples.shape[0] <= window:
        return l2_normalise(_embed_samples(extractor, samples, sample_rate))

    shortest = max(1, int(_MIN_WINDOW_SECONDS * sample_rate))
    vectors: List[np.ndarray] = []
    for first in range(0, samples.shape[0], window):
        piece = samples[first:first + window]
        if piece.shape[0] < shortest and vectors:
            continue
        vectors.append(_embed_samples(extractor, piece, sample_rate))
    if not vectors:  # region shorter than the tail limit — take it whole
        return l2_normalise(_embed_samples(extractor, samples, sample_rate))
    return l2_normalise(np.mean(np.vstack(vectors), axis=0))


def embed_audio(
    waveform: np.ndarray,
    sample_rate: int,
    start: float = 0.0,
    end: Optional[float] = None,
) -> np.ndarray:
    """L2-normalised embedding of one region of the waveform (seconds)."""
    samples = np.ascontiguousarray(waveform, dtype=np.float32)
    total = samples.shape[0] / float(sample_rate)
    start = max(0.0, float(start))
    stop = total if end is None else min(float(end), total)
    if stop <= start:
        raise ValueError(f"empty audio region: [{start}, {stop})")
    first = int(start * sample_rate)
    last = int(stop * sample_rate)
    extractor = _get_extractor()
    return _embed_region(extractor, samples[first:last], sample_rate)


def embed_segments(
    waveform: np.ndarray,
    sample_rate: int,
    segments: Sequence[Tuple[float, float]],
    min_duration: float = 0.5,
) -> List[Optional[np.ndarray]]:
    """Embeddings for time regions; ``None`` where the region is too short.

    Segments shorter than ``min_duration`` seconds carry too little speech to
    give a stable portrait, so they are skipped (noise dominates the vector).
    Longer than :data:`MAX_EMBED_SECONDS` is fine: such a region is embedded
    window by window and averaged.
    """
    samples = np.ascontiguousarray(waveform, dtype=np.float32)
    total = samples.shape[0] / float(sample_rate)
    extractor = _get_extractor()
    result: List[Optional[np.ndarray]] = []
    for start, end in segments:
        start = max(0.0, float(start))
        end = min(float(end), total)
        if end - start < float(min_duration):
            result.append(None)
            continue
        first = int(start * sample_rate)
        last = int(end * sample_rate)
        if last <= first:
            result.append(None)
            continue
        result.append(
            _embed_region(extractor, samples[first:last], sample_rate)
        )
    return result


def cluster_vectors(
    waveform: np.ndarray,
    sample_rate: int,
    segments: Sequence[Segment],
    min_segment_sec: float = 0.5,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[np.ndarray]]]:
    """Duration-weighted mean embedding per speaker label + per-segment vectors.

    Returns ``(clusters, per_segment)`` where ``clusters`` maps a speaker label
    (``SPEAKER_00``) to the weighted average of its segment embeddings, and
    ``per_segment`` maps the label to the list of its segment embeddings
    (short segments excluded). Both are L2-normalised.
    """
    regions = [(segment.start, segment.end) for segment in segments]
    vectors = embed_segments(
        waveform, sample_rate, regions, min_duration=min_segment_sec
    )
    sums: Dict[str, np.ndarray] = {}
    weights: Dict[str, float] = {}
    per_segment: Dict[str, List[np.ndarray]] = {}
    for segment, vector in zip(segments, vectors):
        if vector is None:
            continue
        duration = max(0.0, float(segment.end) - float(segment.start))
        label = segment.speaker
        sums[label] = sums.get(label, np.zeros_like(vector)) + vector * duration
        weights[label] = weights.get(label, 0.0) + duration
        per_segment.setdefault(label, []).append(vector)
    clusters: Dict[str, np.ndarray] = {}
    for label, total in sums.items():
        weight = weights.get(label, 0.0)
        mean = total / weight if weight > 0 else total
        clusters[label] = l2_normalise(mean)
    return clusters, per_segment


# ------------------------------------------------------------------ npz cache
def embeddings_path(root: Path, record_id: str) -> Path:
    """Where the per-recording embedding cache lives.

    ``root`` is the embeddings directory itself (e.g.
    ``%LOCALAPPDATA%\\Nestor\\embeddings``), not the data root.
    """
    return Path(root) / f"{record_id}.npz"


def save_record_embeddings(
    root: Path,
    record_id: str,
    clusters: Dict[str, np.ndarray],
    per_segment: Dict[str, List[np.ndarray]],
    segment_speakers: Optional[Dict[int, str]] = None,
) -> Path:
    """Persist cluster means and segment vectors atomically as ``.npz``.

    Layout: ``cl_<label>`` = cluster mean, ``seg_<label>`` = matrix of that
    label's per-segment embeddings, ``speakers`` = JSON list of labels.
    ``np.savez`` into a temp file + ``os.replace`` so a crash cannot leave a
    half-written cache behind.
    """
    path = embeddings_path(root, record_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(clusters) | set(per_segment))
    arrays: Dict[str, np.ndarray] = {}
    for label in labels:
        safe = json.dumps(label, ensure_ascii=False)  # quotes keep labels unique
        if label in clusters:
            arrays[f"cl_{safe}"] = np.asarray(clusters[label], dtype=np.float32)
        vectors = per_segment.get(label) or []
        if vectors:
            arrays[f"seg_{safe}"] = np.vstack(
                [np.asarray(v, dtype=np.float32) for v in vectors]
            )
    arrays["speakers"] = np.array(
        [json.dumps(labels, ensure_ascii=False)], dtype=np.str_
    )
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            time.sleep(0.01 + 0.01 * attempt)
    else:
        # readers would not go away — overwrite in place
        path.write_bytes(tmp.read_bytes())
        tmp.unlink(missing_ok=True)
    return path


def load_record_embeddings(
    root: Path, record_id: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[np.ndarray]]]:
    """Load a cached embedding file; empty dicts when missing or damaged."""
    path = embeddings_path(root, record_id)
    if not path.exists():
        return {}, {}
    try:
        with np.load(path, allow_pickle=False) as data:
            if "speakers" not in data:
                return {}, {}
            labels = json.loads(str(data["speakers"][0]))
            clusters: Dict[str, np.ndarray] = {}
            per_segment: Dict[str, List[np.ndarray]] = {}
            for label in labels:
                safe = json.dumps(label, ensure_ascii=False)
                if f"cl_{safe}" in data:
                    clusters[label] = l2_normalise(data[f"cl_{safe}"])
                if f"seg_{safe}" in data:
                    per_segment[label] = [
                        l2_normalise(row) for row in data[f"seg_{safe}"]
                    ]
            return clusters, per_segment
    except Exception:  # noqa: BLE001 - damaged cache behaves like "no cache"
        return {}, {}
