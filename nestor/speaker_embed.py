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
        start = float(start)
        end = float(end)
        # Judge the length before clamping to the file: a segment that only
        # looks short because the recording ends mid-word still belongs to the
        # cluster, and callers rely on one embedding per segment for alignment.
        if end - start < float(min_duration):
            result.append(None)
            continue
        start = max(0.0, start)
        end = min(end, total)
        if end <= start:
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


OUTLIER_COS = 0.30
"""Agreement below this means the segment is not this voice at all."""

OUTLIER_RATIO = 0.5
"""...and it must also be far worse than the next segment to be dropped.

One interjection without voice ("э-э, ну вот") has an agreement of 0.08 in a
cluster whose real segments sit at 0.23 and 0.33, so it stands clearly apart.
Two real short segments, on the other hand, may agree with each other only
weakly (0.48 pairwise) - dropping either of them would destroy a tiny cluster's
centroid, which is exactly the case this guard exists for.
"""


def _weighted_mean(pairs: Sequence[Tuple[np.ndarray, float]]) -> np.ndarray:
    total = np.zeros_like(pairs[0][0])
    weight = 0.0
    for vector, duration in pairs:
        total = total + vector * duration
        weight += duration
    return total / weight if weight > 0 else total


def _drop_outliers(
    pairs: Sequence[Tuple[np.ndarray, float]],
    floor: float = OUTLIER_COS,
    ratio: float = OUTLIER_RATIO,
) -> List[Tuple[np.ndarray, float]]:
    """Remove the one segment that clearly does not belong to the cluster.

    Agreement is pairwise - the mean cosine to the other segments of the same
    label. The cosine to the cluster's own centroid is not usable here: on a
    small cluster the outlier drags the centroid towards itself and then looks
    almost respectable there (0.56 in the case that started this).
    """
    count = len(pairs)
    if count < 3:
        # Two vectors always agree or disagree equally with each other, so
        # there is no majority to trust: leave the cluster alone.
        return list(pairs)
    matrix = np.asarray(
        [np.asarray(vector, dtype=np.float64) for vector, _ in pairs]
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 0.0] = 1.0
    matrix = matrix / norms
    sims = matrix @ matrix.T
    agreement = (sims.sum(axis=1) - 1.0) / (count - 1)
    order = np.argsort(agreement)
    worst = int(order[0])
    runner_up = float(agreement[order[1]])
    if agreement[worst] >= floor or agreement[worst] >= runner_up * ratio:
        return list(pairs)
    return [pair for index, pair in enumerate(pairs) if index != worst]


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

    The centroid is robust against segments with no voice in them: segments far
    below :data:`OUTLIER_COS` from the rest of their cluster are left out of the
    mean. ``per_segment`` still lists every segment, in segment order, because
    later steps match those rows to segments by position.
    """
    regions = [(segment.start, segment.end) for segment in segments]
    vectors = embed_segments(
        waveform, sample_rate, regions, min_duration=min_segment_sec
    )
    pairs: Dict[str, List[Tuple[np.ndarray, float]]] = {}
    for segment, vector in zip(segments, vectors):
        if vector is None:
            continue
        duration = max(0.0, float(segment.end) - float(segment.start))
        pairs.setdefault(segment.speaker, []).append((vector, duration))
    clusters: Dict[str, np.ndarray] = {}
    per_segment: Dict[str, List[np.ndarray]] = {}
    for label, rows in pairs.items():
        clusters[label] = l2_normalise(_weighted_mean(_drop_outliers(rows)))
        per_segment[label] = [vector for vector, _ in rows]
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
