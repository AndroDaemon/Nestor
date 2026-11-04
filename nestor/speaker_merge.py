"""Merge diarization clusters that actually describe the same voice.

sherpa-onnx's fast clustering can leave a long recording with dozens of
near-identical labels: the same person gets a fresh ``SPEAKER_NN`` whenever the
clustering threshold is not met, and very short utterances often end up in
clusters of their own. The fix is a second, conservative pass *after*
diarization: the voice embeddings of all clusters are compared, and only
clusters that are extremely similar are folded together.

The merge is greedy and pairwise — the globally most similar pair is merged
first, and only pairs at or above ``threshold`` are ever merged. This is much
safer than single-linkage chaining, where every pair above the threshold
transitively collapses a whole recording into one speaker.

Clusters whose total speech is below ``min_weight`` seconds are unreliable by
construction (their centroid comes from a handful of frames), so they are
absorbed into their most similar neighbour whatever the similarity.

When the individual segment embeddings are supplied (``segment_vectors``), a
candidate pair must also pass a floor test: the mean similarity between the
smaller cluster's own segments and the bigger cluster's centroid must reach
``segment_floor``. Two different people may have similarly positioned centroids
(both were recorded in the same room), but their individual utterances still sit
closer to their own voice — that is what keeps a real meeting from collapsing
into one speaker.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["merge_similar_clusters", "cosine", "normalise", "apply_to_segments"]

# A merged cluster keeps at most this many per-segment votes; long recordings
# are subsampled so the agreement test stays cheap.
SEGMENT_VOTE_CAP = 400


def normalise(vector: Sequence[float]) -> np.ndarray:
    """L2-normalised copy of a vector; a zero vector is returned unchanged."""
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        return values.copy()
    return values / norm


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (0.0 when either is a zero vector)."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def merge_similar_clusters(
    vectors: Dict[str, Sequence[float]],
    threshold: float,
    weights: Optional[Dict[str, float]] = None,
    min_weight: float = 0.0,
    segment_vectors: Optional[Dict[str, List[Sequence[float]]]] = None,
    segment_floor: float = 0.0,
) -> Tuple[Dict[str, str], List[List[str]]]:
    """Fold clusters whose centroids are at least ``threshold`` alike.

    ``vectors`` maps a cluster label to its centroid. ``weights`` (normally the
    total speech seconds per cluster) decides which label represents the merged
    group — the heaviest one, ties broken alphabetically — and how strongly the
    merged centroid leans towards each member.

    ``segment_vectors`` holds the individual embeddings behind each centroid.
    When present, a pair may also be required to pass ``segment_floor`` (see the
    module docstring); ``0.0`` disables that test.

    Returns ``(mapping, groups)``: ``mapping`` sends every input label to the
    representative of its group (itself when nothing merged) and ``groups``
    lists the merged groups.
    """
    labels = sorted(vectors)
    if not labels:
        return {}, []

    matrix = np.asarray([vectors[label] for label in labels], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 0.0] = 1.0
    matrix = matrix / norms

    weight_of = np.asarray(
        [float((weights or {}).get(label, 0.0)) for label in labels], dtype=np.float64
    )
    # heaviest first, then alphabetically: the first cluster of a group wins
    order = sorted(range(len(labels)), key=lambda i: (-weight_of[i], labels[i]))
    labels = [labels[i] for i in order]
    matrix = matrix[order]
    weight_of = np.maximum(weight_of[order], 1e-6)
    size = len(labels)

    segments_of: Optional[List[np.ndarray]] = None
    if segment_vectors:
        segments_of = []
        for label in labels:
            rows = [v for v in segment_vectors.get(label, []) if v is not None]
            if not rows:
                segments_of.append(np.zeros((0, matrix.shape[1])))
                continue
            block = np.asarray(rows, dtype=np.float64)
            block_norms = np.linalg.norm(block, axis=1, keepdims=True)
            block_norms[block_norms <= 0.0] = 1.0
            segments_of.append(block / block_norms)

    def agrees(small: int, big: int) -> bool:
        """Do both clusters' own segments support the merged voice?"""
        if segment_floor <= 0.0 or segments_of is None:
            return True
        own = segments_of[small]
        other = segments_of[big]
        if own.shape[0] == 0 or other.shape[0] == 0:
            return False
        forward = float(np.mean(own @ matrix[big]))
        backward = float(np.mean(other @ matrix[small]))
        return forward >= segment_floor and backward >= segment_floor

    alive = np.ones(size, dtype=bool)
    parent = np.arange(size)

    def similarities() -> np.ndarray:
        table = matrix @ matrix.T
        np.fill_diagonal(table, -2.0)
        table[~alive, :] = -2.0
        table[:, ~alive] = -2.0
        return table

    def merge(small: int, big: int) -> None:
        total = weight_of[small] + weight_of[big]
        matrix[big] = normalise(
            (matrix[small] * weight_of[small] + matrix[big] * weight_of[big]) / total
        )
        weight_of[big] = total
        parent[small] = big
        alive[small] = False
        if segments_of is not None:
            base = segments_of[big]
            grown = segments_of[small]
            if base.shape[0] == 0:
                segments_of[big] = grown
            elif grown.shape[0]:
                segments_of[big] = np.vstack([base, grown])
            # keep the per-segment vote affordable on very long recordings
            if segments_of[big].shape[0] > SEGMENT_VOTE_CAP:
                step = segments_of[big].shape[0] // SEGMENT_VOTE_CAP
                segments_of[big] = segments_of[big][::step][:SEGMENT_VOTE_CAP]

    def apply_merge(small: int, big: int) -> None:
        merge(small, big)
        row = matrix @ matrix[big]
        table[big, :] = row
        table[:, big] = row
        np.fill_diagonal(table, -2.0)
        table[~alive, :] = -2.0
        table[:, ~alive] = -2.0

    table = similarities()

    if min_weight > 0.0:
        # tiny clusters are unreliable: hand them to the most similar survivor
        for small in [i for i in range(size) if weight_of[i] < min_weight]:
            if not alive[small]:
                continue
            row = table[small]
            target = int(np.argmax(row))
            if row[target] <= -2.0:
                continue
            keeper, loser = (small, target) if small < target else (target, small)
            apply_merge(loser, keeper)

    while True:
        order_list = np.argsort(table, axis=None)[::-1]
        chosen = None
        for flat in order_list:
            first, second = divmod(int(flat), size)
            if first == second or not alive[first] or not alive[second]:
                continue
            if table[first, second] < threshold:
                break  # everything left is below the threshold
            if agrees(min(first, second), max(first, second)):
                chosen = (first, second)
            # a rejected pair must not be retried forever: its similarity is
            # lowered just below the threshold so the scan moves on
            table[first, second] = table[second, first] = min(
                table[first, second], threshold - 1e-9
            )
            break
        if chosen is None:
            break
        first, second = chosen
        # cheaper clusters come first in the matrix, so the lower index is the
        # heavier cluster and it keeps the merged group
        keeper, loser = (first, second) if first < second else (second, first)
        apply_merge(loser, keeper)

    groups: Dict[int, List[str]] = {i: [labels[i]] for i in range(size) if alive[i]}
    for i in range(size):
        if alive[i]:
            continue
        root = i
        while parent[root] != root:
            root = parent[root]
        groups.setdefault(root, []).append(labels[i])

    mapping: Dict[str, str] = {}
    members: List[List[str]] = []
    for root in sorted(groups):
        group = sorted(groups[root])
        for label in group:
            mapping[label] = labels[root]
        members.append(group)
    return mapping, members


def apply_to_segments(
    segments: Sequence,
    clusters: Dict[str, Sequence[float]],
    per_segment: Optional[Dict[str, List[Sequence[float]]]] = None,
    threshold: float = 0.45,
    min_weight: float = 1.5,
    segment_floor: float = 0.0,
    report: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Merge duplicate speakers in place and return the refreshed vectors.

    ``segments`` are the pipeline's :class:`nestor.types.Segment` objects: the
    speaker of every segment inside a merged group is rewritten to the group's
    representative label. ``clusters`` / ``per_segment`` are the embedding maps
    keyed by the old labels; both are returned re-keyed by the new ones (means
    are recombined with their speech weights), so the later voiceprint matching
    and the cached ``.npz`` describe the merged speakers.
    """
    if not clusters:
        return clusters, per_segment or {}

    weights = {
        label: (segments and sum(
            max(0.0, seg.end - seg.start)
            for seg in segments
            if seg.speaker == label
        )) or 0.0
        for label in clusters
    }
    if not any(weights.values()):
        # no segment information: fall back to plain cluster counting
        weights = {label: 1.0 for label in clusters}

    mapping, members = merge_similar_clusters(
        clusters,
        threshold,
        weights=weights,
        min_weight=min_weight,
        segment_vectors=per_segment,
        segment_floor=segment_floor,
    )
    merged_groups = [group for group in members if len(group) > 1]
    if not merged_groups:
        return clusters, per_segment or {}

    for segment in segments:
        segment.speaker = mapping.get(segment.speaker, segment.speaker)
        for word in getattr(segment, "words", []) or []:
            word.speaker = segment.speaker

    def recombine(values: Dict[str, Sequence[float]], combine) -> Dict[str, list]:
        grouped: Dict[str, list] = {}
        for label, vector in values.items():
            if vector is None:
                continue
            grouped.setdefault(mapping.get(label, label), []).append((label, vector))
        out: Dict[str, list] = {}
        for representative, items in grouped.items():
            total = sum(weights.get(label, 1.0) for label, _ in items) or 1.0
            accumulated = None
            for label, vector in items:
                weighted = np.asarray(vector, dtype=np.float64) * weights.get(label, 1.0)
                accumulated = weighted if accumulated is None else accumulated + weighted
            if accumulated is not None:
                out[representative] = combine(accumulated / total)
        return out

    new_clusters = recombine(clusters, normalise)
    new_segments: Dict[str, list] = {}
    if per_segment:
        for label, rows in per_segment.items():
            if not rows:
                continue
            new_segments.setdefault(mapping.get(label, label), []).extend(rows)

    if report is not None:
        for group in sorted(merged_groups, key=lambda g: -len(g)):
            report(
                f"  merged speakers: {', '.join(group)} -> {mapping[group[0]]}"
            )
    return new_clusters, new_segments
