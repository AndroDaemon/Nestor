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
construction (their centroid comes from a handful of frames). They may still be
attached to another cluster, but never below ``threshold``: attaching a short
utterance to whoever happens to be closest is what folds different people into
one speaker, so the similarity test is never bypassed. A tiny cluster is also
only merged into a cluster that is itself substantial (``>= min_weight``),
otherwise it stays a speaker of its own. Raising ``min_weight`` therefore makes
the pass more cautious, not more aggressive.

When the individual segment embeddings are supplied (``segment_vectors``), a
candidate pair must also pass a floor test: the mean similarity between the
smaller cluster's own segments and the bigger cluster's centroid must reach
``segment_floor``. Two different people may have similarly positioned centroids
(both were recorded in the same room), but their individual utterances still sit
closer to their own voice — that is what keeps a real meeting from collapsing
into one speaker.

``min_coh`` (0.0 = off) is a stricter, opt-in guard on top of the floor: a pair
is only merged when the *merged* group's own segments still average at least
``min_coh`` cosine to the merged centroid. On long recordings average linkage
can chain a whole continuum of voices into one huge block; the guard refuses any
merge that would make the group internally inhomogeneous, trading a giant blob
for more, smaller speakers.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "merge_similar_clusters",
    "cosine",
    "normalise",
    "apply_to_segments",
    "attach_untimed_clusters",
    "group_cohesion",
    "split_inhomogeneous_clusters",
]

# A merged cluster keeps at most this many per-segment votes; long recordings
# are subsampled so the agreement test stays cheap.
SEGMENT_VOTE_CAP = 400

# --- auto-split of a too-large, inhomogeneous merged cluster -----------------
# Before average linkage is allowed to fold a big cluster together, the cluster
# is checked with a 2-means split. It is only cut when the two halves are each
# homogeneous AND clearly different voices; otherwise the barrier is too high and
# nothing changes. These numbers are deliberately strict: a wrong split is worse
# than a big cluster the user can cut by hand.
SPLIT_COHESION_FLOOR = 0.75   # every half's mean cosine to its own centroid
SPLIT_MAX_CENTROID_COS = 0.60  # the halves must be this unlike each other
SPLIT_MIN_HALF_SHARE = 0.05    # smallest half >= 5% of the cluster
SPLIT_KMEANS_SEED = 0          # fixed seed: deterministic split
SPLIT_KMEANS_ITERS = 25


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
    min_coh: float = 0.0,
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
        own = _drop_isolated(np.asarray(segments_of[small], dtype=np.float64))
        other = _drop_isolated(np.asarray(segments_of[big], dtype=np.float64))
        if own.shape[0] == 0 or other.shape[0] == 0:
            return False
        forward = float(np.mean(own @ matrix[big]))
        backward = float(np.mean(other @ matrix[small]))
        return forward >= segment_floor and backward >= segment_floor

    def merged_cohesion(small: int, big: int) -> float:
        """Mean cosine of the merged group's own segments to its centroid.

        ``1.0`` (i.e. "no objection") when there are no per-segment vectors to
        judge with - the strict-cohesion guard is only meaningful then.
        """
        if segments_of is None:
            return 1.0
        own = segments_of[small]
        other = segments_of[big]
        if own.shape[0] == 0 and other.shape[0] == 0:
            return 1.0
        if own.shape[0] == 0:
            rows = other
        elif other.shape[0] == 0:
            rows = own
        else:
            rows = np.vstack([own, other])
        if rows.shape[0] > SEGMENT_VOTE_CAP:
            step = rows.shape[0] // SEGMENT_VOTE_CAP
            rows = rows[::step][:SEGMENT_VOTE_CAP]
        total = weight_of[small] + weight_of[big]
        centroid = normalise(
            (matrix[small] * weight_of[small] + matrix[big] * weight_of[big]) / total
        )
        return float(np.mean(rows @ centroid))

    def cohesion_ok(small: int, big: int) -> bool:
        """Strict-cohesion guard (opt-in): is the merged group homogeneous?"""
        if min_coh <= 0.0:
            return True
        return merged_cohesion(small, big) >= min_coh

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
        # Tiny clusters are unreliable on their own, but "unreliable" must not
        # mean "merge with whoever is closest": folding a short utterance of one
        # person into another person's cluster is exactly the failure this pass
        # has to avoid. So a tiny cluster is attached only when it clears the
        # same similarity threshold as everyone else, and only to a cluster that
        # is itself substantial (>= min_weight). Otherwise it stays a speaker of
        # its own, which the user can merge by hand from the UI.
        for small in [i for i in range(size) if weight_of[i] < min_weight]:
            if not alive[small]:
                continue
            row = table[small]
            eligible = [
                i
                for i in range(size)
                if alive[i] and i != small and weight_of[i] >= min_weight
            ]
            if eligible:
                target = max(eligible, key=lambda i: float(row[i]))
            else:
                target = int(np.argmax(row))
            if target == small or row[target] < threshold or not cohesion_ok(small, target):
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
            if agrees(min(first, second), max(first, second)) and cohesion_ok(
                min(first, second), max(first, second)
            ):
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


SEGMENT_OUTLIER_COS = 0.30
"""Agreement below this means a segment is not this speaker at all."""

SEGMENT_OUTLIER_RATIO = 0.5
"""...and it must be far worse than the next one to be left out."""


def _drop_isolated(matrix: np.ndarray) -> np.ndarray:
    """Rows that agree with none of their neighbours in the same cluster.

    Agreement is the mean cosine to the other rows. Such a row is an
    interjection with no voice in it, and the cosine to the other cluster's
    centroid says nothing about the speaker - it only adds noise to the floor
    test. Only the single clearly isolated row is dropped, and never from a
    cluster of two, where there is no majority to trust.
    """
    count = int(matrix.shape[0])
    if count < 3:
        return matrix
    sims = matrix @ matrix.T
    agreement = (sims.sum(axis=1) - 1.0) / (count - 1)
    order = np.argsort(agreement)
    worst = int(order[0])
    if agreement[worst] >= SEGMENT_OUTLIER_COS:
        return matrix
    if agreement[worst] >= float(agreement[order[1]]) * SEGMENT_OUTLIER_RATIO:
        return matrix
    keep = [index for index in range(count) if index != worst]
    return matrix[keep]


def apply_to_segments(
    segments: Sequence,
    clusters: Dict[str, Sequence[float]],
    per_segment: Optional[Dict[str, List[Sequence[float]]]] = None,
    threshold: float = 0.45,
    min_weight: float = 1.5,
    segment_floor: float = 0.0,
    report: Optional[Callable[[str], None]] = None,
    min_coh: float = 0.0,
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
        min_coh=min_coh,
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


# ------------------------------------------------------- untimed cluster attach
# A label whose every utterance is shorter than the embedding minimum never
# gets a centroid, so the merge pass cannot see it and it survives as a
# one-utterance "speaker". On the real long recordings 12 of 13 such labels had
# a genuine speaker talking within three seconds, so the closest voice in time
# is attached instead of guessed by similarity (there is nothing to compare).
ATTACH_WINDOW_SEC = 3.0


def attach_untimed_clusters(
    segments: Sequence,
    clusters: Dict[str, Sequence[float]],
    per_segment: Optional[Dict[str, List[Sequence[float]]]] = None,
    window: float = ATTACH_WINDOW_SEC,
    report: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Fold labels that never got an embedding into their temporal neighbour.

    ``clusters`` holds a centroid only for labels with at least one utterance
    long enough to embed; a label missing from it is a "phantom" (an um, a
    cough, a half-word) that the similarity merge literally cannot see. Each
    such label is attached to the nearest other label in time, but only when
    the gap is at most ``window`` seconds - without a neighbour close enough the
    label is left untouched, because there is no evidence to place it.

    ``clusters`` / ``per_segment`` are returned unchanged: phantom labels never
    had vectors to recombine.
    """
    if not segments or not clusters or window <= 0.0:
        return clusters, per_segment or {}

    spans: Dict[str, List[Tuple[float, float]]] = {}
    for segment in segments:
        label = str(segment.speaker)
        spans.setdefault(label, []).append((float(segment.start), float(segment.end)))

    phantoms = [label for label in spans if label not in clusters]
    if not phantoms:
        return clusters, per_segment or {}

    # smallest first: a tiny fragment must not swallow a longer neighbour
    phantoms.sort(key=lambda label: sum(end - start for start, end in spans[label]))

    for label in phantoms:
        own = spans.get(label)
        if not own:
            continue
        low = min(start for start, _ in own)
        high = max(end for _, end in own)
        best_label = None
        best_gap = float("inf")
        for other, other_spans in spans.items():
            if other == label or other not in clusters:
                continue
            for start, end in other_spans:
                gap = max(0.0, max(start - high, low - end))
                if gap < best_gap:
                    best_gap = gap
                    best_label = other
        if best_label is None or best_gap > window:
            if report is not None:
                report(
                    f"  attach {label}: no speaker within {window:.0f}s - left alone"
                )
            continue
        for segment in segments:
            if str(segment.speaker) == label:
                segment.speaker = best_label
                for word in getattr(segment, "words", []) or []:
                    word.speaker = best_label
        spans[best_label].extend(own)
        spans.pop(label, None)
        if report is not None:
            report(f"  attached {label} -> {best_label} (gap {best_gap:.2f}s)")
    return clusters, per_segment or {}


# ---------------------------------------------------------- cluster cohesion
def _rows_matrix(rows: Sequence[Sequence[float]], dim: int) -> np.ndarray:
    """L2-normalised matrix of the given embedding rows (empty input -> 0xdim)."""
    clean = [np.asarray(row, dtype=np.float64) for row in rows if row is not None]
    if not clean:
        return np.zeros((0, dim))
    block = np.asarray(clean, dtype=np.float64)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    norms[norms <= 0.0] = 1.0
    return block / norms


def group_cohesion(rows: Sequence[Sequence[float]], dim: int) -> float:
    """Mean cosine of a group's own segment vectors to their centroid.

    ``1.0`` for an empty/one-element group. This is the number the strict-cohesion
    guard (``min_coh``) and the auto-split barriers both judge clusters by.
    """
    matrix = _rows_matrix(rows, dim)
    if matrix.shape[0] <= 1:
        return 1.0
    centroid = normalise(matrix.mean(axis=0))
    return float(np.mean(matrix @ centroid))


def _kmeans_two(matrix: np.ndarray, seed: int, iters: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic 2-means (cosine) on an L2-normalised matrix.

    Returns ``(labels, centroids)`` where ``labels[i]`` is 0/1 for row ``i`` and
    ``centroids`` holds the two L2-normalised centroids.
    """
    count = matrix.shape[0]
    rng = np.random.default_rng(seed)
    # k-means++ style seeding, but deterministic: pick one point, then the point
    # farthest from it, then refine.
    first = int(rng.integers(count))
    similarity = matrix @ matrix[first]
    second = int(np.argmin(similarity))
    if second == first:
        second = (first + 1) % count
    centroids = normalise(np.vstack([matrix[first], matrix[second]]))
    labels = np.zeros(count, dtype=int)
    for _ in range(iters):
        scores = matrix @ centroids.T  # (count, 2), cosine
        new_labels = np.argmax(scores, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for group in (0, 1):
            members = matrix[labels == group]
            if members.shape[0]:
                centroids[group] = normalise(members.mean(axis=0))
    return labels, centroids


def split_inhomogeneous_clusters(
    segments: Sequence,
    clusters: Dict[str, Sequence[float]],
    per_segment: Optional[Dict[str, List[Sequence[float]]]] = None,
    report: Optional[Callable[[str], None]] = None,
    min_segments: int = 50,
    min_segment_sec: float = 0.5,
) -> Tuple[Dict[str, Sequence[float]], Dict[str, List[Sequence[float]]]]:
    """Cut clusters that are large and clearly made of two voices.

    A cluster with at least ``min_segments`` segment embeddings is split when a
    fixed-seed 2-means finds two halves that are *both* homogeneous (each half's
    mean cosine to its own centroid ``>= SPLIT_COHESION_FLOOR``), *and* unlike
    each other (centroid cosine ``<= SPLIT_MAX_CENTROID_COS``), and the smaller
    half is at least ``SPLIT_MIN_HALF_SHARE`` of the cluster. Otherwise nothing
    changes - the barrier is intentionally high, so a single speaker with a wide
    range of voice is never carved up.

    ``per_segment`` lists each label's segment embeddings in the order
    :func:`nestor.speaker_embed.cluster_vectors` produced them: one per segment
    of that label whose duration reaches ``min_segment_sec`` (shorter segments
    carry no embedding). A cut rewrites the matching ``segment.speaker`` values;
    cut labels get a ``__SPLIT<k>`` suffix that the caller renumbers to
    sequential ``SPEAKER_NN`` afterwards. Every check (cut, and refused with the
    failing numbers) is reported through ``report``.
    """
    if not clusters or not per_segment or min_segments <= 0:
        return clusters, per_segment or {}

    def say(message: str) -> None:
        if report is not None:
            report(message)

    # label -> the segments that actually carry an embedding, in order
    carriers: Dict[str, List] = {}
    for segment in segments:
        label = segment.speaker
        if not per_segment.get(label):
            continue
        if max(0.0, float(segment.end) - float(segment.start)) < min_segment_sec:
            continue
        carriers.setdefault(label, []).append(segment)

    new_clusters: Dict[str, Sequence[float]] = {}
    new_per_segment: Dict[str, List[Sequence[float]]] = {}
    for label, vector in clusters.items():
        rows = [row for row in per_segment.get(label, []) if row is not None]
        if len(rows) < min_segments:
            new_clusters[label] = vector
            if rows:
                new_per_segment[label] = rows
            continue
        dim = len(vector)
        matrix = _rows_matrix(rows, dim)
        label_segments = carriers.get(label, [])
        if len(label_segments) != matrix.shape[0]:
            # Rows are matched to segments by position, so a mismatch (some
            # segments were dropped as noise) would mislabel the halves.
            new_clusters[label] = vector
            new_per_segment[label] = rows
            say(
                f"  split skipped {label}: {matrix.shape[0]} embeddings for "
                f"{len(label_segments)} segments"
            )
            continue
        labels, centroids = _kmeans_two(matrix, SPLIT_KMEANS_SEED, SPLIT_KMEANS_ITERS)
        halves = [matrix[labels == 0], matrix[labels == 1]]
        if not halves[0].shape[0] or not halves[1].shape[0]:
            new_clusters[label] = vector
            new_per_segment[label] = rows
            say(f"  split skipped {label}: empty half")
            continue
        cohesion_first = group_cohesion(halves[0], dim)
        cohesion_second = group_cohesion(halves[1], dim)
        centroid_cos = float(centroids[0] @ centroids[1])
        smaller = min(halves[0].shape[0], halves[1].shape[0])
        share = smaller / float(matrix.shape[0])
        reasons = []
        if min(cohesion_first, cohesion_second) < SPLIT_COHESION_FLOOR:
            reasons.append(
                f"halves too mixed (cohesion {cohesion_first:.2f}/{cohesion_second:.2f}"
                f" < {SPLIT_COHESION_FLOOR:.2f})"
            )
        if centroid_cos > SPLIT_MAX_CENTROID_COS:
            reasons.append(
                f"halves alike (centroid cos {centroid_cos:.2f}"
                f" > {SPLIT_MAX_CENTROID_COS:.2f})"
            )
        if share < SPLIT_MIN_HALF_SHARE:
            reasons.append(
                f"small half {share * 100:.1f}% < {SPLIT_MIN_HALF_SHARE * 100:.0f}%"
            )
        if reasons:
            new_clusters[label] = vector
            new_per_segment[label] = rows
            say(
                f"  split checked {label} ({matrix.shape[0]} segments): "
                f"not cut - " + "; ".join(reasons)
            )
            continue
        say(
            f"  split {label} ({matrix.shape[0]} segments) into 2: "
            f"sizes {halves[0].shape[0]}/{halves[1].shape[0]}, "
            f"cohesion {cohesion_first:.2f}/{cohesion_second:.2f}, "
            f"centroid cos {centroid_cos:.2f}"
        )
        # heaviest half keeps the original label; the other gets a suffix
        order = [0, 1] if halves[0].shape[0] >= halves[1].shape[0] else [1, 0]
        for index, group in enumerate(order):
            suffix = "" if index == 0 else f"__SPLIT{index}"
            new_label = label + suffix
            members = matrix[labels == group]
            if not members.shape[0]:
                continue
            # rewrite the segments that produced this half (same order as rows)
            # because cluster_vectors appended per-label embeddings in segment order
            mask = labels == group
            row_index = 0
            for segment in label_segments:
                if row_index >= mask.shape[0]:
                    break
                if mask[row_index]:
                    segment.speaker = new_label
                row_index += 1
            new_clusters[new_label] = normalise(members.mean(axis=0))
            new_per_segment[new_label] = [row for row in members]
    return new_clusters, new_per_segment or (per_segment or {})
