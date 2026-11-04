"""Matching diarization clusters against the voiceprint library.

The decision rule ("rule of three") — a cluster gets a name only when ALL of:

1. its cosine similarity to the best voiceprint reaches ``abs_threshold``;
2. it leads the runner-up by at least ``margin`` (no ties);
3. at least ``segment_agreement`` of its segments (per-segment embeddings)
   agree on the same winner — the cluster average alone can be misleading.

Anything else keeps the raw ``SPEAKER_NN`` label: no guessing. One voiceprint
cannot be assigned to two clusters in the same run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MatchThresholds:
    """Thresholds of the matching rule (defaults live in config.py)."""

    abs_threshold: float = 0.75
    margin: float = 0.05
    segment_agreement: float = 0.60


@dataclass
class MatchResult:
    """Outcome of matching one cluster against the library."""

    assigned_name: Optional[str] = None
    confidence: float = 0.0
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "assigned_name": self.assigned_name,
            "confidence": round(self.confidence, 4),
            "candidates": [
                [name, round(score, 4)] for name, score in self.candidates
            ],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MatchResult":
        try:
            return cls(
                assigned_name=data.get("assigned_name"),
                confidence=float(data.get("confidence", 0.0)),
                candidates=[
                    (str(name), float(score))
                    for name, score in data.get("candidates", [])
                ],
                reason=str(data.get("reason", "")),
            )
        except (TypeError, ValueError):
            return cls()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (assumes L2-normalised inputs)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _segment_winner(
    segment_vector: np.ndarray,
    entries: Sequence[Tuple[str, str, np.ndarray]],
) -> Optional[Tuple[str, str, float]]:
    """Best (slug, name, score) for one segment vector, or None if no voices."""
    best: Optional[Tuple[str, str, float]] = None
    for slug, name, vector in entries:
        score = cosine(segment_vector, vector)
        if best is None or score > best[2]:
            best = (slug, name, score)
    return best


def match_clusters(
    cluster_embeddings: Dict[str, np.ndarray],
    library,
    thresholds: Optional[MatchThresholds] = None,
    segment_embeddings: Optional[Dict[str, List[np.ndarray]]] = None,
) -> Dict[str, MatchResult]:
    """Match every cluster embedding against the library.

    Parameters
    ----------
    cluster_embeddings:
        Speaker label -> cluster mean embedding (L2-normalised).
    library:
        :class:`~nestor.voiceprints.VoiceprintLibrary`.
    thresholds:
        The three thresholds of the rule.
    segment_embeddings:
        Speaker label -> per-segment embeddings (L2-normalised). Used for the
        agreement condition; when absent, only the average is checked.

    Returns
    -------
    A result per speaker label; ``assigned_name`` is ``None`` whenever the
    rule did not pass (including an empty library).
    """
    thresholds = thresholds or MatchThresholds()
    results: Dict[str, MatchResult] = {}
    voiceprints = library.vectors() if library is not None else {}
    names = library.names() if library is not None else {}
    entries: List[Tuple[str, str, np.ndarray]] = [
        (slug, names.get(slug, slug), vector) for slug, vector in voiceprints.items()
    ]

    taken: Dict[str, str] = {}  # slug -> label that already claimed the voice
    for label in sorted(cluster_embeddings):
        cluster = np.asarray(cluster_embeddings[label], dtype=np.float32).reshape(-1)
        candidates = sorted(
            ((name, cosine(cluster, vector)) for _, name, vector in entries),
            key=lambda item: item[1],
            reverse=True,
        )
        result = MatchResult(candidates=candidates[:3])

        if not entries:
            result.reason = "empty library"
            results[label] = result
            continue

        best_name, best_score = candidates[0]
        best_slug = next(
            slug for slug, name, _ in entries if name == best_name
        )
        result.confidence = best_score

        # rule 1: absolute threshold
        if best_score < thresholds.abs_threshold:
            result.reason = (
                f"below threshold ({best_score:.2f} < {thresholds.abs_threshold:.2f})"
            )
            results[label] = result
            continue

        # rule 2: margin over the runner-up
        if len(candidates) > 1:
            runner_up = candidates[1][1]
            if best_score - runner_up < thresholds.margin:
                result.reason = (
                    f"tie ({best_score:.2f} vs {runner_up:.2f}, "
                    f"margin {thresholds.margin:.2f})"
                )
                results[label] = result
                continue

        # rule 3: per-segment agreement
        agreeing = segment_embeddings.get(label) if segment_embeddings else None
        if agreeing:
            votes: Dict[str, int] = {}
            for vector in agreeing:
                winner = _segment_winner(vector, entries)
                if winner is None:
                    continue
                votes[winner[0]] = votes.get(winner[0], 0) + 1
            if votes:
                required = max(1, math.ceil(len(agreeing) * thresholds.segment_agreement))
                winner_slug = max(votes, key=votes.get)
                if votes[winner_slug] < required or winner_slug != best_slug:
                    result.reason = (
                        f"segment agreement {votes.get(best_slug, 0)}/{len(agreeing)} "
                        f"< {thresholds.segment_agreement:.0%}"
                    )
                    results[label] = result
                    continue

        # one voiceprint serves one cluster per run
        if best_slug in taken and taken[best_slug] != label:
            result.reason = f"voice already assigned to {taken[best_slug]}"
            results[label] = result
            continue

        taken[best_slug] = label
        result.assigned_name = best_name
        result.reason = "matched"
        results[label] = result

    return results
