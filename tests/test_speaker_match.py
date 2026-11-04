"""Tests for the voiceprint matching rule (nestor/speaker_match.py).

Synthetic embeddings only - the real TitaNet model is never loaded here.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.speaker_match import (  # noqa: E402
    MatchThresholds,
    cosine,
    match_clusters,
)
from nestor.voiceprints import VoiceprintLibrary  # noqa: E402


def unit(seed: int, dim: int = 64) -> np.ndarray:
    """Deterministic unit vector."""
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def shifted(base: np.ndarray, noise_seed: int, amount: float = 0.1) -> np.ndarray:
    """A unit vector close to ``base`` (small deterministic perturbation)."""
    rng = np.random.default_rng(noise_seed)
    vector = base + amount * rng.standard_normal(base.shape[0]).astype(np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture()
def library(tmp_path):
    lib = VoiceprintLibrary(tmp_path)
    lib.add_or_update("Иван", unit(1), duration_sec=100.0)
    lib.add_or_update("Мария", unit(2), duration_sec=100.0)
    return lib


def test_confident_match(library):
    clusters = {"SPEAKER_00": shifted(unit(1), 10, amount=0.05)}
    segments = {"SPEAKER_00": [shifted(unit(1), 11, 0.05), shifted(unit(1), 12, 0.05)]}
    result = match_clusters(clusters, library, segment_embeddings=segments)

    assert result["SPEAKER_00"].assigned_name == "Иван"
    assert result["SPEAKER_00"].reason == "matched"
    assert result["SPEAKER_00"].confidence >= 0.65
    assert result["SPEAKER_00"].candidates[0][0] == "Иван"


def test_below_threshold_stays_unmatched(library):
    # a direction far from both stored voices
    far = unit(99)
    clusters = {"SPEAKER_00": far}
    result = match_clusters(clusters, library)
    assert result["SPEAKER_00"].assigned_name is None
    assert "below threshold" in result["SPEAKER_00"].reason


def test_tie_is_rejected_by_margin(library):
    # exactly between Ivan and Maria -> margin condition fails
    mid = unit(1) + unit(2)
    mid = (mid / np.linalg.norm(mid)).astype(np.float32)
    clusters = {"SPEAKER_00": mid}
    result = match_clusters(clusters, library)
    assert result["SPEAKER_00"].assigned_name is None
    assert "tie" in result["SPEAKER_00"].reason


def test_segment_disagreement_rejects(library):
    # the cluster average is close to Ivan, but half the segments vote Maria
    close_to_ivan = shifted(unit(1), 10, 0.02)
    maria = shifted(unit(2), 20, 0.02)
    clusters = {"SPEAKER_00": close_to_ivan}
    segments = {
        "SPEAKER_00": [
            shifted(unit(1), 11, 0.02),
            shifted(unit(1), 12, 0.02),
            maria,  # disagrees
            maria,  # disagrees  -> 2/4 = 50% < 60%
        ]
    }
    result = match_clusters(clusters, library, segment_embeddings=segments)
    assert result["SPEAKER_00"].assigned_name is None
    assert "segment agreement" in result["SPEAKER_00"].reason


def test_segment_agreement_passes_at_exactly_60_percent(library):
    close_to_ivan = shifted(unit(1), 10, 0.02)
    clusters = {"SPEAKER_00": close_to_ivan}
    segments = {
        "SPEAKER_00": [
            shifted(unit(1), 11, 0.02),
            shifted(unit(1), 12, 0.02),
            shifted(unit(1), 13, 0.02),
            shifted(unit(2), 14, 0.02),  # 3/4 = 75% >= 60% -> match
        ]
    }
    result = match_clusters(clusters, library, segment_embeddings=segments)
    assert result["SPEAKER_00"].assigned_name == "Иван"


def test_empty_library_matches_nothing(tmp_path):
    empty = VoiceprintLibrary(tmp_path)
    clusters = {"SPEAKER_00": unit(1)}
    result = match_clusters(clusters, empty)
    assert result["SPEAKER_00"].assigned_name is None
    assert result["SPEAKER_00"].reason == "empty library"
    # and with no library object at all
    result = match_clusters(clusters, None)
    assert result["SPEAKER_00"].assigned_name is None


def test_one_voice_serves_one_cluster(library):
    twin_a = shifted(unit(1), 10, 0.02)
    twin_b = shifted(unit(1), 15, 0.03)
    clusters = {"SPEAKER_00": twin_a, "SPEAKER_01": twin_b}
    result = match_clusters(clusters, library)
    winners = [
        label for label, res in result.items() if res.assigned_name == "Иван"
    ]
    assert len(winners) == 1  # the second cluster keeps SPEAKER_NN
    other = [label for label in result if label not in winners][0]
    assert "already assigned" in result[other].reason


def test_thresholds_are_configurable(library):
    clusters = {"SPEAKER_00": shifted(unit(1), 10, amount=0.3)}
    strict = MatchThresholds(abs_threshold=0.95)
    result = match_clusters(clusters, library, thresholds=strict)
    assert result["SPEAKER_00"].assigned_name is None

    loose = MatchThresholds(abs_threshold=0.5, margin=0.0)
    result = match_clusters(clusters, library, thresholds=loose)
    assert result["SPEAKER_00"].assigned_name == "Иван"


def test_roundtrip_dict():
    from nestor.speaker_match import MatchResult

    original = MatchResult(
        assigned_name="Иван",
        confidence=0.81,
        candidates=[("Иван", 0.81), ("Мария", 0.4)],
        reason="matched",
    )
    restored = MatchResult.from_dict(original.to_dict())
    assert restored.assigned_name == "Иван"
    assert restored.confidence == pytest.approx(0.81)
    assert restored.candidates[0] == ("Иван", 0.81)
    assert MatchResult.from_dict({"weird": 1}).assigned_name is None


def test_cosine_of_normalised_vectors():
    a = unit(1)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, -a) == pytest.approx(-1.0)
    assert cosine(a, np.zeros_like(a)) == 0.0
