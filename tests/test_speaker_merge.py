"""Tests for the post-diarization cluster merge (``nestor/speaker_merge.py``).

The merge must fold duplicate clusters without collapsing genuinely different
voices; the cases below pin both directions plus the guard rails (tiny
clusters, per-segment floor, determinism).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nestor.speaker_merge import cosine, merge_similar_clusters, normalise


def unit(angle_deg: float, size: int = 16):
    """A unit vector rotated by ``angle_deg`` from the x axis."""
    angle = math.radians(angle_deg)
    vector = np.zeros(size, dtype=np.float64)
    vector[0] = math.cos(angle)
    vector[1] = math.sin(angle)
    return normalise(vector)


def similarity(a, b) -> float:
    return cosine(a, b)


# --------------------------------------------------------------------- basics
def test_normalise_and_cosine_edge_cases():
    assert list(normalise([0.0, 0.0, 0.0])) == [0.0, 0.0, 0.0]
    assert list(normalise([3.0, 4.0])) == pytest.approx([0.6, 0.8])
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_empty_input():
    assert merge_similar_clusters({}, 0.5) == ({}, [])


def test_single_cluster_is_returned_untouched():
    mapping, groups = merge_similar_clusters({"SPEAKER_00": unit(0)}, 0.5)
    assert mapping == {"SPEAKER_00": "SPEAKER_00"}
    assert groups == [["SPEAKER_00"]]


# ------------------------------------------------------------------ thresholds
def test_similar_clusters_merge_above_threshold():
    vectors = {"SPEAKER_00": unit(0), "SPEAKER_01": unit(20)}  # cosine ~0.94
    mapping, groups = merge_similar_clusters(vectors, 0.9)
    assert len(groups) == 1
    assert set(groups[0]) == {"SPEAKER_00", "SPEAKER_01"}
    assert mapping["SPEAKER_01"] in {"SPEAKER_00", "SPEAKER_01"}
    assert mapping[mapping["SPEAKER_01"]] == mapping["SPEAKER_01"]


def test_similar_clusters_stay_apart_below_threshold():
    vectors = {"SPEAKER_00": unit(0), "SPEAKER_01": unit(20)}  # cosine ~0.94
    mapping, groups = merge_similar_clusters(vectors, 0.99)
    assert len(groups) == 2
    assert mapping == {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01"}


def test_different_voices_never_merge():
    vectors = {"SPEAKER_00": unit(0), "SPEAKER_01": unit(75)}  # cosine ~0.26
    mapping, groups = merge_similar_clusters(vectors, 0.5)
    assert len(groups) == 2
    assert mapping["SPEAKER_00"] == "SPEAKER_00"
    assert mapping["SPEAKER_01"] == "SPEAKER_01"


def test_merge_requires_a_pair_not_a_chain_of_weak_links():
    # A-B and B-C are close, A-C is not: the greedy best-pair order merges the
    # strongest pair first and stops relying on the weak link.
    vectors = {
        "SPEAKER_00": unit(0),
        "SPEAKER_01": unit(15),   # A-B cosine ~0.97
        "SPEAKER_02": unit(55),   # B-C ~0.71, A-C ~0.57
    }
    mapping, groups = merge_similar_clusters(vectors, 0.9)
    assert len(groups) == 2
    assert set(mapping) == set(vectors)
    assert len({mapping[label] for label in vectors}) == 2


def test_threshold_is_inclusive():
    vector_a, vector_b = unit(0), unit(30)  # cosine 0.866
    vectors = {"A": vector_a, "B": vector_b}
    assert len(merge_similar_clusters(vectors, 0.866)[1]) == 1
    assert len(merge_similar_clusters(vectors, 0.87)[1]) == 2


# -------------------------------------------------------------- representative
def test_heavier_cluster_represents_the_group():
    vectors = {"SPEAKER_00": unit(0), "SPEAKER_01": unit(10)}
    weights = {"SPEAKER_00": 1.0, "SPEAKER_01": 50.0}
    mapping, groups = merge_similar_clusters(vectors, 0.9, weights=weights)
    assert len(groups) == 1
    assert mapping["SPEAKER_00"] == "SPEAKER_01"
    assert mapping["SPEAKER_01"] == "SPEAKER_01"


def test_ties_are_broken_alphabetically_and_deterministically():
    vectors = {"SPEAKER_01": unit(0), "SPEAKER_00": unit(10)}
    weights = {"SPEAKER_00": 5.0, "SPEAKER_01": 5.0}
    first = merge_similar_clusters(vectors, 0.9, weights=weights)
    second = merge_similar_clusters(vectors, 0.9, weights=weights)
    assert first == second
    # equal weights: the alphabetically first label keeps the group
    assert first[0]["SPEAKER_01"] == "SPEAKER_00"


def test_mapping_covers_every_label_and_is_idempotent():
    vectors = {
        "SPEAKER_00": unit(0),
        "SPEAKER_01": unit(5),
        "SPEAKER_02": unit(80),
        "SPEAKER_03": unit(-5),
    }
    mapping, groups = merge_similar_clusters(vectors, 0.9)
    assert set(mapping) == set(vectors)
    assert sum(len(group) for group in groups) == len(vectors)
    for label, representative in mapping.items():
        assert mapping[representative] == representative


# ----------------------------------------------------------------- min_weight
def test_min_weight_absorbs_tiny_clusters_regardless_of_threshold():
    vectors = {"SPEAKER_00": unit(0), "SPEAKER_01": unit(70)}  # cosine ~0.34
    weights = {"SPEAKER_00": 60.0, "SPEAKER_01": 0.7}
    mapping, groups = merge_similar_clusters(
        vectors, 0.95, weights=weights, min_weight=1.0
    )
    assert len(groups) == 1
    assert mapping["SPEAKER_01"] == "SPEAKER_00"


def test_min_weight_spares_clusters_with_enough_speech():
    vectors = {"SPEAKER_00": unit(0), "SPEAKER_01": unit(70)}
    weights = {"SPEAKER_00": 60.0, "SPEAKER_01": 5.0}
    mapping, groups = merge_similar_clusters(
        vectors, 0.95, weights=weights, min_weight=1.0
    )
    assert len(groups) == 2


# ------------------------------------------------------------- segment_floor
def test_segment_floor_can_reject_a_pair_whose_parts_disagree():
    # centroids are close (0.94), but the smaller cluster's own fragments are
    # much closer to their own centroid than to the other one
    vectors = {"SPEAKER_00": unit(0), "SPEAKER_01": unit(20)}
    segments = {
        "SPEAKER_00": [unit(0) for _ in range(10)],
        "SPEAKER_01": [unit(60) for _ in range(10)],
    }
    # mean(SPEAKER_01 fragments · SPEAKER_00 centroid) = cos(60°) = 0.5
    mapping, groups = merge_similar_clusters(
        vectors, 0.9, segment_vectors=segments, segment_floor=0.8
    )
    assert len(groups) == 2
    # the same pair passes when the floor is low enough
    mapping, groups = merge_similar_clusters(
        vectors, 0.9, segment_vectors=segments, segment_floor=0.4
    )
    assert len(groups) == 1


def test_segment_floor_uses_accumulated_member_segments():
    vectors = {"SPEAKER_00": unit(0), "SPEAKER_01": unit(10)}
    segments = {
        "SPEAKER_00": [unit(5)],
        "SPEAKER_01": [unit(10)],
    }
    mapping, groups = merge_similar_clusters(
        vectors, 0.9, segment_vectors=segments, segment_floor=0.9
    )
    assert len(groups) == 1


def test_cluster_without_segments_fails_the_floor():
    vectors = {"SPEAKER_00": unit(0), "SPEAKER_01": unit(10)}
    segments = {"SPEAKER_00": [unit(0)], "SPEAKER_01": []}
    mapping, groups = merge_similar_clusters(
        vectors, 0.9, segment_vectors=segments, segment_floor=0.5
    )
    assert len(groups) == 2


# ----------------------------------------------------------------- real-world
def test_lecture_like_many_duplicate_clusters_collapse():
    """Many slightly rotated copies of one voice plus two outliers."""
    vectors = {f"SPEAKER_{i:02d}": unit(i) for i in range(0, 40, 2)}  # 0..38 deg
    vectors["SPEAKER_90"] = unit(90)
    vectors["SPEAKER_95"] = unit(-95)
    mapping, groups = merge_similar_clusters(vectors, 0.55, min_weight=0.0)
    representatives = set(mapping.values())
    assert len(representatives) < len(vectors)
    # the two clearly different voices stay their own speakers
    assert len({mapping["SPEAKER_90"], mapping["SPEAKER_95"]}) == 2
    assert all(
        group and len(group) >= 1 for group in groups
    ), "every group lists at least its representative"


def test_many_clusters_scale_without_blowing_up():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(12, 24))
    vectors = {}
    for index in range(180):
        vectors[f"SPEAKER_{index:03d}"] = base[index % 12] + rng.normal(size=24) * 0.05
    weights = {label: 1.0 + (index % 7) for index, label in enumerate(sorted(vectors))}
    mapping, groups = merge_similar_clusters(
        vectors, 0.5, weights=weights, min_weight=1.5
    )
    assert set(mapping) == set(vectors)
    assert len(groups) <= 30
    assert sum(len(group) for group in groups) == len(vectors)
