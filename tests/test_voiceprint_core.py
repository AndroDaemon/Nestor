"""Tests for the robust voiceprint core and explicit samples.

A saved voice must not be dragged towards a noisy or mis-assigned line that
happened to land in the same cluster, so the portrait is the mean of the
utterances closest to their own centroid (``_robust_core_mean``). When the user
explicitly picks the samples (``segment_ids``) that choice is trusted as is and
the core filter is skipped.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nestor.web.app as app_module  # noqa: E402
from nestor.library import Library, Recording, Segment  # noqa: E402
from nestor.speaker_embed import save_record_embeddings  # noqa: E402


def unit(angle_deg: float, size: int = 8) -> np.ndarray:
    """A unit vector rotated by ``angle_deg`` from the x axis."""
    angle = math.radians(angle_deg)
    vector = np.zeros(size, dtype=np.float64)
    vector[0] = math.cos(angle)
    vector[1] = math.sin(angle)
    return vector / np.linalg.norm(vector)


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _endpoint(app, method: str, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", ()):
            return route.endpoint
    raise AssertionError(f"route {method} {path} not found")


# ------------------------------------------------------------ robust core mean
def test_robust_core_mean_drops_the_outliers():
    """Eight utterances of one voice plus two stray ones: the strays are cut."""
    vectors = [unit(0) for _ in range(8)] + [unit(90), unit(-90)]
    core, kept, total = app_module._robust_core_mean(vectors)
    assert total == 10
    assert kept == max(app_module.VP_CORE_MIN, round(10 * app_module.VP_CORE_SHARE))
    assert cos(core, unit(0)) > 0.999
    assert abs(cos(core, unit(90))) < 0.05


def test_robust_core_mean_keeps_a_tight_voice_whole():
    """A perfectly consistent speaker is returned unchanged."""
    vectors = [unit(30) for _ in range(5)]
    core, kept, total = app_module._robust_core_mean(vectors)
    assert total == 5
    assert kept == max(app_module.VP_CORE_MIN, round(5 * app_module.VP_CORE_SHARE))
    assert cos(core, unit(30)) > 0.999


def test_robust_core_mean_single_and_empty():
    core, kept, total = app_module._robust_core_mean([unit(12)])
    assert (kept, total) == (1, 1)
    assert cos(core, unit(12)) > 0.999
    assert app_module._robust_core_mean([]) == (None, 0, 0)
    assert app_module._robust_core_mean([None, None]) == (None, 0, 0)


# ------------------------------------------------------------- explicit samples
def _cache_two_voices(library: Library) -> Recording:
    """One speaker with three clean lines and one stray line; cached rows."""
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [
        Segment("s1", 0.0, 2.0, "SPEAKER_00", "а"),
        Segment("s2", 2.0, 4.0, "SPEAKER_00", "б"),
        Segment("s3", 4.0, 6.0, "SPEAKER_00", "в"),
        Segment("s4", 6.0, 8.0, "SPEAKER_00", "г"),
    ]
    library.save(recording)
    rows = [unit(0), unit(0), unit(0), unit(90)]
    centroid = np.mean([np.asarray(r) for r in rows], axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    save_record_embeddings(
        library.embeddings_dir,
        "rec1",
        {"SPEAKER_00": centroid},
        {"SPEAKER_00": rows},
    )
    return recording


def test_explicit_samples_are_used_as_is(tmp_path):
    """A chosen stray line is kept verbatim; the core filter is not applied."""
    library = Library(tmp_path)
    recording = _cache_two_voices(library)

    stray, reason = app_module._voiceprint_vector(
        library, "rec1", recording, "SPEAKER_00", ["s4"]
    )
    assert reason is None
    assert cos(stray, unit(90)) > 0.999

    clean, _ = app_module._voiceprint_vector(
        library, "rec1", recording, "SPEAKER_00", ["s1", "s2"]
    )
    assert cos(clean, unit(0)) > 0.999


def test_default_keeps_the_robust_core(tmp_path):
    """Without a choice the stray line is dropped by the core filter."""
    library = Library(tmp_path)
    recording = _cache_two_voices(library)
    core, reason = app_module._voiceprint_vector(library, "rec1", recording, "SPEAKER_00")
    assert reason is None
    assert cos(core, unit(0)) > 0.999


def test_unknown_sample_falls_back_to_the_whole_cluster(tmp_path):
    """A segment_ids list with nothing usable must not lose the voice."""
    library = Library(tmp_path)
    recording = _cache_two_voices(library)
    core, reason = app_module._voiceprint_vector(
        library, "rec1", recording, "SPEAKER_00", ["missing-id"]
    )
    assert reason is None
    assert core is not None
    assert cos(core, unit(0)) > 0.999


# ------------------------------------------------------------------ e2e over API
def test_voiceprint_endpoint_honours_segment_ids(tmp_path, monkeypatch):
    """POST /api/voiceprints with segment_ids saves only the chosen sample."""
    library = Library(tmp_path)
    recording = _cache_two_voices(library)
    recording.speaker_names = {"SPEAKER_00": "Иван"}
    library.save(recording)
    app = app_module.create_app(work_dir=tmp_path)

    called = {"n": 0}

    def failing_decode(path, on_progress=None):
        called["n"] += 1
        raise AssertionError("the cache is valid, decode must not run")

    monkeypatch.setattr(app_module, "decode", failing_decode)
    add = _endpoint(app, "POST", "/api/voiceprints")

    class Body:
        async def json(self):
            return {
                "name": "Иван",
                "recording_id": "rec1",
                "speaker": "SPEAKER_00",
                "segment_ids": ["s4"],
            }

    result = asyncio.run(add(request=Body()))
    assert result["saved"] == "Иван"
    assert called["n"] == 0

    from nestor.voiceprints import VoiceprintLibrary

    vectors = VoiceprintLibrary(tmp_path).vectors()
    saved = next(iter(vectors.values()))
    assert cos(saved, unit(90)) > 0.999
