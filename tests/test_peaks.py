"""Waveform points for the player: the count must follow the recording length."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.web.app import compute_peaks, peaks_bucket_count  # noqa: E402

SR = 16000


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0.0, 1200),
        (-5.0, 1200),
        (60.0, 1200),
        (300.0, 1200),
        (600.0, 2400),
        (3600.0, 14400),
        (3926.9, 15708),
        (6000.0, 24000),
        (18000.0, 72000),
        (86400.0, 288000),
    ],
)
def test_bucket_count_follows_duration(duration: float, expected: int):
    assert peaks_bucket_count(duration) == expected


def test_bucket_count_is_never_coarser_than_quarter_second():
    """The old fixed 1200 points gave 3.2 s per point on a one-hour take."""
    for duration in (600.0, 3600.0, 7120.0, 18000.0):
        assert duration / peaks_bucket_count(duration) <= 0.25 + 1e-9


def test_compute_peaks_uses_duration_by_default():
    duration = 700.0
    waveform = np.sin(2 * np.pi * 220 * np.arange(int(SR * duration)) / SR).astype(
        np.float32
    )

    peaks = compute_peaks(waveform)

    assert len(peaks) == peaks_bucket_count(duration)
    assert max(peaks) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in peaks)


def test_compute_peaks_respects_explicit_bucket_count():
    rng = np.random.default_rng(0)
    waveform = rng.normal(0, 0.2, size=SR).astype(np.float32)

    assert len(compute_peaks(waveform, buckets=100)) == 100
    assert len(compute_peaks(waveform, buckets=1)) == 1


def test_compute_peaks_empty_waveform():
    assert compute_peaks(np.zeros(0, dtype=np.float32)) == []


def test_compute_peaks_chunks_without_changing_result(monkeypatch):
    """Blocked passes must give exactly the single-pass answer."""
    import nestor.web.app as app_module

    rng = np.random.default_rng(7)
    waveform = rng.normal(0, 0.3, size=50_000).astype(np.float32)
    buckets = 777
    edges = np.linspace(0, waveform.size, buckets + 1).astype(np.int64)
    naive = np.maximum.reduceat(np.abs(waveform), edges[:-1]).astype(np.float32)
    expected = [round(float(value / naive.max()), 4) for value in naive]

    monkeypatch.setattr(app_module, "_PEAKS_BLOCK", 512)

    assert compute_peaks(waveform, buckets=buckets) == expected
