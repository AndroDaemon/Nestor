"""Embeddings must survive regions longer than the model's 12288-frame limit.

TitaNet-small dies with ``Where node ... 12288 by N`` as soon as one call gets
more than 122,88 s of audio. Long speaker turns (and the whole file, in the
``voiceprints add`` path) are exactly that, so regions are windowed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor import speaker_embed  # noqa: E402

SR = 16000


class _FakeStream:
    def __init__(self) -> None:
        self.samples = None
        self.rate = None

    def accept_waveform(self, rate, samples):
        self.rate = rate
        self.samples = np.asarray(samples, dtype=np.float32)

    def input_finished(self):
        pass


class _FakeExtractor:
    """Records the length of every chunk the real model would have received."""

    def __init__(self) -> None:
        self.lengths = []
        self.rates = []

    def create_stream(self):
        return _FakeStream()

    def compute(self, stream):
        self.lengths.append(int(stream.samples.shape[0]))
        self.rates.append(int(stream.rate))
        mean = float(np.mean(stream.samples)) if stream.samples.size else 0.0
        return np.array([mean, 1.0, 2.0], dtype=np.float32)


def test_short_region_goes_in_one_piece():
    extractor = _FakeExtractor()
    samples = np.full(SR * 5, 0.1, dtype=np.float32)

    vector = speaker_embed._embed_region(extractor, samples, SR)

    assert extractor.lengths == [SR * 5]
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6


def test_long_region_is_cut_into_windows():
    extractor = _FakeExtractor()
    samples = np.full(int(SR * 176.91), 0.1, dtype=np.float32)

    vector = speaker_embed._embed_region(extractor, samples, SR)

    window = int(speaker_embed.MAX_EMBED_SECONDS * SR)
    assert max(extractor.lengths) <= window
    assert len(extractor.lengths) == int(np.ceil(samples.shape[0] / window))
    # модель считает фреймы мел-спектрограммы (100 в секунду) и падает на 12288
    assert speaker_embed.MAX_EMBED_SECONDS * 100 < 12288
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6


def test_short_tail_is_dropped():
    extractor = _FakeExtractor()
    samples = np.zeros(int(SR * 40.1), dtype=np.float32)  # два полных окна + 0.1 с

    speaker_embed._embed_region(extractor, samples, SR)

    assert len(extractor.lengths) == 2


def test_windows_are_averaged():
    extractor = _FakeExtractor()
    window = int(speaker_embed.MAX_EMBED_SECONDS * SR)
    samples = np.concatenate(
        [np.full(window, 0.2, dtype=np.float32), np.full(window, 0.4, dtype=np.float32)]
    )

    vector = speaker_embed._embed_region(extractor, samples, SR)

    # среднее по окнам: (0.2, 1, 2) и (0.4, 1, 2) -> (0.3, 1, 2), затем L2
    raw = np.array([0.3, 1.0, 2.0], dtype=np.float32)
    expected = raw / np.linalg.norm(raw)
    assert np.allclose(vector, expected, atol=1e-6)


def test_embed_segments_window_long_segment(monkeypatch):
    extractor = _FakeExtractor()
    monkeypatch.setattr(speaker_embed, "_get_extractor", lambda: extractor)
    samples = np.full(int(SR * 200), 0.1, dtype=np.float32)

    vectors = speaker_embed.embed_segments(samples, SR, [(0.0, 200.0), (10.0, 10.2)])

    assert vectors[0] is not None
    assert vectors[1] is None  # короче min_duration
    window = int(speaker_embed.MAX_EMBED_SECONDS * SR)
    assert max(extractor.lengths) <= window


def test_embed_audio_accepts_a_whole_long_file(monkeypatch):
    extractor = _FakeExtractor()
    monkeypatch.setattr(speaker_embed, "_get_extractor", lambda: extractor)
    samples = np.full(int(SR * 600), 0.05, dtype=np.float32)

    vector = speaker_embed.embed_audio(samples, SR)

    window = int(speaker_embed.MAX_EMBED_SECONDS * SR)
    assert max(extractor.lengths) <= window
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6
