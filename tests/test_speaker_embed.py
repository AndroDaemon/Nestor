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


# ---------------------------------------------------------------- robust mean
class _VoiceExtractor:
    """Returns a direction that depends on whether the chunk carries sound.

    Loud chunks map to ``(1, 0, 0)``, silent ones to ``(0, 1, 0)`` — i.e. the
    cosine between a voice and an interjection is zero, like in the real case
    where a "э-э, ну вот" segment got 0.05 against its own speaker.
    """

    def create_stream(self):
        return _FakeStream()

    def compute(self, stream):
        level = float(np.abs(stream.samples).mean()) if stream.samples.size else 0.0
        if level > 0.2:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


class _Piece:
    def __init__(self, start, end, speaker):
        self.start = start
        self.end = end
        self.speaker = speaker


def _blocks(*spec):
    return np.concatenate(
        [np.full(int(SR * length), level, dtype=np.float32) for length, level in spec]
    )


def test_segment_without_voice_is_left_out_of_the_centroid(monkeypatch):
    monkeypatch.setattr(speaker_embed, "_get_extractor", lambda: _VoiceExtractor())
    samples = _blocks((3.0, 0.5), (3.0, 0.5), (1.2, 0.0))
    segments = [_Piece(0, 3, "A"), _Piece(3, 6, "A"), _Piece(6, 7.2, "A")]

    clusters, per_segment = speaker_embed.cluster_vectors(samples, SR, segments)

    assert np.allclose(clusters["A"], [1.0, 0.0, 0.0], atol=1e-6)
    # строки остаются на месте: по позиции их сопоставляют с сегментами
    assert len(per_segment["A"]) == 3


def test_matching_segments_are_all_kept(monkeypatch):
    monkeypatch.setattr(speaker_embed, "_get_extractor", lambda: _VoiceExtractor())
    samples = _blocks((3.0, 0.5), (1.5, 0.6))
    segments = [_Piece(0, 3, "A"), _Piece(3, 4.5, "A")]

    clusters, per_segment = speaker_embed.cluster_vectors(samples, SR, segments)

    assert len(per_segment["A"]) == 2
    assert np.allclose(clusters["A"], [1.0, 0.0, 0.0], atol=1e-6)


def test_lone_segment_is_never_dropped(monkeypatch):
    monkeypatch.setattr(speaker_embed, "_get_extractor", lambda: _VoiceExtractor())
    samples = _blocks((1.2, 0.0))

    clusters, per_segment = speaker_embed.cluster_vectors(
        samples, SR, [_Piece(0, 1.2, "A")]
    )

    assert len(per_segment["A"]) == 1
    assert np.allclose(clusters["A"], [0.0, 1.0, 0.0], atol=1e-6)


def test_pairwise_agreement_drops_the_odd_one_out():
    voice = np.array([1.0, 0.0, 0.0])
    rows = [(voice, 3.0), (voice, 3.0), (np.array([0.0, 1.0, 0.0]), 1.2)]

    kept = speaker_embed._drop_outliers(rows)

    assert len(kept) == 2
    assert all(np.allclose(vector, voice) for vector, _ in kept)


def test_outlier_filter_keeps_a_cluster_with_no_majority():
    # Все трое одинаково далеки друг от друга — «worst» ничем не выделяется.
    rows = [
        (np.array([1.0, 0.0, 0.0]), 1.0),
        (np.array([0.0, 1.0, 0.0]), 1.0),
        (np.array([0.0, 0.0, 1.0]), 1.0),
    ]

    assert len(speaker_embed._drop_outliers(rows)) == 3


def test_only_the_odd_segment_is_dropped():
    # Векторы из реального случая SPEAKER_18: два слабо похожих настоящих
    # сегмента (0.23 и 0.33) и вставка «э-э, ну вот» (0.08).
    rows = [
        (np.array([1.0, 0.0, 0.0]), 1.2),
        (np.array([0.477, 0.879, 0.0]), 2.84),
        (np.array([-0.0166, 0.1759, 0.9844]), 1.2),
    ]

    kept = speaker_embed._drop_outliers(rows)

    assert len(kept) == 2
    assert all(not np.allclose(pair[0], [-0.0166, 0.1759, 0.9844]) for pair in kept)


def test_two_segments_are_never_filtered():
    # два вектора «согласны» друг с другом одинаково — большинства нет
    rows = [
        (np.array([1.0, 0.0, 0.0]), 1.0),
        (np.array([0.0, 1.0, 0.0]), 1.0),
    ]

    assert len(speaker_embed._drop_outliers(rows)) == 2


def test_segment_clipped_by_the_end_of_the_file_still_gets_a_vector(monkeypatch):
    extractor = _FakeExtractor()
    monkeypatch.setattr(speaker_embed, "_get_extractor", lambda: extractor)
    samples = np.full(int(SR * 10), 0.1, dtype=np.float32)

    vectors = speaker_embed.embed_segments(samples, SR, [(9.4, 10.4)])

    assert vectors[0] is not None
