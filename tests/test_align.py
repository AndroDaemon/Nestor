"""Unit tests for the alignment logic (no heavy dependencies required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.align import (  # noqa: E402
    assign_speakers,
    build_chunks,
    merge_words,
    relabel_speakers,
    speaker_order,
)
from nestor.types import Segment, Turn, Word  # noqa: E402


def test_build_chunks_merges_close_turns():
    turns = [Turn(0.0, 5.0, "A"), Turn(5.2, 9.0, "B")]
    chunks = build_chunks(turns, max_chunk=20.0, max_gap=0.5)
    assert chunks == [(0.0, 9.0)]


def test_build_chunks_splits_long_turns():
    turns = [Turn(0.0, 50.0, "A")]
    chunks = build_chunks(turns, max_chunk=20.0)
    assert len(chunks) == 3
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == 50.0


def test_build_chunks_keeps_distant_turns_apart():
    turns = [Turn(0.0, 2.0, "A"), Turn(30.0, 33.0, "A")]
    chunks = build_chunks(turns, max_chunk=20.0, max_gap=0.5)
    assert chunks == [(0.0, 2.0), (30.0, 33.0)]


def test_assign_speakers_uses_overlap():
    words = [Word("привет", 1.0, 1.4), Word("мир", 11.0, 11.3)]
    turns = [Turn(0.0, 5.0, "A"), Turn(10.0, 15.0, "B")]
    assigned = assign_speakers(words, turns)
    assert assigned[0].speaker == "A"
    assert assigned[1].speaker == "B"


def test_assign_speakers_falls_back_to_nearest_turn():
    words = [Word("эхо", 7.0, 7.2)]
    turns = [Turn(0.0, 5.0, "A"), Turn(10.0, 15.0, "B")]
    assigned = assign_speakers(words, turns)
    assert assigned[0].speaker in {"A", "B"}


def test_build_chunks_keeps_short_utterances():
    """A short word ("Привет", ~0.4 s) must not be dropped."""
    turns = [Turn(0.2, 0.66, "A"), Turn(1.7, 3.0, "A")]
    chunks = build_chunks(turns, max_chunk=20.0, max_gap=0.5)
    assert len(chunks) == 2
    assert chunks[0] == (0.2, 0.66)


def test_build_chunks_drops_noise():
    turns = [Turn(0.0, 0.05, "A"), Turn(1.0, 3.0, "A")]
    chunks = build_chunks(turns, max_chunk=20.0, max_gap=0.5, min_chunk=0.2)
    assert chunks == [(1.0, 3.0)]


def test_merge_words_groups_by_speaker():
    words = [
        Word("привет", 0.0, 0.5, "A"),
        Word("мир", 0.6, 1.0, "A"),
        Word("да", 5.0, 5.2, "B"),
    ]
    segments = merge_words(words, max_gap=1.0)
    assert len(segments) == 2
    assert segments[0].text == "привет мир"
    assert segments[0].speaker == "A"
    assert segments[1].speaker == "B"


def test_relabel_speakers():
    segments = [
        Segment(0.0, 1.0, "SPEAKER_00", "a"),
        Segment(1.0, 2.0, "SPEAKER_01", "b"),
    ]
    relabel_speakers(segments, ["Иван", "Мария"])
    assert segments[0].speaker == "Иван"
    assert segments[1].speaker == "Мария"
    assert speaker_order(segments) == ["Иван", "Мария"]


def test_renumber_speakers_orders_by_first_appearance():
    from nestor.align import renumber_speakers
    from nestor.types import Segment, Word

    segments = [
        Segment(30.0, 32.0, "SPEAKER_71", "второй", [Word("в", 30.0, 32.0)]),
        Segment(0.0, 2.0, "SPEAKER_296", "первый", [Word("п", 0.0, 2.0)]),
        Segment(34.0, 36.0, "SPEAKER_71", "второй", [Word("в", 34.0, 36.0)]),
    ]
    mapping = renumber_speakers(segments)
    assert mapping == {"SPEAKER_71": "SPEAKER_00", "SPEAKER_296": "SPEAKER_01"}
    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [w.speaker for s in segments for w in s.words] == [
        "SPEAKER_00", "SPEAKER_01", "SPEAKER_00",
    ]


def test_renumber_speakers_handles_library_segments():
    """Library segments have no .words - renumbering must still work."""
    from dataclasses import dataclass

    from nestor.align import renumber_speakers

    @dataclass
    class LibSegment:
        start: float
        end: float
        speaker: str
        text: str

    segments = [LibSegment(5.0, 6.0, "SPEAKER_71", "б"), LibSegment(0.0, 1.0, "SPEAKER_296", "а")]
    mapping = renumber_speakers(segments)
    assert mapping == {"SPEAKER_71": "SPEAKER_00", "SPEAKER_296": "SPEAKER_01"}
    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
