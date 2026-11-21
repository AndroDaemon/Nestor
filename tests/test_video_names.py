"""Tests for the roster read off video frames (nestor/video_names.py).

Only the pure text logic is exercised: no ONNX session is created and no video
is decoded, so these tests run anywhere.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.ocr import TextLine  # noqa: E402
from nestor.video_names import (  # noqa: E402
    _boxes_confined_to_bottom,
    _drop_fragments,
    _name_candidates,
    _same_person,
    collect_names,
    names_from_video,
    rank_names,
)


def line(text: str, confidence: float = 0.95, y: int = 0, height: int = 12) -> TextLine:
    return TextLine(text=text, confidence=confidence, x=0, y=y, width=300, height=height)


def tally():
    return Counter(), Counter(), Counter()


# --------------------------------------------------------------- candidates
def test_clean_plate_is_one_whole_name():
    candidates = _name_candidates("Шаталов Дмитрий Олегович")
    assert candidates == [("Шаталов Дмитрий Олегович", True)]


def test_parenthesis_status_is_removed():
    # "Ирина (Протей)" is a plate with a status tag; the tag must not break it.
    assert _name_candidates("Анна Вайденталь (Протей)") == [("Анна Вайденталь", True)]


def test_glued_plates_are_split_on_capitals():
    candidates = _name_candidates(
        "Александрова Юлия ПетровнаИванов Борис Александрович"
    )
    names = [name for name, _ in candidates]
    assert "Александрова Юлия Петровна" in names
    assert "Иванов Борис Александрович" in names
    # Recovered from a glued line, so never reported as a clean whole line.
    assert all(whole is False for name, whole in candidates if " " in name)


def test_window_title_tail_is_a_whole_name():
    # Screen shares put the presenter name at the end of the window title.
    candidates = _name_candidates(
        "Формулы Данные Шаталов Дмитрий Олегович", allow_tail=True
    )
    assert ("Шаталов Дмитрий Олегович", True) in candidates


def test_tail_is_not_taken_without_permission():
    candidates = _name_candidates("Формулы Данные Шаталов Дмитрий Олегович")
    assert all(whole is False for _, whole in candidates)


def test_interface_labels_are_not_name_parts():
    # "Файл" and "Правка" are menu items, not surnames.
    assert _name_candidates("Файл Правка Вид") == []


# ------------------------------------------------------------------ collect
def test_collect_counts_a_candidate_once_per_frame():
    whole, window, confidence = tally()
    lines = [line("Иванов Борис Александрович"), line("Иванов Борис Александрович")]
    collect_names(lines, whole, window, confidence)
    assert whole["Иванов Борис Александрович"] == 1


def test_collect_skips_low_confidence_whole_lines():
    whole, window, confidence = tally()
    collect_names([line("Иванов Борис Александрович", confidence=0.6)], whole, window, confidence)
    assert not whole


def test_collect_drops_lines_below_candidate_confidence():
    whole, window, confidence = tally()
    collect_names([line("Иванов Борис Александрович", confidence=0.2)], whole, window, confidence)
    assert not whole and not window


# ------------------------------------------------------------------- helpers
def test_same_person_needs_equal_word_count():
    assert _same_person("Шаталов Дмитрий", "Шаталов Дмитрий Олегович") is False


def test_same_person_tolerates_ocr_typo():
    assert _same_person("Шаталов Дмитрий Олегович", "Шаталов Дмйтрий Олегович") is True


def test_fragments_are_dropped():
    from nestor.video_names import NameCandidate

    items = [
        NameCandidate("Шаталов Дмитрий Олегович", 9, 0.9),
        NameCandidate("Шаталов Дмитрий", 3, 0.9),
    ]
    assert [item.name for item in _drop_fragments(items)] == ["Шаталов Дмитрий Олегович"]


def test_boxes_confined_to_bottom_is_false_for_a_gallery():
    boxes = [line("", y=40, height=10), line("", y=380, height=10)]
    assert _boxes_confined_to_bottom(boxes, frame_height=720, tall=22) is False


def test_boxes_confined_to_bottom_is_true_for_a_share_strip():
    boxes = [line("", y=600, height=10), line("", y=640, height=10)]
    assert _boxes_confined_to_bottom(boxes, frame_height=720, tall=22) is True


def test_boxes_confined_to_bottom_ignores_captions():
    boxes = [line("", y=200, height=40), line("", y=640, height=10)]
    assert _boxes_confined_to_bottom(boxes, frame_height=720, tall=22) is True


# ------------------------------------------------------------------- ranking
def test_name_seen_in_two_frames_is_kept():
    whole, window, confidence = tally()
    whole["Иванов Борис Александрович"] = 2
    confidence["Иванов Борис Александрович"] = 0.9
    ranked = rank_names(whole, window, confidence, frames=8)
    assert [item.name for item in ranked] == ["Иванов Борис Александрович"]


def test_single_sighting_is_not_enough():
    whole, window, confidence = tally()
    whole["Иванов Борис Александрович"] = 1
    assert rank_names(whole, window, confidence, frames=8) == []


def test_one_frame_recordings_accept_a_single_sighting():
    whole, window, confidence = tally()
    whole["Иванов Борис Александрович"] = 1
    assert len(rank_names(whole, window, confidence, frames=1)) == 1


def test_clean_name_wins_over_glued_garbage():
    whole, window, confidence = tally()
    whole["Шаталов Дмитрий Олегович"] = 3
    window["Дднтые Рценирс Шаталов"] = 5
    confidence["Шаталов Дмитрий Олегович"] = 0.9
    ranked = rank_names(whole, window, confidence, frames=8)
    assert [item.name for item in ranked] == ["Шаталов Дмитрий Олегович"]


def test_repeated_window_survives_when_nothing_was_clean():
    whole, window, confidence = tally()
    window["Шаталов Дмитрий Олегович"] = 4
    confidence["Шаталов Дмитрий Олегович"] = 0.8
    ranked = rank_names(whole, window, confidence, frames=8)
    assert [item.name for item in ranked] == ["Шаталов Дмитрий Олегович"]


def test_latin_interface_label_is_dropped():
    whole, window, confidence = tally()
    whole["Location Request"] = 2
    whole["Иванов Борис Александрович"] = 2
    ranked = rank_names(whole, window, confidence, frames=8)
    assert [item.name for item in ranked] == ["Иванов Борис Александрович"]


def test_latin_name_is_kept_when_it_repeats():
    whole, window, confidence = tally()
    whole["Elizaveta Khabaeva"] = 3
    ranked = rank_names(whole, window, confidence, frames=8)
    assert [item.name for item in ranked] == ["Elizaveta Khabaeva"]


def test_ranking_prefers_the_more_often_seen_name():
    whole, window, confidence = tally()
    whole["Иванов Борис Александрович"] = 5
    whole["Котов Анатолий Александрович"] = 2
    ranked = rank_names(whole, window, confidence, frames=8)
    assert [item.name for item in ranked] == [
        "Иванов Борис Александрович",
        "Котов Анатолий Александрович",
    ]


# ------------------------------------------------------------------ wrapper
def test_missing_file_yields_no_names(tmp_path):
    assert names_from_video(tmp_path / "nope.webm") == []
