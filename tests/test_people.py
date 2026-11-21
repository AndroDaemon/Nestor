"""Tests for the people layer over the voiceprints.

A person is a name plus every spelling of it, plus the voices recorded for
them.  The layer must never lose a voice: a missing ``people.json`` rebuilds
into one person per stored voice, and the invariant «one voice — one person»
survives every edit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.people import PeopleLibrary, Person, same_name  # noqa: E402
from nestor.voiceprints import VoiceprintLibrary  # noqa: E402


def unit(index: int, size: int = 8) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index % size] = 1.0
    return vector


def voices_with(tmp_path, *names: str) -> VoiceprintLibrary:
    library = VoiceprintLibrary(tmp_path)
    for index, name in enumerate(names):
        library.add_or_update(name, unit(index), duration_sec=5.0)
    return library


# ------------------------------------------------------------------- same_name
def test_same_name_matches_full_and_short_spelling():
    assert same_name("Шаталов Дмитрий", "Шаталов Дмитрий Олегович")
    assert same_name("Шаталов Дмитрий Олегович", "Шаталов Дмитрий")


def test_same_name_is_case_insensitive_and_tolerates_a_slip():
    assert same_name("шаталов дмитрий", "Шаталов Дмитрий")
    assert same_name("Шаталов Дмитрий", "Шаталов Дмитрии")


def test_same_name_rejects_different_people_and_empties():
    assert not same_name("Шаталов Дмитрий", "Пронин Игорь")
    assert not same_name("", "Шаталов")
    assert not same_name("Шаталов", "")


# ------------------------------------------------------------------ spellings
def test_spellings_start_with_the_display_name_and_dedupe():
    person = Person(id="p", display_name="Шаталов Дмитрий Олегович",
                    aliases=["шаталов дмитрий олегович", "Шаталов Дмитрий"])
    assert person.spellings() == ["Шаталов Дмитрий Олегович", "Шаталов Дмитрий"]


# ------------------------------------------------------------- rebuild on load
def test_reconcile_builds_one_person_per_voice(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий", "Пронин Игорь")
    people = PeopleLibrary(tmp_path)
    assert len(people.list()) == 2
    assert {person.display_name for person in people.list()} == {"Шаталов Дмитрий", "Пронин Игорь"}
    assert (people.dir / "people.json").exists()


def test_reconcile_is_idempotent(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    assert people.reconcile() is False


def test_damaged_people_json_is_rebuilt(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    (tmp_path / "voiceprints" / "people.json").write_text("{ not json", encoding="utf-8")
    people = PeopleLibrary(tmp_path)
    assert [person.display_name for person in people.list()] == ["Шаталов Дмитрий"]


def test_reconcile_forgets_a_deleted_voice(tmp_path):
    voices = voices_with(tmp_path, "Шаталов Дмитрий", "Пронин Игорь")
    people = PeopleLibrary(tmp_path, voices=voices)
    voices.delete("Пронин Игорь")
    people.reconcile()
    assert [person.display_name for person in people.list()] == ["Шаталов Дмитрий"]


# -------------------------------------------------------------- rename/aliases
def test_rename_keeps_the_previous_name_as_a_spelling(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    person = people.list()[0]
    people.rename(person.id, "Шаталов Дмитрий Олегович")
    assert person.display_name == "Шаталов Дмитрий Олегович"
    assert "Шаталов Дмитрий" in person.spellings()
    assert people.find_by_name("Шаталов Дмитрий") is person


def test_set_display_switches_the_shown_spelling(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    person = people.list()[0]
    people.add_alias(person.id, "Шаталов Дмитрий Олегович")
    people.set_display(person.id, "Шаталов Дмитрий Олегович")
    assert person.display_name == "Шаталов Дмитрий Олегович"
    assert "Шаталов Дмитрий" in person.spellings()


def test_add_and_remove_alias(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    person = people.list()[0]
    people.add_alias(person.id, "Дмитрий")
    assert "Дмитрий" in person.aliases
    people.remove_alias(person.id, "Дмитрий")
    assert "Дмитрий" not in person.aliases


# -------------------------------------------------------------------- voices
def test_attach_moves_a_voice_off_another_person(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий", "Пронин Игорь")
    people = PeopleLibrary(tmp_path)
    by_name = {person.display_name: person for person in people.list()}
    target = by_name["Шаталов Дмитрий"]
    other = by_name["Пронин Игорь"]
    slug = other.voices[0]
    people.attach_voice(target.id, slug)
    assert slug in target.voices
    assert slug not in other.voices


def test_detaching_the_last_voice_removes_the_person(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    person = people.list()[0]
    assert people.detach_voice(person.id, person.voices[0]) is None
    assert people.list() == []


def test_person_of_voice_and_names(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    person = people.list()[0]
    slug = person.voices[0]
    assert people.person_of_voice(slug) is person
    assert people.names() == {slug: "Шаталов Дмитрий"}
    assert people.display_for(slug) == "Шаталов Дмитрий"


# --------------------------------------------------------------------- merge
def test_merge_folds_spellings_and_voices(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий", "Шаталов Дмитрий Олегович")
    people = PeopleLibrary(tmp_path)
    short, long = people.list()
    if short.display_name != "Шаталов Дмитрий":
        short, long = long, short
    people.merge(short.id, long.id)
    assert [person.id for person in people.list()] == [long.id]
    assert len(long.voices) == 2
    assert "Шаталов Дмитрий" in long.spellings()


def test_merge_with_itself_is_a_no_op(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    person = people.list()[0]
    assert people.merge(person.id, person.id) is person


# ------------------------------------------------------------ similar_person
def test_similar_person_finds_a_known_spelling(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    person = people.list()[0]
    people.add_alias(person.id, "Шаталов Дмитрий Олегович")
    found = people.similar_person("Шаталов Дмитрии Олегович")
    assert found is not None
    matched, spelling = found
    assert matched is person
    assert spelling == "Шаталов Дмитрий Олегович"


def test_similar_person_returns_none_for_a_stranger(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    assert people.similar_person("Неизвестный Голос") is None


# ---------------------------------------------------------------- delete
def test_delete_removes_the_voices_too(tmp_path):
    voices = voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path, voices=voices)
    person = people.list()[0]
    assert people.delete(person.id) is True
    assert voices.names() == {}
    assert people.list() == []


def test_delete_can_keep_the_voices(tmp_path):
    voices = voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path, voices=voices)
    person = people.list()[0]
    people.delete(person.id, with_voices=False)
    assert "Шаталов Дмитрий" in voices.names().values()
    assert people.list() == []


# ---------------------------------------------------------------- exchange
def test_export_and_import_round_trip(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    source = PeopleLibrary(tmp_path)
    person = source.list()[0]
    source.add_alias(person.id, "Шаталов Дмитрий Олегович")
    source.set_comment(person.id, "докладчик")
    exported = source.export_data()
    assert exported[0]["aliases"] == ["Шаталов Дмитрий Олегович"]
    assert exported[0]["voices"] == ["Шаталов Дмитрий"]

    other = tmp_path / "copy"
    other.mkdir()
    (other / "voiceprints").mkdir()
    json.dump({"schema_version": 2, "voiceprints": []}, (other / "voiceprints" / "voiceprints.json").open("w", encoding="utf-8"))
    fresh = PeopleLibrary(other)
    report = fresh.import_data(exported)
    assert report["added"] == ["Шаталов Дмитрий"]
    assert fresh.list()[0].comment == "докладчик"


def test_import_merges_into_an_existing_person(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    report = people.import_data([{
        "display_name": "Шаталов Дмитрий",
        "aliases": ["Шаталов Дмитрий Олегович"],
        "voices": ["Шаталов Дмитрий"],
    }])
    assert report["merged"] == ["Шаталов Дмитрий"]
    assert len(people.list()) == 1
    assert "Шаталов Дмитрий Олегович" in people.list()[0].spellings()


def test_import_skips_an_entry_without_a_name(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    report = people.import_data([{"aliases": ["x"]}])
    assert report["added"] == []
    assert len(report["skipped"]) == 1


def test_import_of_a_non_list_is_an_empty_report(tmp_path):
    voices_with(tmp_path, "Шаталов Дмитрий")
    people = PeopleLibrary(tmp_path)
    assert people.import_data("nonsense") == {"added": [], "merged": [], "skipped": []}
