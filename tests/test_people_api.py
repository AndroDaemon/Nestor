"""The /api/people layer: names on top of the voice library.

The UI works with people - one shown name plus the spellings it is known by -
while matching still runs on voices. These tests pin down the contract:

* a person is rebuilt from the voices that exist (nothing to migrate);
* renaming a person carries the new name into the recordings that the voice
  library had named, and leaves hand-typed names alone;
* the "similar" hint recognises the same person under another spelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.library import Library, Recording, Segment  # noqa: E402
from nestor.voiceprints import VoiceprintLibrary  # noqa: E402
from nestor.web.app import create_app  # noqa: E402


def _vec(seed: float) -> np.ndarray:
    v = np.full(8, seed, dtype=np.float32)
    return v / np.linalg.norm(v)


def _voices(tmp_path, *names) -> VoiceprintLibrary:
    lib = VoiceprintLibrary(tmp_path)
    for index, name in enumerate(names, start=1):
        lib.add_or_update(name, _vec(float(index)), duration_sec=60.0)
    return lib


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(work_dir=tmp_path))


def _recording(tmp_path, **names) -> Library:
    """A done recording whose names came from voice matching."""
    lib = Library(tmp_path)
    recording = Recording(id="rec1", filename="a.wav", status="done")
    recording.segments = [
        Segment("s1", 0.0, 2.0, "SPEAKER_00", "привет"),
        Segment("s2", 2.0, 4.0, "SPEAKER_01", "пока"),
    ]
    recording.speaker_names = names["speaker_names"]
    recording.speaker_match = names.get("speaker_match", {})
    lib.save(recording)
    return lib


def test_people_are_rebuilt_from_voices(tmp_path):
    _voices(tmp_path, "Шаталов Дмитрий", "Иванов Борис")
    people = _client(tmp_path).get("/api/people").json()["people"]

    assert sorted(p["name"] for p in people) == ["Иванов Борис", "Шаталов Дмитрий"]
    assert all(len(p["voices"]) == 1 for p in people)


def test_rename_person_reaches_the_recordings_but_not_hand_typed_names(tmp_path):
    _voices(tmp_path, "Шаталов Дмитрий")
    _recording(
        tmp_path,
        speaker_names={"SPEAKER_00": "Шаталов Дмитрий", "SPEAKER_01": "Пётр"},
        speaker_match={"SPEAKER_00": {"assigned_name": "Шаталов Дмитрий", "confidence": 0.9}},
    )
    client = _client(tmp_path)
    person = client.get("/api/people").json()["people"][0]

    answer = client.patch(
        f"/api/people/{person['id']}", json={"name": "Шаталов Дмитрий Олегович"}
    ).json()
    assert answer["name"] == "Шаталов Дмитрий Олегович"
    assert answer["changed"] == 1
    assert "Шаталов Дмитрий" in answer["aliases"]

    saved = Library(tmp_path).load("rec1")
    assert saved.speaker_names["SPEAKER_00"] == "Шаталов Дмитрий Олегович"
    # the hand-typed name must survive untouched
    assert saved.speaker_names["SPEAKER_01"] == "Пётр"
    assert saved.speaker_match["SPEAKER_00"]["assigned_name"] == "Шаталов Дмитрий Олегович"


def test_display_switch_does_not_touch_recordings(tmp_path):
    _voices(tmp_path, "Шаталов Дмитрий")
    _recording(
        tmp_path,
        speaker_names={"SPEAKER_00": "Шаталов Дмитрий"},
        speaker_match={"SPEAKER_00": {"assigned_name": "Шаталов Дмитрий"}},
    )
    client = _client(tmp_path)
    person = client.get("/api/people").json()["people"][0]
    client.post(
        f"/api/people/{person['id']}/aliases", json={"alias": "Шаталов Д. О."}
    )

    answer = client.patch(
        f"/api/people/{person['id']}", json={"display": "Шаталов Д. О."}
    ).json()
    assert answer["name"] == "Шаталов Д. О."
    assert answer["changed"] == 0

    saved = Library(tmp_path).load("rec1")
    assert saved.speaker_names["SPEAKER_00"] == "Шаталов Дмитрий"


def test_similar_person_finds_the_same_person_under_a_longer_name(tmp_path):
    _voices(tmp_path, "Шаталов Дмитрий")
    client = _client(tmp_path)

    answer = client.get(
        "/api/people/similar", params={"name": "Шаталов Дмитрий Олегович"}
    ).json()
    assert answer["person"] is not None
    assert answer["person"]["name"] == "Шаталов Дмитрий"
    assert answer["spelling"] == "Шаталов Дмитрий"

    stranger = client.get("/api/people/similar", params={"name": "Совсем Другой"}).json()
    assert stranger["person"] is None


def test_merge_people_folds_spellings_voices_and_recording_names(tmp_path):
    _voices(tmp_path, "Шаталов Дмитрий", "Шаталов Дмитрий Олегович")
    _recording(
        tmp_path,
        speaker_names={"SPEAKER_00": "Шаталов Дмитрий Олегович"},
        speaker_match={"SPEAKER_00": {"assigned_name": "Шаталов Дмитрий Олегович"}},
    )
    client = _client(tmp_path)
    people = client.get("/api/people").json()["people"]
    short = next(p for p in people if p["name"] == "Шаталов Дмитрий")
    long = next(p for p in people if p["name"] == "Шаталов Дмитрий Олегович")

    answer = client.post(
        "/api/people/merge", json={"source": short["id"], "target": long["id"]}
    ).json()
    assert answer["name"] == "Шаталов Дмитрий Олегович"
    assert len(answer["voices"]) == 2
    assert "Шаталов Дмитрий" in answer["aliases"]

    remaining = client.get("/api/people").json()["people"]
    assert len(remaining) == 1


def test_create_attach_detach_and_delete(tmp_path):
    voices = _voices(tmp_path, "Иван Петров", "Мария")
    client = _client(tmp_path)

    created = client.post("/api/people", json={"name": "Новый Человек"}).json()
    assert created["name"] == "Новый Человек"
    assert created["voices"] == []

    slug = voices.find("Мария").slug
    attached = client.post(
        f"/api/people/{created['id']}/voices", json={"voice": slug}
    ).json()
    assert [v["slug"] for v in attached["voices"]] == [slug]
    # the voice now belongs to the new person and to nobody else
    assert len(client.get("/api/people").json()["people"]) == 2

    # detaching leaves the person with nothing behind it, so it disappears and
    # the voice gets its own person back
    gone = client.post(
        f"/api/people/{created['id']}/voices", json={"voice": slug, "detach": True}
    ).json()
    assert gone.get("gone") is True
    names = sorted(p["name"] for p in client.get("/api/people").json()["people"])
    assert names == ["Иван Петров", "Мария"]


def test_delete_person_removes_its_voices(tmp_path):
    voices = _voices(tmp_path, "Иван Петров")
    client = _client(tmp_path)
    person = client.get("/api/people").json()["people"][0]

    assert client.delete(f"/api/people/{person['id']}").status_code == 200
    assert client.get("/api/people").json()["people"] == []
    assert VoiceprintLibrary(tmp_path).find("Иван Петров") is None


def test_aliases_can_be_added_and_dropped(tmp_path):
    _voices(tmp_path, "Шаталов Дмитрий")
    client = _client(tmp_path)
    person = client.get("/api/people").json()["people"][0]

    added = client.post(
        f"/api/people/{person['id']}/aliases", json={"alias": "Шаталов Д.О."}
    ).json()
    assert "Шаталов Д.О." in added["aliases"]

    dropped = client.post(
        f"/api/people/{person['id']}/aliases",
        json={"alias": "Шаталов Д.О.", "remove": True},
    ).json()
    assert "Шаталов Д.О." not in dropped["aliases"]


def test_missing_person_is_a_404(tmp_path):
    client = _client(tmp_path)
    assert client.patch("/api/people/nobody", json={"name": "X"}).status_code == 404
    assert client.delete("/api/people/nobody").status_code == 404
