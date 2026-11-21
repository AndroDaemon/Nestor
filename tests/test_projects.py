"""Projects (folders) and the review state of a recording.

Two axes live here:

* **Projects** group recordings by topic.  A project is just a name and a
  colour; the recording keeps the link, so deleting a project never deletes a
  recording — it only becomes uncategorized.
* **Review state** is separate from the processing ``status``: "not started"
  (nothing hand-edited), "editing" (the transcript was changed) and "finished"
  (the user marked it).  Editing a finished recording brings it back to
  "editing".

The tests pin down the storage layer (``ProjectLibrary``) and the API contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.library import Library, Recording, Segment  # noqa: E402
from nestor.organizer import ProjectLibrary  # noqa: E402
from nestor.web.app import create_app  # noqa: E402


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(work_dir=tmp_path))


def _recording(tmp_path, record_id="rec1", **fields) -> Library:
    lib = Library(tmp_path)
    recording = Recording(id=record_id, filename="a.wav", status="done", **fields)
    recording.segments = [
        Segment("s1", 0.0, 2.0, "SPEAKER_00", "привет"),
        Segment("s2", 2.0, 4.0, "SPEAKER_01", "пока"),
    ]
    recording.speaker_names = {"SPEAKER_00": "Пётр"}
    lib.save(recording)
    return lib


# ------------------------------------------------------------------ library
def test_project_library_creates_and_persists(tmp_path):
    projects = ProjectLibrary(tmp_path)
    project = projects.create("Босфор")

    assert project.id == "bosfor"
    assert project.color  # a colour is picked automatically
    assert ProjectLibrary(tmp_path).get("bosfor").name == "Босфор"


def test_project_names_are_unique(tmp_path):
    projects = ProjectLibrary(tmp_path)
    first = projects.create("Босфор")
    again = projects.create("  Босфор  ")

    assert first.id == again.id
    assert len(projects.list()) == 1


def test_projects_get_distinct_colors(tmp_path):
    projects = ProjectLibrary(tmp_path)
    colors = {projects.create(name).color for name in ("A", "B", "C")}

    assert len(colors) == 3


def test_rename_keeps_the_id_and_delete_is_reported(tmp_path):
    projects = ProjectLibrary(tmp_path)
    project = projects.create("Старое")

    renamed = projects.rename(project.id, "Новое")
    assert renamed.id == project.id and renamed.name == "Новое"
    assert projects.delete(project.id) is True
    assert projects.get(project.id) is None
    assert projects.delete(project.id) is False


def test_damaged_organizer_file_is_not_fatal(tmp_path):
    (tmp_path / "organizer.json").write_text("{oops", encoding="utf-8")

    assert ProjectLibrary(tmp_path).list() == []


def test_recording_carries_project_and_review_fields(tmp_path):
    lib = _recording(tmp_path, project="bosfor", edited_at="2026-01-01T00:00:00+00:00")
    loaded = Library(tmp_path).load("rec1")

    assert loaded.project == "bosfor"
    assert loaded.edited_at == "2026-01-01T00:00:00+00:00"
    assert loaded.done_at is None


# ---------------------------------------------------------------------- api
def test_api_creates_lists_and_renames_projects(tmp_path):
    client = _client(tmp_path)
    created = client.post("/api/projects", json={"name": "Босфор"}).json()
    assert created["name"] == "Босфор"
    assert created["count"] == 0
    assert created["color"]

    listed = client.get("/api/projects").json()["projects"]
    assert [p["name"] for p in listed] == ["Босфор"]

    renamed = client.patch(
        f"/api/projects/{created['id']}", json={"name": "Босфор 2026"}
    ).json()
    assert renamed["name"] == "Босфор 2026"


def test_api_rejects_empty_project_name(tmp_path):
    response = _client(tmp_path).post("/api/projects", json={"name": "   "})
    assert response.status_code == 400


def test_api_assigns_a_recording_to_a_project(tmp_path):
    lib = _recording(tmp_path)
    client = _client(tmp_path)
    project = client.post("/api/projects", json={"name": "Босфор"}).json()

    answer = client.patch(
        "/api/recordings/rec1/meta", json={"project": project["id"]}
    ).json()
    assert answer["project"] == project["id"]
    assert answer["project_color"] == project["color"]
    assert lib.load("rec1").project == project["id"]

    # listing shows the project and the count of its recordings
    listed = client.get("/api/projects").json()["projects"][0]
    assert listed["count"] == 1

    # an unknown project is refused
    assert client.patch(
        "/api/recordings/rec1/meta", json={"project": "nope"}
    ).status_code == 404


def test_api_unassigns_a_recording(tmp_path):
    _recording(tmp_path, project="bosfor")
    client = _client(tmp_path)
    client.post("/api/projects", json={"name": "Босфор"})

    answer = client.patch("/api/recordings/rec1/meta", json={"project": ""}).json()
    assert answer["project"] is None
    assert Library(tmp_path).load("rec1").project is None


def test_deleting_a_project_unlinks_recordings(tmp_path):
    lib = _recording(tmp_path, project="bosfor")
    client = _client(tmp_path)
    client.post("/api/projects", json={"name": "Босфор"})

    answer = client.delete("/api/projects/bosfor").json()
    assert answer["deleted"] == "bosfor"
    assert answer["moved"] == 1
    assert lib.load("rec1").project is None


def test_deleting_a_project_can_move_recordings(tmp_path):
    lib = _recording(tmp_path, project="bosfor")
    client = _client(tmp_path)
    client.post("/api/projects", json={"name": "Босфор"})
    target = client.post("/api/projects", json={"name": "Другое"}).json()

    answer = client.delete(
        f"/api/projects/bosfor?move_to={target['id']}"
    ).json()
    assert answer["moved"] == 1
    assert lib.load("rec1").project == target["id"]


def test_marking_a_recording_finished_and_back(tmp_path):
    lib = _recording(tmp_path)
    client = _client(tmp_path)

    done = client.patch("/api/recordings/rec1/meta", json={"done": True}).json()
    assert done["done_at"]
    assert done["edited_at"] is None            # not edited, only finished

    undone = client.patch("/api/recordings/rec1/meta", json={"done": False}).json()
    assert undone["done_at"] is None
    assert lib.load("rec1").done_at is None


def test_editing_the_transcript_marks_it_and_reopens_finished(tmp_path):
    lib = _recording(tmp_path)
    client = _client(tmp_path)
    client.patch("/api/recordings/rec1/meta", json={"done": True})

    payload = {
        "segments": [
            {"id": "s1", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "привет!"},
            {"id": "s2", "start": 2.0, "end": 4.0, "speaker": "SPEAKER_01", "text": "пока"},
        ],
        "speaker_names": {"SPEAKER_00": "Пётр"},
        "speaker_keep": [],
    }
    answer = client.put("/api/recordings/rec1", json=payload).json()
    assert answer["edited_at"]                 # the edit is remembered
    assert answer["done_at"] is None           # finished -> back to editing

    saved = lib.load("rec1")
    assert saved.edited_at and saved.done_at is None


def test_meta_only_save_does_not_count_as_editing(tmp_path):
    lib = _recording(tmp_path)
    client = _client(tmp_path)
    client.patch("/api/recordings/rec1/meta", json={"title": "Новое имя"})

    assert lib.load("rec1").edited_at is None
