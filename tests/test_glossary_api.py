"""Tests for the glossary management API (nestor/web/app.py).

Covers listing, adding, deleting, export/import and the dry-run preview of the
term dictionary that the web UI's «Словарь терминов» panel drives.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from nestor.web.app import create_app


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(work_dir=tmp_path))


def test_get_returns_bundled_glossary(tmp_path):
    client = _client(tmp_path)
    data = client.get("/api/glossary").json()
    assert data["source"] == "bundled"
    assert data["count"] >= 100
    assert any(t["term"] == "Docker" for t in data["terms"])


def test_add_creates_user_file_and_flips_source(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/api/glossary/term", json={"term": "TestTerm", "aliases": ["тесттерм", "testterm"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["term"] == "TestTerm"
    assert body["aliases"] == ["тесттерм", "testterm"]
    assert body["count"] >= 101

    after = client.get("/api/glossary").json()
    assert after["source"] == "user"
    assert any(t["term"] == "TestTerm" for t in after["terms"])
    # the user file exists on disk and other terms were preserved
    assert (tmp_path / "hotwords.yaml").exists()


def test_add_accepts_comma_string_aliases(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/api/glossary/term", json={"term": "VPN2", "aliases": "впн2, vpn2"}
    )
    assert response.status_code == 200
    assert response.json()["aliases"] == ["впн2", "vpn2"]


def test_add_requires_term_and_alias(tmp_path):
    client = _client(tmp_path)
    assert client.post("/api/glossary/term", json={"aliases": ["x"]}).status_code == 400
    assert client.post("/api/glossary/term", json={"term": "X"}).status_code == 400


def test_update_replaces_aliases(tmp_path):
    client = _client(tmp_path)
    client.post("/api/glossary/term", json={"term": "Docker", "aliases": ["докер", "докерс"]})
    data = client.get("/api/glossary").json()
    docker = next(t for t in data["terms"] if t["term"] == "Docker")
    assert docker["aliases"] == ["докер", "докерс"]


def test_delete_term(tmp_path):
    client = _client(tmp_path)
    client.post("/api/glossary/term", json={"term": "TestTerm", "aliases": ["тесттерм"]})
    response = client.delete("/api/glossary/term", params={"term": "TestTerm"})
    assert response.status_code == 200
    assert response.json()["removed"] == "TestTerm"
    assert not any(t["term"] == "TestTerm" for t in client.get("/api/glossary").json()["terms"])
    # unknown term -> 404, empty -> 400
    assert client.delete("/api/glossary/term", params={"term": "нет"}).status_code == 404
    assert client.delete("/api/glossary/term", params={"term": ""}).status_code == 400


def test_export_is_yaml_attachment(tmp_path):
    client = _client(tmp_path)
    response = client.get("/api/glossary/export")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert "hotwords" in response.headers["content-disposition"]
    assert "Docker" in response.text


def test_import_merges_terms(tmp_path):
    client = _client(tmp_path)
    payload = '- term: "Imported"\n  aliases: ["импортед"]\n'.encode("utf-8")
    response = client.post(
        "/api/glossary/import",
        files={"file": ("terms.yaml", payload, "application/x-yaml")},
    )
    assert response.status_code == 200
    assert response.json()["added"] >= 1
    assert any(
        t["term"] == "Imported" for t in client.get("/api/glossary").json()["terms"]
    )


def test_import_empty_file_is_rejected(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/api/glossary/import", files={"file": ("empty.yaml", b"", "application/x-yaml")}
    )
    assert response.status_code == 400


def test_preview_shows_replacements(tmp_path):
    client = _client(tmp_path)
    data = client.get("/api/glossary/preview", params={"text": "Разверни докер и проверь днс"}).json()
    assert "Docker" in data["text"]
    assert {"from": "докер", "to": "Docker"} in data["changes"]


def test_preview_without_text(tmp_path):
    client = _client(tmp_path)
    data = client.get("/api/glossary/preview", params={"text": ""}).json()
    assert data == {"text": "", "changes": []}
