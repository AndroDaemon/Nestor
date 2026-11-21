"""Tests for the local-server access guard (session token + Host check).

The desktop app binds to 127.0.0.1 only, so remote machines cannot connect in
principle. The guard removes the two remaining local vectors: blind CSRF
requests from a malicious web page in the user's browser and DNS rebinding.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nestor.web.app import create_app, generate_session_token


def test_desktop_module_uses_guard_helpers():
    """Regression: desktop.py must import the token helper it calls (the exe
    crashed at startup with NameError when this import was missing)."""
    import nestor.desktop as desktop

    assert callable(desktop.generate_session_token)


@pytest.fixture()
def guarded(tmp_path):
    token = generate_session_token()
    app = create_app(work_dir=tmp_path, token=token)
    return TestClient(app), token


def test_without_token_is_rejected(guarded):
    client, _ = guarded
    assert client.get("/api/status").status_code == 403
    assert client.get("/api/recordings").status_code == 403
    # blind POST (no token) must not change any state either
    assert client.post("/api/recordings/cancel/rec1").status_code == 403


def test_wrong_token_is_rejected(guarded):
    client, _ = guarded
    assert client.get("/api/status", headers={"X-Nestor-Token": "nope"}).status_code == 403


def test_accepts_header_query_and_cookie(guarded):
    client, token = guarded
    assert client.get("/api/status", headers={"X-Nestor-Token": token}).status_code == 200
    assert client.get(f"/api/status?token={token}").status_code == 200
    client.cookies.set("nestor_token", token)
    assert client.get("/api/status").status_code == 200


def test_guard_sets_httponly_cookie(guarded):
    client, token = guarded
    response = client.get(f"/api/status?token={token}")
    assert response.status_code == 200
    cookie_header = response.headers.get("set-cookie", "")
    assert "nestor_token=" in cookie_header
    assert "httponly" in cookie_header.lower()
    assert "samesite=lax" in cookie_header.lower()


def test_foreign_host_is_rejected(guarded):
    """DNS rebinding: a foreign hostname resolving to 127.0.0.1 still fails."""
    client, token = guarded
    response = client.get(f"/api/status?token={token}", headers={"Host": "evil.example"})
    assert response.status_code == 403


def test_localhost_host_is_accepted(guarded):
    client, token = guarded
    assert client.get(f"/api/status?token={token}", headers={"Host": "localhost"}).status_code == 200
    assert client.get(
        f"/api/status?token={token}", headers={"Host": "127.0.0.1:8000"}
    ).status_code == 200


def test_token_is_random_per_run(tmp_path):
    """A fresh run must never reuse the previous run's token."""
    first = generate_session_token()
    second = generate_session_token()
    assert first != second
    assert len(first) >= 32
