"""Regression tests for the voiceprints export endpoint selection (QA К3).

The UI offers «Экспорт выбранных» and sends ``?slug=a&slug=b``. The endpoint
must honour that selection instead of silently returning the whole library.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from nestor.voiceprints import VoiceprintLibrary
from nestor.web.app import create_app


def _vec(seed: float) -> np.ndarray:
    v = np.full(8, seed, dtype=np.float32)
    return v / np.linalg.norm(v)


def _client_with_two_voices(tmp_path) -> TestClient:
    lib = VoiceprintLibrary(tmp_path)
    lib.add_or_update("Иван", _vec(1.0), duration_sec=60.0)
    lib.add_or_update("Пётр", _vec(2.0), duration_sec=60.0)
    return TestClient(create_app(work_dir=tmp_path))


def test_export_all_without_query(tmp_path):
    client = _client_with_two_voices(tmp_path)
    data = client.get("/api/voiceprints/export").json()
    assert data["count"] == 2


def test_export_only_the_chosen_slugs(tmp_path):
    client = _client_with_two_voices(tmp_path)
    slug = VoiceprintLibrary(tmp_path).find("Иван").slug
    data = client.get(f"/api/voiceprints/export?slug={slug}").json()
    assert data["count"] == 1
    names = [v["meta"]["display_name"] for v in data["voices"]]
    assert names == ["Иван"]


def test_export_several_slugs(tmp_path):
    client = _client_with_two_voices(tmp_path)
    lib = VoiceprintLibrary(tmp_path)
    slugs = [lib.find("Иван").slug, lib.find("Пётр").slug]
    query = "&".join(f"slug={s}" for s in slugs)
    data = client.get(f"/api/voiceprints/export?{query}").json()
    assert data["count"] == 2
