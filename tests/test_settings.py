"""Tests for the 1.1.4 persistent settings (nestor/settings.py + API).

The WebView2 localStorage proved volatile, so all user preferences now live
in ``<root>/settings.json`` — written atomically, quarantined when damaged.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.settings import DEFAULTS, SCHEMA_VERSION, SettingsStore  # noqa: E402
from nestor.web.app import create_app  # noqa: E402


def test_defaults_and_roundtrip(tmp_path):
    store = SettingsStore(tmp_path)
    assert store.as_public() == DEFAULTS
    store.update({"num_speakers": 3, "sound_done": True})
    # a fresh store reads the same values from disk
    assert SettingsStore(tmp_path).as_public()["num_speakers"] == 3
    assert SettingsStore(tmp_path).as_public()["sound_done"] is True


def test_unknown_and_invalid_keys_ignored_or_rejected(tmp_path):
    store = SettingsStore(tmp_path)
    # unknown keys are ignored silently
    store.update({"what_is_this": 42})
    assert "what_is_this" not in store.as_public()
    # invalid type / out of range raise
    with pytest.raises(ValueError):
        store.update({"vp_abs_threshold": "much"})
    with pytest.raises(ValueError):
        store.update({"vp_abs_threshold": 5.0})
    with pytest.raises(ValueError):
        store.update({"auto_start": "yes"})
    # the stored file is untouched after failures
    assert store.as_public()["vp_abs_threshold"] == DEFAULTS["vp_abs_threshold"]


def test_merge_settings_roundtrip_and_validation(tmp_path):
    """The duplicate-speaker merge has its own documented settings."""
    store = SettingsStore(tmp_path)
    assert store.as_public()["merge_speakers"] is True
    assert store.as_public()["merge_min_weight"] == 1.5
    assert store.as_public()["merge_threshold"] == 0.45
    assert store.as_public()["merge_segment_floor"] == 0.5
    assert store.as_public()["diar_threshold"] == 0.5

    store.update({"merge_speakers": False, "merge_min_weight": 3.0})
    assert store.as_public()["merge_speakers"] is False
    assert SettingsStore(tmp_path).as_public()["merge_min_weight"] == 3.0

    with pytest.raises(ValueError):
        store.update({"merge_speakers": "yes"})
    with pytest.raises(ValueError):
        store.update({"merge_min_weight": -1.0})
    with pytest.raises(ValueError):
        store.update({"merge_threshold": 2.0})


def test_glossary_setting_roundtrip(tmp_path):
    """The glossary toggle is a plain bool setting, off by default."""
    store = SettingsStore(tmp_path)
    assert store.as_public()["glossary"] is False
    store.update({"glossary": True})
    assert store.as_public()["glossary"] is True
    assert SettingsStore(tmp_path).as_public()["glossary"] is True
    with pytest.raises(ValueError):
        store.update({"glossary": "yes"})


def test_damaged_json_quarantined(tmp_path):
    (tmp_path / "settings.json").write_text("{not json at all", encoding="utf-8")
    store = SettingsStore(tmp_path)
    assert store.as_public() == DEFAULTS  # survived with defaults
    broken = list(tmp_path.glob("settings.json.broken-*"))
    assert len(broken) == 1  # the original is kept for inspection
    assert "not json" in broken[0].read_text(encoding="utf-8")


def test_old_schema_migrated_to_new_diarization_defaults(tmp_path):
    """A settings.json written by an older build (schema 1) is re-tuned.

    The old file carried diar_threshold=0.3 explicitly; that value over-split
    long meetings, so the migration resets it (and adds the merge floor).
    """
    (tmp_path / "settings.json").write_text(
        json.dumps({"schema_version": 1, "diar_threshold": 0.3, "num_speakers": 3}),
        encoding="utf-8",
    )
    store = SettingsStore(tmp_path)
    assert store.as_public()["diar_threshold"] == DEFAULTS["diar_threshold"]
    assert store.as_public()["merge_segment_floor"] == DEFAULTS["merge_segment_floor"]
    assert store.as_public()["num_speakers"] == 3  # user values survive
    # the file is rewritten at the new schema version
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == SCHEMA_VERSION


def test_schema3_file_gets_the_raised_merge_floor(tmp_path):
    """A 1.5.x file carries segment_floor=0.35 explicitly; the migration lifts it.

    Without the reset the old value would keep folding two voices together,
    because a settings file wins over the defaults.
    """
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {"schema_version": 3, "merge_segment_floor": 0.35, "num_speakers": 2}
        ),
        encoding="utf-8",
    )
    store = SettingsStore(tmp_path)
    assert store.as_public()["merge_segment_floor"] == DEFAULTS["merge_segment_floor"]
    assert store.as_public()["num_speakers"] == 2  # other user values survive
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == SCHEMA_VERSION


def test_atomic_write_under_concurrency(tmp_path):
    """Parallel updates must not corrupt the file or lose both changes."""
    store = SettingsStore(tmp_path)
    errors = []

    def worker(n: int) -> None:
        try:
            for i in range(10):
                store.update({"vp_margin": 0.05 + 0.001 * ((n * 10 + i) % 50)})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    data = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert 0.05 <= data["vp_margin"] <= 0.1
    # no leftover temp files
    assert list(tmp_path.glob("*.tmp")) == []


def test_reset_restores_defaults(tmp_path):
    store = SettingsStore(tmp_path)
    store.update({"num_speakers": 2, "vp_margin": 0.07})
    assert store.as_public()["num_speakers"] == 2
    store.reset()
    assert store.as_public() == DEFAULTS


# ------------------------------------------------------------------------ API
@pytest.fixture()
def client(tmp_path):
    return create_app(work_dir=tmp_path), tmp_path


def _api(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_api_get_patch_reset(client):
    app, root = client
    api = _api(app)
    body = api.get("/api/settings").json()["settings"]
    assert body["vp_abs_threshold"] == DEFAULTS["vp_abs_threshold"]

    patched = api.patch("/api/settings", json={"num_speakers": 3, "vp_margin": 0.12})
    assert patched.status_code == 200
    assert patched.json()["settings"]["num_speakers"] == 3
    # persisted on disk
    on_disk = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    assert on_disk["num_speakers"] == 3
    assert on_disk["vp_margin"] == 0.12

    # invalid payload is a client error (400 from validation, 422 when the
    # body itself is not a JSON object), never a 500
    assert api.patch("/api/settings", json={"vp_margin": "oops"}).status_code == 400
    assert api.patch("/api/settings", json=["not an object"]).status_code in (400, 422)

    reset = api.post("/api/settings/reset")
    assert reset.status_code == 200
    assert reset.json()["settings"] == DEFAULTS
