"""Tests for the local voiceprint library (nestor/voiceprints.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.voiceprints import SCHEMA_VERSION, VoiceprintLibrary, make_slug  # noqa: E402


def vec(seed: float, dim: int = 192) -> np.ndarray:
    """Deterministic pseudo-embedding with a non-trivial direction."""
    rng = np.random.default_rng(int(seed * 1000))
    return rng.standard_normal(dim).astype(np.float32)


def near(base: np.ndarray, seed: int, amount: float = 0.1) -> np.ndarray:
    """A unit vector close to ``base``."""
    rng = np.random.default_rng(seed)
    vector = base + amount * rng.standard_normal(base.shape[0]).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_slug_transliterates_cyrillic():
    assert make_slug("Иван Петров") == "ivan-petrov"
    assert make_slug("Мария") == "mariya"
    assert make_slug("  Анна  ") == "anna"
    # unsafe characters collapse to dashes
    assert "/" not in make_slug("a/b\\c:d")
    # empty name still yields a usable slug
    assert make_slug("???") == "voice"


def test_add_and_list_roundtrip(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    library.add_or_update("Иван", vec(1.0), source_recording_id="rec1", duration_sec=120.0)

    # a fresh instance reads the same data from disk
    second = VoiceprintLibrary(tmp_path)
    entries = second.list()
    assert len(entries) == 1
    assert entries[0].display_name == "Иван"
    assert entries[0].num_utterances == 1
    assert entries[0].total_duration_sec == pytest.approx(120.0)

    vectors = second.vectors()
    assert len(vectors) == 1
    vector = next(iter(vectors.values()))
    assert vector.dtype == np.float32
    assert np.isclose(float(np.linalg.norm(vector)), 1.0)  # L2-normalised


def test_running_average_blends_not_overwrites(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    first = vec(1.0)
    second_vec = near(first, 11, amount=0.3)  # same direction, perturbed
    library.add_or_update("Иван", first, duration_sec=100.0)
    library.add_or_update("Иван", second_vec, duration_sec=100.0)

    meta = library.find("Иван")
    assert meta.num_utterances == 2
    stored = library.vectors()[meta.slug]
    # the stored vector lies "between" the two utterances
    from nestor.speaker_match import cosine

    assert cosine(stored, first) > 0.9
    assert cosine(stored, second_vec) > 0.9
    # and it is not a plain copy of either
    assert not np.allclose(stored, first)
    assert not np.allclose(stored, second_vec)


def test_duration_weighting(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    short = vec(2.0)
    long_one = vec(3.0)  # different direction, but 10x more speech
    library.add_or_update("А", short, duration_sec=10.0)
    library.add_or_update("А", long_one, duration_sec=1000.0)
    stored = library.vectors()[library.find("А").slug]
    from nestor.speaker_match import cosine

    assert cosine(stored, long_one) > cosine(stored, short)


def test_zero_duration_voice_is_not_overwritten(tmp_path):
    """A voice whose stored duration is unknowable (0) must not be replaced."""
    library = VoiceprintLibrary(tmp_path)
    from nestor.speaker_match import cosine

    first = vec(1.0)
    other = vec(2.0)  # unrelated direction
    library.add_or_update("Импорт", first, duration_sec=0.0)
    slug = library.find("Импорт").slug
    library.add_or_update("Импорт", other, duration_sec=50.0)
    stored = library.vectors()[slug]
    # equal mix, not a copy of the newcomer
    assert cosine(stored, first) > 0.5
    assert cosine(stored, other) > 0.5
    assert not np.allclose(stored, other, atol=1e-6)
    # and the mirror case: an update without duration onto a lasting voice
    library.add_or_update("Импорт", other, duration_sec=0.0)
    stored2 = library.vectors()[slug]
    # a full overwrite would give ~0.0 here; the mix keeps a real share
    assert cosine(stored2, first) > 0.3
    assert not np.allclose(stored2, other, atol=1e-6)


def test_delete_removes_vector_and_meta(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    library.add_or_update("Мария", vec(4.0))
    slug = library.find("Мария").slug
    assert (tmp_path / "voiceprints" / f"{slug}.npy").exists()

    assert library.delete("мария") is True  # case-insensitive
    assert library.delete("мария") is False
    assert library.list() == []
    assert not (tmp_path / "voiceprints" / f"{slug}.npy").exists()
    # survives a reload
    assert VoiceprintLibrary(tmp_path).list() == []


def test_damaged_json_starts_empty(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    library.add_or_update("Иван", vec(5.0))
    (tmp_path / "voiceprints" / "voiceprints.json").write_text("{broken", encoding="utf-8")
    revived = VoiceprintLibrary(tmp_path)
    assert revived.list() == []  # not a crash: an empty library


def test_older_schema_version_loads_with_warning(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    library.add_or_update("Иван", vec(6.0))
    path = tmp_path / "voiceprints" / "voiceprints.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = SCHEMA_VERSION - 1
    payload["voiceprints"][0]["num_utterances"] = 3  # old field layout tolerated
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    revived = VoiceprintLibrary(tmp_path)
    assert len(revived.list()) == 1
    # the warning went to log.txt next to the library
    log_text = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "schema version" in log_text


def test_future_schema_version_does_not_crash(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    library.add_or_update("Иван", vec(7.0))
    path = tmp_path / "voiceprints" / "voiceprints.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = SCHEMA_VERSION + 5
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert len(VoiceprintLibrary(tmp_path).list()) == 1


def test_russian_names_survive_roundtrip(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    names = ["Иван Петров", "Ёжик", "Анна-Мария", "Ольга Фёдоровна"]
    for index, name in enumerate(names):
        library.add_or_update(name, vec(10.0 + index))
    revived = VoiceprintLibrary(tmp_path)
    stored_names = {meta.display_name for meta in revived.list()}
    assert stored_names == set(names)
    for name in names:
        assert revived.find(name.lower()) is not None


def test_zero_embedding_rejected(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    with pytest.raises(ValueError):
        library.add_or_update("Иван", np.zeros(192, dtype=np.float32))
    with pytest.raises(ValueError):
        library.add_or_update("", vec(8.0))
    assert library.list() == []


# ------------------------------------------------------------------ comments
def test_comment_roundtrip(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    library.add_or_update("Иван", vec(1.0))
    slug = library.find("Иван").slug

    updated = library.set_comment(slug, "Коллега, инженер")
    assert updated is not None and updated.comment == "Коллега, инженер"
    # survives a reload
    revived = VoiceprintLibrary(tmp_path)
    assert revived.find("Иван").comment == "Коллега, инженер"
    # clearing
    revived.set_comment(slug, "")
    assert VoiceprintLibrary(tmp_path).find("Иван").comment == ""
    assert revived.set_comment("no-such-slug", "x") is None


# --------------------------------------------------------- export / import
def test_export_import_roundtrip(tmp_path):
    source = VoiceprintLibrary(tmp_path)
    source.add_or_update("Иван", vec(1.0), duration_sec=120.0)
    source.set_comment(source.find("Иван").slug, "Коллега")
    source.add_or_update("Пётр", vec(2.0), duration_sec=30.0)

    payload = source.export_data()
    assert payload["format"] == "nestor-voices"
    assert payload["count"] == 2

    # a different machine = a different data root
    other = VoiceprintLibrary(tmp_path / "elsewhere")
    report = other.import_data(payload)
    assert sorted(report["added"]) == ["Иван", "Пётр"]
    assert not report["skipped"]

    imported = other.find("Иван")
    assert imported.comment == "Коллега"
    assert imported.total_duration_sec == pytest.approx(120.0)
    from nestor.speaker_match import cosine

    assert cosine(
        other.vectors()[imported.slug], source.vectors()[source.find("Иван").slug]
    ) > 0.999


def test_import_merges_same_name(tmp_path):
    source = VoiceprintLibrary(tmp_path)
    base = vec(1.0)
    source.add_or_update("Иван", base, duration_sec=100.0)

    other = VoiceprintLibrary(tmp_path / "elsewhere")
    other.add_or_update("иван", vec(1.0), duration_sec=50.0)  # case-insensitive

    report = other.import_data(source.export_data())
    assert report["merged"] == ["Иван"]
    meta = other.find("Иван")
    assert meta.num_utterances == 2  # blended, not overwritten


def test_import_rejects_garbage(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    with pytest.raises(ValueError):
        library.import_data({"format": "something-else"})
    with pytest.raises(ValueError):
        library.import_data({"format": "nestor-voices", "voices": "not-a-list"})
    # a broken entry is skipped, the rest lands
    payload = {
        "format": "nestor-voices",
        "version": 1,
        "voices": [
            {"meta": {"display_name": "Без вектора"}, "vector_b64": ""},
            {
                "meta": {"display_name": "Норм"},
                "vector_b64": __import__("base64").b64encode(
                    np.ascontiguousarray(vec(3.0)).tobytes()
                ).decode("ascii"),
                "dim": 192,
            },
        ],
    }
    report = library.import_data(payload)
    assert report["added"] == ["Норм"]
    assert len(report["skipped"]) == 1
