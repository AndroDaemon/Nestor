"""Tests for the glossary post-correction pass (nestor/glossary.py).

Layer B: rewrite heard forms ("докер", "днс") to the canonical term ("Docker",
"DNS"). Fully offline and deterministic; the decoder is never involved.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.glossary import (  # noqa: E402
    Glossary,
    Term,
    apply_to_segments,
    build_alias_map,
    dump_glossary,
    ensure_glossary,
    load_glossary,
    merge_glossary,
    postcorrect,
    remove_term,
    resolve_glossary_path,
    save_glossary,
    seed_glossary_text,
    upsert_term,
)
from nestor.types import Segment, Word  # noqa: E402


def _glossary(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "hotwords.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_shipped_glossary_parses():
    # the term list that seeds a fresh data root on first use
    glossary = Glossary.parse(seed_glossary_text())
    assert len(glossary) >= 100
    assert any(t.term == "Docker" for t in glossary.terms)


def test_simple_alias_replaced():
    glossary = Glossary([Term(term="Docker", aliases=["докер"])])
    text, changes = postcorrect("Разверни докер на контуре.", build_alias_map(glossary))
    assert "Docker" in text
    assert "докер" not in text
    assert changes == [("докер", "Docker")]


def test_multiword_alias(tmp_path):
    path = _glossary(tmp_path, '''
        - term: "IP-адрес"
          aliases: ["айпи адрес", "айпи"]
    ''')
    alias_map = build_alias_map(load_glossary(path))
    text, changes = postcorrect("Задай айпи адрес и маску.", alias_map)
    assert "IP-адрес" in text
    assert changes == [("айпи адрес", "IP-адрес")]


def test_short_alias_only_uppercase(tmp_path):
    path = _glossary(tmp_path, '''
        - term: "NAS"
          aliases: ["нас"]
    ''')
    alias_map = build_alias_map(load_glossary(path))
    # a lower-case ordinary word is NOT rewritten (would corrupt speech)
    text, changes = postcorrect("Он сказал нас об этом.", alias_map)
    assert changes == []
    assert "нас" in text
    # an upper-case token is
    text, changes = postcorrect("Подключи НАС к сети.", alias_map)
    assert changes == [("НАС", "NAS")]
    assert "NAS" in text


def test_short_latin_alias_matches_any_case(tmp_path):
    path = _glossary(tmp_path, '''
        - term: "DNS"
          aliases: ["днс", "dns"]
    ''')
    alias_map = build_alias_map(load_glossary(path))
    # Cyrillic must be upper-case...
    _, changes = postcorrect("проверь днс", alias_map)
    assert changes == []
    # ...but a short Latin alias is not an ordinary Russian word: any case.
    text, changes = postcorrect("проверь dns", alias_map)
    assert changes == [("dns", "DNS")]
    assert text == "проверь DNS"
    # already the canonical spelling: no-op, not counted as a change
    text, changes = postcorrect("проверь DNS", alias_map)
    assert changes == []
    assert text == "проверь DNS"


def test_alias_equal_to_term_is_skipped():
    glossary = Glossary([Term(term="DNS", aliases=["DNS"])])
    assert build_alias_map(glossary) == []


def test_longest_phrase_wins(tmp_path):
    path = _glossary(tmp_path, '''
        - term: "IP-адрес"
          aliases: ["айпи адрес"]
        - term: "PI"
          aliases: ["айпи"]
    ''')
    alias_map = build_alias_map(load_glossary(path))
    # "айпи адрес" must win over "айпи" (which is too short anyway)
    text, _ = postcorrect("дай айпи адрес", alias_map)
    assert text == "дай IP-адрес"


def test_apply_to_segments_updates_text_and_words():
    glossary = Glossary([Term(term="Docker", aliases=["докер"])])
    seg = Segment(
        start=0.0, end=1.0, speaker="SPEAKER_00",
        text="разверни докер тут",
        words=[Word("разверни", 0.0, 0.2), Word("докер", 0.2, 0.5), Word("тут", 0.5, 1.0)],
    )
    changed = apply_to_segments([seg], glossary)
    assert changed == 1
    assert seg.text == "разверни Docker тут"
    assert seg.words[1].text == "Docker"


def test_missing_and_broken_glossary_are_empty(tmp_path):
    assert len(load_glossary(tmp_path / "nope.yaml")) == 0
    broken = tmp_path / "broken.yaml"
    broken.write_text("::: not: valid: yaml\n  - [", encoding="utf-8")
    assert len(load_glossary(broken)) == 0  # never raises


def test_resolve_creates_and_returns_the_single_file(tmp_path):
    # there is exactly one glossary: <data root>/hotwords.yaml, created on use
    target = resolve_glossary_path(tmp_path)
    assert target == tmp_path / "hotwords.yaml"
    assert target.exists()
    assert len(load_glossary(target)) >= 100
    # a second call returns the same file rather than re-seeding it
    assert resolve_glossary_path(tmp_path) == target


def test_realistic_sentence(tmp_path):
    path = _glossary(tmp_path, '''
        - term: "DNS"
          aliases: ["днс", "ДНС"]
        - term: "Kubernetes"
          aliases: ["кубернетес"]
        - term: "Docker"
          aliases: ["докер"]
    ''')
    alias_map = build_alias_map(load_glossary(path))
    text, changes = postcorrect(
        "Разверни докер и кубернетес, потом проверь ДНС на шлюзе.", alias_map
    )
    assert "Docker" in text and "Kubernetes" in text and "DNS" in text
    assert len(changes) == 3


# --------------------------------------------------------------------------- #
# Editing: parse / dump / save / upsert / remove / merge
# --------------------------------------------------------------------------- #
def test_parse_and_dump_roundtrip():
    glossary = Glossary.parse('- term: "API"\n  aliases: [апи, api]\n')
    assert len(glossary) == 1
    assert glossary.terms[0].term == "API"
    assert glossary.terms[0].aliases == ["апи", "api"]
    dumped = dump_glossary(glossary)
    assert '- term: "API"' in dumped
    assert 'aliases: ["апи", "api"]' in dumped
    # the dump is valid YAML again
    assert dump_glossary(Glossary.parse(dumped)) == dumped


def test_parse_skips_malformed_entries():
    glossary = Glossary.parse(
        "- term: \"Ok\"\n  aliases: [a]\n- no term here\n- term: \"\"\n"
    )
    assert [t.term for t in glossary.terms] == ["Ok"]


def test_upsert_adds_then_replaces(tmp_path):
    path = _glossary(tmp_path, '- term: "Docker"\n  aliases: ["докер"]\n')
    glossary = load_glossary(path)
    assert upsert_term(glossary, "DNS", ["днс", "dns"]) is glossary
    assert len(glossary) == 2
    # replacing by name keeps the count but swaps aliases
    upsert_term(glossary, "Docker", ["докер", "докерс"])
    assert len(glossary) == 2
    docker = next(t for t in glossary.terms if t.term == "Docker")
    assert docker.aliases == ["докер", "докерс"]


def test_remove_term(tmp_path):
    path = _glossary(tmp_path, '''
        - term: "Docker"
          aliases: ["докер"]
        - term: "DNS"
          aliases: ["днс"]
    ''')
    glossary = load_glossary(path)
    assert remove_term(glossary, "Docker") is True
    assert [t.term for t in glossary.terms] == ["DNS"]
    assert remove_term(glossary, "нет-такого") is False


def test_save_then_reload_reflects_change(tmp_path):
    path = tmp_path / "hotwords.yaml"
    glossary = Glossary([Term(term="Kubernetes", aliases=["кубернетес"])])
    save_glossary(glossary, path)
    reloaded = load_glossary(path)
    assert [t.term for t in reloaded.terms] == ["Kubernetes"]
    # the mtime cache was invalidated: a second write is visible too
    upsert_term(reloaded, "Docker", ["докер"])
    save_glossary(reloaded, path)
    assert len(load_glossary(path)) == 2


def test_merge_glossary_counts_new_terms():
    base = Glossary([Term(term="Docker", aliases=["докер"])])
    extra = Glossary([
        Term(term="Docker", aliases=["докер", "докерс"]),  # replaces, not new
        Term(term="DNS", aliases=["днс"]),  # new
    ])
    added = merge_glossary(base, extra)
    assert added == 1
    assert len(base) == 2
    assert next(t for t in base.terms if t.term == "Docker").aliases == ["докер", "докерс"]


def test_ensure_glossary_seeds_once(tmp_path):
    target = ensure_glossary(tmp_path)
    assert target == tmp_path / "hotwords.yaml"
    assert target.exists()
    # it was seeded with the shipped term list
    assert len(load_glossary(target)) >= 100
    # a second call is a no-op, not a re-seed
    upsert_term(load_glossary(target), "TestTerm", ["тесттерм"])
    save_glossary(load_glossary(target), target)
    assert "TestTerm" in ensure_glossary(tmp_path).read_text(encoding="utf-8")

