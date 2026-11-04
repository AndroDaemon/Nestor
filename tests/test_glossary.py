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
    default_glossary_path,
    load_glossary,
    postcorrect,
    resolve_glossary_path,
)
from nestor.types import Segment, Word  # noqa: E402


def _glossary(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "hotwords.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_bundled_glossary_loads():
    path = default_glossary_path()
    assert path.exists(), f"bundled glossary missing: {path}"
    glossary = load_glossary(path)
    assert len(glossary) >= 100


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


def test_resolve_prefers_user_file(tmp_path):
    user = _glossary(tmp_path, '- term: "X"\n  aliases: ["y"]\n')
    assert resolve_glossary_path(tmp_path) == user
    # nothing there -> bundled file
    assert resolve_glossary_path(tmp_path / "empty") == default_glossary_path()


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
