"""Glossary-based post-correction of recognised terms.

Nestor often hears an acronym the way it is *said* — "докер", "днс",
"айпи адрес" — instead of the canonical spelling used in documents. This
module rewrites those heard forms back to their canonical terms, fully
offline and deterministically: a straight alias -> term substitution over the
recognised text, with no model involved.

This is *layer B* only. The alternative — biasing the decoder itself with
sherpa-onnx hotwords (``modified_beam_search``) — was measured on Russian
speech and rejected: it added almost nothing over the post-correction, cost
~1.5x decode time, and sometimes turned a correct word into a wrong one.

The glossary is a YAML file:

    - term: "API"                # canonical form, what lands in the transcript
      aliases: ["апи", "api"]    # heard / mis-spelled forms to replace

A user copy lives at ``<data root>/hotwords.yaml``; when absent the bundled
:func:`default_glossary_path` is used. Nothing here touches the network.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Words (terms/aliases) shorter than this are only rewritten when the
#: recognised token is upper-case -- but only for Cyrillic aliases: a short
#: Cyrillic token is ambiguous with ordinary Russian words ("нас" = "us" vs
#: NAS), while a short Latin token ("dns", "5g") never is.
SHORT_ALIAS_CHARS = 4

#: Aliases shorter than this are ignored entirely ("пр", "си").
MIN_ALIAS_CHARS = 3

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


# --------------------------------------------------------------------------- #
# Glossary
# --------------------------------------------------------------------------- #
@dataclass
class Term:
    """A canonical term and the forms it may be heard as."""

    term: str
    aliases: List[str] = field(default_factory=list)


@dataclass
class Glossary:
    """A parsed glossary file."""

    terms: List[Term]

    @classmethod
    def load(cls, path: "str | Path") -> "Glossary":
        """Read a YAML glossary; malformed entries are skipped silently."""
        import yaml  # imported lazily so the rest of Nestor never needs it

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls._from_data(data)

    @classmethod
    def parse(cls, text: str) -> "Glossary":
        """Parse a glossary from YAML (or JSON - JSON is valid YAML)."""
        import yaml

        return cls._from_data(yaml.safe_load(text))

    @classmethod
    def _from_data(cls, data: object) -> "Glossary":
        terms: List[Term] = []
        if not isinstance(data, list):
            return cls(terms)
        for entry in data:
            if not isinstance(entry, dict) or "term" not in entry:
                continue
            term = str(entry["term"]).strip()
            if not term:
                continue
            aliases = [
                str(a).strip()
                for a in (entry.get("aliases") or [])
                if str(a).strip()
            ]
            terms.append(Term(term=term, aliases=aliases))
        return cls(terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __bool__(self) -> bool:
        return bool(self.terms)


# --------------------------------------------------------------------------- #
# Paths and caching
# --------------------------------------------------------------------------- #
def default_glossary_path() -> Path:
    """The glossary shipped with the package (used when the user has none)."""
    return Path(__file__).resolve().parent / "data" / "hotwords.yaml"


def resolve_glossary_path(root: "str | Path | None" = None) -> Path:
    """Prefer a user glossary in ``root``; fall back to the bundled one."""
    if root is not None:
        candidate = Path(root) / "hotwords.yaml"
        if candidate.exists():
            return candidate
    return default_glossary_path()


_CACHE: Dict[Tuple[str, float], Optional[Glossary]] = {}


def load_glossary(path: "str | Path | None" = None) -> Glossary:
    """Load and cache a glossary. A missing/broken file yields an empty one.

    The cache is keyed by the resolved path and its modification time, so an
    edit to ``hotwords.yaml`` is picked up without restarting the app.
    """
    resolved = Path(path) if path is not None else default_glossary_path()
    try:
        stamp = resolved.stat().st_mtime
    except OSError:
        return Glossary([])
    key = (str(resolved), stamp)
    if key in _CACHE:
        return _CACHE[key] or Glossary([])
    try:
        glossary = Glossary.load(resolved)
    except Exception:  # noqa: BLE001 - a broken glossary must never fail a job
        glossary = Glossary([])
    _CACHE.clear()
    _CACHE[key] = glossary
    return glossary


# --------------------------------------------------------------------------- #
# Layer B — alias -> canonical replacement
# --------------------------------------------------------------------------- #
def _alias_words(alias: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(alias)]


def build_alias_map(
    glossary: Glossary,
) -> List[Tuple[List[str], str, bool]]:
    """Alias phrase -> canonical term, longest phrase first.

    Returns ``(words, canonical, require_upper)`` where ``words`` is the
    lower-cased alias split into words. ``require_upper`` marks short aliases
    that are only applied to an upper-case token, so that ordinary words
    ("нас", "лак", "сид") are not rewritten by accident.
    """
    pairs: List[Tuple[List[str], str, bool]] = []
    for term in glossary.terms:
        canonical = term.term.strip()
        if not canonical:
            continue
        for alias in term.aliases:
            alias = alias.strip()
            if not alias:
                continue
            if alias == canonical:
                continue  # exact same spelling: nothing to correct
            words = _alias_words(alias)
            if not words:
                continue
            joined_len = sum(len(w) for w in words)
            if joined_len < MIN_ALIAS_CHARS:
                continue
            # Short Cyrillic aliases are ambiguous with ordinary words, so they
            # must be recognised in upper case; short Latin ones ("dns", "5g")
            # are not words in Russian and may match case-insensitively.
            require_upper = joined_len < SHORT_ALIAS_CHARS and any(
                "а" <= ch.lower() <= "я" or ch.lower() == "ё" for w in words for ch in w
            )
            pairs.append((words, canonical, require_upper))
    # longest phrases first so "эй пи ай" wins over "пи ай"
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def _tokens_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _is_upper(token: str) -> bool:
    return token.isupper() and any(ch.isalpha() for ch in token)


def postcorrect(
    text: str, alias_map: Sequence[Tuple[List[str], str, bool]]
) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace alias phrases with canonical terms.

    Returns ``(new_text, changes)`` where ``changes`` is a list of
    ``(original, canonical)`` pairs for reporting. Replacement is whole-word,
    longest-phrase-first and non-overlapping; separators between the tokens of
    a phrase (spaces, hyphens) are consumed with it.
    """
    if not text or not alias_map:
        return text, []

    tokens = _tokens_with_spans(text)
    words = [t[0].lower() for t in tokens]
    n = len(words)
    consumed = [False] * n
    replacements: List[Tuple[int, int, str]] = []  # (token_start, token_end, canonical)

    for idx in range(n):
        if consumed[idx]:
            continue
        for phrase, canonical, require_upper in alias_map:
            k = len(phrase)
            if idx + k > n:
                continue
            if any(consumed[idx + j] for j in range(k)):
                continue
            if words[idx:idx + k] != phrase:
                continue
            if require_upper and not all(_is_upper(tokens[idx + j][0]) for j in range(k)):
                continue
            # already the canonical spelling (only its case could differ): skip,
            # so the change count reflects real corrections only
            if text[tokens[idx][1]:tokens[idx + k - 1][2]] == canonical:
                continue
            for j in range(k):
                consumed[idx + j] = True
            replacements.append((idx, idx + k, canonical))
            break

    if not replacements:
        return text, []

    changes: List[Tuple[str, str]] = []
    out: List[str] = []
    cursor = 0
    for start, end, canonical in replacements:
        out.append(text[cursor:tokens[start][1]])
        original = text[tokens[start][1]:tokens[end - 1][2]]
        changes.append((original, canonical))
        out.append(canonical)
        cursor = tokens[end - 1][2]
    out.append(text[cursor:])
    return "".join(out), changes


def apply_to_segments(segments: Sequence, glossary: Glossary) -> int:
    """Rewrite terms in each segment, in place. Returns the number of changes.

    ``segment.text`` (what the user reads and every export writes) gets the
    full treatment, including multi-word aliases. ``segment.words`` are
    corrected word by word as well so word-level data stays in sync wherever a
    single word maps to a single term.
    """
    alias_map = build_alias_map(glossary)
    if not alias_map:
        return 0
    total = 0
    for segment in segments:
        new_text, changes = postcorrect(getattr(segment, "text", "") or "", alias_map)
        if changes:
            segment.text = new_text
            total += len(changes)
        for word in getattr(segment, "words", ()) or ():
            word_text, word_changes = postcorrect(word.text, alias_map)
            if word_changes:
                word.text = word_text
    return total


# --------------------------------------------------------------------------- #
# Editing — the user glossary file
# --------------------------------------------------------------------------- #
def user_glossary_path(root: "str | Path") -> Path:
    """Where a user's own glossary lives (``<data root>/hotwords.yaml``)."""
    return Path(root) / "hotwords.yaml"


def glossary_source(root: "str | Path | None" = None) -> str:
    """``"user"`` when the data-root file exists, else ``"bundled"``."""
    if root is not None and user_glossary_path(root).exists():
        return "user"
    return "bundled"


def editable_glossary_path(root: "str | Path") -> Path:
    """The file edits should go to: the user file, seeded from the bundled one.

    The bundled glossary lives inside the installed package and would be
    overwritten by an update, so the first edit copies it to the data root and
    every later edit works on that copy. Returns the path (not yet written).
    """
    target = user_glossary_path(root)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        bundled = default_glossary_path()
        if bundled.exists():
            target.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def dump_glossary(glossary: Glossary) -> str:
    """Render a glossary back to YAML text (no PyYAML dependency needed)."""
    lines: List[str] = []
    for term in glossary.terms:
        canonical = term.term.strip()
        if not canonical:
            continue
        lines.append(f'- term: {_yaml_dq(canonical)}')
        aliases = [a.strip() for a in term.aliases if a.strip()]
        if aliases:
            rendered = ", ".join(_yaml_dq(a) for a in aliases)
            lines.append(f'  aliases: [{rendered}]')
        else:
            lines.append("  aliases: []")
    return "\n".join(lines) + "\n"


def _yaml_dq(text: str) -> str:
    """A double-quoted YAML scalar (escaping just what YAML needs)."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_glossary(glossary: Glossary, path: "str | Path") -> Path:
    """Write a glossary to ``path`` atomically. Returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(dump_glossary(glossary), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, target)
            break
        except PermissionError:
            import time as _time

            _time.sleep(0.01 + 0.01 * attempt)
    else:  # last resort: never lose the data
        target.write_text(dump_glossary(glossary), encoding="utf-8")
        tmp.unlink(missing_ok=True)
    # the loader caches by mtime; a fresh write must not serve stale terms
    _CACHE.clear()
    return target


def upsert_term(glossary: Glossary, term: str, aliases: Sequence[str]) -> Glossary:
    """Add a term, or replace the aliases of an existing one (by name)."""
    name = term.strip()
    cleaned = [a.strip() for a in aliases if a.strip()]
    existing = next((t for t in glossary.terms if t.term == name), None)
    if existing is not None:
        existing.aliases = cleaned
        return glossary
    glossary.terms.append(Term(term=name, aliases=cleaned))
    return glossary


def remove_term(glossary: Glossary, term: str) -> bool:
    """Drop a term by name; True when something was removed."""
    before = len(glossary.terms)
    glossary.terms = [t for t in glossary.terms if t.term != term.strip()]
    return len(glossary.terms) != before


def merge_glossary(base: Glossary, extra: Glossary) -> int:
    """Fold ``extra`` into ``base`` (extra wins on a name clash). Returns count."""
    added = 0
    for term in extra.terms:
        exists = any(t.term == term.term for t in base.terms)
        upsert_term(base, term.term, term.aliases)
        if not exists:
            added += 1
    return added


__all__ = [
    "Glossary",
    "Term",
    "MIN_ALIAS_CHARS",
    "SHORT_ALIAS_CHARS",
    "apply_to_segments",
    "build_alias_map",
    "default_glossary_path",
    "dump_glossary",
    "editable_glossary_path",
    "glossary_source",
    "load_glossary",
    "merge_glossary",
    "postcorrect",
    "remove_term",
    "resolve_glossary_path",
    "save_glossary",
    "upsert_term",
    "user_glossary_path",
]
