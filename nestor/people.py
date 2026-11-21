"""People: the humans behind the voiceprints.

A person is a name plus every spelling of it met in practice — «Шаталов
Дмитрий» and «Шаталов Дмитрий Олегович» are one person — plus the voiceprints
recorded for them.  ``voiceprints.py`` keeps the vectors; this module keeps the
identity on top of them.

Layout inside the data root (default ``%LOCALAPPDATA%\\Nestor``)::

    voiceprints/people.json      people (names, aliases, voice slugs)

The invariant is simple: **every voiceprint belongs to exactly one person**, and
every person owns at least one voiceprint.  ``reconcile()`` enforces it, so a
missing or damaged ``people.json`` is never fatal: the library rebuilds itself
with one person per stored voice.

The voice matching in ``speaker_match.py`` knows nothing about people — it keeps
working with voiceprints as before.  Names from people are applied on top of the
matching result, which keeps the familiar engines untouched.
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .library import default_root
from .voiceprints import VoiceprintLibrary, make_slug

PEOPLE_SCHEMA_VERSION = 1
PEOPLE_FILE = "people.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_name(name: str) -> str:
    return " ".join((name or "").split())


def same_name(left: str, right: str) -> bool:
    """True when two spellings name the same person.

    Russian full names are the reason this exists: «Шаталов Дмитрий» is a prefix
    of «Шаталов Дмитрий Олегович», so a plain string ratio is not enough.  Two
    multi-word names match when one is a word-prefix of the other, or when they
    are almost letter-for-letter the same (an OCR slip).
    """
    a = _clean_name(left).casefold()
    b = _clean_name(right).casefold()
    if not a or not b:
        return False
    if a == b:
        return True
    words_a, words_b = a.split(), b.split()
    if len(words_a) >= 2 and len(words_b) >= 2:
        shorter, longer = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
        if longer[: len(shorter)] == shorter:
            return True
    return SequenceMatcher(None, a, b).ratio() >= 0.93


@dataclass
class Person:
    """One person: the shown name, the other spellings, and their voices."""

    id: str
    display_name: str
    aliases: List[str] = field(default_factory=list)
    voices: List[str] = field(default_factory=list)
    comment: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = ""
    schema_version: int = PEOPLE_SCHEMA_VERSION

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Person":
        aliases = data.get("aliases") or []
        voices = data.get("voices") or []
        return cls(
            id=str(data.get("id", "")),
            display_name=str(data.get("display_name", "")),
            aliases=[str(item) for item in aliases if str(item).strip()],
            voices=[str(item) for item in voices if str(item).strip()],
            comment=str(data.get("comment", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            schema_version=int(data.get("schema_version", PEOPLE_SCHEMA_VERSION)),
        )

    def spellings(self) -> List[str]:
        """Display name first, then the other known spellings."""
        seen = {self.display_name.casefold()}
        result = [self.display_name]
        for alias in self.aliases:
            key = alias.casefold()
            if key not in seen:
                seen.add(key)
                result.append(alias)
        return result


class PeopleLibrary:
    """File-backed storage of people, layered over the voiceprint library."""

    def __init__(
        self,
        root: Optional[Path] = None,
        voices: Optional[VoiceprintLibrary] = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.dir = self.root / "voiceprints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.dir / PEOPLE_FILE
        self.voices = voices if voices is not None else VoiceprintLibrary(self.root)
        self._people: Dict[str, Person] = {}
        self.load()
        self.reconcile()

    # ------------------------------------------------------------------- i/o
    def save(self) -> None:
        payload = {
            "schema_version": PEOPLE_SCHEMA_VERSION,
            "people": [person.to_dict() for person in self._people.values()],
        }
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.json_path.with_name(self.json_path.name + ".tmp")
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(tmp, self.json_path)
                break
            except PermissionError:
                time.sleep(0.01 + 0.01 * attempt)
        else:
            self.json_path.write_text(text, encoding="utf-8")
            tmp.unlink(missing_ok=True)

    def load(self) -> None:
        """Load people; tolerate a missing or damaged file."""
        self._people = {}
        if not self.json_path.exists():
            return
        try:
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._warn("people.json is damaged, rebuilding from the voiceprints")
            return
        entries = payload.get("people", []) if isinstance(payload, dict) else []
        version = payload.get("schema_version", PEOPLE_SCHEMA_VERSION) if isinstance(payload, dict) else PEOPLE_SCHEMA_VERSION
        if version != PEOPLE_SCHEMA_VERSION:
            self._warn(f"people.json schema version {version} differs from the expected {PEOPLE_SCHEMA_VERSION}; loading anyway")
        for entry in entries:
            try:
                person = Person.from_dict(entry)
            except Exception:  # noqa: BLE001 - skip broken entries
                continue
            if person.id and person.display_name:
                self._people[person.id] = person

    def _warn(self, message: str) -> None:
        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = self.root / "log.txt"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] people: {message}\n")
        except OSError:
            pass

    # -------------------------------------------------------------- internal
    def _unique_id(self, name: str) -> str:
        base = make_slug(name) or "person"
        candidate = base
        suffix = 2
        while candidate in self._people:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def _touch(self, person: Person) -> None:
        person.updated_at = _now()

    @staticmethod
    def _worth_keeping(person: Person) -> bool:
        """A person with no voices is still worth keeping when something else
        was filled in by hand — an imported list of names, for instance.  Only
        a bare name with nothing behind it is dropped."""
        return bool(person.voices or person.aliases or person.comment)

    # ------------------------------------------------------------------- api
    def list(self) -> List[Person]:
        """All people, alphabetically by shown name."""
        return sorted(self._people.values(), key=lambda item: item.display_name.casefold())

    def get(self, person_id: str) -> Optional[Person]:
        return self._people.get(person_id)

    def person_of_voice(self, slug: str) -> Optional[Person]:
        for person in self._people.values():
            if slug in person.voices:
                return person
        return None

    def names(self) -> Dict[str, str]:
        """Voice slug -> the name to show for it (the person's name)."""
        result: Dict[str, str] = {}
        for person in self._people.values():
            for slug in person.voices:
                result[slug] = person.display_name
        return result

    def vectors(self) -> Dict[str, "np.ndarray"]:
        """Voice slug -> voiceprint vector, like the voice library itself.

        Matching only ever looks at voices; this keeps :class:`PeopleLibrary`
        a drop-in for :class:`~nestor.voiceprints.VoiceprintLibrary`.
        """
        return self.voices.vectors()

    def display_for(self, slug: str, fallback: str = "") -> str:
        person = self.person_of_voice(slug)
        if person is not None:
            return person.display_name
        return fallback or self.voices.names().get(slug, slug)

    def find_by_name(self, name: str) -> Optional[Person]:
        """Person whose shown name or any spelling matches (case-insensitive)."""
        wanted = _clean_name(name).casefold()
        if not wanted:
            return None
        for person in self._people.values():
            if any(spelling.casefold() == wanted for spelling in person.spellings()):
                return person
        return None

    def similar_person(self, name: str) -> Optional[Tuple[Person, str]]:
        """Best guess for a newly seen name: (person, the spelling it matched)."""
        cleaned = _clean_name(name)
        if not cleaned:
            return None
        exact = self.find_by_name(cleaned)
        if exact is not None:
            return exact, cleaned
        best: Optional[Tuple[Person, str]] = None
        best_score = 0.0
        for person in self._people.values():
            for spelling in person.spellings():
                if same_name(cleaned, spelling):
                    score = SequenceMatcher(None, cleaned.casefold(), spelling.casefold()).ratio()
                    # a word-prefix match is strong even when the ratio is low
                    if cleaned.casefold().split()[0:1] == spelling.casefold().split()[0:1]:
                        score = max(score, 0.9)
                    if score > best_score:
                        best_score = score
                        best = (person, spelling)
        return best

    def create(self, name: str, voices: Optional[List[str]] = None) -> Person:
        cleaned = _clean_name(name)
        if not cleaned:
            raise ValueError("имя человека не может быть пустым")
        person = Person(
            id=self._unique_id(cleaned),
            display_name=cleaned,
            voices=[slug for slug in (voices or []) if slug],
        )
        self._people[person.id] = person
        for slug in person.voices:
            self._detach_from_others(slug, keep=person.id)
        self.save()
        return person

    def rename(self, person_id: str, new_name: str) -> Optional[Person]:
        """Change the shown name; the previous one stays as a known spelling."""
        person = self._people.get(person_id)
        cleaned = _clean_name(new_name)
        if person is None or not cleaned:
            return None
        if cleaned.casefold() == person.display_name.casefold():
            return person
        self._remember(person, person.display_name)
        person.display_name = cleaned
        person.aliases = [alias for alias in person.aliases if alias.casefold() != cleaned.casefold()]
        self._touch(person)
        self.save()
        return person

    def add_alias(self, person_id: str, alias: str) -> Optional[Person]:
        person = self._people.get(person_id)
        cleaned = _clean_name(alias)
        if person is None or not cleaned:
            return None
        person.aliases = [item for item in person.aliases if item.casefold() != cleaned.casefold()]
        if cleaned.casefold() not in {item.casefold() for item in person.spellings()}:
            person.aliases.append(cleaned)
        self._touch(person)
        self.save()
        return person

    def remove_alias(self, person_id: str, alias: str) -> Optional[Person]:
        person = self._people.get(person_id)
        if person is None:
            return None
        wanted = _clean_name(alias).casefold()
        person.aliases = [item for item in person.aliases if item.casefold() != wanted]
        self._touch(person)
        self.save()
        return person

    def set_display(self, person_id: str, name: str) -> Optional[Person]:
        """Show one of the known spellings in the transcripts."""
        person = self._people.get(person_id)
        cleaned = _clean_name(name)
        if person is None or not cleaned:
            return None
        if cleaned.casefold() != person.display_name.casefold():
            self._remember(person, person.display_name)
        person.display_name = cleaned
        person.aliases = [item for item in person.aliases if item.casefold() != cleaned.casefold()]
        self._touch(person)
        self.save()
        return person

    def _remember(self, person: Person, name: str) -> None:
        """Keep a spelling around.  The current shown name is not excluded: a
        rename remembers the name being replaced, which is still the display
        name at the moment this is called."""
        cleaned = _clean_name(name)
        if not cleaned:
            return
        known = {alias.casefold() for alias in person.aliases}
        if cleaned.casefold() not in known:
            person.aliases.append(cleaned)

    def attach_voice(self, person_id: str, slug: str) -> Optional[Person]:
        person = self._people.get(person_id)
        if person is None or not slug:
            return None
        self._detach_from_others(slug, keep=person_id)
        if slug not in person.voices:
            person.voices.append(slug)
            self._touch(person)
            self.save()
        return person

    def detach_voice(self, person_id: str, slug: str) -> Optional[Person]:
        """Unhook a voice; if the person is left voiceless with nothing else
        filled in, it disappears."""
        person = self._people.get(person_id)
        if person is None:
            return None
        person.voices = [item for item in person.voices if item != slug]
        if not self._worth_keeping(person):
            self._people.pop(person.id, None)
            self.save()
            return None
        self._touch(person)
        self.save()
        return person

    def set_comment(self, person_id: str, comment: str) -> Optional[Person]:
        person = self._people.get(person_id)
        if person is None:
            return None
        person.comment = (comment or "").strip()
        self._touch(person)
        self.save()
        return person

    def merge(self, source_id: str, target_id: str) -> Optional[Person]:
        """Fold one person into another: voices and spellings move over."""
        if source_id == target_id:
            return self._people.get(target_id)
        source = self._people.get(source_id)
        target = self._people.get(target_id)
        if source is None or target is None:
            return None
        for spelling in source.spellings():
            if spelling.casefold() != target.display_name.casefold():
                self._remember(target, spelling)
        for slug in source.voices:
            if slug not in target.voices:
                target.voices.append(slug)
        if not target.comment and source.comment:
            target.comment = source.comment
        self._people.pop(source_id, None)
        self._touch(target)
        self.save()
        return target

    def delete(self, person_id: str, with_voices: bool = True) -> bool:
        """Remove a person; by default their voiceprints go too."""
        person = self._people.get(person_id)
        if person is None:
            return False
        if with_voices:
            for slug in list(person.voices):
                self.voices.delete(slug)
        self._people.pop(person_id, None)
        self.save()
        return True

    def reconcile(self) -> bool:
        """Restore the invariant: one person per voice, no abandoned people.

        Called on load, so a missing ``people.json`` (a fully working library
        from before this feature) simply becomes one person per stored voice.
        A person left without voices is dropped unless a name list or a note
        was filled in by hand.
        """
        changed = False
        known_slugs = set(self.voices.names())
        for person in list(self._people.values()):
            kept = [slug for slug in person.voices if slug in known_slugs]
            if len(kept) != len(person.voices):
                person.voices = kept
                changed = True
            clean_aliases = [
                alias for alias in person.aliases
                if alias.casefold() != person.display_name.casefold()
            ]
            if len(clean_aliases) != len(person.aliases):
                person.aliases = clean_aliases
                changed = True
            if not self._worth_keeping(person):
                self._people.pop(person.id, None)
                changed = True
        taken = {slug for person in self._people.values() for slug in person.voices}
        for slug, name in self.voices.names().items():
            if slug in taken:
                continue
            self._people[self._unique_id(name)] = Person(
                id=self._unique_id(name), display_name=name, voices=[slug],
            )
            changed = True
        if changed:
            self.save()
        return changed

    # ------------------------------------------------------------ exchange
    def export_data(self) -> List[Dict]:
        """People as plain JSON: names, spellings and their voice names."""
        voice_names = self.voices.names()
        items: List[Dict] = []
        for person in self.list():
            items.append({
                "display_name": person.display_name,
                "aliases": list(person.aliases),
                "voices": [voice_names.get(slug, slug) for slug in person.voices],
                "comment": person.comment,
                "created_at": person.created_at,
                "updated_at": person.updated_at,
            })
        return items

    def import_data(self, items: List[Dict]) -> Dict:
        """Merge exported people back in, matching voiceprints by name."""
        report = {"added": [], "merged": [], "skipped": []}
        if not isinstance(items, list):
            return report
        for item in items:
            try:
                if not isinstance(item, dict):
                    raise ValueError("неверная запись")
                name = _clean_name(str(item.get("display_name") or ""))
                if not name:
                    raise ValueError("нет имени")
                voice_names = [str(entry) for entry in (item.get("voices") or []) if str(entry).strip()]
                slugs = []
                for voice_name in voice_names:
                    meta = self.voices.find(voice_name) or self.voices.find(voice_name.strip())
                    if meta is not None:
                        slugs.append(meta.slug)
                existing = self.find_by_name(name)
                if existing is not None:
                    for alias in item.get("aliases") or []:
                        self.add_alias(existing.id, str(alias))
                    for slug in slugs:
                        self.attach_voice(existing.id, slug)
                    if not existing.comment and item.get("comment"):
                        self.set_comment(existing.id, str(item["comment"]))
                    report["merged"].append(name)
                else:
                    person = self.create(name, voices=slugs)
                    for alias in item.get("aliases") or []:
                        self.add_alias(person.id, str(alias))
                    if item.get("comment"):
                        self.set_comment(person.id, str(item["comment"]))
                    report["added"].append(name)
            except Exception as exc:  # noqa: BLE001 - one bad entry must not sink the import
                report["skipped"].append(f"{(item or {}).get('display_name', '?')}: {exc}")
        self.reconcile()
        return report

    def _detach_from_others(self, slug: str, keep: str) -> None:
        for person in self._people.values():
            if person.id != keep and slug in person.voices:
                person.voices = [item for item in person.voices if item != slug]
                self._touch(person)
