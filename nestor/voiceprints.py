"""Persistent library of speaker voiceprints (embeddings).

Layout inside the data root (default ``%LOCALAPPDATA%\\Nestor``)::

    voiceprints/<slug>.npy      L2-normalised float32 embedding (dim 192)
    voiceprints/voiceprints.json  metadata (schema_version, names, ...)

A voiceprint is the duration-weighted average of cluster embeddings from one
or more recordings, L2-normalised. Matching computes cosine similarity, so the
normalisation guarantees the dot product equals the cosine.
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .library import default_root

SCHEMA_VERSION = 2
LIBRARY_FILE = "voiceprints.json"

# Simple Cyrillic -> Latin transliteration used to build file slugs
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def make_slug(name: str) -> str:
    """Build a filesystem-safe slug from a display name.

    Cyrillic is transliterated, anything that is not a letter or digit becomes
    a dash. Examples: ``Иван Петров`` -> ``ivan-petrov``, ``Марья-Искусница``
    -> ``marya-iskusnica``.
    """
    lowered = name.strip().lower()
    pieces = []
    for char in lowered:
        if char in _TRANSLIT:
            pieces.append(_TRANSLIT[char])
        elif char.isascii() and (char.isalnum() or char in "-_"):
            pieces.append(char)
        elif unicodedata.category(char).startswith("L"):
            # other non-ASCII letters (e.g. Latin-1 accents) keep them
            pieces.append(char)
        else:
            pieces.append("-")
    slug = "".join(pieces).strip("-_")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "voice"


@dataclass
class VoiceprintMeta:
    """Metadata of one stored voiceprint (the ``.npy`` holds the vector)."""

    slug: str
    display_name: str
    comment: str = ""
    source_recording_id: str = ""
    num_utterances: int = 0
    total_duration_sec: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "VoiceprintMeta":
        return cls(
            slug=str(data.get("slug", "")),
            display_name=str(data.get("display_name", "")),
            comment=str(data.get("comment", "")),
            source_recording_id=str(data.get("source_recording_id", "")),
            num_utterances=int(data.get("num_utterances", 0)),
            total_duration_sec=float(data.get("total_duration_sec", 0.0)),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


class VoiceprintLibrary:
    """File-backed storage of speaker voiceprints."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.dir = self.root / "voiceprints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.dir / LIBRARY_FILE
        self._meta: Dict[str, VoiceprintMeta] = {}
        # embeddings live only in memory; vectors are loaded on demand
        self._cache: Dict[str, "np.ndarray"] = {}
        self.load()

    # ------------------------------------------------------------------- i/o
    def save(self) -> None:
        """Write metadata atomically (temp + os.replace, same as library.py)."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "voiceprints": [meta.to_dict() for meta in self._meta.values()],
        }
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.json_path.with_name(self.json_path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for attempt in range(10):
            try:
                os.replace(tmp, self.json_path)
                break
            except PermissionError:
                time.sleep(0.01 + 0.01 * attempt)
        else:
            self.json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.unlink(missing_ok=True)

    def load(self) -> None:
        """Load metadata; tolerate a missing or damaged file.

        Unknown (older or newer) schema versions are loaded with a warning
        instead of failing: the vectors are still readable, only the metadata
        may be incomplete.
        """
        self._meta = {}
        if not self.json_path.exists():
            return
        try:
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # damaged file: keep the in-memory library empty rather than crash
            self._log_damaged()
            return
        entries = payload.get("voiceprints", []) if isinstance(payload, dict) else []
        version = payload.get("schema_version", 1) if isinstance(payload, dict) else 1
        if version != SCHEMA_VERSION:
            self._log_schema_mismatch(int(version))
        for entry in entries:
            try:
                meta = VoiceprintMeta.from_dict(entry)
            except Exception:  # noqa: BLE001 - skip broken entries
                continue
            if meta.slug and meta.display_name:
                self._meta[meta.slug] = meta

    def _log_damaged(self) -> None:
        self._warn("voiceprints.json is damaged, starting with an empty library")

    def _log_schema_mismatch(self, found: int) -> None:
        self._warn(
            f"voiceprints.json schema version {found} differs from the expected "
            f"{SCHEMA_VERSION}; loading anyway"
        )

    def _warn(self, message: str) -> None:
        """Append a note to log.txt next to the library root, best effort."""
        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = self.root / "log.txt"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] voiceprints: {message}\n")
        except OSError:
            pass

    # -------------------------------------------------------------- vectors
    def _vector_path(self, slug: str) -> Path:
        return self.dir / f"{slug}.npy"

    def _get_vector(self, slug: str) -> Optional["np.ndarray"]:
        if slug in self._cache:
            return self._cache[slug]
        path = self._vector_path(slug)
        if not path.exists():
            return None
        try:
            import numpy as np

            vector = np.load(path)
        except Exception:  # noqa: BLE001 - damaged vector: treat as missing
            self._warn(f"vector {slug}.npy is damaged, ignoring")
            return None
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            self._warn(f"vector {slug}.npy has a wrong shape, ignoring")
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            self._warn(f"vector {slug}.npy is zero, ignoring")
            return None
        self._cache[slug] = vector
        return vector

    # ------------------------------------------------------------------- api
    def add_or_update(
        self,
        name: str,
        embedding: "np.ndarray",
        source_recording_id: str = "",
        duration_sec: float = 0.0,
    ) -> VoiceprintMeta:
        """Add a new voiceprint or fold a new utterance into an existing one.

        The stored vector is the running average of all utterances seen so far
        (duration-weighted), so a single bad sample never overwrites the
        accumulated portrait.
        """
        import numpy as np

        name = (name or "").strip()
        if not name:
            raise ValueError("voiceprint name must not be empty")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("embedding must not be empty")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("embedding must not be zero")
        vector = vector / norm

        # find existing by display name (case-insensitive)
        existing = self.find(name)
        if existing is not None:
            meta = existing
            old = self._get_vector(meta.slug)
        else:
            slug = self._unique_slug(name)
            meta = VoiceprintMeta(slug=slug, display_name=name)
            old = None

        if old is not None:
            # running average weighted by total seconds
            w_old = max(meta.total_duration_sec, 0.0)
            w_new = max(float(duration_sec), 0.0)
            if w_old > 0 and w_new > 0:
                mixed = (old * w_old + vector * w_new) / (w_old + w_new)
            else:
                # one side carries no duration info (e.g. a voice imported
                # with total_duration_sec = 0): an equal mix - the stored
                # portrait must never be fully replaced by one utterance
                mixed = (old + vector) / 2.0
            norm = float(np.linalg.norm(mixed))
            vector = mixed / norm if norm > 0 else vector

        np.save(self._vector_path(meta.slug), vector.astype(np.float32))
        self._cache[meta.slug] = vector.astype(np.float32)
        meta.num_utterances += 1
        meta.total_duration_sec += max(float(duration_sec), 0.0)
        if source_recording_id:
            meta.source_recording_id = str(source_recording_id)
        meta.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta.schema_version = SCHEMA_VERSION
        self._meta[meta.slug] = meta
        self.save()
        return meta

    def _unique_slug(self, name: str) -> str:
        base = make_slug(name)
        slug = base
        suffix = 2
        while slug in self._meta:
            slug = f"{base}-{suffix}"
            suffix += 1
        return slug

    def list(self) -> List[VoiceprintMeta]:
        """All voiceprints, alphabetically by display name."""
        return sorted(self._meta.values(), key=lambda m: m.display_name.lower())

    def find(self, name: str) -> Optional[VoiceprintMeta]:
        """Metadata by display name (case-insensitive), None if unknown."""
        wanted = (name or "").strip().lower()
        for meta in self._meta.values():
            if meta.display_name.lower() == wanted:
                return meta
        return None

    def find_by_slug(self, slug: str) -> Optional[VoiceprintMeta]:
        return self._meta.get(slug)

    def vectors(self) -> Dict[str, "np.ndarray"]:
        """All stored vectors keyed by slug (loads lazily)."""
        result: Dict[str, "np.ndarray"] = {}
        for meta in self._meta.values():
            vector = self._get_vector(meta.slug)
            if vector is not None:
                result[meta.slug] = vector
        return result

    def names(self) -> Dict[str, str]:
        """Slug -> display name."""
        return {meta.slug: meta.display_name for meta in self._meta.values()}

    def delete(self, name: str) -> bool:
        """Remove a voiceprint by display name; True when something was removed."""
        meta = self.find(name)
        if meta is None:
            meta = self.find_by_slug(name or "")
        if meta is None:
            return False
        self._meta.pop(meta.slug, None)
        self._cache.pop(meta.slug, None)
        self._vector_path(meta.slug).unlink(missing_ok=True)
        self.save()
        return True

    def set_comment(self, slug: str, comment: str) -> Optional[VoiceprintMeta]:
        """Attach a free-form note to a voiceprint; None if unknown slug."""
        meta = self._meta.get(slug)
        if meta is None:
            return None
        meta.comment = (comment or "").strip()
        self.save()
        return meta

    def rename(self, old_name: str, new_name: str) -> Optional[VoiceprintMeta]:
        """Rename a voice everywhere: the library entry and the vector file.

        If ``new_name`` is already taken, the renamed voice FOLDS INTO the
        existing one (vectors merge like add_or_update does, weighted by
        duration; utterance counts and source recording are kept from the
        surviving entry). Returns the resulting meta, None when ``old_name``
        is unknown. Slug follows the new name; the vector file is moved so
        no orphan ``.npy`` is left behind.
        """
        import numpy as np

        source = self.find(old_name)
        if source is None:
            return None
        new_name = (new_name or "").strip()
        if not new_name or new_name == source.display_name:
            return source
        target = self.find(new_name)
        if target is None:
            # plain rename: new slug, vector file moved under it
            self._meta.pop(source.slug, None)
            self._cache.pop(source.slug, None)
            vector = self._get_vector(source.slug)
            meta = VoiceprintMeta(
                slug=self._unique_slug(new_name),
                display_name=new_name,
                comment=source.comment,
                source_recording_id=source.source_recording_id,
                num_utterances=source.num_utterances,
                total_duration_sec=source.total_duration_sec,
                created_at=source.created_at,
                schema_version=SCHEMA_VERSION,
            )
            if vector is not None:
                np.save(self._vector_path(meta.slug), vector.astype(np.float32))
                self._cache[meta.slug] = vector.astype(np.float32)
            old_path = self._vector_path(source.slug)
            old_path.unlink(missing_ok=True)
            self._meta[meta.slug] = meta
            self.save()
            return meta
        # target exists: fold the source voice into it (duration-weighted)
        source_vector = self._get_vector(source.slug)
        target_vector = self._get_vector(target.slug)
        if source_vector is not None:
            if target_vector is not None:
                w_old = max(target.total_duration_sec, 0.0)
                w_new = max(source.total_duration_sec, 0.0)
                if w_old > 0 and w_new > 0:
                    mixed = (target_vector * w_old + source_vector * w_new) / (w_old + w_new)
                else:
                    mixed = (target_vector + source_vector) / 2.0
                norm = float(np.linalg.norm(mixed))
                target_vector = mixed / norm if norm > 0 else source_vector
            else:
                target_vector = source_vector
            np.save(self._vector_path(target.slug), target_vector.astype(np.float32))
            self._cache[target.slug] = target_vector.astype(np.float32)
            target.total_duration_sec += source.total_duration_sec
            target.num_utterances += source.num_utterances
        self._meta.pop(source.slug, None)
        self._cache.pop(source.slug, None)
        self._vector_path(source.slug).unlink(missing_ok=True)
        target.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.save()
        return target

    # ---------------------------------------------------- exchange (portable)
    EXPORT_FORMAT = "nestor-voices"
    EXPORT_VERSION = 1

    def export_data(self, slugs: Optional[List[str]] = None) -> Dict:
        """Portable payload: metadata + base64 vectors of chosen voices.

        ``slugs=None`` exports everything. Vectors ride base64-encoded float32
        arrays - the exchange file stays plain JSON, readable by hand."""
        import base64

        import numpy as np

        chosen = [
            meta for meta in self.list()
            if slugs is None or meta.slug in set(slugs)
        ]
        items = []
        for meta in chosen:
            vector = self._get_vector(meta.slug)
            if vector is None:
                continue
            items.append({
                "meta": {
                    "display_name": meta.display_name,
                    "comment": meta.comment,
                    "num_utterances": meta.num_utterances,
                    "total_duration_sec": meta.total_duration_sec,
                    "created_at": meta.created_at,
                    "updated_at": meta.updated_at,
                },
                "vector_b64": base64.b64encode(
                    np.asarray(vector, dtype=np.float32).tobytes()
                ).decode("ascii"),
                "dim": int(vector.size),
            })
        return {
            "format": self.EXPORT_FORMAT,
            "version": self.EXPORT_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "count": len(items),
            "voices": items,
        }

    def import_data(
        self, payload: Dict, skip_unknown_version: bool = True
    ) -> Dict:
        """Merge a portable payload into the library; returns a report.

        Matching by display name (case-insensitive): an imported voice with
        an existing name is treated as one person - the vectors and durations
        are merged via add_or_update (duration-weighted average). A dimension
        mismatch (a foreign embedding model) is skipped and reported.
        """
        import base64

        import numpy as np

        if not isinstance(payload, dict) or payload.get("format") != self.EXPORT_FORMAT:
            raise ValueError("Это не файл голосов Nestor")
        version = int(payload.get("version", 0))
        if version > self.EXPORT_VERSION and skip_unknown_version:
            # newer schema: still try - the vector layout has been stable
            pass
        items = payload.get("voices")
        if not isinstance(items, list):
            raise ValueError("В файле нет списка голосов")

        report = {"added": [], "merged": [], "skipped": []}
        for item in items:
            try:
                meta_in = item.get("meta") or {}
                name = str(meta_in.get("display_name") or "").strip()
                if not name:
                    raise ValueError("нет имени")
                raw = base64.b64decode(str(item.get("vector_b64") or ""))
                dim = int(item.get("dim") or 0)
                vector = np.frombuffer(raw, dtype=np.float32)
                if vector.size == 0:
                    raise ValueError("пустой вектор")
                if dim and vector.size != dim:
                    raise ValueError("повреждённый вектор")
                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    raise ValueError("нулевой вектор")
                vector = (vector / norm).astype(np.float32)

                existing = self.find(name)
                if existing is not None:
                    self.add_or_update(
                        name, vector,
                        duration_sec=float(meta_in.get("total_duration_sec") or 0.0),
                    )
                    # keep the richer comment: prefer a non-empty incoming one
                    merged = self.find(name)
                    if merged is not None and not merged.comment and meta_in.get("comment"):
                        self.set_comment(merged.slug, str(meta_in["comment"]))
                    report["merged"].append(name)
                else:
                    meta = self.add_or_update(
                        name, vector,
                        duration_sec=float(meta_in.get("total_duration_sec") or 0.0),
                    )
                    if meta_in.get("comment"):
                        self.set_comment(meta.slug, str(meta_in["comment"]))
                    report["added"].append(name)
            except Exception as exc:  # noqa: BLE001 - one bad voice must not sink the import
                report["skipped"].append(
                    f"{(item.get('meta') or {}).get('display_name', '?')}: {exc}"
                )
        return report
