"""Projects: grouping recordings by topic.

A *project* is just a name and a colour.  It is the "folder" for recordings
that belong to one topic — "Босфор", "P01", "Клиент Иванов" — so a long list of
interspersed files can be read as groups instead of one flat column.

Only the registry lives here: which projects exist, what they are called and
which colour they carry.  The recording itself keeps the link
(``Recording.project`` holds a project id), so deleting a project never touches
a recording — it simply becomes "uncategorized" (``project`` empty).

Layout inside the data root (default ``%LOCALAPPDATA%\\Nestor``)::

    organizer.json      projects (id, name, colour, timestamps)

The file is optional: a missing or damaged one is never fatal, the app just
starts with no projects.  Colours are picked from a fixed palette by order of
creation, so a new project is visually distinct without any extra dialog.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .library import default_root
from .voiceprints import make_slug

ORGANIZER_SCHEMA_VERSION = 1
ORGANIZER_FILE = "organizer.json"

# Same palette the speaker chips use: a project gets the first colour that is
# not taken yet, so groups look different without any manual choice.
PROJECT_PALETTE = [
    "#60a5fa",
    "#f472b6",
    "#34d399",
    "#fbbf24",
    "#c084fc",
    "#fb923c",
    "#22d3ee",
    "#a3e635",
    "#f87171",
    "#818cf8",
    "#2dd4bf",
    "#e879f9",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_name(name: str) -> str:
    return " ".join((name or "").split())


@dataclass
class Project:
    """A named group of recordings."""

    id: str = ""
    name: str = ""
    color: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    schema_version: int = ORGANIZER_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Project":
        return cls(
            id=str(data.get("id", "")),
            name=_clean_name(str(data.get("name", ""))),
            color=str(data.get("color", "")),
            created_at=str(data.get("created_at", "")) or _now(),
            updated_at=str(data.get("updated_at", "")) or _now(),
            schema_version=int(data.get("schema_version", ORGANIZER_SCHEMA_VERSION)),
        )


class ProjectLibrary:
    """File-backed registry of projects (folders for recordings)."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.json_path = self.root / ORGANIZER_FILE
        self._projects: Dict[str, Project] = {}
        self.load()

    # ------------------------------------------------------------------- i/o
    def save(self) -> None:
        payload = {
            "schema_version": ORGANIZER_SCHEMA_VERSION,
            "projects": [project.to_dict() for project in self._projects.values()],
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
        """Load projects; tolerate a missing or damaged file."""
        self._projects = {}
        if not self.json_path.exists():
            return
        try:
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._warn("organizer.json is damaged, starting without projects")
            return
        entries = payload.get("projects", []) if isinstance(payload, dict) else []
        version = (
            payload.get("schema_version", ORGANIZER_SCHEMA_VERSION)
            if isinstance(payload, dict)
            else ORGANIZER_SCHEMA_VERSION
        )
        if version != ORGANIZER_SCHEMA_VERSION:
            self._warn(
                f"organizer.json schema version {version} differs from the "
                f"expected {ORGANIZER_SCHEMA_VERSION}; loading anyway"
            )
        for entry in entries:
            try:
                project = Project.from_dict(entry)
            except Exception:  # noqa: BLE001 - skip broken entries
                continue
            if project.id and project.name:
                self._projects[project.id] = project

    def _warn(self, message: str) -> None:
        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = self.root / "log.txt"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] organizer: {message}\n")
        except OSError:
            pass

    # -------------------------------------------------------------- internal
    def _unique_id(self, name: str) -> str:
        base = make_slug(name) or "project"
        candidate = base
        suffix = 2
        while candidate in self._projects:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def _free_color(self) -> str:
        used = {project.color for project in self._projects.values()}
        for color in PROJECT_PALETTE:
            if color not in used:
                return color
        return PROJECT_PALETTE[len(self._projects) % len(PROJECT_PALETTE)]

    # ------------------------------------------------------------------- api
    def list(self) -> List[Project]:
        """All projects, alphabetically by name."""
        return sorted(self._projects.values(), key=lambda item: item.name.casefold())

    def get(self, project_id: str) -> Optional[Project]:
        return self._projects.get(str(project_id or ""))

    def find_by_name(self, name: str) -> Optional[Project]:
        key = _clean_name(name).casefold()
        if not key:
            return None
        for project in self._projects.values():
            if project.name.casefold() == key:
                return project
        return None

    def create(self, name: str) -> Project:
        """Create a project, or return the existing one with that name."""
        clean = _clean_name(name)
        if not clean:
            raise ValueError("project name is required")
        existing = self.find_by_name(clean)
        if existing is not None:
            return existing
        project = Project(id=self._unique_id(clean), name=clean, color=self._free_color())
        self._projects[project.id] = project
        self.save()
        return project

    def rename(self, project_id: str, name: str) -> Optional[Project]:
        """Rename a project in place; recordings keep pointing at its id."""
        project = self.get(project_id)
        clean = _clean_name(name)
        if project is None or not clean:
            return None
        project.name = clean
        project.updated_at = _now()
        self.save()
        return project

    def set_color(self, project_id: str, color: str) -> Optional[Project]:
        project = self.get(project_id)
        if project is None:
            return None
        project.color = str(color or "")
        project.updated_at = _now()
        self.save()
        return project

    def delete(self, project_id: str) -> bool:
        """Remove a project.  Recordings are expected to be unlinked by caller."""
        if self.get(project_id) is None:
            return False
        self._projects.pop(str(project_id), None)
        self.save()
        return True

    def color_for(self, project_id: Optional[str]) -> str:
        project = self.get(project_id) if project_id else None
        return project.color if project is not None else ""
