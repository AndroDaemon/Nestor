"""Persistent user settings (``settings.json`` in the data root).

The desktop app stores them next to the library (default
``%LOCALAPPDATA%\\Nestor\\settings.json``). Writes are atomic (tmp file +
``os.replace``) so a crash can never leave a half-written file; a damaged
file is moved aside (``settings.json.broken-<ts>``) and defaults are used
instead - the app never fails to start because of settings.

Unknown keys sent to the API are ignored; values of the wrong type are
rejected with 400 by the endpoint (and by :meth:`SettingsStore.update`).
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

SCHEMA_VERSION = 4
SETTINGS_FILE = "settings.json"

# key -> (kind, constraint); anything not listed here is ignored on update.
# kind: "bool" | "int" | "float" | "str"
# constraint: for str - allowed values; for int/float - (min, max) inclusive.
SCHEMA: Dict[str, Tuple[str, object]] = {
    "num_speakers": ("int", (0, 6)),  # 0 = determine automatically
    "auto_start": ("bool", None),
    "vp_match": ("bool", None),
    "vp_abs_threshold": ("float", (0.0, 1.0)),
    "vp_margin": ("float", (0.0, 1.0)),
    "vp_segment_agreement": ("float", (0.0, 1.0)),
    "sound_done": ("bool", None),  # play a sound when processing finishes
    # cluster merge: fold labels that describe the same voice after diarization
    "merge_speakers": ("bool", None),
    "merge_threshold": ("float", (0.0, 1.0)),
    "merge_min_weight": ("float", (0.0, 30.0)),
    # segment-agreement floor for the merge pass (0 = old lenient behaviour);
    # 0.5 keeps two similar voices apart where 0.35 collapsed them into one
    "merge_segment_floor": ("float", (0.0, 1.0)),
    # strict-cohesion guard for the merge pass (opt-in, 0 = off): refuse merges
    # that would leave the group internally inhomogeneous
    "merge_min_coh": ("float", (0.0, 1.0)),
    # audio enhancement before ASR (highpass + level normalization)
    "audio_enhance": ("bool", None),
    # diarization clustering (sherpa distance threshold; 0.5 = sherpa default)
    "diar_threshold": ("float", (0.0, 1.0)),
    "diar_min_duration_on": ("float", (0.0, 5.0)),
    "diar_min_duration_off": ("float", (0.0, 5.0)),
    # diarize long recordings in overlapping blocks (0 = one call, high memory):
    # sherpa-onnx copies the whole waveform, so a single call on a multi-hour
    # file costs a second full-length buffer and dies with `bad allocation`
    "diar_block_seconds": ("int", (0, 3600)),
    "diar_block_overlap": ("int", (0, 600)),
    # rescue pass: re-transcribe silent holes between words
    "rescue_holes": ("bool", None),
    # glossary post-correction: rewrite heard forms ("докер") to canonical
    # terms ("Docker"); offline text pass, no effect on speed
    "glossary": ("bool", None),
    # worker threads for ASR/diarization models (0 = auto)
    "threads": ("int", (0, 64)),
    # run heavy jobs at ABOVE_NORMAL priority (opt-in; can starve the UI on
    # small laptops, so off by default)
    "process_priority": ("bool", None),
    # on hybrid CPUs keep the heavy worker on the performance cores; off by
    # default - measured slower than letting Windows spread 8 threads over all
    # cores (19:22 pinned to 4 P-cores vs 17:50 unpinned)
    "pin_performance": ("bool", None),
    # Windows toast notification when processing finishes
    "notify_done": ("bool", None),
}

DEFAULTS: Dict[str, object] = {
    "num_speakers": 0,
    "auto_start": True,
    "vp_match": True,
    "vp_abs_threshold": 0.75,
    "vp_margin": 0.05,
    "vp_segment_agreement": 0.6,
    "sound_done": False,
    "merge_speakers": True,
    "merge_threshold": 0.45,
    "merge_min_weight": 1.5,
    "merge_segment_floor": 0.5,
    "merge_min_coh": 0.0,
    "audio_enhance": True,
    "diar_threshold": 0.5,
    "diar_min_duration_on": 0.5,
    "diar_min_duration_off": 0.0,
    "diar_block_seconds": 1200,
    "diar_block_overlap": 60,
    "rescue_holes": True,
    "glossary": False,
    "threads": 0,
    "process_priority": False,
    "pin_performance": False,
    "notify_done": True,
}


def _validate(key: str, value: object) -> Optional[str]:
    """Error message when ``value`` does not fit the schema, else None."""
    kind, constraint = SCHEMA[key]
    if kind == "bool":
        if not isinstance(value, bool):
            return f"{key}: ожидалось true/false"
        return None
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{key}: ожидалось целое число"
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{key}: ожидалось число"
    else:  # str
        if not isinstance(value, str):
            return f"{key}: ожидалась строка"
    if constraint is not None:
        if kind == "str":
            allowed = ", ".join(str(item) for item in constraint)
            if value not in constraint:
                return f"{key}: допустимые значения: {allowed}"
        else:
            low, high = constraint
            if not (low <= value <= high):
                return f"{key}: значение должно быть от {low} до {high}"
    return None


def _coerce(key: str, value: object) -> object:
    """Normalise a validated value for storage (e.g. int -> float)."""
    kind, _ = SCHEMA[key]
    if kind == "float":
        return float(value)
    return value


class SettingsStore:
    """File-backed user settings with atomic writes."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / SETTINGS_FILE
        self._lock = threading.Lock()
        self._data: Dict[str, object] = dict(DEFAULTS)
        self._load()

    # ------------------------------------------------------------------ i/o
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._quarantine()
            return
        if not isinstance(payload, dict):
            self._quarantine()
            return
        for key, value in payload.items():
            if key in SCHEMA and _validate(key, value) is None:
                self._data[key] = _coerce(key, value)
        version = payload.get("schema_version")
        if not isinstance(version, int) or version < SCHEMA_VERSION:
            self._migrate(version if isinstance(version, int) else 0)

    def _migrate(self, from_version: int) -> None:
        """Bring an older settings file up to the current schema.

        1 -> 2: the diarization cluster threshold was re-tuned (0.3 over-split
        long meetings) and the merge pass gained a segment-agreement floor;
        files written by older builds carry the old values explicitly, so they
        are reset to the new defaults here.

        2 -> 3: the merge pass gained an opt-in strict-cohesion guard
        (``merge_min_coh``); older files simply take the default (off), and a
        large-cluster auto-split was added that is always on.

        3 -> 4: the segment-agreement floor was raised from 0.35 to 0.5.
        Measurements on real recordings showed 0.35 folded a two-person talk
        into a single label holding 71% of the words, while 0.5 keeps the
        voices apart; files written earlier carry the old value explicitly.
        """
        if from_version < 2:
            self._data["diar_threshold"] = DEFAULTS["diar_threshold"]
            self._data["merge_segment_floor"] = DEFAULTS["merge_segment_floor"]
        if from_version < 3:
            self._data["merge_min_coh"] = DEFAULTS["merge_min_coh"]
        if from_version < 4:
            self._data["merge_segment_floor"] = DEFAULTS["merge_segment_floor"]
        with self._lock:
            self._write()

    def _quarantine(self) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            self.path.replace(self.path.with_name(f"settings.json.broken-{stamp}"))
        except OSError:
            pass
        self._warn("settings.json is damaged, moved aside; starting with defaults")

    def _warn(self, message: str) -> None:
        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with (self.root / "log.txt").open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] settings: {message}\n")
        except OSError:
            pass

    def _write(self) -> None:
        payload = {"schema_version": SCHEMA_VERSION, **self._data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(tmp, self.path)
                break
            except PermissionError:
                time.sleep(0.01 + 0.01 * attempt)
        else:
            # give up on the atomic path but never lose the values
            self.path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ api
    def as_public(self) -> Dict[str, object]:
        """Current values (a copy - safe to return from an endpoint)."""
        return dict(self._data)

    def update(self, changes: Dict) -> Dict[str, object]:
        """Validate, merge and persist; raises ValueError on bad input."""
        if not isinstance(changes, dict):
            raise ValueError("Ожидался JSON-объект с настройками")
        checked: Dict[str, object] = {}
        for key, value in changes.items():
            if key not in SCHEMA:
                continue  # unknown fields are ignored, not an error
            problem = _validate(key, value)
            if problem:
                raise ValueError(problem)
            checked[key] = _coerce(key, value)
        with self._lock:
            self._data.update(checked)
            self._write()
            return dict(self._data)

    def reset(self) -> Dict[str, object]:
        """Back to factory defaults (persisted)."""
        with self._lock:
            self._data = dict(DEFAULTS)
            self._write()
            return dict(self._data)
