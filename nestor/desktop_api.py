"""Native helpers exposed to the web interface through pywebview.

The WebView2 engine has no browser download UI, so exporting goes through
Python: a native "Save As" dialog lets the user choose location and filename.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:  # the GUI backend is only needed to actually open the desktop window;
    import webview  # noqa: F401  (kept importable so tests run headless)
except ImportError:  # pragma: no cover - exercised only without pywebview
    webview = None  # type: ignore[assignment]

from .library import Library


def downloads_folder() -> Path:
    """Where exported files are placed by default."""
    env = os.environ.get("NESTOR_EXPORT_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        user = os.environ.get("USERPROFILE") or str(Path.home())
        return Path(user) / "Downloads"
    return Path.home() / "Downloads"


def unique_path(folder: Path, name: str) -> Path:
    """Avoid overwriting an existing export."""
    candidate = folder / name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        candidate = folder / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


class DesktopApi:
    """The object bound to ``window.pywebview.api``."""

    def __init__(
        self,
        library: Library,
        port: Optional[int] = None,
        token: Optional[str] = None,
    ) -> None:
        self.library = library
        self._window: Optional[webview.Window] = None
        # loopback endpoint of our own server (set by desktop.run); lets the
        # native Save As build the same payload as the web endpoint
        self._port = port
        self._token = token or ""

    def set_window(self, window: webview.Window) -> None:
        """Store the window reference for file dialogs."""
        self._window = window

    # ------------------------------------------------------------------ export
    def save_export_as(self, record_id: str, fmt: str = "docx") -> Optional[str]:
        """
        Show a native Save As dialog, write the export to the chosen location.

        Returns the chosen path as string, or None if user cancelled.
        Does NOT open the file after saving.
        """
        from .web.app import build_export

        if not self._window:
            return None

        recording = self.library.load(record_id)
        if not recording:
            return None

        source = build_export(self.library, record_id, fmt)
        default_name = f"{Path(recording.filename).stem}.{fmt}"

        # File type description for the dialog
        file_types = {
            "docx": "Документ Word (*.docx)",
            "txt": "Текстовый файл (*.txt)",
            "md": "Markdown (*.md)",
            "json": "JSON (*.json)",
            "srt": "Субтитры SRT (*.srt)",
            "vtt": "Субтитры VTT (*.vtt)",
        }
        desc = file_types.get(fmt, f"{fmt.upper()} (*.{fmt})")

        try:
            # FileDialog.SAVE is the non-deprecated constant (== SAVE_DIALOG)
            save_const = getattr(webview, "FileDialog", None)
            dialog_type = save_const.SAVE if save_const else webview.SAVE_DIALOG
            result = self._window.create_file_dialog(
                dialog_type,
                directory=str(downloads_folder()),
                save_filename=default_name,
                file_types=(desc,),
            )
        except Exception:
            return None

        if not result:
            return None

        target = Path(result[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return str(target)

    def save_export(self, record_id: str, fmt: str = "docx") -> str:
        """Legacy: write to Downloads folder and open (for backward compat)."""
        from .web.app import build_export

        source = build_export(self.library, record_id, fmt)
        target = unique_path(downloads_folder(), source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        self._open(target)
        return str(target)

    def save_report_as(self, text: str, filename: str = "") -> Optional[str]:
        """Native Save As for a diagnostics report (plain text).

        The run report used to be saved with a plain browser download, but
        WebView2 has no download UI: the file landed in Downloads silently and
        the user could not find it. Reports now go through the same native
        dialog as the other exports. Returns the chosen path, or None if the
        dialog was cancelled or unavailable."""
        if not self._window:
            return None
        name = (filename or "").strip() or "nestor-report.txt"
        # keep a single path component: never let a caller traverse folders
        name = Path(name).name
        if not name.lower().endswith(".txt"):
            name += ".txt"
        try:
            save_const = getattr(webview, "FileDialog", None)
            dialog_type = save_const.SAVE if save_const else webview.SAVE_DIALOG
            result = self._window.create_file_dialog(
                dialog_type,
                directory=str(downloads_folder()),
                save_filename=name,
                file_types=("Текстовый файл (*.txt)",),
            )
        except Exception:  # noqa: BLE001 - a cancelled/broken dialog is not fatal
            return None
        if not result:
            return None
        target = Path(result if isinstance(result, str) else result[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return str(target)

    def export_formats(self) -> List[str]:
        """Formats the interface can offer."""
        return ["docx", "txt", "md", "json", "srt"]

    # ------------------------------------------------------------ open dialog
    AUDIO_SUFFIXES = (
        ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wma",
        ".aiff", ".aif",
    )
    VIDEO_SUFFIXES = (".webm", ".mp4", ".mkv", ".mov", ".avi", ".3gp")

    @classmethod
    def _wildcards(cls, suffixes) -> str:
        return ";".join("*" + s for s in suffixes)

    @classmethod
    def open_file_filters(cls) -> tuple:
        """Filter sets for the native Open dialog, in display order."""
        everything = cls.AUDIO_SUFFIXES + cls.VIDEO_SUFFIXES
        return (
            f"Все поддерживаемые форматы ({cls._wildcards(everything)})",
            f"Аудиофайлы ({cls._wildcards(cls.AUDIO_SUFFIXES)})",
            f"Видеофайлы ({cls._wildcards(cls.VIDEO_SUFFIXES)})",
            "Все файлы (*.*)",
        )

    def open_files_dialog(self) -> List[str]:
        """Native Open dialog for media files; returns chosen paths."""
        if not self._window:
            return []
        save_const = getattr(webview, "FileDialog", None)
        dialog_type = save_const.OPEN if save_const else webview.OPEN_DIALOG
        try:
            result = self._window.create_file_dialog(
                dialog_type,
                allow_multiple=True,
                file_types=self.open_file_filters(),
            )
        except Exception:  # noqa: BLE001 - a cancelled/broken dialog is not fatal
            return []
        return [str(p) for p in (result or [])]

    # ------------------------------------------------------- voices exchange
    def save_voices_as(self, query: str = "") -> Optional[str]:
        """Native Save As for the portable voices file (JSON).

        Builds the payload from the library and writes it to the chosen
        location (same pattern as save_export_as). ``query`` is an optional
        ``?slug=...&slug=...`` selection — the panel's checkbox export;
        empty means the full backup. Returns the chosen path, or None if
        cancelled."""
        from .voiceprints import VoiceprintLibrary

        if not self._window:
            return None
        payload: Optional[dict] = None
        if self._port:
            try:
                import urllib.request

                url = (
                    f"http://127.0.0.1:{self._port}/api/voiceprints/export"
                    + (query or "")
                )
                req = urllib.request.Request(url, headers={"X-Nestor-Token": self._token})
                with urllib.request.urlopen(req, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception:  # noqa: BLE001 - fall back to the whole library
                payload = None
        if not isinstance(payload, dict):
            payload = VoiceprintLibrary(self.library.root).export_data()
        count = payload.get("count") or 0
        stamp = datetime.now().strftime("%Y-%m-%d")
        default_name = f"nestor-voices{'-' + str(count) if query else ''}-{stamp}.json"
        try:
            save_const = getattr(webview, "FileDialog", None)
            dialog_type = save_const.SAVE if save_const else webview.SAVE_DIALOG
            result = self._window.create_file_dialog(
                dialog_type,
                directory=str(downloads_folder()),
                save_filename=default_name,
                file_types=("Файл голосов Nestor (*.json)",),
            )
        except Exception:  # noqa: BLE001 - a cancelled/broken dialog is not fatal
            return None
        if not result:
            return None
        target = Path(result if isinstance(result, str) else result[0])
        payload = VoiceprintLibrary(self.library.root).export_data()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return str(target)

    def open_voices_file(self) -> Optional[str]:
        """Native Open dialog for a voices exchange file; path or None."""
        if not self._window:
            return None
        try:
            save_const = getattr(webview, "FileDialog", None)
            dialog_type = save_const.OPEN if save_const else webview.OPEN_DIALOG
            result = self._window.create_file_dialog(
                dialog_type,
                file_types=(
                    "Файл голосов Nestor (*.json)",
                    "Все файлы (*.*)",
                ),
            )
        except Exception:  # noqa: BLE001 - a cancelled/broken dialog is not fatal
            return None
        if not result:
            return None
        return str(result if isinstance(result, str) else result[0])

    # ------------------------------------------------------------------ misc
    def open_data_folder(self) -> str:
        """Open the folder with recordings in Explorer."""
        folder = self.library.root
        folder.mkdir(parents=True, exist_ok=True)
        self._open(folder)
        return str(folder)

    def open_recordings_folder(self) -> str:
        """Open the folder with the exports of a recording."""
        folder = self.library.output_dir
        folder.mkdir(parents=True, exist_ok=True)
        self._open(folder)
        return str(folder)

    def open_log_file(self) -> str:
        """Open the application log (log.txt) in the default viewer."""
        path = self.library.log_path
        if not path.exists():
            path.write_text("", encoding="utf-8")
        self._open(path)
        return str(path)

    def show_in_folder(self, path: str) -> str:
        """Reveal a file in Explorer."""
        target = Path(path)
        if target.exists() and sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(target)])
        else:
            self._open(target.parent)
        return str(target)

    @staticmethod
    def _open(path: Path) -> None:
        """Open a file or folder with the default application."""
        try:
            if sys.platform == "win32":
                if path.is_dir():
                    os.startfile(str(path))  # noqa: S606
                else:
                    os.startfile(str(path))  # noqa: S606
            elif sys.platform == "darwin":
                import subprocess as sp

                sp.Popen(["open", str(path)])
            else:
                import subprocess as sp

                sp.Popen(["xdg-open", str(path)])
        except Exception:  # noqa: BLE001 - opening a file must never crash the app
            pass