"""Native helpers exposed to the web interface through pywebview.

The WebView2 engine has no browser download UI, so exporting goes through
Python: a native "Save As" dialog lets the user choose location and filename.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote, urlparse

try:  # the GUI backend is only needed to actually open the desktop window;
    import webview  # noqa: F401  (kept importable so tests run headless)
except ImportError:  # pragma: no cover - exercised only without pywebview
    webview = None  # type: ignore[assignment]

from .library import Library


def dialog_constant(name: str):
    """pywebview file-dialog constant (``SAVE``/``OPEN``).

    ``webview.FileDialog`` is the current API, ``webview.SAVE_DIALOG`` the
    legacy one. When pywebview is unavailable (headless CI), fall back to the
    plain name so the call path still works against a stub window."""
    if webview is not None:
        file_dialog = getattr(webview, "FileDialog", None)
        if file_dialog is not None and hasattr(file_dialog, name):
            return getattr(file_dialog, name)
        legacy = getattr(webview, f"{name}_DIALOG", None)
        if legacy is not None:
            return legacy
    return name


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
        # Presenter screen: created lazily on the first click. The JS bridge can
        # deliver two clicks in parallel threads, and pywebview's load_url waits
        # for the window to be shown (up to 20 s), so creation, URL loads and
        # closing are all serialized by a lock and the very first URL is applied
        # from the "shown" event instead of blocking the bridge call.
        self._watch_window: Optional[webview.Window] = None
        self._watch_lock = threading.Lock()
        self._watch_shown = False
        self._watch_pending: Optional[str] = None
        self._watch_pending_close: Optional[webview.Window] = None

    def set_window(self, window: webview.Window) -> None:
        """Store the window reference for file dialogs."""
        self._window = window

    # --------------------------------------------------------- presenter screen
    def _watch_url(self, record_id: str) -> str:
        url = f"http://127.0.0.1:{self._port}/watch.html?id={quote(str(record_id))}"
        if self._token:
            url += f"&token={quote(self._token)}"
        return url

    def open_watch_window(self, record_id: str) -> bool:
        """Open the presenter screen, or bring the existing one to the front.

        Called from the JS bridge, i.e. off the GUI main thread - exactly what
        pywebview needs for windows created after ``start()``.
        """
        if self._port is None:
            return False
        url = self._watch_url(record_id)
        with self._watch_lock:
            window = self._watch_window
            if window is not None:
                if not self._watch_shown:
                    # still starting up: load_url would block the bridge thread
                    # for up to 20 s waiting for ``shown``; defer it instead
                    self._watch_pending = url
                    return True
                try:
                    window.load_url(url)
                    # show() == Show + Activate: bring it forward every time,
                    # not just on the first click (restore() only un-minimizes)
                    window.show()
                    return True
                except Exception:
                    self._watch_window = None
                    self._watch_shown = False
            if webview is None:
                return False
            try:
                new_window = webview.create_window(
                    "Экран докладчика",
                    url,
                    width=1024,
                    height=640,
                    min_size=(420, 260),
                    background_color="#0d0f12",
                    text_select=False,
                    js_api=self,
                    on_top=False,
                )
            except Exception:
                return False
            if new_window is None:
                return False
            self._watch_window = new_window
            self._watch_pending = None

            def on_shown() -> None:
                # runs in its own thread (pywebview fires events that way), so
                # load_url may safely Invoke into the GUI thread here
                with self._watch_lock:
                    is_current = self._watch_window is new_window
                    if is_current:
                        self._watch_shown = True
                        pending = self._watch_pending
                        self._watch_pending = None
                    else:
                        pending = None
                    close_pending = self._watch_pending_close
                    self._watch_pending_close = None
                if pending:
                    try:
                        new_window.load_url(pending)
                    except Exception:
                        pass
                if close_pending is new_window:
                    # closed while we were still starting up: destroy() would
                    # have blocked on "shown"; now it is safe
                    try:
                        new_window.destroy()
                    except Exception:
                        pass

            def forget() -> None:
                with self._watch_lock:
                    if self._watch_window is new_window:
                        self._watch_window = None
                        self._watch_shown = False
                    if self._watch_pending_close is new_window:
                        self._watch_pending_close = None

            try:
                new_window.events.shown += on_shown
            except Exception:
                pass
            try:
                new_window.events.closed += forget
            except Exception:
                pass
            return True

    def close_watch_window(self) -> bool:
        """Close the presenter screen (the pane and this window are exclusive)."""
        with self._watch_lock:
            window = self._watch_window
            if window is None:
                return False
            self._watch_window = None
            self._watch_pending = None
            shown = self._watch_shown
            self._watch_shown = False
            if not shown:
                # still starting up: destroy() would block the bridge thread
                # waiting for "shown"; let the shown handler finish the job
                self._watch_pending_close = window
                return True
            self._watch_pending_close = None
        try:
            window.destroy()
            return True
        except Exception:
            return False

    def close_extra_windows(self) -> bool:
        """Close every window opened in addition to the main one.

        pywebview tears the application down only once the *last* window is
        gone (winforms calls ``Application.Exit`` when no forms remain). So if
        the main window is closed while the presenter screen is still open, the
        process — and its local server — would linger with a lone video window
        and no main window. ``desktop.run`` calls this from the main window's
        ``closed`` event so everything goes down together.
        """
        return self.close_watch_window()

    def watch_window_open(self) -> bool:
        """Whether the presenter screen still exists (the user may have closed it)."""
        return self._watch_window is not None

    def set_watch_on_top(self, flag: bool) -> bool:
        """Toggle always-on-top for the presenter screen.

        pywebview applies ``window.on_top`` by assigning ``Form.TopMost`` from
        whatever thread happens to call it. Called through the JS bridge that
        assignment deadlocks the WebView2 message loop and froze the whole app
        whenever «Поверх» was clicked (the app had to be killed).

        ``SetWindowPos`` is documented as safe to call from any thread, so we
        drive the native window handle directly and never touch WinForms from a
        foreign thread. ``SWP_NOACTIVATE`` keeps the toggle from stealing focus.
        """
        window = self._watch_window
        if window is None:
            return False
        form = getattr(window, "native", None)
        handle = getattr(form, "Handle", None)
        if handle is None:
            return False
        try:
            hwnd = handle.ToInt32()
        except Exception:
            return False

        topmost = -1 if flag else -2  # HWND_TOPMOST / HWND_NOTOPMOST
        flags = 0x0001 | 0x0002 | 0x0010  # NOSIZE | NOMOVE | NOACTIVATE
        try:
            user32 = ctypes.windll.user32
            user32.SetWindowPos.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint,
            ]
            user32.SetWindowPos.restype = ctypes.c_int
            result = user32.SetWindowPos(
                ctypes.c_void_p(hwnd), ctypes.c_void_p(topmost), 0, 0, 0, 0, flags
            )
        except Exception:
            return False
        return bool(result)

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
            dialog_type = dialog_constant("SAVE")
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
        # keep a single path component: never let a caller traverse folders.
        # Split on both separators so the guard also holds on POSIX hosts,
        # where backslash is an ordinary filename character.
        name = name.replace("\\", "/").split("/")[-1]
        if not name.lower().endswith(".txt"):
            name += ".txt"
        try:
            dialog_type = dialog_constant("SAVE")
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
        dialog_type = dialog_constant("OPEN")
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
            dialog_type = dialog_constant("SAVE")
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
            dialog_type = dialog_constant("OPEN")
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

    def save_glossary_as(self) -> Optional[str]:
        """Native Save As for the effective glossary (YAML); path or None."""
        from .glossary import dump_glossary, load_glossary, resolve_glossary_path

        if not self._window:
            return None
        glossary = load_glossary(resolve_glossary_path(self.library.root))
        stamp = datetime.now().strftime("%Y-%m-%d")
        default_name = f"nestor-hotwords-{stamp}.yaml"
        try:
            dialog_type = dialog_constant("SAVE")
            result = self._window.create_file_dialog(
                dialog_type,
                directory=str(downloads_folder()),
                save_filename=default_name,
                file_types=("Словарь терминов Nestor (*.yaml)", "Все файлы (*.*)"),
            )
        except Exception:  # noqa: BLE001 - a cancelled/broken dialog is not fatal
            return None
        if not result:
            return None
        target = Path(result if isinstance(result, str) else result[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dump_glossary(glossary), encoding="utf-8")
        return str(target)

    def open_glossary_file(self) -> Optional[str]:
        """Native Open dialog for a glossary YAML/JSON file; path or None."""
        if not self._window:
            return None
        try:
            dialog_type = dialog_constant("OPEN")
            result = self._window.create_file_dialog(
                dialog_type,
                file_types=(
                    "Словарь терминов (*.yaml;*.yml;*.json)",
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

    def open_guide(self) -> bool:
        """Open the offline user guide shipped next to the program.

        The installer puts the rendered guide in <install dir>\\docs\\index.html,
        a sibling of the running executable in a frozen build. When the file is
        unavailable (running from source), report failure so the interface can
        fall back to its built-in help."""
        if not getattr(sys, "frozen", False):
            return False
        base = Path(sys.executable).resolve().parent
        for candidate in (
            base / "docs" / "index.html",
            base.parent / "docs" / "index.html",
        ):
            if candidate.exists():
                self._open(candidate)
                return True
        return False

    def show_in_folder(self, path: str) -> str:
        """Reveal a file in Explorer."""
        target = Path(path)
        if target.exists() and sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(target)])
        else:
            self._open(target.parent)
        return str(target)

    def open_external(self, url: str) -> bool:
        """Open an http(s) link in the user's browser, not inside the app.

        The help window links to the Telegram channel; letting the WebView
        handle it would navigate away from the app (and there is no Back).
        Only http/https is accepted, so a malformed value cannot launch a
        local file or a custom protocol handler.
        """
        parsed = urlparse(str(url or ""))
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False
        try:
            return bool(webbrowser.open(parsed.geturl()))
        except Exception:  # noqa: BLE001 - a bad link must never crash the app
            return False

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