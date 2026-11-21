"""Native desktop window for Nestor.

Runs the FastAPI backend on a local port and opens it in a real Windows window
(Microsoft Edge WebView2) through pywebview. Only one instance runs at a time:
a second launch focuses the window that is already open.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import __version__
from .library import default_root
from .web.app import generate_session_token

WINDOW_TITLE = "Nestor"

# Named mutex held for the lifetime of the process. The installer and the
# uninstaller watch this name (AppMutex in packaging/Nestor.iss): it is how they
# know the application is running and files must not be replaced mid-upgrade.
# The single-instance logic itself uses instance.json — see run().
MUTEX_NAME = "Nestor.Desktop.SingleInstance"
_mutex_handle = None


def acquire_mutex(name: str = MUTEX_NAME) -> bool:
    """Create the named mutex the installer waits on (Windows only, best effort).

    Returns True when the mutex is held. On a non-Windows host, or if the call
    fails, returns False and the caller carries on: the mutex must never stop
    the application from starting."""
    global _mutex_handle
    if os.name != "nt":
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            return False
        _mutex_handle = handle  # held for the process lifetime
        return True
    except Exception:  # noqa: BLE001 - best effort
        return False


def free_port() -> int:
    """Pick a free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def instance_file(root: Optional[Path] = None) -> Path:
    root = Path(root) if root else default_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / "instance.json"


def _port_alive(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def focus_existing(title: str = WINDOW_TITLE) -> bool:
    """Bring an already running window to the front (Windows only)."""
    if os.name != "nt":
        return False
    try:
        user32 = ctypes.windll.user32
        handle = user32.FindWindowW(None, title)
        if handle:
            user32.ShowWindow(handle, 9)  # SW_RESTORE
            user32.SetForegroundWindow(handle)
            return True
    except Exception:  # noqa: BLE001 - best effort
        pass
    return False


def _read_instance(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ui_log(message: str, root: Optional[Path] = None) -> None:
    """Append a line to ui.log next to the data root (best effort).

    The packaged build runs with ``console=False``, so every ``print`` in this
    module disappears into nothing. Anything that can explain a window that
    closed or crashed has to go to a file we can actually read afterwards.
    """
    try:
        directory = Path(root) if root else default_root()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with (directory / "ui.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")
    except OSError:
        pass


def _install_gui_diagnostics(root: Optional[Path] = None) -> List[str]:
    """Route uncaught exceptions and GUI lifecycle events to ui.log.

    Returns the previous ``sys.excepthook``/``threading.excepthook`` so tests can
    restore them. Without this a crash inside the GUI thread leaves no trace at
    all: Windows Error Reporting is off on many machines and the frozen app has
    no console.
    """
    previous = [sys.excepthook, threading.excepthook]

    def handle_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous[0](exc_type, exc_value, exc_tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        ui_log(f"uncaught exception:\n{text}", root)
        previous[0](exc_type, exc_value, exc_tb)

    def handle_thread_exception(args):
        text = "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )
        ui_log(f"uncaught exception in thread {args.thread.name}:\n{text}", root)
        previous[1](args)

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception
    return previous


def _restore_excepthooks(previous: List[object]) -> None:
    sys.excepthook = previous[0]
    threading.excepthook = previous[1]


def _window_alive(window) -> Optional[bool]:
    """True/False when we can read the native handle, None when we cannot.

    Only meaningful on Windows with the winforms backend; any other case
    returns None and the watchdog stays quiet.
    """
    if os.name != "nt":
        return None
    try:
        form = getattr(window, "native", None)
        handle = getattr(form, "Handle", None)
        if handle is None:
            return None
        hwnd = handle.ToInt32()
        if not hwnd:
            return None
        return bool(ctypes.windll.user32.IsWindow(ctypes.c_void_p(hwnd)))
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return None


def _start_window_watchdog(window, processor, root: Optional[Path]) -> None:
    """Record, every minute, whether the window still exists while jobs run.

    The evidence that was missing after the P01 crash: the transcription kept
    writing progress while the window had already disappeared. This thread
    leaves a timeline in ui.log so the next report can be diagnosed instead of
    guessed at.
    """

    def watch() -> None:
        while True:
            time.sleep(60)
            try:
                alive = _window_alive(window)
                active = processor.active() if hasattr(processor, "active") else []
            except Exception:  # noqa: BLE001
                alive, active = None, []
            if alive is False:
                ui_log(f"watchdog: main window handle is gone; active jobs={active}", root)
                return
            if active:
                ui_log(f"watchdog: window alive={alive}, active jobs={active}", root)

    threading.Thread(target=watch, name="nestor-ui-watchdog", daemon=True).start()


def _confirm_close_while_busy(window, processor, root: Optional[Path] = None) -> bool:
    """Ask before closing the window while a job is still running.

    Registered on ``window.events.closing``. That event is created with
    ``should_lock=True``, so this runs on the GUI thread and its return value is
    honoured: returning ``False`` cancels the close (pywebview's winforms
    ``on_closing`` sets ``args.Cancel``). Without this the window vanishes while
    the non-daemon worker threads keep transcribing — the process stays alive
    with no window and looks like a crash.
    """
    try:
        active = processor.active() if hasattr(processor, "active") else []
    except Exception:  # noqa: BLE001 - the guard must never block closing
        return True
    if not active:
        return True

    text = (
        f"Идёт обработка ({len(active)} задани{'е' if len(active) == 1 else 'я'}). "
        "Если закрыть окно сейчас, приложение остановит работу и завершится.\n\n"
        "Остановить обработку и закрыть?"
    )
    try:
        ui_log(f"close requested while busy: {active}", root)
        # We are on the GUI thread (closing is a should_lock event) and the
        # window is already shown, so the @_shown_call wrapper returns at once
        # and create_confirmation_dialog is a plain modal MessageBox:
        # OK = stop and close, Cancel = keep working.
        keep_open = not window.create_confirmation_dialog("Nestor", text)
    except Exception:  # noqa: BLE001 - if the dialog fails, fall back to closing
        ui_log("close confirmation dialog failed; closing", root)
        return True

    if keep_open:
        ui_log("close cancelled by user", root)
        return False

    # The user chose to stop: signal the workers so the process can exit once
    # they reach their next checkpoint instead of lingering without a window.
    try:
        for record_id in active:
            processor.cancel(record_id)
        ui_log(f"closing: cancelled jobs {active}", root)
    except Exception:  # noqa: BLE001
        ui_log("closing: cancelling jobs failed", root)
    return True


def run(
    work_dir: Optional[Path] = None,
    title: str = WINDOW_TITLE,
    width: int = 1320,
    height: int = 860,
    debug: bool = False,
) -> int:
    """Open the desktop application."""
    import uvicorn
    import webview

    from .desktop_api import DesktopApi
    from .web.app import create_app

    lock = instance_file()
    existing = _read_instance(lock)
    if existing and _port_alive(int(existing.get("port", 0))):
        if focus_existing(title):
            print("Nestor is already running — the existing window was focused.")
            return 0

    # Let the installer/uninstaller see that the application is up.
    acquire_mutex()

    port = free_port()
    token = generate_session_token()
    app = create_app(work_dir=work_dir, token=token)
    lib = app.state.library
    processor = app.state.processor

    hooks = _install_gui_diagnostics(lib.root)
    ui_log(f"launch: version={__version__} port={port}", lib.root)

    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="info", access_log=False
    )
    server = uvicorn.Server(config)

    def serve() -> None:
        try:
            server.run()
        except Exception as exc:  # noqa: BLE001 - surfaced in the console
            print(f"Local server stopped: {type(exc).__name__}: {exc}", file=sys.stderr)

    thread = threading.Thread(target=serve, name="nestor-server", daemon=True)
    thread.start()

    deadline = time.time() + 30
    while time.time() < deadline and not _port_alive(port):
        if not thread.is_alive():
            print("The local server thread stopped unexpectedly.", file=sys.stderr)
            return 1
        time.sleep(0.1)
    if not _port_alive(port):
        print("Could not start the local server.", file=sys.stderr)
        return 1

    lock.write_text(
        json.dumps({"port": port, "pid": os.getpid(), "title": title}),
        encoding="utf-8",
    )

    url = f"http://127.0.0.1:{port}/?token={token}"
    print(f"Nestor is running locally on http://127.0.0.1:{port}/", flush=True)

    def on_closed() -> None:
        print("Window closed, stopping the local server.", flush=True)
        ui_log("window closed", lib.root)
        ui_log(f"jobs still active at close: {processor.active()}", lib.root)
        # The presenter screen (and any other extra window) must go down with
        # the main one: pywebview only exits the process once the last window
        # is gone, so a leftover video window would keep the app (and its
        # server) alive without a main window.
        try:
            if api.close_extra_windows():
                ui_log("closed extra windows with the main one", lib.root)
        except Exception:  # noqa: BLE001 - never block shutdown
            ui_log("closing extra windows failed", lib.root)
        server.should_exit = True

    api = DesktopApi(lib, port=port, token=token)
    window = webview.create_window(
        title,
        url,
        width=width,
        height=height,
        min_size=(900, 600),
        text_select=True,
        js_api=api,
    )
    api.set_window(window)
    window.events.loaded += lambda: (print("Interface loaded.", flush=True), ui_log("interface loaded", lib.root))
    # NOTE: webview.start(func) runs func right after the GUI appears, so the
    # "window closed" hook has to be registered as an event, not passed to start().
    window.events.closed += on_closed
    window.events.closing += lambda: _confirm_close_while_busy(window, processor, lib.root)
    _start_window_watchdog(window, processor, lib.root)

    try:
        webview.start(debug=debug)
        print("Event loop finished.", flush=True)
        ui_log("event loop finished", lib.root)
    except Exception as exc:  # noqa: BLE001 - a GUI crash must be recorded, not swallowed
        ui_log(f"GUI event loop crashed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}", lib.root)
        raise
    finally:
        _restore_excepthooks(hooks)
        # Safety net: if the event loop ends for any other reason (crash,
        # stop()), make sure no extra window keeps the process alive.
        try:
            api.close_extra_windows()
        except Exception:  # noqa: BLE001
            pass
        server.should_exit = True
        lock.unlink(missing_ok=True)
        ui_log("server stopping, instance unlocked", lib.root)
    return 0
