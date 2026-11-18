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
from pathlib import Path
from typing import Optional

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
    window.events.loaded += lambda: print("Interface loaded.", flush=True)
    # NOTE: webview.start(func) runs func right after the GUI appears, so the
    # "window closed" hook has to be registered as an event, not passed to start().
    window.events.closed += on_closed

    try:
        webview.start(debug=debug)
        print("Event loop finished.", flush=True)
    finally:
        server.should_exit = True
        lock.unlink(missing_ok=True)
    return 0
