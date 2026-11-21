"""Closing the desktop window while a transcription job is running.

The window used to vanish while the non-daemon worker threads kept going, so
the process stayed alive with no window and looked like a crash. These tests
pin the guard: with an active job the user is asked; "cancel" keeps the window
and the job, "stop" cancels the jobs so the process can exit.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import nestor.desktop as desktop


class FakeWindow:
    def __init__(self, confirm_result=True, error=None):
        self.confirm_result = confirm_result
        self.error = error
        self.dialogs = []

    def create_confirmation_dialog(self, title, message):
        self.dialogs.append((title, message))
        if self.error is not None:
            raise self.error
        return self.confirm_result


class FakeProcessor:
    def __init__(self, active=(), fail=False):
        self._active = list(active)
        self.fail = fail
        self.cancelled = []

    def active(self):
        if self.fail:
            raise RuntimeError("boom")
        return list(self._active)

    def cancel(self, record_id):
        self.cancelled.append(record_id)
        return "running"


def test_no_jobs_closes_without_asking(tmp_path):
    window = FakeWindow()
    processor = FakeProcessor(active=[])

    assert desktop._confirm_close_while_busy(window, processor, tmp_path) is True
    assert window.dialogs == []
    assert processor.cancelled == []


def test_cancel_keeps_window_and_job(tmp_path):
    window = FakeWindow(confirm_result=False)
    processor = FakeProcessor(active=["rec1"])

    # Returning False makes pywebview set args.Cancel and keep the window open.
    assert desktop._confirm_close_while_busy(window, processor, tmp_path) is False
    assert len(window.dialogs) == 1
    assert processor.cancelled == []          # the job keeps running


def test_stop_cancels_jobs_and_allows_close(tmp_path):
    window = FakeWindow(confirm_result=True)
    processor = FakeProcessor(active=["rec1", "rec2"])

    assert desktop._confirm_close_while_busy(window, processor, tmp_path) is True
    assert processor.cancelled == ["rec1", "rec2"]


def test_dialog_failure_does_not_block_closing(tmp_path):
    window = FakeWindow(error=RuntimeError("no gui"))
    processor = FakeProcessor(active=["rec1"])

    assert desktop._confirm_close_while_busy(window, processor, tmp_path) is True


def test_processor_without_active_closes(tmp_path):
    """A processor without the helper (older object) must not block closing."""
    window = FakeWindow()

    class Bare:
        pass

    assert desktop._confirm_close_while_busy(window, Bare(), tmp_path) is True


def test_ui_log_writes_a_line(tmp_path):
    desktop.ui_log("hello", tmp_path)
    text = (tmp_path / "ui.log").read_text(encoding="utf-8")
    assert "hello" in text


def test_ui_log_never_raises(tmp_path):
    bad = tmp_path / "file.txt"
    bad.write_text("x", encoding="utf-8")
    desktop.ui_log("boom", bad)  # a file where a directory is expected


def test_excepthooks_are_installed_and_restored(tmp_path):
    before = (sys.excepthook, threading.excepthook)
    hooks = desktop._install_gui_diagnostics(tmp_path)
    assert sys.excepthook is not before[0]
    assert threading.excepthook is not before[1]

    try:
        raise ValueError("kaboom")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    desktop._restore_excepthooks(hooks)
    assert sys.excepthook is before[0]
    assert threading.excepthook is before[1]
    assert "kaboom" in (tmp_path / "ui.log").read_text(encoding="utf-8")


class FakeHandle:
    def __init__(self, value):
        self._value = value

    def ToInt32(self):
        return self._value


class FakeNativeWindow:
    def __init__(self, hwnd):
        self.native = SimpleNamespace(Handle=FakeHandle(hwnd))


class NoNativeWindow:
    pass


def test_window_alive_without_native_returns_none():
    """No native handle (not started / not winforms) -> we say nothing."""
    assert desktop._window_alive(NoNativeWindow()) is None


def test_window_alive_reads_the_handle(monkeypatch):
    window = FakeNativeWindow(0x1234)
    seen = {}

    class FakeUser32:
        def IsWindow(self, hwnd):
            seen["hwnd"] = hwnd.value
            return 1

    fake_ctypes = SimpleNamespace(
        windll=SimpleNamespace(user32=FakeUser32()),
        c_void_p=__import__("ctypes").c_void_p,
    )
    monkeypatch.setattr(desktop, "ctypes", fake_ctypes)

    assert desktop._window_alive(window) is True
    assert seen["hwnd"] == 0x1234


def test_window_alive_never_raises(monkeypatch):
    class Boom:
        @property
        def native(self):
            raise RuntimeError("nope")

    assert desktop._window_alive(Boom()) is None

