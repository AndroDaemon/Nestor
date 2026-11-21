"""The presenter screen, a second native window of DesktopApi.

pywebview allows ``create_window`` after ``start`` only from a non-main thread,
which is exactly how the JS bridge calls in, so the window is created lazily on
the first click. The bridge may deliver two clicks from parallel threads while
the window is still starting, and ``load_url`` blocks until the window is shown
(up to 20 s in pywebview), so the first URL is applied from the ``shown`` event.
These tests pin the URL, the deferral, the reuse (load + bring forward) and the
close/on-top helpers without importing pywebview.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import nestor.desktop_api as desktop_api
from nestor.desktop_api import DesktopApi


class FakeEvent:
    """Minimal stand-in for pywebview's ``Event`` (supports ``+=``)."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def fire(self):
        for handler in list(self.handlers):
            handler()


class FakeEvents:
    """Minimal stand-in for pywebview's ``Window.events`` container."""

    def __init__(self):
        self.shown = FakeEvent()
        self.closed = FakeEvent()


class FakeWindow:
    def __init__(self, url=None, **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.events = FakeEvents()
        self.on_top = bool(kwargs.get("on_top", False))
        self.loaded = []
        self.shown = 0
        self.destroyed = 0

    def load_url(self, url):
        self.loaded.append(url)
        self.url = url

    def show(self):
        self.shown += 1

    def destroy(self):
        self.destroyed += 1


class FakeHandle:
    """Stand-in for the .NET handle returned by ``Form.Handle``."""

    def __init__(self, value):
        self._value = value

    def ToInt32(self):
        return self._value


class FakeNative:
    """Stand-in for the winforms form attached to a pywebview window."""

    def __init__(self, hwnd):
        self.Handle = FakeHandle(hwnd)


class FakeWebview:
    def __init__(self, window=None, error=None):
        self.created = []
        self.error = error
        self.window = window

    def create_window(self, title=None, url=None, **kwargs):
        if self.error is not None:
            raise self.error
        self.created.append({"title": title, "url": url, **kwargs})
        if self.window is not None:
            return self.window
        window = FakeWindow(url, **kwargs)
        self.window = window
        return window


def make_api(port=8123, token="tok"):
    api = DesktopApi.__new__(DesktopApi)  # skip the Library.__init__ signature
    api._window = None
    api._watch_window = None
    api._port = port
    api._token = token
    api._watch_lock = threading.Lock()
    api._watch_shown = False
    api._watch_pending = None
    api._watch_pending_close = None
    return api


def with_webview(monkeypatch, fake):
    monkeypatch.setattr(desktop_api, "webview", fake)
    return fake


def test_creates_window_for_the_record(monkeypatch):
    fake = with_webview(monkeypatch, FakeWebview())
    api = make_api()

    assert api.open_watch_window("rec-1") is True

    assert len(fake.created) == 1
    created = fake.created[0]
    assert created["title"] == "Экран докладчика"
    assert created["url"] == "http://127.0.0.1:8123/watch.html?id=rec-1&token=tok"
    assert created["on_top"] is False
    assert created["js_api"] is api
    assert api._watch_window is not None
    assert api.watch_window_open() is True


def test_record_id_is_escaped(monkeypatch):
    fake = with_webview(monkeypatch, FakeWebview())
    api = make_api(token="")

    api.open_watch_window("a b&c")

    url = fake.created[0]["url"]
    assert "id=a%20b%26c" in url
    assert "token=" not in url


def test_second_record_while_starting_is_deferred(monkeypatch):
    """Before ``shown`` fires, load_url would block the bridge thread."""
    window = FakeWindow("first")
    fake = with_webview(monkeypatch, FakeWebview(window))
    api = make_api()

    api.open_watch_window("rec-1")
    assert api.open_watch_window("rec-2") is True

    assert len(fake.created) == 1          # no second window
    assert window.loaded == []             # and no blocking load yet
    assert api._watch_pending.endswith("id=rec-2&token=tok")

    window.events.shown.fire()             # the window finally appears
    assert window.loaded == ["http://127.0.0.1:8123/watch.html?id=rec-2&token=tok"]
    assert api._watch_shown is True


def test_reuses_the_window_and_brings_it_forward(monkeypatch):
    window = FakeWindow("first")
    fake = with_webview(monkeypatch, FakeWebview(window))
    api = make_api()

    api.open_watch_window("rec-1")
    window.events.shown.fire()

    assert api.open_watch_window("rec-2") is True

    assert len(fake.created) == 1          # no second window
    assert window.url == "http://127.0.0.1:8123/watch.html?id=rec-2&token=tok"
    assert window.shown == 1               # Show + Activate, not just restore


def test_closing_the_window_forgets_it(monkeypatch):
    fake = with_webview(monkeypatch, FakeWebview())
    api = make_api()

    api.open_watch_window("rec-1")
    window = fake.window or api._watch_window
    assert len(window.events.closed.handlers) == 1

    window.events.closed.fire()
    assert api._watch_window is None
    assert api.watch_window_open() is False


def test_close_watch_window_destroys_and_forgets(monkeypatch):
    fake = with_webview(monkeypatch, FakeWebview())
    api = make_api()

    assert api.close_watch_window() is False       # nothing open yet

    api.open_watch_window("rec-1")
    window = fake.window or api._watch_window
    window.events.shown.fire()                     # the window is up
    assert api.close_watch_window() is True
    assert window.destroyed == 1
    assert api._watch_window is None


def test_close_while_starting_is_deferred_to_shown(monkeypatch):
    """destroy() also waits for ``shown``; a close during startup must not block."""
    fake = with_webview(monkeypatch, FakeWebview())
    api = make_api()

    api.open_watch_window("rec-1")
    window = fake.window or api._watch_window

    assert api.close_watch_window() is True
    assert window.destroyed == 0            # not yet - it would block
    assert api._watch_window is None

    window.events.shown.fire()              # the window finally appears
    assert window.destroyed == 1


def test_without_port_or_webview_returns_false(monkeypatch):
    with_webview(monkeypatch, FakeWebview())
    api = make_api(port=None)
    assert api.open_watch_window("rec-1") is False

    with_webview(monkeypatch, None)
    api = make_api()
    assert api.open_watch_window("rec-1") is False


def test_creation_error_is_not_fatal(monkeypatch):
    with_webview(monkeypatch, FakeWebview(error=RuntimeError("boom")))
    api = make_api()

    assert api.open_watch_window("rec-1") is False
    assert api._watch_window is None


def test_set_watch_on_top(monkeypatch):
    fake = with_webview(monkeypatch, FakeWebview())
    api = make_api()

    assert api.set_watch_on_top(True) is False   # nothing to toggle yet

    api.open_watch_window("rec-1")
    window = fake.window or api._watch_window
    window.native = FakeNative(0x1234)

    calls = []

    def set_window_pos(hwnd, insert_after, x, y, cx, cy, flags):
        calls.append((hwnd.value, insert_after.value, flags))
        return 1

    class FakeUser32:
        pass

    class FakeWindll:
        pass

    FakeUser32.SetWindowPos = set_window_pos
    FakeWindll.user32 = FakeUser32
    real = __import__("ctypes")
    fake = SimpleNamespace(
        windll=FakeWindll(),
        c_void_p=real.c_void_p,
        c_int=real.c_int,
        c_uint=real.c_uint,
    )
    monkeypatch.setattr(desktop_api, "ctypes", fake)

    assert api.set_watch_on_top(True) is True
    assert calls[-1] == (0x1234, 0xFFFFFFFFFFFFFFFF, 0x0001 | 0x0002 | 0x0010)
    assert api.set_watch_on_top(False) is True
    assert calls[-1] == (0x1234, 0xFFFFFFFFFFFFFFFE, 0x0001 | 0x0002 | 0x0010)


def test_set_watch_on_top_without_native_handle(monkeypatch):
    """A window that has no winforms form yet must not raise."""
    with_webview(monkeypatch, FakeWebview())
    api = make_api()
    api.open_watch_window("rec-1")

    assert api.set_watch_on_top(True) is False


def test_close_extra_windows_closes_the_presenter_screen(monkeypatch):
    """Closing the main window must take the presenter screen down too.

    pywebview only exits once the last window is gone, so a lingering video
    window would keep the app alive without its main window.
    """
    fake = with_webview(monkeypatch, FakeWebview())
    api = make_api()

    assert api.close_extra_windows() is False      # nothing open yet

    api.open_watch_window("rec-1")
    window = fake.window or api._watch_window
    window.events.shown.fire()

    assert api.close_extra_windows() is True
    assert window.destroyed == 1
    assert api._watch_window is None


def test_close_extra_windows_is_safe_without_any_window(monkeypatch):
    with_webview(monkeypatch, FakeWebview())
    api = make_api()

    assert api.close_extra_windows() is False
    assert api.watch_window_open() is False
