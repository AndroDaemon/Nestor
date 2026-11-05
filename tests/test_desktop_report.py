"""Native "save report as" helper in DesktopApi.

The run report must not rely on a browser download: the desktop window is
WebView2, which has no download UI, so the file would land in Downloads with
no word to the user. These tests pin the dialog-based path.
"""

from __future__ import annotations

from pathlib import Path

from nestor.desktop_api import DesktopApi, unique_path


class FakeWindow:
    """Stands in for pywebview.Window: records the dialog call."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def create_file_dialog(self, dialog_type, **kwargs):
        self.calls.append({"dialog_type": dialog_type, **kwargs})
        return self.result


def make_api(tmp_path, result):
    api = DesktopApi.__new__(DesktopApi)   # skip the Library.__init__ signature
    api._window = FakeWindow(result)
    return api


def test_saves_chosen_path_and_returns_it(tmp_path):
    target = tmp_path / "nestor-отчёт.txt"
    api = make_api(tmp_path, (str(target),))

    path = api.save_report_as("отчёт\nстрока", "nestor-отчёт-2026-11-03 15-05.txt")

    assert path == str(target)
    assert target.read_text(encoding="utf-8") == "отчёт\nстрока"
    call = api._window.calls[0]
    assert call["save_filename"].endswith(".txt")
    assert "Текстовый" in call["file_types"][0]


def test_cancelled_dialog_changes_nothing(tmp_path):
    api = make_api(tmp_path, None)

    assert api.save_report_as("text", "r.txt") is None
    assert list(tmp_path.iterdir()) == []


def test_without_window_returns_none(tmp_path):
    api = DesktopApi.__new__(DesktopApi)
    api._window = None

    assert api.save_report_as("text", "r.txt") is None


def test_filename_suffix_and_traversal_are_tamed(tmp_path):
    target = tmp_path / "ok.txt"
    api = make_api(tmp_path, (str(target),))

    api.save_report_as("x", "..\\..\\evil")

    call = api._window.calls[0]
    assert call["save_filename"] == "evil.txt"
    assert "\\" not in call["save_filename"]


def test_blank_filename_falls_back_to_default(tmp_path):
    target = tmp_path / "ok.txt"
    api = make_api(tmp_path, (str(target),))

    api.save_report_as("x", "   ")

    assert api._window.calls[0]["save_filename"] == "nestor-report.txt"


def test_unique_path_does_not_overwrite(tmp_path):
    first = tmp_path / "a.txt"
    first.write_text("1", encoding="utf-8")

    assert unique_path(tmp_path, "a.txt") == tmp_path / "a (2).txt"
    assert Path(first).read_text(encoding="utf-8") == "1"


def test_open_external_uses_system_browser(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "nestor.desktop_api.webbrowser.open",
        lambda url: seen.append(url) or True,
    )
    api = DesktopApi.__new__(DesktopApi)

    assert api.open_external("https://t.me/Nestor_transcribe") is True
    assert seen == ["https://t.me/Nestor_transcribe"]


def test_open_external_rejects_non_web_schemes(monkeypatch):
    called = []
    monkeypatch.setattr(
        "nestor.desktop_api.webbrowser.open",
        lambda url: called.append(url) or True,
    )
    api = DesktopApi.__new__(DesktopApi)

    for bad in (
        "file:///C:/Windows/System32/calc.exe",
        "javascript:alert(1)",
        "not a url",
        "",
    ):
        assert api.open_external(bad) is False
    assert called == []
