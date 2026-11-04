"""Tests for the export formats and the native save helper."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.exporters.vtt_export import export as export_vtt  # noqa: E402
from nestor.exporters.vtt_export import format_timestamp  # noqa: E402
from nestor.types import Segment, Transcript, Word  # noqa: E402


def _transcript() -> Transcript:
    word = Word("привет", 0.0, 0.5, "SPEAKER_00")
    segment = Segment(0.0, 0.5, "SPEAKER_00", "привет", [word])
    return Transcript(
        source="sample.mp3",
        duration=1.0,
        model="gigaam-v3-punct",
        segments=[segment],
    )


def test_vtt_timestamp_format():
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(61.5) == "00:01:01.500"
    assert format_timestamp(3661.25) == "01:01:01.250"


def test_vtt_export(tmp_path: Path):
    path = export_vtt(_transcript(), tmp_path / "out.vtt")
    content = path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:00.500" in content
    assert "SPEAKER_00: привет" in content


def test_unique_path_avoids_overwriting(tmp_path: Path):
    from nestor.desktop_api import unique_path

    first = tmp_path / "file.txt"
    first.write_text("1", encoding="utf-8")
    second = unique_path(tmp_path, "file.txt")
    assert second != first
    assert second.name == "file (2).txt"


def test_downloads_folder_is_configurable(monkeypatch, tmp_path: Path):
    from nestor import desktop_api

    monkeypatch.setenv("NESTOR_EXPORT_DIR", str(tmp_path))
    assert desktop_api.downloads_folder() == tmp_path
