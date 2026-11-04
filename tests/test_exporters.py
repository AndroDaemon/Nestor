"""Tests for the export formats (JSON / SRT / TXT)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.exporters.json_export import export as export_json  # noqa: E402
from nestor.exporters.srt_export import export as export_srt  # noqa: E402
from nestor.exporters.txt_export import export as export_txt  # noqa: E402
from nestor.types import Segment, Transcript, Word  # noqa: E402


def _transcript() -> Transcript:
    word = Word("привет", 0.0, 0.5, "SPEAKER_00")
    segment = Segment(0.0, 0.5, "SPEAKER_00", "привет", [word])
    return Transcript(
        source="sample.mp3",
        duration=12.5,
        model="gigaam-v3-punct",
        created_at="2026-01-01T00:00:00+00:00",
        segments=[segment],
        rtf=0.5,
        processing_seconds=6.25,
    )


def test_json_export(tmp_path: Path):
    path = export_json(_transcript(), tmp_path / "out.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["asr_model"] == "gigaam-v3-punct"
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert payload["segments"][0]["words"][0]["text"] == "привет"


def test_srt_export(tmp_path: Path):
    path = export_srt(_transcript(), tmp_path / "out.srt")
    content = path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:00,500" in content
    assert "SPEAKER_00: привет" in content


def test_txt_export(tmp_path: Path):
    path = export_txt(_transcript(), tmp_path / "out.txt")
    content = path.read_text(encoding="utf-8")
    assert "SPEAKER_00: привет" in content
