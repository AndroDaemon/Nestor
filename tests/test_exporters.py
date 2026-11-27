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


def test_talk_stats_counts_speakers():
    from nestor.exporters import format_talk, talk_stats

    transcript = _transcript()
    transcript.segments = [
        Segment(0.0, 1.0, "SPEAKER_00", "привет", []),
        Segment(2.0, 3.0, "SPEAKER_00", "ещё", []),
        Segment(4.0, 5.0, "SPEAKER_01", "пока", []),
    ]
    total, rows = talk_stats(transcript)
    assert total == 3.0
    assert rows[0]["speaker"] == "SPEAKER_00"
    assert rows[0]["turns"] == 2 and rows[0]["seconds"] == 2.0
    assert rows[0]["longest"] == 1.0 and rows[0]["avg"] == 1.0
    assert rows[1]["speaker"] == "SPEAKER_01"
    assert round(rows[1]["share"], 2) == 0.33
    assert format_talk(3725) == "1:02:05"
    assert format_talk(65) == "1:05"


def test_txt_export_lists_talk_time(tmp_path: Path):
    transcript = _transcript()
    transcript.segments = [
        Segment(0.0, 2.0, "SPEAKER_00", "привет", []),
        Segment(3.0, 4.0, "SPEAKER_01", "пока", []),
    ]
    content = export_txt(transcript, tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "Речь по участникам" in content
    assert "SPEAKER_00" in content and "0:02" in content
