"""The rescue pass only re-decodes holes that are louder than the room.

The pass exists for quiet speech the first ASR run missed. Re-running it over
pure silence invites the model to invent words, so holes whose loudest frame
stays at the recording's own noise floor are skipped before the model is even
asked. These tests drive ``_rescue_holes`` with a fake ASR that always returns
a fixed word, so any word in the result can only have come from a hole the
pass actually decided to re-decode.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import nestor.pipeline as pipeline_module  # noqa: E402
from nestor.constants import SAMPLE_RATE  # noqa: E402
from nestor.types import Word  # noqa: E402


class RecordingAsr:
    """Returns one word that sits entirely inside the requested slice.

    The rescue pass keeps only words that fit inside the hole, so the fake
    records every call it receives — that call log is the real assertion.
    """

    def __init__(self, text: str = "тихо"):
        self.text = text
        self.calls: list[tuple[float, float]] = []

    def transcribe(self, waveform, offset: float = 0.0):
        length = len(waveform) / float(SAMPLE_RATE)
        self.calls.append((offset, offset + length))
        # centre the word in the slice so it lies inside the hole: the pass
        # keeps only words that fit between the neighbouring words
        middle = offset + length / 2.0
        return [Word(self.text, middle - 0.2, middle + 0.2)]


def _run(monkeypatch, waveform, words, asr):
    """Call the private pass with inert reporting hooks."""
    monkeypatch.setattr(pipeline_module, "RESCUE_CONTEXT", 0.2)
    report_lines: list[str] = []
    return pipeline_module._rescue_holes(
        waveform,
        words,
        asr,
        report=report_lines.append,
        cancelled=lambda: False,
        stage=lambda *a, **k: None,
    ), report_lines


def _tone(seconds: float, amplitude: float) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / np.float32(SAMPLE_RATE)
    return (np.float32(amplitude) * np.sin(np.float32(2.0 * np.pi * 220.0) * t)).astype(np.float32)


def test_silent_hole_is_not_re_decoded(monkeypatch):
    """A hole that never rises above the floor is left alone."""
    quiet = _tone(4.0, 0.0005)          # -66 dBFS: room noise
    words = [Word("до", 0.5, 1.0), Word("после", 3.0, 3.5)]
    asr = RecordingAsr()

    result, lines = _run(monkeypatch, quiet, words, asr)

    assert asr.calls == []              # the model was never asked
    assert [w.text for w in result] == ["до", "после"]
    assert any("pure silence" in line for line in lines)


def test_quiet_speech_in_the_hole_is_still_recovered(monkeypatch):
    """The gate must not swallow quiet speech — the reason the pass exists."""
    waveform = _tone(4.0, 0.0005)
    # the missed phrase is ~30 dB above the floor, a distant participant
    waveform[int(1.2 * SAMPLE_RATE):int(2.6 * SAMPLE_RATE)] = _tone(1.4, 0.02)
    words = [Word("до", 0.5, 1.0), Word("после", 3.0, 3.5)]
    asr = RecordingAsr("восстановлено")

    result, _ = _run(monkeypatch, waveform, words, asr)

    assert len(asr.calls) == 1
    # the slice starts RESCUE_CONTEXT (0.2 s) before the hole at 1.0 s
    assert 0.75 < asr.calls[0][0] < 0.85
    assert "восстановлено" in [w.text for w in result]


def test_short_and_long_gaps_are_skipped(monkeypatch):
    """Gaps outside the 2..20 s window stay untouched, as before."""
    loud = _tone(40.0, 0.05)
    words = [
        Word("а", 1.0, 1.3),            # 1.3 -> 2.3 is a 1 s pause: too short
        Word("б", 2.3, 2.6),
        # 2.6 -> 30.0 is 27.4 s: beyond RESCUE_MAX_HOLE
        Word("в", 30.0, 30.3),
    ]
    asr = RecordingAsr()

    result, _ = _run(monkeypatch, loud, words, asr)

    assert asr.calls == []
    assert [w.text for w in result] == ["а", "б", "в"]


def test_edge_context_does_not_answer_for_the_hole(monkeypatch):
    """The word bordering a hole must not make a silent hole look audible."""
    waveform = _tone(6.0, 0.0005)
    words = [Word("before", 1.0, 2.0), Word("after", 4.0, 5.0)]
    # make the word tails loud, but leave 2.0..4.0 itself at the floor
    waveform[int(1.6 * SAMPLE_RATE):int(2.0 * SAMPLE_RATE)] = _tone(0.4, 0.4)
    waveform[int(4.0 * SAMPLE_RATE):int(4.4 * SAMPLE_RATE)] = _tone(0.4, 0.4)
    asr = RecordingAsr()

    _, lines = _run(monkeypatch, waveform, words, asr)

    assert asr.calls == []
    assert any("pure silence" in line for line in lines)
