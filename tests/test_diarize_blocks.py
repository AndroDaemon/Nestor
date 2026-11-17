"""Block-wise diarization must skip blocks that are pure digital silence.

A recording made with a paused dictaphone holds hours of silence; running
pyannote over it costs minutes and lets it invent speaker turns out of the
noise floor — the turn then goes to the embedding model and, if it is longer
than 122,88 s, kills the job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.diarize_local import (  # noqa: E402
    LocalDiarizer,
    block_level_db,
    is_silent_block,
)
from nestor.types import Turn  # noqa: E402

SR = 16000


def test_digital_silence_is_silent():
    assert is_silent_block(np.zeros(SR * 5, dtype=np.float32))
    assert block_level_db(np.zeros(SR, dtype=np.float32)) == -120.0


def test_noise_floor_of_a_muted_recorder_is_silent():
    # -100 dBFS: единичные биты на дне 16-битной шкалы
    samples = np.full(SR * 5, 1e-5, dtype=np.float32)
    assert is_silent_block(samples)


def test_real_speech_is_not_silent():
    samples = (0.05 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR)).astype(np.float32)
    assert not is_silent_block(samples)


def test_empty_block_is_silent():
    assert is_silent_block(np.zeros(0, dtype=np.float32))


def test_blocks_of_silence_are_not_sent_to_the_diarizer():
    diarizer = object.__new__(LocalDiarizer)
    calls = []

    def fake_call(piece, sample_rate, offset, progress, span=1.0, base=0.0):
        calls.append(offset)
        return [Turn(offset, offset + 1.0, "SPEAKER_00")]

    diarizer._diarize_call = fake_call  # type: ignore[assignment]
    diarizer._cluster_centroids = lambda piece, sr, turns, offset=0.0: {}  # type: ignore[assignment]

    block = SR  # 1 секунда
    samples = np.concatenate(
        [
            np.zeros(block, dtype=np.float32),  # тишина
            np.full(block, 0.05, dtype=np.float32),  # «речь»
            np.zeros(block, dtype=np.float32),  # тишина
        ]
    )

    turns = diarizer._diarize_blocks(samples, SR, block_seconds=1.0, overlap=0.0, progress=None)

    assert calls == [1.0]
    assert len(turns) == 1
