"""Check the token-free sherpa-onnx diarizer on a sample file."""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from nestor import audio
from nestor.diarize_local import LocalDiarizer

path = sys.argv[1] if len(sys.argv) > 1 else "samples/dialogue.mp3"
num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None

waveform = audio.decode(path)
print(f"audio: {audio.duration(waveform):.1f}s")

started = time.perf_counter()
diarizer = LocalDiarizer(num_speakers=num_speakers)
print(f"loaded in {time.perf_counter() - started:.1f}s: {diarizer.model_id}")

started = time.perf_counter()
turns = diarizer.diarize(waveform, 16000)
elapsed = time.perf_counter() - started

import collections

print(f"diarized in {elapsed:.1f}s (RTF {elapsed / audio.duration(waveform):.3f})")
print("speakers:", dict(collections.Counter(t.speaker for t in turns)))
for turn in turns:
    print(f"  {turn.speaker}  {turn.start:6.2f} - {turn.end:6.2f}")
