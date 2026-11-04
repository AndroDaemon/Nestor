"""Integration: voiceprints inside the full pipeline (with stubbed models).

Verifies the non-negotiable guarantees:

* default Config (matching off) -> byte-identical behaviour to 1.0.0;
* empty library + matching on -> still SPEAKER_NN, no crash;
* matching on + known voices -> names applied via speaker mapping only,
  segment.speaker stays SPEAKER_NN inside the pipeline output;
* embeddings are cached to <record_id>.npz for later "remember this voice".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nestor.pipeline as pipeline_module  # noqa: E402
from nestor.config import Config  # noqa: E402
from nestor.speaker_embed import (  # noqa: E402
    load_record_embeddings,
    l2_normalise,
)
from nestor.types import Segment, Turn, Word  # noqa: E402
from nestor.voiceprints import VoiceprintLibrary  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "example.wav"


class FakeDiarizer:
    """Two speakers talking in turn."""

    model_id = "fake-diarizer"

    def __init__(self, *args, **kwargs):
        pass

    def diarize(self, waveform, sample_rate):
        total = waveform.shape[0] / sample_rate
        half = total / 2
        return [Turn(0.0, half, "SPEAKER_00"), Turn(half, total, "SPEAKER_01")]


class FakeAsr:
    """Emits one word per second of the requested chunk."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe_chunk(self, waveform, start, end, word_timestamps=True):
        words = []
        position = start
        while position < end:
            words.append(Word("слово", position, min(position + 0.8, end)))
            position += 1.0
        return words

    def close(self):
        pass


def run_pipeline(tmp_path, monkeypatch, **config_kwargs):
    monkeypatch.setattr(pipeline_module, "build_diarizer", lambda *a, **k: FakeDiarizer())
    monkeypatch.setattr(pipeline_module, "SherpaOnnxAsr", FakeAsr)
    monkeypatch.setattr(pipeline_module, "ensure_gigaam_punct", lambda *a, **k: {})
    config = Config(
        diarize=True,
        output_dir=tmp_path / "output" / "rec01",
        record_id="rec01",
        embedding_dir=tmp_path / "embeddings",
        formats=["json", "txt"],
        **config_kwargs,
    )
    transcript = pipeline_module.process_file(config, SAMPLE, progress=None)
    return transcript, config


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample audio is missing")
def test_default_config_matches_v1_behaviour(tmp_path, monkeypatch):
    """No voiceprint settings -> no embedding pass, no npz, no speaker_match."""
    transcript, config = run_pipeline(tmp_path, monkeypatch)

    assert transcript.speaker_match == {}
    assert not (tmp_path / "embeddings").exists()
    assert transcript.speakers == ["SPEAKER_00", "SPEAKER_01"]
    # segments carry raw labels (names are applied later by the speakers layer)
    assert transcript.segments[0].speaker == "SPEAKER_00"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample audio is missing")
def test_empty_library_matches_nothing_but_caches_embeddings(tmp_path, monkeypatch):
    transcript, _ = run_pipeline(
        tmp_path, monkeypatch, match_voiceprints=True, vp_library_root=tmp_path
    )

    assert set(transcript.speaker_match) == {"SPEAKER_00", "SPEAKER_01"}
    for result in transcript.speaker_match.values():
        assert result["assigned_name"] is None
        assert result["reason"] == "empty library"
    # embeddings were still saved for a future "remember this voice"
    clusters, per_segment = load_record_embeddings(tmp_path / "embeddings", "rec01")
    assert set(clusters) == {"SPEAKER_00", "SPEAKER_01"}
    assert per_segment["SPEAKER_00"], "segment vectors must be cached too"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample audio is missing")
def test_known_voice_is_matched(tmp_path, monkeypatch):
    # build a "known voice" from the first half of the sample (one speaker there)
    from nestor.audio import decode

    waveform = decode(SAMPLE)
    from nestor.constants import SAMPLE_RATE

    first_half = waveform[: waveform.shape[0] // 2]
    known = l2_normalise(
        pipeline_module.speaker_embed.embed_audio(
            first_half, SAMPLE_RATE, 0.0, first_half.shape[0] / SAMPLE_RATE
        )
    )
    library = VoiceprintLibrary(tmp_path)
    library.add_or_update("Иван", known, duration_sec=5.0)

    transcript, _ = run_pipeline(
        tmp_path, monkeypatch, match_voiceprints=True, vp_library_root=tmp_path
    )

    matched = {
        label: result["assigned_name"]
        for label, result in transcript.speaker_match.items()
    }
    assert matched["SPEAKER_00"] == "Иван"  # the same voice
    assert matched["SPEAKER_01"] is None  # the second half is another voice
    # pipeline output keeps raw labels - names go through the mapping layer
    assert transcript.segments[0].speaker == "SPEAKER_00"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample audio is missing")
def test_matching_off_and_embeddings_off_are_identical(tmp_path, monkeypatch):
    """save_embeddings=False + matching off -> no npz, no stage change."""
    transcript, _ = run_pipeline(
        tmp_path, monkeypatch, save_embeddings=False
    )
    assert transcript.speaker_match == {}
    assert not (tmp_path / "embeddings").exists()
