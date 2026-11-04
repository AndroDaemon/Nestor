"""End-to-end pipeline test with stubbed models.

Only ffmpeg and numpy are needed: the diarizer and the ASR are replaced by
fakes, so we verify the wiring (chunking -> transcription -> speaker
attribution -> segments) without downloading any checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import nestor.pipeline as pipeline_module  # noqa: E402
from nestor.config import Config  # noqa: E402
from nestor.types import Turn, Word  # noqa: E402

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


def test_pipeline_produces_two_speakers(tmp_path, monkeypatch):
    if not SAMPLE.exists():
        import pytest

        pytest.skip("sample audio is missing")

    monkeypatch.setattr(pipeline_module, "build_diarizer", lambda *a, **k: FakeDiarizer())
    monkeypatch.setattr(pipeline_module, "SherpaOnnxAsr", FakeAsr)
    monkeypatch.setattr(pipeline_module, "ensure_gigaam_punct", lambda *a, **k: {})

    config = Config(
        diarize=True,
        output_dir=tmp_path,
        formats=["json", "srt", "txt"],
        speaker_labels=["Иван", "Мария"],
    )
    transcript = pipeline_module.process_file(config, SAMPLE, progress=None)

    assert transcript.speakers == ["Иван", "Мария"]
    assert transcript.segments
    assert transcript.segments[0].speaker == "Иван"
    assert transcript.segments[-1].speaker == "Мария"
    assert transcript.duration > 0

    from nestor.exporters import export_all

    written = export_all(transcript, config)
    assert {path.suffix for path in written} == {".json", ".srt", ".txt"}
    assert all(path.exists() for path in written)


def test_pipeline_without_diarization(tmp_path, monkeypatch):
    if not SAMPLE.exists():
        import pytest

        pytest.skip("sample audio is missing")

    monkeypatch.setattr(pipeline_module, "SherpaOnnxAsr", FakeAsr)
    monkeypatch.setattr(pipeline_module, "ensure_gigaam_punct", lambda *a, **k: {})

    config = Config(diarize=False, output_dir=tmp_path, formats=["txt"])
    transcript = pipeline_module.process_file(config, SAMPLE, progress=None)

    assert transcript.speakers == ["SPEAKER_00"]
    assert transcript.diarization_model is None


def test_pipeline_reports_model_loading_stage(tmp_path, monkeypatch):
    """The first transcription loads the model (10-20 s) and the UI must hear
    about it, otherwise the progress bar looks frozen on "transcription"."""
    if not SAMPLE.exists():
        import pytest

        pytest.skip("sample audio is missing")

    monkeypatch.setattr(pipeline_module, "build_diarizer", lambda *a, **k: FakeDiarizer())
    monkeypatch.setattr(pipeline_module, "SherpaOnnxAsr", FakeAsr)
    monkeypatch.setattr(pipeline_module, "ensure_gigaam_punct", lambda *a, **k: {})
    # the caches are process-wide and other tests may have filled them
    pipeline_module._ASR_CACHE.clear()
    pipeline_module._DIARIZER_CACHE.clear()

    seen: list[tuple[str, float]] = []
    config = Config(diarize=True, output_dir=tmp_path, formats=["txt"])
    pipeline_module.process_file(
        config,
        SAMPLE,
        progress=None,
        on_stage=lambda stage, fraction: seen.append((stage, fraction)),
    )

    stages = [stage for stage, _ in seen]
    # both models are cold, so the UI is told before each blocking load; the
    # notice lands right after the stage marker it belongs to
    loadings = [i for i, s in enumerate(stages) if s == "loading"]
    assert len(loadings) == 2
    assert loadings[0] == stages.index("diarization") + 1
    assert loadings[1] == stages.index("transcription") + 1


def test_pipeline_merges_duplicate_speakers(tmp_path, monkeypatch):
    """The post-diarization merge folds labels that are the same voice.

    The fake diarizer splits the audio into two halves with distinct labels;
    a stubbed embedding stage returns two *identical* centroids, so the merge
    stage must fold them into one speaker - and ``merge_speakers=False`` must
    leave the two labels alone.
    """
    if not SAMPLE.exists():
        import pytest

        pytest.skip("sample audio is missing")

    import nestor.speaker_embed as speaker_embed

    monkeypatch.setattr(pipeline_module, "build_diarizer", lambda *a, **k: FakeDiarizer())
    monkeypatch.setattr(pipeline_module, "SherpaOnnxAsr", FakeAsr)
    monkeypatch.setattr(pipeline_module, "ensure_gigaam_punct", lambda *a, **k: {})
    pipeline_module._ASR_CACHE.clear()
    pipeline_module._DIARIZER_CACHE.clear()

    same = [1.0] + [0.0] * 191

    def fake_clusters(waveform, sample_rate, segments, min_segment_sec=0.5):
        labels = sorted({seg.speaker for seg in segments})
        clusters = {label: same for label in labels}
        per_segment = {label: [same] for label in labels}
        return clusters, per_segment

    monkeypatch.setattr(speaker_embed, "cluster_vectors", fake_clusters)
    # the pipeline imports cluster_vectors through the module object, so the
    # patch above is seen - but keep the cached singleton out of the way
    monkeypatch.setattr(
        speaker_embed, "save_record_embeddings", lambda *a, **k: None
    )

    merged = pipeline_module.process_file(
        Config(
            diarize=True,
            merge_speakers=True,
            merge_threshold=0.9,
            merge_min_weight=0.0,
            output_dir=tmp_path,
            formats=["txt"],
        ),
        SAMPLE,
        progress=None,
    )
    assert merged.speakers == ["SPEAKER_00"], "identical voices must merge"
    assert {seg.speaker for seg in merged.segments} == {"SPEAKER_00"}

    kept = pipeline_module.process_file(
        Config(
            diarize=True,
            merge_speakers=False,
            output_dir=tmp_path / "off",
            formats=["txt"],
        ),
        SAMPLE,
        progress=None,
    )
    assert kept.speakers == ["SPEAKER_00", "SPEAKER_01"]
