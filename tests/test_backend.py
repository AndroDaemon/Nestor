"""Tests for the choice of the diarization backend."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor.config import Config  # noqa: E402
from nestor.diarize import build_diarizer, SingleSpeakerDiarizer  # noqa: E402


class _FakeLocal:
    """Stands in for the sherpa-onnx diarizer."""

    model_id = "fake-local"

    def diarize(self, waveform, sample_rate):
        return []


def test_unknown_backend_is_rejected():
    config = Config(diarization_backend="wat")
    try:
        build_diarizer(config)
    except ValueError as exc:
        assert "Unknown diarization backend" in str(exc)
    else:  # pragma: no cover - the call must fail
        raise AssertionError("an unknown backend must raise")


def test_auto_prefers_local_when_models_are_present(monkeypatch):
    import nestor.models as models_module

    monkeypatch.setattr(models_module, "has_local_diarization", lambda *a, **k: True)

    import nestor.diarize_local as local_module

    monkeypatch.setattr(local_module, "LocalDiarizer", lambda *a, **k: _FakeLocal())

    config = Config(diarization_backend="auto")
    assert isinstance(build_diarizer(config), _FakeLocal)


def test_auto_falls_back_to_single_speaker_when_no_local_models(monkeypatch):
    import nestor.models as models_module

    monkeypatch.setattr(models_module, "has_local_diarization", lambda *a, **k: False)

    config = Config(diarization_backend="auto")
    diarizer = build_diarizer(config)
    assert isinstance(diarizer, SingleSpeakerDiarizer)


def test_explicit_none_backend_uses_single_speaker(monkeypatch):
    import nestor.models as models_module

    monkeypatch.setattr(models_module, "has_local_diarization", lambda *a, **k: True)

    import nestor.diarize_local as local_module

    monkeypatch.setattr(local_module, "LocalDiarizer", lambda *a, **k: _FakeLocal())

    config = Config(diarization_backend="none")
    diarizer = build_diarizer(config)
    assert isinstance(diarizer, SingleSpeakerDiarizer)


def test_explicit_local_backend_requires_models(monkeypatch):
    import nestor.models as models_module

    monkeypatch.setattr(models_module, "has_local_diarization", lambda *a, **k: False)

    config = Config(diarization_backend="local")
    try:
        build_diarizer(config)
    except ValueError as exc:
        assert "local" in str(exc).lower() or "model" in str(exc).lower()
    else:
        raise AssertionError("local backend without models must raise")