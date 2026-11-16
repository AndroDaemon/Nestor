"""Speaker diarization backends.

Available backends:
* ``local`` — sherpa-onnx (pyannote-segmentation-3.0 + TitaNet-small), token-free, offline.
* ``none`` — single speaker fallback (when diarization is disabled).
"""

from __future__ import annotations

from typing import List

import numpy as np

from .cpuinfo import resolve_diar_threads
from .types import DEFAULT_SPEAKER, Turn


class SingleSpeakerDiarizer:
    """Fallback used when diarization is disabled: one speaker for everything."""

    model_id = "none"

    def diarize(self, waveform: np.ndarray, sample_rate: int) -> List[Turn]:
        total = waveform.shape[0] / float(sample_rate)
        if total <= 0:
            return []
        return [Turn(start=0.0, end=total, speaker=DEFAULT_SPEAKER)]


def _accepts_progress(diarizer) -> bool:
    """Whether ``diarizer.diarize`` supports the ``progress`` keyword."""
    try:
        import inspect

        return "progress" in inspect.signature(diarizer.diarize).parameters
    except (TypeError, ValueError):
        return False


def _accepts_blocking(diarizer) -> bool:
    """Whether ``diarizer.diarize`` supports block-wise long recordings."""
    try:
        import inspect

        return "block_seconds" in inspect.signature(diarizer.diarize).parameters
    except (TypeError, ValueError):
        return False


def build_diarizer(config, report=None):
    """Create the diarizer described by the configuration.

    ``auto`` uses the local sherpa-onnx stack if models are available,
    otherwise falls back to single-speaker mode.
    """
    from .config import Config
    from .models import has_local_diarization

    assert isinstance(config, Config)
    backend = (config.diarization_backend or "auto").strip().lower()

    if backend == "auto":
        backend = "local" if has_local_diarization() else "none"

    if backend == "local":
        if not has_local_diarization():
            raise ValueError(
                "Local diarization models not found. Run 'nestor models' to download them, "
                "or use 'auto' backend to fall back to single-speaker mode."
            )
        from .diarize_local import LocalDiarizer

        if report:
            report("Using local diarization (sherpa-onnx, no token needed).")
        return LocalDiarizer(
            num_speakers=config.num_speakers,
            num_threads=resolve_diar_threads(config.threads),
            threshold=getattr(config, "diar_threshold", 0.5),
            min_duration_on=getattr(config, "diar_min_duration_on", 0.5),
            min_duration_off=getattr(config, "diar_min_duration_off", 0.0),
        )

    if backend == "none" or not config.diarize:
        if report:
            report("Diarization disabled — single speaker mode.")
        return SingleSpeakerDiarizer()

    raise ValueError(
        f"Unknown diarization backend '{config.diarization_backend}'. "
        "Use auto or local."
    )