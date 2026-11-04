"""Token-free speaker diarization with sherpa-onnx.

Uses two public ONNX models:

* ``sherpa-onnx-pyannote-segmentation-3-0`` — who speaks when
* ``nemo_en_titanet_small.onnx`` — voice embeddings

Everything is bundled locally, so no Hugging Face token and no account are
needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .models import ensure_local_diarization, local_diarization_models
from .types import Turn


class LocalDiarizationError(RuntimeError):
    """Raised when the local diarization stack cannot be used."""


class LocalDiarizer:
    """Diarizer that works offline, without any model hub access."""

    model_id = "sherpa-onnx: pyannote-segmentation-3.0 + TitaNet-small"

    def __init__(
        self,
        model_root: Optional[Path] = None,
        num_threads: int = 2,
        num_speakers: Optional[int] = None,
        threshold: float = 0.5,
        min_duration_on: float = 0.25,
        min_duration_off: float = 0.4,
        provider: str = "cpu",
    ) -> None:
        models = local_diarization_models(model_root)
        if models is None:
            try:
                models = ensure_local_diarization()
            except Exception as exc:  # noqa: BLE001
                raise LocalDiarizationError(
                    "The local diarization models are missing and could not be "
                    f"downloaded: {exc}"
                ) from exc

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise LocalDiarizationError(
                "sherpa-onnx is not installed. Run: pip install sherpa-onnx"
            ) from exc

        pyannote_kwargs = {"model": str(models["segmentation"])}
        # window_shift_ratio appeared in sherpa-onnx 2.x; 1.13.x does not have it
        try:
            sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(models["segmentation"]), window_shift_ratio=0.25
            )
            pyannote_kwargs["window_shift_ratio"] = 0.25
        except TypeError:
            pass
        pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            **pyannote_kwargs
        )
        if not pyannote.validate():
            raise LocalDiarizationError(
                f"sherpa-onnx rejected the segmentation model: {models['segmentation']}"
            )

        segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig()
        segmentation.pyannote = pyannote
        segmentation.num_threads = num_threads
        segmentation.provider = provider

        embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(models["embedding"]),
            num_threads=num_threads,
            provider=provider,
        )
        if not embedding.validate():
            raise LocalDiarizationError(
                f"sherpa-onnx rejected the embedding model: {models['embedding']}"
            )

        clustering = sherpa_onnx.FastClusteringConfig(
            num_clusters=-1 if num_speakers is None else int(num_speakers),
            threshold=threshold,
        )

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation,
            embedding,
            clustering,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

        if not config.validate():
            raise LocalDiarizationError(
                "sherpa-onnx rejected the diarization configuration"
            )

        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)

    def diarize(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        progress: Optional[Callable[[float], int]] = None,
    ) -> List[Turn]:
        """Return speaker turns for a 16 kHz mono waveform.

        ``progress`` receives the fraction of processed chunks (``0..1``) as
        reported by sherpa-onnx, so long jobs can update the UI. Returning a
        truthy value from ``progress`` aborts the run (sherpa-onnx stops when
        the callback returns non-zero); the result is then empty/partial and
        the caller must check its own stop flag.
        """
        samples = np.ascontiguousarray(waveform, dtype=np.float32)
        if samples.size == 0:
            return []

        callback = None
        if progress is not None:
            def callback(done: int, total: int) -> int:
                try:
                    stop = progress(done / total if total > 0 else 0.0)
                except Exception:  # noqa: BLE001 - never abort the job from UI
                    stop = None
                return 1 if stop else 0  # non-zero aborts diarization

        result = self._diarizer.process(samples, callback)
        turns: List[Turn] = []
        for segment in result.sort_by_start_time():
            start = float(segment.start)
            end = float(segment.end)
            if end <= start:
                continue
            turns.append(
                Turn(start=round(start, 3), end=round(end, 3), speaker=f"SPEAKER_{int(segment.speaker):02d}")
            )
        return turns
