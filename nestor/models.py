"""Where the local models live and how to fetch them.

The local diarization stack needs two ONNX models that are published without any
access restrictions:

* ``sherpa-onnx-pyannote-segmentation-3-0`` — speaker segmentation
* ``nemo_en_titanet_small.onnx`` — speaker embeddings (TitaNet-small)
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, Optional

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/nemo_en_titanet_small.onnx"
)

SEGMENTATION_FILES = ("model.onnx", "model.int8.onnx")
EMBEDDING_FILE = "nemo_en_titanet_small.onnx"

# GigaAM v3 with punctuation, as an ONNX transducer for sherpa-onnx.
# Published by the k2-fsa project; the encoder metadata carries
# model_type=EncDecRNNTBPEModel so sherpa-onnx must be told
# model_type="nemo_transducer" (all parameters are read from the encoder).
GIGAAM_PUNCT_DIR = "gigaam-v3-punct"
GIGAAM_PUNCT_URL = (
    "https://huggingface.co/csukuangfj/"
    "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16/resolve/main/"
)
GIGAAM_PUNCT_FILES = ("encoder.int8.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")


def bundled_models_dir() -> Optional[Path]:
    """Folder with the models shipped next to the executable, if any."""
    if not getattr(sys, "frozen", False):
        return None
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "models")
    candidates.append(Path(sys.executable).resolve().parent / "models")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def model_root() -> Path:
    """Directory where the models are stored."""
    env = os.environ.get("NESTOR_MODEL_DIR")
    if env:
        return Path(env)

    bundled = bundled_models_dir()
    if bundled is not None:
        return bundled

    local = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Nestor" / "models"
    if local.exists():
        return local

    project = Path(__file__).resolve().parent.parent / "models"
    if project.exists():
        return project

    return local


def download_root() -> Path:
    """Where newly downloaded models should be stored."""
    env = os.environ.get("NESTOR_MODEL_DIR")
    if env:
        return Path(env)
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Nestor" / "models"


def local_diarization_models(root: Optional[Path] = None) -> Optional[Dict[str, Path]]:
    """Paths of the local diarization models, or ``None`` if something is missing."""
    roots = [Path(root)] if root else [model_root(), download_root()]
    for candidate_root in roots:
        segmentation = candidate_root / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
        embedding = candidate_root / EMBEDDING_FILE
        if segmentation.exists() and embedding.exists():
            return {"segmentation": segmentation, "embedding": embedding}
    return None


def has_local_diarization(root: Optional[Path] = None) -> bool:
    """Whether the token-free diarization stack is ready to use."""
    return local_diarization_models(root) is not None


def local_gigaam_punct(root: Optional[Path] = None) -> Optional[Dict[str, Path]]:
    """Paths of the GigaAM punct ONNX model, or ``None`` if incomplete."""
    roots = [Path(root)] if root else [model_root(), download_root()]
    for candidate_root in roots:
        directory = candidate_root / GIGAAM_PUNCT_DIR
        files = {name: directory / name for name in GIGAAM_PUNCT_FILES}
        if all(path.exists() for path in files.values()):
            return files
    return None


def has_gigaam_punct(root: Optional[Path] = None) -> bool:
    """Whether the punctuated ONNX model is already on disk."""
    return local_gigaam_punct(root) is not None


def ensure_gigaam_punct(root: Optional[Path] = None, force: bool = False) -> Dict[str, Path]:
    """Download the punctuated ONNX model if needed and return its file paths."""
    existing = local_gigaam_punct(root)
    if existing and not force:
        return existing

    root = Path(root) if root else download_root()
    directory = root / GIGAAM_PUNCT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for name in GIGAAM_PUNCT_FILES:
        target = directory / name
        if force or not target.exists():
            _download(GIGAAM_PUNCT_URL + name, target)

    models = local_gigaam_punct(root)
    if models is None:
        raise FileNotFoundError(
            f"Could not prepare the GigaAM punct model in {directory}. "
            "Check the network connection and try again."
        )
    return models


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...", flush=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_local_diarization(root: Optional[Path] = None, force: bool = False) -> Dict[str, Path]:
    """Download the models if needed and return their paths."""
    root = Path(root) if root else download_root()
    root.mkdir(parents=True, exist_ok=True)

    segmentation_dir = root / "sherpa-onnx-pyannote-segmentation-3-0"
    segmentation = segmentation_dir / "model.onnx"
    embedding = root / EMBEDDING_FILE

    if force or not segmentation.exists():
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "seg.tar.bz2"
            _download(SEGMENTATION_URL, archive)
            with tarfile.open(archive, "r:bz2") as tar:
                tar.extractall(tmp)
            extracted = Path(tmp) / "sherpa-onnx-pyannote-segmentation-3-0"
            if not extracted.exists():
                # the archive may contain a single folder with a different name
                candidates = [p for p in Path(tmp).iterdir() if p.is_dir()]
                extracted = candidates[0] if candidates else Path(tmp)
            segmentation_dir.mkdir(parents=True, exist_ok=True)
            for name in SEGMENTATION_FILES:
                source = extracted / name
                if source.exists():
                    shutil.copyfile(source, segmentation_dir / name)

    if force or not embedding.exists():
        _download(EMBEDDING_URL, embedding)

    models = local_diarization_models(root)
    if models is None:
        raise FileNotFoundError(
            f"Could not prepare the local diarization models in {root}. "
            "Check the network connection and try again."
        )
    return models
