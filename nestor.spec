# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build script for the Nestor desktop application.

    python -m nestor build-exe

Produces dist/Nestor/Nestor.exe (onedir build: the exe needs its neighbouring
folder, that is how PyInstaller ships the heavy libraries).
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve()
ONEDIR = True

# --------------------------------------------------------------------- datas
datas = []
# web interface
datas += [(str(ROOT / "nestor" / "web" / "static"), "nestor/web/static")]
# bundled glossary (hotwords) for the term-correction pass
datas += [(str(ROOT / "nestor" / "data" / "hotwords.yaml"), "nestor/data")]
# demo recording for the "Попробовать на примере" button (served by /api/sample)
_sample = ROOT / "samples" / "dialogue.webm"
if _sample.exists():
    datas.append((str(_sample), "sample"))
    print(f"[spec] bundling sample audio from {_sample}")

binaries = []

# Local diarization models (sherpa-onnx): token-free stack
local_models = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Nestor" / "models"
if local_models.exists():
    embedding = local_models / "nemo_en_titanet_small.onnx"
    segmentation = local_models / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    if embedding.exists():
        datas.append((str(embedding), "models"))
    if segmentation.exists():
        datas.append((str(segmentation), "models/sherpa-onnx-pyannote-segmentation-3-0"))
    print(f"[spec] bundling local diarization models from {local_models}")
else:
    print("[spec] no local diarization models found — run: nestor models")

# GigaAM v3 punct transducer (sherpa-onnx, ONNX): the default ASR with
# punctuation.  ~220 MB; bundled when already downloaded (nestor models).
punct_dir = local_models / "gigaam-v3-punct" if local_models.exists() else None
PUNCT_FILES = ("encoder.int8.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")
if punct_dir and punct_dir.exists():
    found = [name for name in PUNCT_FILES if (punct_dir / name).exists()]
    if len(found) == len(PUNCT_FILES):
        for name in PUNCT_FILES:
            datas.append((str(punct_dir / name), "models/gigaam-v3-punct"))
        print("[spec] bundling gigaam-v3-punct ONNX models")
    else:
        print(f"[spec] gigaam-v3-punct incomplete ({len(found)}/4 files) — run: nestor models")
else:
    print("[spec] no gigaam-v3-punct models — the app will download them on first run")

# ------------------------------------------------------------------ hiddenimports
hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "websockets",
]

# VC++ 2015-2022 runtime + Universal CRT: onnxruntime/sherpa-onnx DLLs depend
# on them, but the target machine may not have the redistributable installed
# (e.g. a work computer without admin rights). Bundled DLLs are found next to
# the native libraries, so they satisfy WinError 126 "the specified module
# could not be found".
CRT_DLLS = (
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
    "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll",
)
UCRT_DIRS = (
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Windows Kits" / "10" / "Redist" / "10.0.26100.0" / "ucrt" / "DLLs" / "x64",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Windows Kits" / "10" / "Redist" / "ucrt" / "DLLs" / "x64",
)


def collect_windows_runtime() -> None:
    """Copy the VC++ runtime and the UCRT into _internal (loaded on demand)."""
    target = ROOT / "dist" / "Nestor" / "_internal"
    if not target.exists():
        print("[spec] dist not built yet — runtime DLLs will be added post-build")
        return
    copied = 0
    for name in CRT_DLLS:
        source = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / name
        if source.exists():
            shutil.copyfile(source, target / name)
            copied += 1
    for ucrt_dir in UCRT_DIRS:
        if ucrt_dir.exists():
            for source in ucrt_dir.glob("*.dll"):
                dest = target / source.name
                if not dest.exists():
                    shutil.copyfile(source, dest)
                    copied += 1
            break
    print(f"[spec] bundled {copied} runtime DLLs (VC++ redist + UCRT) into _internal")


import shutil  # noqa: E402  (used by collect_windows_runtime)

# sherpa-onnx/onnxruntime ship native libraries; collect them completely.
# onnxruntime also carries developer utilities (onnxruntime.tools.*,
# onnxruntime.transformers.*, onnxruntime.quantization.*) that are never used
# at runtime but drag in torch/pyannote/lightning; skip those submodules.
ONNX_SKIP_PREFIXES = (
    "onnxruntime.tools",
    "onnxruntime.transformers",
    "onnxruntime.quantization",
)
for package in (
    "onnxruntime",
    "sherpa_onnx",
    "av",                 # PyAV — bundles libav* shared libraries
    "webview",            # pywebview for desktop window
):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        if package == "onnxruntime":
            pkg_hidden = [
                name for name in pkg_hidden
                if not name.startswith(ONNX_SKIP_PREFIXES)
            ]
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] could not collect {package}: {exc}")



# --------------------------------------------------------------------- excludes
excludes = [
    "IPython", "jupyter", "notebook", "pytest", "sphinx", "tkinter",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "cv2", "numba",
    "sounddevice", "pdb", "doctest",
    # not used at runtime; keep the build lean and free of the torch stack
    "torch", "torchaudio", "torchvision", "torchcodec", "torchmetrics",
    "torch_audiomentations", "torch_pitch_shift", "pytorch_metric_learning",
    "lightning", "pytorch_lightning", "pyannote", "pyannote.audio",
    "pandas", "matplotlib", "scipy", "sklearn", "sympy", "sqlalchemy",
    "tensorboard", "hydra", "omegaconf",
]

a = Analysis(
    [str(ROOT / "packaging" / "desktop_main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Nestor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # no black console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "packaging" / "nestor.ico") if (ROOT / "packaging" / "nestor.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Nestor",
)

# After COLLECT: drop the VC++/UCRT runtime next to the bundled libraries so
# machines without the redistributable (no admin rights) can still run the
# native onnxruntime/sherpa-onnx DLLs.
collect_windows_runtime()
