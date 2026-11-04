"""Command line interface for Nestor."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .config import (
    DEFAULT_FORMATS,
    Config,
)

SAMPLE_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nestor",
        description=(
            "Offline transcription and speaker diarization on CPU "
            "(GigaAM v3 + sherpa-onnx diarization)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ transcribe
    transcribe = subparsers.add_parser(
        "transcribe", help="transcribe and diarize one or more audio files"
    )
    transcribe.add_argument("inputs", nargs="+", help="audio file(s) to process")
    transcribe.add_argument(
        "-o",
        "--output-dir",
        default="out",
        help="directory for exported transcripts (default: out)",
    )
    transcribe.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help=f"comma separated list of formats: {', '.join(DEFAULT_FORMATS)}",
    )
    transcribe.add_argument(
        "--no-diarize",
        action="store_true",
        help="skip diarization and attribute everything to a single speaker",
    )
    transcribe.add_argument(
        "--diarization-backend",
        default="auto",
        choices=["auto", "local"],
        help=(
            "auto — локальная диаризация (sherpa-onnx, без токена); "
            "local — только sherpa-onnx"
        ),
    )
    transcribe.add_argument("--num-speakers", type=int, default=None)
    transcribe.add_argument("--min-speakers", type=int, default=None)
    transcribe.add_argument("--max-speakers", type=int, default=None)
    transcribe.add_argument(
        "--speaker-labels",
        default="",
        help='comma separated names, e.g. "Иван,Мария"',
    )
    transcribe.add_argument(
        "--threads", type=int, default=0, help="threads for sherpa-onnx (0 = auto)"
    )
    transcribe.add_argument(
        "--chunk-seconds",
        type=float,
        default=20.0,
        help="maximum ASR chunk length in seconds (default: 20)",
    )
    transcribe.add_argument(
        "--match-voiceprints",
        dest="match_voiceprints",
        action="store_true",
        default=None,
        help="substitute known speaker names from the voiceprint library",
    )
    transcribe.add_argument(
        "--no-match-voiceprints",
        dest="match_voiceprints",
        action="store_false",
        help="disable voiceprint matching for this run",
    )
    transcribe.set_defaults(match_voiceprints=None)
    transcribe.add_argument("--quiet", action="store_true")
    transcribe.set_defaults(func=cmd_transcribe)

    # ----------------------------------------------------------------- voiceprints
    vp = subparsers.add_parser(
        "voiceprints", help="manage the local speaker voice library"
    )
    vp_sub = vp.add_subparsers(dest="vp_command", required=True)

    vp_list = vp_sub.add_parser("list", help="show saved voices")
    vp_list.set_defaults(func=cmd_voiceprints_list)

    vp_add = vp_sub.add_parser("add", help="save a voice from a recording or file")
    vp_add.add_argument("name", help="display name, e.g. \"Иван Петров\"")
    vp_add.add_argument(
        "--from-recording",
        default=None,
        help="record id in the library (see: nestor voiceprints recordings)",
    )
    vp_add.add_argument(
        "--speaker",
        default="SPEAKER_00",
        help="speaker label inside the recording (default: SPEAKER_00)",
    )
    vp_add.add_argument(
        "--from-audio",
        default=None,
        help="single-speaker audio file to take the voice from",
    )
    vp_add.set_defaults(func=cmd_voiceprints_add)

    vp_remove = vp_sub.add_parser("remove", help="delete a saved voice")
    vp_remove.add_argument("name", help="display name of the voice")
    vp_remove.set_defaults(func=cmd_voiceprints_remove)

    vp_listings = vp_sub.add_parser("recordings", help="list known record ids")
    vp_listings.set_defaults(func=cmd_voiceprints_recordings)
    vp.set_defaults(func=None)

    # -------------------------------------------------------------- download-sample
    sample = subparsers.add_parser(
        "download-sample", help="download a short Russian sample audio file"
    )
    sample.add_argument("-o", "--output-dir", default="samples")
    sample.set_defaults(func=cmd_download_sample)

    # ----------------------------------------------------------------------- serve
    serve = subparsers.add_parser(
        "serve", help="run the local web interface (FastAPI + uvicorn)"
    )
    serve.add_argument(
        "--port", type=int, default=8000,
        help="local port (the server always listens on 127.0.0.1 only)",
    )
    serve.add_argument(
        "--work-dir",
        default=None,
        help="where recordings are stored (default: %%LOCALAPPDATA%%\\Nestor)",
    )
    serve.add_argument(
        "--workers",
        type=int,
        default=1,
        help="how many transcription jobs to run in parallel (1 = safest on CPU)",
    )
    serve.add_argument("--reload", action="store_true", help="auto-reload (dev)")
    serve.set_defaults(func=cmd_serve)

    # -------------------------------------------------------------------- desktop
    desktop = subparsers.add_parser(
        "desktop", help="open the application in a native Windows window"
    )
    desktop.add_argument(
        "--work-dir",
        default=None,
        help="where recordings are stored (default: %%LOCALAPPDATA%%\\Nestor)",
    )
    desktop.add_argument("--width", type=int, default=1320)
    desktop.add_argument("--height", type=int, default=860)
    desktop.add_argument("--debug", action="store_true")
    desktop.set_defaults(func=cmd_desktop)

    # ------------------------------------------------------------------- models
    models = subparsers.add_parser(
        "models",
        help="download the local diarization models (no token needed)",
    )
    models.add_argument(
        "--force", action="store_true", help="download again even if present"
    )
    models.set_defaults(func=cmd_models)

    # ------------------------------------------------------------------- build
    build = subparsers.add_parser(
        "build-exe", help="package the application into a Windows .exe (PyInstaller)"
    )
    build.add_argument(
        "--dist",
        default="dist",
        help="output directory (default: dist)",
    )
    build.add_argument(
        "--clean",
        action="store_true",
        help="remove previous build artefacts first",
    )
    build.set_defaults(func=cmd_build)

    return parser


def cmd_transcribe(args: argparse.Namespace) -> int:
    from . import cpuinfo, speaker_embed
    from .exporters import export_all
    from .pipeline import process_file

    labels = [name.strip() for name in args.speaker_labels.split(",") if name.strip()]
    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]

    config = Config(
        threads=args.threads,
        chunk_seconds=args.chunk_seconds,
        diarize=not args.no_diarize,
        diarization_backend=args.diarization_backend,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        output_dir=Path(args.output_dir),
        formats=formats,
        speaker_labels=labels,
        quiet=args.quiet,
    )
    if args.match_voiceprints is not None:
        config.match_voiceprints = args.match_voiceprints

    def report(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    speaker_embed.configure(config.threads)
    for input_path in args.inputs:
        report(f"=== {input_path} ===")
        with cpuinfo.tuned(config.threads):
            transcript = process_file(config, input_path, progress=report)
        matches = transcript.speaker_match or {}
        if matches:
            # keep manual --speaker-labels; fill the rest with matched names
            order = _speaker_order(transcript)
            for i, speaker in enumerate(order):
                name = matches.get(speaker, {}).get("assigned_name")
                if name and i >= len(labels):
                    labels.append(name)
            config.speaker_labels = labels
            line = ", ".join(
                f"{label} -> {result['assigned_name']} ({result['confidence']:.2f})"
                if result.get("assigned_name")
                else f"{label} -> (не узнан)"
                for label, result in sorted(matches.items())
            )
            report(f"  [voiceprints] {line}")
        written = export_all(transcript, config)
        report(
            f"  done: {len(transcript.segments)} segments, "
            f"{len(transcript.speakers)} speakers, "
            f"RTF={transcript.rtf} ({transcript.processing_seconds}s)"
        )
        for path in written:
            report(f"  wrote {path}")
    return 0


def _speaker_order(transcript) -> List[str]:
    """Speaker labels in order of first appearance."""
    seen: List[str] = []
    for segment in transcript.segments:
        if segment.speaker not in seen:
            seen.append(segment.speaker)
    return seen


def _vp_library():
    from .voiceprints import VoiceprintLibrary

    return VoiceprintLibrary()


def cmd_voiceprints_list(args: argparse.Namespace) -> int:
    library = _vp_library()
    entries = library.list()
    if not entries:
        print("Библиотека голосов пуста")
        return 0
    print(f"Голосов в библиотеке: {len(entries)}")
    for meta in entries:
        minutes = meta.total_duration_sec / 60.0
        print(
            f"  {meta.display_name:<24} {meta.num_utterances} запис(ей), "
            f"{minutes:.1f} мин, обновлён {meta.updated_at or meta.created_at}"
        )
    return 0


def cmd_voiceprints_add(args: argparse.Namespace) -> int:
    from .audio import decode as decode_audio, duration as audio_duration
    from .constants import SAMPLE_RATE
    from .speaker_embed import cluster_vectors, embed_audio

    library = _vp_library()
    if args.from_recording:
        from .library import Library

        lib = Library()
        recording = lib.load(args.from_recording)
        if recording is None:
            print(f"error: запись {args.from_recording} не найдена", file=sys.stderr)
            return 1
        speaker = args.speaker
        if speaker not in recording.speakers:
            print(
                f"error: в записи нет спикера {speaker} "
                f"(есть: {', '.join(recording.speakers) or 'нет'})",
                file=sys.stderr,
            )
            return 1
        source = lib.source_path(recording.id, recording.filename)
        if not source.exists():
            print(f"error: файл записи не найден: {source}", file=sys.stderr)
            return 1
        print(f"Считаю эмбеддинги записи {recording.filename} ...")
        waveform = decode_audio(source)
        clusters, per_segment = cluster_vectors(waveform, SAMPLE_RATE, recording.segments)
        vector = clusters.get(speaker)
        if vector is None:
            print(f"error: слишком мало речи у спикера {speaker}", file=sys.stderr)
            return 1
        duration = sum(
            max(0.0, seg.end - seg.start)
            for seg in recording.segments
            if seg.speaker == speaker
        )
        meta = library.add_or_update(
            args.name, vector, source_recording_id=recording.id, duration_sec=duration
        )
        print(f"Голос сохранён: {meta.display_name} ({meta.slug}.npy)")
        return 0

    if args.from_audio:
        path = Path(args.from_audio)
        if not path.exists():
            print(f"error: файл не найден: {path}", file=sys.stderr)
            return 1
        print(f"Считаю эмбеддинг файла {path.name} ...")
        waveform = decode_audio(path)
        vector = embed_audio(waveform, SAMPLE_RATE)
        meta = library.add_or_update(
            args.name, vector, duration_sec=audio_duration(waveform)
        )
        print(f"Голос сохранён: {meta.display_name} ({meta.slug}.npy)")
        return 0

    print("error: укажите --from-recording <id> --speaker <SPEAKER_NN> "
          "или --from-audio <file>", file=sys.stderr)
    return 1


def cmd_voiceprints_remove(args: argparse.Namespace) -> int:
    library = _vp_library()
    if not library.delete(args.name):
        print(f"error: голос «{args.name}» не найден", file=sys.stderr)
        return 1
    print(f"Голос удалён: {args.name}")
    return 0


def cmd_voiceprints_recordings(args: argparse.Namespace) -> int:
    from .library import Library

    lib = Library()
    recordings = lib.list()
    if not recordings:
        print("Библиотека записей пуста")
        return 0
    print(f"Записей: {len(recordings)}")
    for recording in recordings:
        speakers = ", ".join(recording.speakers) or "нет спикеров"
        print(f"  {recording.id}  {recording.filename}  [{speakers}]")
    return 0


def cmd_download_sample(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "example.wav"
    if target.exists():
        print(f"{target} already exists")
        return 0
    print(f"Downloading {SAMPLE_URL} ...")
    urllib.request.urlretrieve(SAMPLE_URL, target)
    print(f"Saved {target}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "Web dependencies are missing. Install them with:\n  pip install -e .",
            file=sys.stderr,
        )
        return 1

    from .web.app import run_server

    print("The server always listens on 127.0.0.1 and requires a session token.")
    print("Press Ctrl+C to stop.")
    run_server(
        port=args.port,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        reload=args.reload,
    )
    return 0


def cmd_desktop(args: argparse.Namespace) -> int:
    try:
        import webview  # noqa: F401
    except ImportError:
        print(
            "pywebview is not installed. Install it with:\n  pip install pywebview",
            file=sys.stderr,
        )
        return 1

    from .desktop import run

    return run(
        work_dir=Path(args.work_dir) if args.work_dir else None,
        width=args.width,
        height=args.height,
        debug=args.debug,
    )


def cmd_models(args: argparse.Namespace) -> int:
    """Fetch the models of the token-free diarization stack."""
    from .models import (
        ensure_gigaam_punct,
        ensure_local_diarization,
        model_root,
    )

    root = model_root()
    print(f"Model directory: {root}")
    try:
        paths = ensure_local_diarization(root, force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        print(f"  {name}: {path}")
    try:
        punct = ensure_gigaam_punct(root, force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"gigaam-v3-punct download failed: {exc}", file=sys.stderr)
        return 1
    for name, path in punct.items():
        print(f"  {name}: {path}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Package the desktop application with PyInstaller."""
    import shutil
    import subprocess

    root = Path(__file__).resolve().parent.parent
    spec = root / "nestor.spec"
    if not spec.exists():
        print(f"Build script not found: {spec}", file=sys.stderr)
        return 1

    if args.clean:
        for folder in ("build", args.dist):
            shutil.rmtree(root / folder, ignore_errors=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec),
        "--noconfirm",
        "--distpath",
        str(root / args.dist),
        "--workpath",
        str(root / "build"),
    ]
    print("Building the executable ...")
    result = subprocess.run(command, cwd=str(root))
    if result.returncode != 0:
        return result.returncode

    exe = root / args.dist / "Nestor" / "Nestor.exe"
    print()
    print(f"Ready: {exe}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
