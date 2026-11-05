"""FastAPI application: the backend of the Nestor desktop app.

The same API powers the browser UI (``nestor serve``) and the native window
(``nestor desktop``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, wait as futures_wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .. import align, cpuinfo, speaker_embed
from ..audio import decode, duration, write_wav
from ..constants import SAMPLE_RATE
from ..config import DEFAULT_ASR_MODEL, Config
from ..exporters import transcript_from_recording
from ..glossary import (
    Glossary,
    build_alias_map,
    dump_glossary,
    ensure_glossary,
    load_glossary,
    merge_glossary,
    remove_term,
    resolve_glossary_path,
    save_glossary,
    upsert_term,
)
from ..library import Library, Recording, Segment
from ..pipeline import JobCancelled, process_file
from ..settings import SettingsStore
from ..voiceprints import VoiceprintLibrary

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wma",
    ".aiff", ".aif", ".webm", ".mp4", ".mkv", ".mov", ".avi", ".3gp",
}

# Content types for the built-in player. Every allowed upload suffix must be
# listed here: the browser refuses to decode a media element served as
# application/octet-stream ("MEDIA_ERR_SRC_NOT_SUPPORTED"), so a missing entry
# leaves the recording unplayable even though it was transcribed fine.
MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".wma": "audio/x-ms-wma",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".webm": "audio/webm",
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".3gp": "video/3gpp",
}
EXPORT_FORMATS = ("json", "srt", "vtt", "txt", "md", "docx")


def generate_session_token() -> str:
    """A random per-run token guarding every HTTP request to this app."""
    return secrets.token_urlsafe(32)


def _forbidden_page(reason: str) -> Response:
    """Human-readable 403 page for browsers that stumble on the port.

    ``reason`` is either ``host`` (foreign Host header - DNS rebinding or a
    misconfigured proxy) or ``token`` (missing/wrong session token). Both
    pages are self-contained (inline CSS, no external assets) and say what
    to do in plain Russian.
    """
    if reason == "host":
        title = "Nestor доступен только с этого компьютера"
        hint = (
            "Этот адрес не совпадает с локальным (127.0.0.1). "
            "Nestor — локальное приложение и не отвечает чужим сайтам."
        )
    else:
        title = "Нужен ключ доступа"
        hint = (
            "Nestor защищён одноразовым ключом, чтобы посторонние страницы "
            "не могли обращаться к программе. Откройте Nestor через его окно "
            "(ярлык «Nestor») — откроется правильный адрес автоматически."
        )
    html = (
        "<!DOCTYPE html><html lang=\"ru\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title}</title><style>"
        "body{margin:0;min-height:100vh;display:flex;align-items:center;"
        "justify-content:center;background:#f4f2ee;color:#2d2a26;"
        "font:15px/1.55 system-ui,'Segoe UI',sans-serif}"
        ".card{max-width:460px;padding:28px 32px;background:#fff;"
        "border:1px solid #e2ded6;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.06)}"
        "h1{font-size:18px;margin:0 0 10px}p{margin:0 0 14px}"
        ".logo{font-weight:700;letter-spacing:.5px;color:#8a6d3b;margin-bottom:14px}"
        "code{background:#f4f2ee;border-radius:4px;padding:1px 5px;font-size:13px}"
        "</style></head><body><div class=\"card\">"
        "<div class=\"logo\">Nestor</div><h1>" + title + "</h1>"
        "<p>" + hint + "</p>"
        "<p style=\"color:#8a857c;font-size:13px\">Ошибка 403 · доступ запрещён</p>"
        "</div></body></html>"
    )
    return HTMLResponse(status_code=403, content=html)


def install_access_guard(app: FastAPI, token: str) -> None:
    """Reject requests that did not come from our own window.

    The server only ever listens on 127.0.0.1, so remote machines cannot
    connect in principle. Two remaining theoretical vectors are a malicious
    web page in the user's browser (blind CSRF POSTs / DNS rebinding) and
    other local software. Both are cut off by:

    1. a per-run random session token (unknown to any third-party page);
    2. a strict ``Host`` check (DNS rebinding resolves a foreign hostname to
       127.0.0.1, but its ``Host`` header betrays it).

    Static assets and the HTML shell also require the token (they carry no
    secrets, but gating everything keeps the rule simple); the token rides
    the URL query of the initial page load, then an HttpOnly SameSite cookie
    (set by the guard itself) and/or an ``X-Nestor-Token`` header.
    """
    @app.middleware("http")
    async def access_guard(request: Request, call_next):
        host_header = (request.headers.get("host") or "").lower()
        hostname = host_header.rsplit(":", 1)[0] if host_header else ""
        # "testserver" is what httpx's TestClient sends in tests.
        if hostname not in ("127.0.0.1", "localhost", "testserver"):
            return _forbidden_page("host")
        provided = (
            request.headers.get("x-nestor-token")
            or request.query_params.get("token")
            or request.cookies.get("nestor_token")
            or ""
        )
        if not secrets.compare_digest(provided, token):
            return _forbidden_page("token")
        response = await call_next(request)
        if not request.cookies.get("nestor_token"):
            response.set_cookie(
                "nestor_token",
                token,
                samesite="lax",
                httponly=True,
                max_age=86400,
            )
        return response


# --------------------------------------------------------------------- helpers
def compute_peaks(waveform: np.ndarray, buckets: int = 1200) -> List[float]:
    """Downsample a waveform into normalised peak values for the player."""
    if waveform.size == 0:
        return []
    buckets = max(1, min(buckets, waveform.size))
    edges = np.linspace(0, waveform.size, buckets + 1).astype(np.int64)
    # One vectorised pass: max(|waveform|) per bucket. buckets <= waveform.size
    # keeps every segment non-empty, so reduceat is safe here.
    peaks = np.maximum.reduceat(np.abs(waveform), edges[:-1])
    maximum = peaks.max() if peaks.size else 1.0
    if maximum > 0:
        peaks = peaks / maximum
    return [round(float(value), 4) for value in peaks]


def parse_range(header: str, size: int) -> Optional[tuple]:
    """Parse a ``Range: bytes=start-end`` header."""
    match = re.match(r"bytes=(\d*)-(\d*)", header.strip())
    if not match:
        return None
    start_text, end_text = match.groups()
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    else:
        if not end_text:
            return None
        length = int(end_text)
        start = max(0, size - length)
        end = size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        return None
    return start, end


def ranged_response(path: Path, request: Request, media_type: str) -> Response:
    """Serve a file, honouring HTTP range requests (needed by the audio player)."""
    size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
        )

    parsed = parse_range(range_header, size)
    if parsed is None:
        return Response(
            status_code=416, headers={"Content-Range": f"bytes */{size}"}
        )

    start, end = parsed
    length = end - start + 1

    def iterator():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iterator(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


def build_export(library: Library, record_id: str, fmt: str) -> Path:
    """Create an export file and return its path."""
    recording = library.load(record_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")

    fmt = fmt.lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail="Неизвестный формат")

    target_dir = library.output_dir / record_id
    if fmt == "json":
        return library.record_path(record_id)

    transcript = transcript_from_recording(recording)
    if fmt == "docx":
        from ..exporters import docx_export

        return docx_export.export(transcript, target_dir / "transcript.docx")
    if fmt == "md":
        from ..exporters import md_export

        return md_export.export(transcript, target_dir / "transcript.md")
    if fmt == "srt":
        from ..exporters import srt_export

        return srt_export.export(transcript, target_dir / "transcript.srt")
    if fmt == "vtt":
        from ..exporters import vtt_export

        return vtt_export.export(transcript, target_dir / "transcript.vtt")

    from ..exporters import txt_export

    return txt_export.export(transcript, target_dir / "transcript.txt")


# ------------------------------------------------------------------- processor
class Processor:
    """Runs transcription jobs one by one in a background worker."""

    def __init__(self, library: Library, max_workers: int = 1) -> None:
        self.library = library
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="nestor"
        )
        # record id -> stop flag / future of the job owned by this process
        self._events: Dict[str, threading.Event] = {}
        self._futures: Dict[str, Future] = {}
        # record id -> wall-clock start (time.time()); live "elapsed" in /api
        self._started: Dict[str, float] = {}

    def submit(self, recording: Recording, config: Config, mode: str = "full") -> None:
        event = threading.Event()
        with self._lock:
            self._events[recording.id] = event
        future = self._executor.submit(self._run, recording, config, event, mode)
        with self._lock:
            if recording.id in self._events:
                self._futures[recording.id] = future
            else:
                # finished before its future was registered — nothing to track
                self._futures.pop(recording.id, None)

    def cancel(self, record_id: str) -> str:
        """Stop a queued or running job.

        Returns ``"queued"`` when the job will never start (the caller then
        persists the ``cancelled`` status itself), ``"running"`` when the worker
        stops at its next checkpoint, or ``"none"`` when this process owns no
        such job (e.g. a leftover from a previous run).
        """
        with self._lock:
            event = self._events.get(record_id)
            future = self._futures.get(record_id)
        if event is None and future is None:
            return "none"
        if future is not None and future.cancel():
            # still waiting in the queue: _run will never be called
            with self._lock:
                self._futures.pop(record_id, None)
                self._events.pop(record_id, None)
            return "queued"
        if event is not None:
            event.set()
            return "running"
        return "none"


    def wait(self, record_id: str, timeout: float = 20.0) -> bool:
        """Block until the job has left the worker; true when it is safe."""
        with self._lock:
            future = self._futures.get(record_id)
        if future is None:
            return True
        done, _ = futures_wait({future}, timeout=timeout)
        return future in done

    def _mark_cancelled(self, recording: Recording) -> None:
        recording.status = "cancelled"
        recording.stage = "cancelled"
        self.library.save(recording)

    def _maybe_notify(self, recording: Recording, text: str = "Расшифровка завершена") -> None:
        """Windows toast after a successful run (respects the notify_done setting)."""
        try:
            enabled = bool(
                SettingsStore(self.library.root).as_public().get("notify_done", True)
            )
        except Exception:  # noqa: BLE001 - a broken settings file must not fail the job
            enabled = True
        if enabled:
            _notify_done("Nestor", f"{text}: {recording.filename}")

    def _run(self, recording: Recording, config: Config, event: threading.Event, mode: str = "full") -> None:
        with self._lock:
            self._started[recording.id] = time.time()
        priority, pin = _runtime_tuning(self.library)
        pinning = cpuinfo.describe_pinning(config.threads, pin)
        # Everything a benchmark needs, captured at start and kept with the
        # recording: the UI shows it in the expanded card, so nobody has to dig
        # through log.txt / timings.json to compare two runs.
        recording.run_meta = {
            "app_version": __version__,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "threads_setting": config.threads,
            "threads_used": cpuinfo.resolve_threads(config.threads),
            "cpu_physical_cores": cpuinfo.physical_cores(),
            "cpu_logical_processors": os.cpu_count(),
            "cpu_hybrid": cpuinfo.is_hybrid(),
            # only meaningful on a hybrid CPU - a flat "P-core count" on a
            # normal processor would read as "this machine has N fast cores"
            "cpu_perf_cores": cpuinfo.performance_logical_count() if cpuinfo.is_hybrid() else None,
            "process_priority": priority,
            "pin_performance": pin,
            "pin_applied": pinning["applied"],
            "pin_message": pinning["message"],
            "model": config.model_name,
            "settings": _settings_snapshot(config),
        }
        # one line per run keeps the file log useful as well
        try:
            self.library.log(
                f"tuning {recording.filename}: "
                f"threads={cpuinfo.resolve_threads(config.threads)} "
                f"physical={cpuinfo.physical_cores()} "
                f"hybrid={cpuinfo.is_hybrid()} "
                f"priority={priority} pin={pin}"
            )
        except OSError:
            pass
        try:
            # keep the embedding extractor on the same thread budget as ASR
            speaker_embed.configure(config.threads)
            with cpuinfo.tuned(
                config.threads, priority=priority, pin_performance=pin
            ):
                if mode == "speakers":
                    self._run_speakers(recording, config, event)
                else:
                    self._run_job(recording, config, event)
        finally:
            with self._lock:
                self._events.pop(recording.id, None)
                self._futures.pop(recording.id, None)
                self._started.pop(recording.id, None)

    def elapsed_sec(self, record_id: str) -> Optional[float]:
        """Seconds since the job started, for a live "работает: ..." line."""
        with self._lock:
            started = self._started.get(record_id)
        if started is None:
            return None
        return max(0.0, time.time() - started)

    def _run_job(
        self, recording: Recording, config: Config, event: threading.Event
    ) -> None:
        if event.is_set():
            # stopped while it was still waiting in the queue
            self._mark_cancelled(recording)
            self.library.log(f"stopped {recording.filename} (before start)")
            return

        recording.status = "running"
        recording.stage = "starting"
        try:
            self.library.save(recording)
            self.library.log(f"start {recording.filename} (model {config.model_name})")
        except OSError as exc:
            # without this the worker thread would die silently and the
            # record would stay "queued" forever
            recording.status = "error"
            recording.stage = "error"
            recording.error = f"Не удалось сохранить статус записи: {exc}"
            with contextlib.suppress(OSError):
                self.library.save(recording)
            return

        # Throttle disk writes during the run: persist a progress tick at most
        # a few times a second (or when the stage changes / the job completes).
        # The final state is always written after the run, so nothing is lost.
        last_save = time.monotonic()
        last_stage = "starting"

        # per-stage durations for output/<id>/timings.json (wall clock)
        timings_started = time.time()
        timings_stages: Dict[str, float] = {}
        timings_last = [last_save, last_stage]

        def timings_tick(stage: str) -> None:
            # Attribute the interval that just elapsed to the stage it was
            # spent in. Ticks arrive several times a second while decoding and
            # once per ASR chunk, so accumulating only on a *change* of stage
            # (the old behaviour) kept just the last fragment of each stage -
            # a 17-minute transcription was recorded as 1.8 seconds.
            now = time.time()
            prev_stage = timings_last[1]
            if prev_stage != "starting":
                timings_stages[prev_stage] = timings_stages.get(prev_stage, 0.0) + (
                    now - timings_last[0]
                )
            timings_last[0] = now
            timings_last[1] = stage

        def write_timings(final_stage: str, status: str) -> None:
            """Persist the per-stage breakdown; best effort, never fatal."""
            timings_tick(final_stage)
            try:
                payload = {
                    "schema_version": 1,
                    "started_at": datetime.fromtimestamp(
                        timings_started, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    "finished_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "total_sec": round(time.time() - timings_started, 2),
                    "status": status,
                    "stages": {
                        k: round(v, 2) for k, v in sorted(timings_stages.items())
                    },
                }
                path = self.library.output_dir / recording.id / "timings.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                # also keep the breakdown in record.json so the UI can show it
                recording.stage_seconds = dict(payload["stages"])
                # the diagnostics of this run travel with the timings file too
                if recording.run_meta:
                    payload["run_meta"] = recording.run_meta
                    path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                with contextlib.suppress(OSError):
                    self.library.save(recording)
            except OSError:
                pass

        def on_stage(stage: str, fraction: float) -> None:
            nonlocal last_save, last_stage
            timings_tick(stage)
            recording.stage = stage
            fraction = max(0.0, min(1.0, fraction))
            recording.progress = fraction
            now = time.monotonic()
            if (
                stage != last_stage
                or fraction >= 1.0
                or now - last_save >= 0.4
            ):
                self.library.save(recording)
                last_save = now
                last_stage = stage

        waveform = None
        try:
            # decoding a long recording takes a while: show the real stage
            # with a live fraction (same band the pipeline uses for decode)
            recording.stage = "decode"
            recording.progress = 0.02
            self.library.save(recording)

            def on_decode(fraction: float) -> None:
                on_stage("decode", 0.02 + 0.04 * max(0.0, min(1.0, fraction)))

            waveform, from_cache = load_or_decode(self.library, recording, on_decode)
            if from_cache:
                on_stage("decode", 0.06)  # cache hit: decode band done instantly
            transcript = process_file(
                config,
                self.library.recording_source(recording),
                on_stage=on_stage,
                waveform=waveform,
                cancel=event.is_set,
            )

            recording.duration = transcript.duration
            recording.model = transcript.model
            recording.diarization_model = transcript.diarization_model
            recording.rtf = transcript.rtf
            recording.processing_seconds = transcript.processing_seconds
            recording.segments = [
                Segment(
                    id=uuid.uuid4().hex[:8],
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    text=segment.text,
                )
                for segment in transcript.segments
            ]

            # voiceprints: store provenance and apply recognised names only
            # where the user has not renamed the speaker manually
            recording.speaker_match = dict(transcript.speaker_match or {})
            for label, result in (transcript.speaker_match or {}).items():
                name = result.get("assigned_name")
                if not name:
                    continue
                current = recording.speaker_names.get(label, "")
                if not current:  # keep manual renames intact
                    recording.speaker_names[label] = str(name)

            if event.is_set():
                # stopped mid-transcription: keep what was recognised so far
                self.library.save_peaks(recording.id, compute_peaks(waveform))
                self._mark_cancelled(recording)
                write_timings("cancelled", "cancelled")
                self.library.log(
                    f"stopped {recording.filename}: "
                    f"kept {len(recording.segments)} segments"
                )
                return

            recording.status = "done"
            recording.stage = "done"
            recording.progress = 1.0
            recording.error = None

            self.library.save_peaks(recording.id, compute_peaks(waveform))
            # Persist the per-stage breakdown *before* the final status save:
            # whoever observes "done" on disk must already see stage_seconds in
            # the record. write_timings() sets stage_seconds and saves the
            # record itself; the save below then re-persists the same state.
            write_timings("done", "done")
            self.library.save(recording)
            self.library.log(
                f"done {recording.filename}: {len(recording.segments)} segments, "
                f"rtf={recording.rtf}"
            )
            self._maybe_notify(recording)
        except JobCancelled:
            if waveform is not None:
                self.library.save_peaks(recording.id, compute_peaks(waveform))
            self._mark_cancelled(recording)
            write_timings("cancelled", "cancelled")
            self.library.log(f"stopped {recording.filename}: no transcript kept")
        except Exception as exc:  # noqa: BLE001 - reported in the UI
            recording.status = "error"
            recording.stage = "error"
            recording.error = str(exc)
            self.library.save(recording)
            write_timings("error", "error")
            self.library.log(f"error {recording.filename}: {exc}")
        finally:
            recording.progress = 1.0 if recording.status == "done" else recording.progress

    def _run_speakers(
        self, recording: Recording, config: Config, event: threading.Event
    ) -> None:
        """Reassign speakers from a fresh diarization pass - no re-recognition.

        The text, manual edits and bookmarks stay untouched; every segment is
        attributed to the speaker whose turn overlaps it the most (the same
        rule the original pipeline used for words). Voiceprint matching is
        re-run on the new clusters.
        """
        recording.status = "running"
        recording.stage = "starting"
        # snapshot for the diff shown after the run («12 → 9 · перенесено 143 реплики»)
        before_counts = Counter(s.speaker for s in recording.segments if s.speaker)
        self._moved_segments = 0
        try:
            self.library.save(recording)
            self.library.log(f"start speakers-only {recording.filename}")
        except OSError as exc:
            recording.status = "error"
            recording.stage = "error"
            recording.error = f"Не удалось сохранить статус записи: {exc}"
            with contextlib.suppress(OSError):
                self.library.save(recording)
            return

        def on_progress(fraction: float) -> bool:
            # sherpa-onnx reports the fraction of processed chunks (0..1).
            # The truthy return value aborts the diarization run (see
            # LocalDiarizer.diarize) when the job was asked to stop.
            nonlocal last_save
            recording.stage = "diarization"
            recording.progress = 0.05 + max(0.0, min(1.0, fraction)) * 0.90
            now = time.monotonic()
            if now - last_save >= 0.4:
                with contextlib.suppress(OSError):
                    self.library.save(recording)
                last_save = now
            return event.is_set()

        last_save = time.monotonic()
        try:
            recording.stage = "decode"
            recording.progress = 0.02
            self.library.save(recording)

            def on_decode(fraction: float) -> None:
                nonlocal last_save
                recording.stage = "decode"
                recording.progress = 0.02 + 0.03 * max(0.0, min(1.0, fraction))
                now = time.monotonic()
                if now - last_save >= 0.4:
                    with contextlib.suppress(OSError):
                        self.library.save(recording)
                    last_save = now

            waveform, from_cache = load_or_decode(self.library, recording, on_decode)
            if from_cache:
                on_decode(1.0)  # cache hit: decode band done instantly
            from ..pipeline import _accepts_progress, _diarizer_for

            diarizer, diarizer_lock, _loaded = _diarizer_for(config, lambda _m: None)
            turns = []
            recording.stage = "diarization"
            self.library.save(recording)
            with diarizer_lock:
                if event.is_set():
                    raise JobCancelled("stopped before diarization")
                if _accepts_progress(diarizer):
                    turns = diarizer.diarize(
                        waveform, SAMPLE_RATE, progress=on_progress
                    )
                else:
                    turns = diarizer.diarize(waveform, SAMPLE_RATE)
            if event.is_set():
                # the diarizer aborted mid-run (empty/partial turns) — discard
                raise JobCancelled("stopped while diarizing")

            new_labels = {}
            for seg in recording.segments:
                if event.is_set():
                    raise JobCancelled("stopped during speaker assignment")
                best_speaker, best_overlap = None, 0.0
                for turn in turns:
                    overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
                    if overlap > best_overlap:
                        best_overlap, best_speaker = overlap, turn.speaker
                if best_speaker is None:
                    # no overlapping turn (short/missed speech): nearest turn
                    best_distance = float("inf")
                    for turn in turns:
                        distance = (
                            turn.start - seg.end if seg.end <= turn.start
                            else seg.start - turn.end if seg.start >= turn.end
                            else 0.0
                        )
                        if distance < best_distance:
                            best_distance, best_speaker = distance, turn.speaker
                new_labels[id(seg)] = best_speaker or "SPEAKER_00"

            # preserve per-speaker manual names: map old label -> new label by
            # best time overlap of that speaker's segments
            def majority(new_label: str) -> str:
                votes: Dict[str, float] = {}
                for seg in recording.segments:
                    if new_labels[id(seg)] != new_label:
                        continue
                    votes[seg.speaker] = votes.get(seg.speaker, 0.0) + max(
                        0.0, seg.end - seg.start
                    )
                return max(votes, key=votes.get) if votes else new_label

            rename_map = {}
            for turn in dict.fromkeys(t.speaker for t in turns):
                old = majority(turn)
                if old:
                    rename_map[turn] = old

            for seg in recording.segments:
                new_label = new_labels[id(seg)]
                if new_label != seg.speaker:
                    self._moved_segments += 1
                seg.speaker = new_label
            # keep manual names aligned with the relabelled clusters
            names = {}
            for new_label, old_label in rename_map.items():
                existing = recording.speaker_names.get(old_label, "")
                if existing:
                    names[new_label] = existing
            manual = {
                label: value for label, value in recording.speaker_names.items()
                if label not in rename_map.values()
            }
            recording.speaker_names = {**names, **{
                label: value for label, value in manual.items()
                if label in {s.speaker for s in recording.segments}
            }}

            # voiceprints: re-match the fresh clusters against the library
            if event.is_set():
                raise JobCancelled("stopped before voiceprint matching")
            recording.speaker_match = {}
            try:
                from .. import speaker_embed, speaker_match as sm
                from ..voiceprints import VoiceprintLibrary

                clusters, per_segment = speaker_embed.cluster_vectors(
                    waveform, SAMPLE_RATE, recording.segments,
                    min_segment_sec=config.vp_min_segment_sec,
                )
                if config.merge_speakers and len(clusters) > 1:
                    # fold duplicate labels the same way the full pipeline does,
                    # so a re-run does not reintroduce the dozens of SPEAKER_NN
                    from .. import speaker_merge
                    clusters, per_segment = speaker_merge.apply_to_segments(
                        recording.segments,
                        clusters,
                        per_segment,
                        threshold=config.merge_threshold,
                        min_weight=config.merge_min_weight,
                        segment_floor=config.merge_segment_floor,
                        report=lambda line: self.library.log(line.strip()),
                    )
                    # the merged labels must also update the manual names
                    recording.speaker_names = {
                        label: value
                        for label, value in recording.speaker_names.items()
                        if label in {s.speaker for s in recording.segments}
                    }
                # tidy the raw cluster ids (0, 71, 296...) into sequential
                # SPEAKER_00.. by first appearance; keep manual names aligned
                label_map = align.renumber_speakers(recording.segments)
                if label_map:
                    recording.speaker_names = {
                        label_map.get(label, label): value
                        for label, value in recording.speaker_names.items()
                    }
                speaker_embed.save_record_embeddings(
                    config.embedding_dir, recording.id, clusters, per_segment
                )
                if config.match_voiceprints and clusters:
                    library = VoiceprintLibrary(config.vp_library_root)
                    matches = sm.match_clusters(
                        clusters, library,
                        thresholds=sm.MatchThresholds(
                            abs_threshold=config.vp_abs_threshold,
                            margin=config.vp_margin,
                            segment_agreement=config.vp_segment_agreement,
                        ),
                        segment_embeddings=per_segment,
                    )
                    recording.speaker_match = {
                        label: result.to_dict() for label, result in matches.items()
                    }
                    for label, result in recording.speaker_match.items():
                        name = result.get("assigned_name")
                        if name and not recording.speaker_names.get(label):
                            recording.speaker_names[label] = str(name)
            except JobCancelled:
                raise  # a stop request must not be swallowed by the vp guard
            except Exception as exc:  # noqa: BLE001 - never fail the job here
                self.library.log(f"voiceprints skipped (speakers-only): {exc}")

            if event.is_set():
                raise JobCancelled("stopped after voiceprint matching")
            recording.status = "done"
            recording.stage = "done"
            recording.progress = 1.0
            recording.error = None
            recording.model = config.model_name
            recording.speakers_diff = (
                f"{len(before_counts)} → {len(recording.speakers)}"
                f" · перенесено {self._moved_segments} "
                + _plural_ru(self._moved_segments, "реплика", "реплики", "реплик")
            )
            self.library.save(recording)
            self.library.log(
                f"done speakers-only {recording.filename}: "
                f"{len(recording.segments)} segments, {len(recording.speakers)} speakers"
            )
            self._maybe_notify(recording, "Готово: разбивка по спикерам")
        except JobCancelled:
            self._mark_cancelled(recording)
            self.library.log(f"stopped {recording.filename} (speakers-only)")
        except Exception as exc:  # noqa: BLE001 - reported in the UI
            recording.status = "error"
            recording.stage = "error"
            recording.error = str(exc)
            self.library.save(recording)
            self.library.log(f"error {recording.filename} (speakers-only): {exc}")


# ----------------------------------------------------------------- application
class ProcessOptions(BaseModel):
    """Overrides for a (re)processing job; anything omitted keeps the stored
    value (``Recording.job_settings``) or the built-in default."""

    backend: Optional[str] = None
    num_speakers: Optional[int] = None
    match_voiceprints: Optional[bool] = None
    vp_abs_threshold: Optional[float] = None
    vp_margin: Optional[float] = None
    vp_segment_agreement: Optional[float] = None
    merge_speakers: Optional[bool] = None
    merge_threshold: Optional[float] = None
    merge_min_weight: Optional[float] = None
    merge_segment_floor: Optional[float] = None
    audio_enhance: Optional[bool] = None
    diar_threshold: Optional[float] = None
    diar_min_duration_on: Optional[float] = None
    diar_min_duration_off: Optional[float] = None
    rescue_holes: Optional[bool] = None
    glossary: Optional[bool] = None


class ProcessBody(BaseModel):
    """Body of POST /api/recordings/{id}/process."""

    mode: str = "full"
    options: Optional[ProcessOptions] = None


def _voiceprint_duration(recording: Recording, speaker: str) -> float:
    """Total speech seconds of one speaker in a recording."""
    return sum(
        max(0.0, float(seg.end) - float(seg.start))
        for seg in recording.segments
        if seg.speaker == speaker
    )


def job_config(
    library: Library,
    recording: Recording,
    options: Dict[str, object],
    mode: str = "full",
) -> Config:
    """Build the job Config from stored job settings + user-supplied overrides.

    ``options`` (all optional): backend, num_speakers, diarize,
    match_voiceprints, vp_abs_threshold, vp_margin, vp_segment_agreement.
    """
    stored = recording.job_settings if isinstance(recording.job_settings, dict) else {}
    chosen = dict(stored)
    chosen.update({k: v for k, v in options.items() if v is not None})

    def as_bool(value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        return default

    config = Config(
        model_name=DEFAULT_ASR_MODEL,
        diarize=as_bool(chosen.get("diarize"), True),
        num_speakers=chosen.get("num_speakers") or None,
        diarization_backend=str(chosen.get("backend") or "auto"),
        output_dir=library.output_dir / recording.id,
        match_voiceprints=bool(chosen.get("match_voiceprints", False)),
        save_embeddings=True,
        record_id=recording.id,
        embedding_dir=library.root / "embeddings",
    )
    if mode == "transcribe":
        config.diarize = False
    if mode == "speakers":
        # the speakers-only pass always diarizes, whatever the stored setting
        config.diarize = True
    if chosen.get("vp_abs_threshold") is not None:
        config.vp_abs_threshold = float(chosen["vp_abs_threshold"])
    if chosen.get("vp_margin") is not None:
        config.vp_margin = float(chosen["vp_margin"])
    if chosen.get("vp_segment_agreement") is not None:
        config.vp_segment_agreement = float(chosen["vp_segment_agreement"])
    if chosen.get("merge_speakers") is not None:
        config.merge_speakers = as_bool(chosen.get("merge_speakers"), True)
    if chosen.get("merge_threshold") is not None:
        config.merge_threshold = float(chosen["merge_threshold"])
    if chosen.get("merge_min_weight") is not None:
        config.merge_min_weight = float(chosen["merge_min_weight"])
    if chosen.get("merge_segment_floor") is not None:
        config.merge_segment_floor = float(chosen["merge_segment_floor"])
    if chosen.get("audio_enhance") is not None:
        config.audio_enhance = as_bool(chosen.get("audio_enhance"), True)
    if chosen.get("diar_threshold") is not None:
        config.diar_threshold = float(chosen["diar_threshold"])
    if chosen.get("diar_min_duration_on") is not None:
        config.diar_min_duration_on = float(chosen["diar_min_duration_on"])
    if chosen.get("diar_min_duration_off") is not None:
        config.diar_min_duration_off = float(chosen["diar_min_duration_off"])
    if chosen.get("rescue_holes") is not None:
        config.rescue_holes = as_bool(chosen.get("rescue_holes"), True)
    if chosen.get("glossary") is not None:
        config.glossary = as_bool(chosen.get("glossary"), False)
    # global runtime setting: worker threads for ASR/diarization models
    try:
        threads = int(SettingsStore(library.root).as_public().get("threads", 0))
    except Exception:  # noqa: BLE001 - bad settings.json must not stop jobs
        threads = 0
    if threads > 0:
        config.threads = threads
    return config


VP_MIN_SEGMENT_SEC = 0.5  # keep in sync with Config.vp_min_segment_sec


def _plural_ru(n: int, one: str, few: str, many: str) -> str:
    """Russian plural form for a count (1 реплика / 2 реплики / 5 реплик)."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    n %= 10
    if n == 1:
        return one
    if 2 <= n <= 4:
        return few
    return many


def _notify_done(title: str, body: str) -> None:
    """Windows toast notification, best-effort (no external dependencies).

    Uses the WinRT toast API via an inline PowerShell script; any failure
    (old PowerShell, missing WinRT, policies) is silently ignored. Fired
    after a recording finishes when the user has not turned notifications
    off in settings."""
    try:
        script = (
            "[Windows.UI.Notifications.ToastNotificationManager,"
            " Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
            "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
            "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
            "$x=$t.GetXml();"
            f"$x=$x.Replace('placeholder1','{title}').Replace('placeholder2','{body}');"
            "$n=New-Object Windows.UI.Notifications.ToastNotification $t;"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
            "'Nestor').Show($n)"
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001 - notifications must never break a job
        pass


def _runtime_tuning(library: Library) -> tuple:
    """``(priority, pin_performance)`` from settings; safe defaults on error.

    Both are opt-in on the user's side: priority is off unless asked for, and
    pinning only ever does something on a hybrid CPU (see :mod:`nestor.cpuinfo`).
    """
    priority = False
    pin = True
    try:
        data = SettingsStore(library.root).as_public()
        priority = bool(data.get("process_priority", False))
        pin = bool(data.get("pin_performance", True))
    except Exception:  # noqa: BLE001 - bad settings.json must not stop jobs
        pass
    return priority, pin


def _settings_snapshot(config: Config) -> Dict[str, object]:
    """The recognition/diarization settings of one run, for the report.

    Kept as plain JSON types so it survives ``record.json`` round-trips and is
    easy to copy out of the UI.
    """
    return {
        "diarize": bool(config.diarize),
        "diarization_backend": config.diarization_backend,
        "num_speakers": config.num_speakers,
        "diar_threshold": config.diar_threshold,
        "audio_enhance": bool(config.audio_enhance),
        "rescue_holes": bool(config.rescue_holes),
        "glossary": bool(config.glossary),
        "match_voiceprints": bool(config.match_voiceprints),
        "merge_speakers": bool(config.merge_speakers),
        "merge_threshold": config.merge_threshold,
        "chunk_seconds": config.chunk_seconds,
    }


# --- decoded waveform cache -------------------------------------------------

WAVEFORM_CACHE_NAME = "audio16k.wav"


def waveform_cache_path(library: Library, record_id: str) -> Path:
    """Where the decoded 16 kHz mono waveform of a recording is cached."""
    return library.output_dir / record_id / WAVEFORM_CACHE_NAME


def load_or_decode(
    library: Library,
    recording: Recording,
    on_decode=None,
) -> tuple:
    """Return ``(waveform, from_cache)`` for the recording's audio.

    The first run decodes the source file and caches the result as PCM 16
    inside ``output/<id>/``; later runs (re-diarization, re-transcription)
    load the cached wav, which is much faster than re-decoding the source.
    Sources are immutable after import (the upload is copied into the
    library), so a cache keyed by record id never goes stale. The cache is
    removed together with the recording by ``Library.delete``.
    """
    cache = waveform_cache_path(library, recording.id)
    if cache.exists():
        try:
            # soundfile reads plain PCM wavs ~100x faster than the PyAV
            # decode path used for arbitrary source formats
            import soundfile as sf

            waveform, sample_rate = sf.read(cache, dtype="float32", always_2d=False)
            if sample_rate == SAMPLE_RATE and waveform.ndim == 1:
                return waveform, True
        except Exception:  # noqa: BLE001 - fall through to a fresh decode
            pass
        try:
            return decode(cache), True
        except Exception:  # noqa: BLE001 - fall through to a fresh decode
            pass
    source = library.recording_source(recording)
    waveform = decode(source, on_progress=on_decode)
    try:
        write_wav(cache, waveform)
    except OSError:
        pass  # cache is an optimization, never a requirement
    return waveform, False


def _voiceprint_long_labels(recording: Recording) -> set:
    """Labels owning at least one segment long enough to build a voice."""
    return {
        seg.speaker for seg in recording.segments
        if float(seg.end) - float(seg.start) >= VP_MIN_SEGMENT_SEC
    }


def _voiceprint_vector(
    lib: Library, record_id: str, recording: Recording, speaker: str
) -> tuple:
    """Cluster embedding of one speaker, ``(vector, reason)``.

    Order of trust: the segment labels first, then the npz cache, then a
    fresh re-embedding from the source audio. The cache is used only while
    its labels cover every current long segment - after a speakers re-run
    the same label can belong to a different person, and a stale cache must
    never hand out that person's vector. ``reason`` explains a failure:
    "missing" (no such speaker), "too_short" (all his segments are shorter
    than the embedding minimum), "no_audio" (nothing to recompute from).

    A rebuild embeds only the segments of the requested speaker. Recomputing
    the whole recording is quadratic to its length (a one-hour file spent
    ~45 s in decode + embedding) while the vector of one speaker needs only
    his own segments (~2 s), independent of the others. Such a partial
    result is not written back into the cache: the cache stays valid only
    when every long label is present.
    """
    from ..speaker_embed import (
        cluster_vectors,
        load_record_embeddings,
    )

    if not any(seg.speaker == speaker for seg in recording.segments):
        return None, "missing"
    long_labels = _voiceprint_long_labels(recording)
    if speaker not in long_labels:
        return None, "too_short"

    clusters, per_segment = load_record_embeddings(lib.embeddings_dir, record_id)
    if clusters and long_labels <= set(clusters):
        # a trusting cache always has the label (it covers every long label)
        if speaker in clusters:
            return clusters[speaker], None
        return None, "no_audio"
    # no usable cache (old recording, missing, or a pass whose labels no
    # longer match the segments): rebuild just this speaker from the audio
    source = lib.recording_source(recording)
    if not source.exists():
        return None, "no_audio"
    try:
        waveform = decode(source)
        own = [seg for seg in recording.segments if seg.speaker == speaker]
        own_clusters, _ = cluster_vectors(
            waveform, SAMPLE_RATE, own, min_segment_sec=VP_MIN_SEGMENT_SEC,
        )
    except Exception:  # noqa: BLE001 - broken audio / model failure
        return None, "no_audio"
    return own_clusters.get(speaker), None


def create_app(
    work_dir: Optional[Path] = None,
    max_workers: int = 1,
    library: Optional[Library] = None,
    token: Optional[str] = None,
) -> FastAPI:
    """Build the FastAPI application.

    ``token`` turns on the access guard (session token + Host check). Without
    it (tests) the app is open on purpose - tests talk to it directly.
    """
    app = FastAPI(title="Nestor", version=__version__)
    lib = library or Library(work_dir)
    processor = Processor(lib, max_workers=max_workers)
    voices = VoiceprintLibrary(lib.root)

    if token:
        install_access_guard(app, token)

    app.state.library = lib
    app.state.processor = processor
    app.state.voiceprints = voices
    app.state.session_token = token or ""
    app.state.settings = SettingsStore(lib.root)

    # A job interrupted by a crash or by closing the window would stay
    # "running" forever (frozen progress, pulsing logo). Reset it once.
    # Status "added" is not a job - an unprocessed upload stays as is.
    for recording in lib.list():
        if recording.status in ("queued", "running"):
            recording.status = "cancelled"
            recording.stage = "cancelled"
            recording.error = "Обработка прервана: приложение было закрыто"
            lib.save(recording)
            lib.log(f"interrupted {recording.filename}: status reset on startup")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/status")
    def status() -> Dict[str, object]:
        from ..models import has_local_diarization, model_root

        return {
            "app": "Nestor",
            "version": __version__,
            "data_dir": str(lib.root),
            "recordings": len(lib.list()),
            "sample_rate": SAMPLE_RATE,
            "local_diarization": has_local_diarization(),
            "model_dir": str(model_root()),
            "cpu_physical_cores": cpuinfo.physical_cores(),
            "cpu_logical_processors": os.cpu_count(),
            "cpu_hybrid": cpuinfo.is_hybrid(),
            "cpu_perf_cores": cpuinfo.performance_logical_count(),
            "cpu_auto_threads": cpuinfo.resolve_threads(0),
        }

    @app.get("/api/settings")
    def get_settings() -> Dict[str, object]:
        """Current user settings (server-persisted)."""
        return {"settings": app.state.settings.as_public()}

    @app.patch("/api/settings")
    def patch_settings(body: Dict = Body(...)) -> Dict[str, object]:
        """Merge and persist settings; unknown keys are ignored."""
        try:
            saved = app.state.settings.update(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        lib.log("settings updated: " + ", ".join(sorted(body.keys())))
        return {"settings": saved}

    @app.post("/api/settings/reset")
    def reset_settings() -> Dict[str, object]:
        """Back to factory defaults."""
        saved = app.state.settings.reset()
        lib.log("settings reset to defaults")
        return {"settings": saved}

    @app.get("/api/recordings")
    def list_recordings() -> Dict[str, object]:
        items = []
        for recording in lib.list():
            item = {
                "id": recording.id,
                "filename": recording.filename,
                "created_at": recording.created_at,
                "status": recording.status,
                "stage": recording.stage,
                "progress": recording.progress,
                "error": recording.error,
                "duration": recording.duration,
                "model": recording.model,
                "diarization_model": recording.diarization_model,
                "rtf": recording.rtf,
                "processing_seconds": recording.processing_seconds,
                "stage_seconds": recording.stage_seconds,
                "run_meta": recording.run_meta,
                # full path to the original file, shown in the expanded rail card
                "source_path": recording.source_path,
                "speakers_diff": recording.speakers_diff,
                "title": recording.title,
                "notes": recording.notes,
                "speakers": len(recording.speakers),
                "segments": len(recording.segments),
                "job_settings": recording.job_settings,
            }
            if recording.status == "running":
                # live seconds on the backend, not a client-side timer
                elapsed = processor.elapsed_sec(recording.id)
                if elapsed is not None:
                    item["elapsed_sec"] = round(elapsed, 1)
            elif recording.status == "done" and recording.processing_seconds:
                item["elapsed_sec"] = round(recording.processing_seconds, 1)
            items.append(item)
        return {"recordings": items}

    @app.post("/api/recordings")
    async def create_recording(
        file: UploadFile = File(...),
        diarize: bool = Form(True),
        num_speakers: Optional[int] = Form(None),
        backend: str = Form("auto"),
        match_voiceprints: bool = Form(False),
        vp_abs_threshold: Optional[float] = Form(None),
        vp_margin: Optional[float] = Form(None),
        vp_segment_agreement: Optional[float] = Form(None),
        merge_speakers: Optional[bool] = Form(None),
        merge_threshold: Optional[float] = Form(None),
        merge_min_weight: Optional[float] = Form(None),
        auto_start: bool = Form(True),
    ) -> Dict[str, object]:
        name = Path(file.filename or "audio.wav").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат «{suffix}». "
                f"Подойдут: {', '.join(sorted(ALLOWED_SUFFIXES))}",
            )

        record_id = uuid.uuid4().hex[:12]
        staging = lib.root / "uploads" / record_id / name
        staging.parent.mkdir(parents=True, exist_ok=True)

        def _stash() -> None:
            # sync disk copy in a worker thread: doing it inline would freeze
            # the event loop, so the whole UI (polling, stop button) stalls
            # until a large file is written
            with staging.open("wb") as target:
                shutil.copyfileobj(file.file, target, length=1024 * 1024)

        await asyncio.to_thread(_stash)
        await file.close()

        recording = Recording(id=record_id, filename=name, status="queued")
        if not auto_start:
            # "add only": the file waits in the records rail until the user
            # starts processing from the context menu
            recording.status = "added"
        recording.job_settings = {
            "model": DEFAULT_ASR_MODEL,
            "backend": backend or "auto",
            "num_speakers": num_speakers,
            "match_voiceprints": match_voiceprints,
        }
        lib.save(recording)
        lib.log(f"upload {name} ({staging.stat().st_size} bytes)")

        config = job_config(lib, recording, {}, mode="full")
        if vp_abs_threshold is not None:
            config.vp_abs_threshold = float(vp_abs_threshold)
            recording.job_settings["vp_abs_threshold"] = float(vp_abs_threshold)
        if vp_margin is not None:
            config.vp_margin = float(vp_margin)
            recording.job_settings["vp_margin"] = float(vp_margin)
        if vp_segment_agreement is not None:
            config.vp_segment_agreement = float(vp_segment_agreement)
            recording.job_settings["vp_segment_agreement"] = float(
                vp_segment_agreement
            )
        if merge_speakers is not None and isinstance(merge_speakers, bool):
            recording.job_settings["merge_speakers"] = merge_speakers
            config.merge_speakers = merge_speakers
        if isinstance(merge_threshold, (int, float)) and not isinstance(
            merge_threshold, bool
        ):
            recording.job_settings["merge_threshold"] = float(merge_threshold)
            config.merge_threshold = float(merge_threshold)
        if isinstance(merge_min_weight, (int, float)) and not isinstance(
            merge_min_weight, bool
        ):
            recording.job_settings["merge_min_weight"] = float(merge_min_weight)
            config.merge_min_weight = float(merge_min_weight)
        lib.save(recording)
        if auto_start:
            processor.submit(recording, config)
        return recording.to_dict()

    @app.post("/api/recordings/import")
    def import_recording(
        path: str = Body(..., embed=True),
        auto_start: bool = Body(True, embed=True),
    ) -> Dict[str, object]:
        """Register an existing local file (from the native Open dialog).

        The file is not copied: it already lives on this disk, the record
        just references its location. Reuses the same validation as uploads.
        """
        source = Path(path)
        if not source.is_file():
            raise HTTPException(status_code=404, detail=f"Файл не найден: {path}")
        suffix = source.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат «{suffix}». "
                f"Подойдут: {', '.join(sorted(ALLOWED_SUFFIXES))}",
            )

        record_id = uuid.uuid4().hex[:12]
        recording = Recording(id=record_id, filename=source.name, status="queued")
        if not auto_start:
            recording.status = "added"
        recording.source_path = str(source)
        settings = app.state.settings.as_public()
        recording.job_settings = {
            "model": DEFAULT_ASR_MODEL,
            "backend": "auto",
            "num_speakers": settings.get("num_speakers") or None,
            "match_voiceprints": bool(settings.get("vp_match", False)),
            "vp_abs_threshold": settings.get("vp_abs_threshold"),
            "vp_margin": settings.get("vp_margin"),
            "vp_segment_agreement": settings.get("vp_segment_agreement"),
            "merge_speakers": settings.get("merge_speakers", True),
            "merge_threshold": settings.get("merge_threshold"),
            "merge_min_weight": settings.get("merge_min_weight"),
            "merge_segment_floor": settings.get("merge_segment_floor"),
            "audio_enhance": settings.get("audio_enhance", True),
            "diar_threshold": settings.get("diar_threshold"),
            "diar_min_duration_on": settings.get("diar_min_duration_on"),
            "diar_min_duration_off": settings.get("diar_min_duration_off"),
            "rescue_holes": settings.get("rescue_holes", True),
            "glossary": settings.get("glossary", False),
        }
        lib.save(recording)
        lib.log(f"import {source.name} from {source.parent}")

        config = job_config(lib, recording, {}, mode="full")
        if auto_start:
            processor.submit(recording, config)
        return recording.to_dict()

    @app.post("/api/recordings/{record_id}/process")
    def process_recording(
        record_id: str,
        body: ProcessBody = Body(default=ProcessBody()),
    ) -> Dict[str, object]:
        """(Re)start processing of an already uploaded recording.

        Modes: ``full`` (transcribe + diarize), ``transcribe`` (single speaker,
        no diarization), ``speakers`` (reassign speakers from a fresh
        diarization pass without re-recognising the text).
        """
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        if recording.status in ("queued", "running"):
            raise HTTPException(status_code=409, detail="Обработка уже идёт")
        source = lib.recording_source(recording)
        if not source.exists():
            raise HTTPException(status_code=409, detail="Файл записи не найден")

        body_dict: Dict[str, object] = body.model_dump(exclude_none=True) if body else {}
        mode = str(body_dict.get("mode") or "full")
        if mode not in ("full", "transcribe", "speakers"):
            raise HTTPException(status_code=400, detail=f"Неизвестный режим: {mode}")
        if mode == "speakers" and not recording.segments:
            raise HTTPException(
                status_code=409,
                detail="Сначала нужна расшифровка — распознавать спикеров не по чему",
            )
        options = (
            body_dict.get("options")
            if isinstance(body_dict.get("options"), dict)
            else {}
        )

        # remember the chosen options: the "process again" dialog pre-fills
        # from job_settings on the next run
        stored = dict(recording.job_settings) if isinstance(recording.job_settings, dict) else {}
        stored.update(options)
        # a single ASR model is shipped now: migrate any legacy name
        stored["model"] = DEFAULT_ASR_MODEL
        if mode == "transcribe":
            stored["diarize"] = False
        elif mode == "full":
            stored["diarize"] = True
        recording.job_settings = stored

        recording.status = "queued"
        recording.error = None
        recording.progress = 0.0
        lib.save(recording)
        lib.log(f"process {recording.filename} (mode {mode})")

        config = job_config(lib, recording, options, mode=mode)
        processor.submit(recording, config, mode=mode)
        return recording.to_dict()

    @app.get("/api/recordings/{record_id}")
    def get_recording(record_id: str) -> Dict[str, object]:
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        payload = recording.to_dict()
        if recording.status == "running":
            elapsed = processor.elapsed_sec(recording.id)
            if elapsed is not None:
                payload["elapsed_sec"] = round(elapsed, 1)
        return payload

    @app.patch("/api/recordings/{record_id}/meta")
    def patch_recording_meta(
        record_id: str, body: Dict = Body(...)
    ) -> Dict[str, object]:
        """Rename (title alias) and notes for the records table.

        Allowed while a job runs: title/notes do not touch segments, so a
        concurrent save from the worker cannot clobber them."""
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        if "title" in body:
            title = str(body.get("title") or "").strip()
            recording.title = title or None
        if "notes" in body:
            notes = str(body.get("notes") or "").strip()
            recording.notes = notes or None
        lib.save(recording, create=False)
        return {
            "id": recording.id,
            "title": recording.title,
            "notes": recording.notes,
        }

    @app.put("/api/recordings/{record_id}")
    async def save_recording(record_id: str, request: Request) -> Dict[str, object]:
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        if recording.status in ("queued", "running"):
            # a worker owns the record right now: a save from the editor would
            # be overwritten by the job's own copy (or overwrite its results)
            raise HTTPException(
                status_code=409,
                detail="Идёт обработка — правки сейчас сохранить нельзя",
            )
        payload = await request.json()

        old_composition = [(seg.id, seg.speaker) for seg in recording.segments]
        segments = payload.get("segments")
        if isinstance(segments, list):
            recording.segments = [Segment.from_dict(item) for item in segments]
        names = payload.get("speaker_names")
        if isinstance(names, dict):
            recording.speaker_names = {str(k): str(v) for k, v in names.items()}

        new_composition = [(seg.id, seg.speaker) for seg in recording.segments]
        if new_composition != old_composition:
            # a segment was moved to another label (or split): the npz cache
            # stores per-label means of the OLD composition, so trusting it
            # would save a voice built from the wrong segments - drop it and
            # let the next "remember voice" rebuild from the audio
            (lib.embeddings_dir / f"{record_id}.npz").unlink(missing_ok=True)
            # a label that lost its last segment disappears from the chips:
            # forget its name and match so no orphan entry lingers
            live = {seg.speaker for seg in recording.segments}
            for label in [k for k in recording.speaker_names if k not in live]:
                recording.speaker_names.pop(label, None)
            for label in [k for k in recording.speaker_match if k not in live]:
                recording.speaker_match.pop(label, None)

        # create=False: if the record was deleted while the editor was typing,
        # do not recreate it from the client's stale copy
        lib.save(recording, create=False)
        return recording.to_dict()

    @app.post("/api/recordings/{record_id}/speakers/merge")
    def merge_speakers(
        record_id: str,
        from_label: str = Body(..., embed=True),
        into: str = Body(..., embed=True),
    ) -> Dict[str, object]:
        """Attribute every segment of ``from_label`` to ``into``.

        Both labels keep living in speaker_names (harmless), but the source
        label loses its manual name so the two entries cannot diverge. When
        the labels are the same or the source has no segments - a no-op that
        still returns the fresh recording dict.
        """
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        if recording.status in ("queued", "running"):
            raise HTTPException(
                status_code=409,
                detail="Идёт обработка — спикеров сейчас склеить нельзя",
            )
        if from_label == into:
            raise HTTPException(status_code=400, detail="Нельзя склеить спикера с собой")
        if not any(s.speaker == from_label for s in recording.segments):
            raise HTTPException(
                status_code=404,
                detail=f"Спикер {from_label} не найден в этой записи",
            )

        moved = 0
        for segment in recording.segments:
            if segment.speaker == from_label:
                segment.speaker = into
                moved += 1
        # the merged label's name moves to the target when the target is
        # still unnamed; otherwise it is dropped so duplicates cannot linger
        from_name = recording.speaker_names.get(from_label)
        if from_name and not recording.speaker_names.get(into):
            recording.speaker_names[into] = from_name
        recording.speaker_names.pop(from_label, None)
        recording.speaker_match.pop(from_label, None)
        # the embedding cache holds the old labels' means - drop it so the
        # next "remember voice" rebuilds the target's voice from all segments
        (lib.embeddings_dir / f"{record_id}.npz").unlink(missing_ok=True)
        lib.save(recording, create=False)
        lib.log(f"merge speakers {from_label} -> {into} ({moved} segments)")
        return recording.to_dict()

    @app.post("/api/recordings/{record_id}/cancel")
    def cancel_recording(record_id: str) -> Dict[str, object]:
        """Stop a queued or running job (item: no way to stop a job today)."""
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        if recording.status not in ("queued", "running"):
            return {"cancel": "noop", "status": recording.status}

        outcome = processor.cancel(record_id)
        if outcome in ("queued", "none"):
            # nothing is running for it: persist the state right away
            recording.status = "cancelled"
            recording.stage = "cancelled"
            lib.save(recording)
            lib.log(f"stopped {recording.filename} (before start)")
            return {"cancel": outcome, "status": "cancelled"}
        # "running": the worker stops at its next checkpoint (between chunks),
        # the UI keeps polling until the status flips to "cancelled"
        return {"cancel": outcome, "status": "stopping"}

    @app.delete("/api/recordings/{record_id}")
    def delete_recording(record_id: str) -> Dict[str, object]:
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")

        if recording.status in ("queued", "running"):
            # a worker still writing would recreate the files right after we
            # removed them ("zombie" record), so stop it first and wait
            outcome = processor.cancel(record_id)
            if outcome == "running" and not processor.wait(record_id, timeout=20):
                raise HTTPException(
                    status_code=409,
                    detail="Обработка ещё не остановлена — попробуйте удалить ещё раз",
                )

        if not lib.delete(record_id):
            # the manifest is gone but locked files remained — the record will
            # not come back; tell the user to clean up after closing the app
            raise HTTPException(
                status_code=409,
                detail="Часть файлов занята — запись удалена частично, попробуйте позже",
            )
        return {"deleted": record_id}

    @app.get("/api/recordings/{record_id}/media")
    def recording_media(record_id: str, request: Request) -> Response:
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        source = lib.recording_source(recording)
        if not source.exists():
            raise HTTPException(status_code=404, detail="Файл записи не найден")
        media_type = MEDIA_TYPES.get(source.suffix.lower(), "application/octet-stream")
        return ranged_response(source, request, media_type)

    @app.get("/api/sample")
    def sample_audio() -> Response:
        """Short demo recording for the «Попробовать на примере» button.

        The file ships inside the exe bundle (nestor.spec datas) and falls
        back to the repository copy in dev mode."""
        candidates = []
        bundle = getattr(sys, "_MEIPASS", None)
        if bundle:
            candidates.append(Path(bundle) / "sample" / "dialogue.webm")
        # dev mode: repo root is two levels up from this file (nestor/web/app.py)
        candidates.append(Path(__file__).resolve().parent.parent.parent / "samples" / "dialogue.webm")
        for sample in candidates:
            if sample.exists():
                return Response(content=sample.read_bytes(), media_type="audio/webm")
        raise HTTPException(status_code=404, detail="Пример недоступен")

    @app.get("/api/recordings/{record_id}/peaks")
    def recording_peaks(record_id: str) -> Dict[str, object]:
        return {"peaks": lib.load_peaks(record_id)}

    @app.get("/api/recordings/{record_id}/export/{fmt}")
    def recording_export(record_id: str, fmt: str) -> Response:
        path = build_export(lib, record_id, fmt)
        media_type = {
            ".json": "application/json",
            ".srt": "application/x-subrip",
            ".vtt": "text/vtt",
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".docx": (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        }.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(
            path,
            media_type=media_type,
            filename=Path(path).name,
        )

    # ---------------------------------------------------------------- glossary
    @app.get("/api/glossary")
    def get_glossary() -> Dict[str, object]:
        """The single list of terms, plus where its file lives."""
        path = resolve_glossary_path(lib.root)
        glossary = load_glossary(path)
        return {
            "path": str(path),
            "count": len(glossary),
            "terms": [
                {"term": t.term, "aliases": list(t.aliases)}
                for t in glossary.terms
            ],
        }

    @app.post("/api/glossary/term")
    async def add_glossary_term(request: Request) -> Dict[str, object]:
        """Add or replace one term in the user's editable glossary."""
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}

        term = str(body.get("term") or "").strip()
        raw_aliases = body.get("aliases") or []
        if isinstance(raw_aliases, str):
            raw_aliases = re.split(r"[,\n;]+", raw_aliases)
        aliases = [str(a).strip() for a in raw_aliases if str(a).strip()]
        if not term:
            raise HTTPException(status_code=400, detail="Укажите термин")
        if not aliases:
            raise HTTPException(status_code=400, detail="Нужен хотя бы один вариант")

        target = ensure_glossary(lib.root)
        glossary = load_glossary(target)
        upsert_term(glossary, term, aliases)
        save_glossary(glossary, target)
        lib.log(f"glossary: term saved «{term}» ({len(aliases)} alias(es))")
        return {"term": term, "aliases": aliases, "count": len(glossary)}

    @app.delete("/api/glossary/term")
    async def delete_glossary_term(term: str) -> Dict[str, object]:
        """Remove a term (by canonical name) from the editable glossary."""
        name = (term or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Укажите термин")
        target = ensure_glossary(lib.root)
        glossary = load_glossary(target)
        removed = remove_term(glossary, name)
        if not removed:
            raise HTTPException(status_code=404, detail="Термин не найден")
        save_glossary(glossary, target)
        lib.log(f"glossary: term removed «{name}»")
        return {"removed": name, "count": len(glossary)}

    @app.get("/api/glossary/export")
    def export_glossary() -> Response:
        """The effective glossary as a portable YAML file."""
        path = resolve_glossary_path(lib.root)
        glossary = load_glossary(path)
        stamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"nestor-hotwords-{stamp}.yaml"
        lib.log(f"glossary exported: {len(glossary)} terms")
        return Response(
            content=dump_glossary(glossary),
            media_type="application/x-yaml",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/glossary/import")
    async def import_glossary(request: Request) -> Dict[str, object]:
        """Merge terms from a YAML/JSON file into the editable glossary.

        Two bodies, mirroring voiceprints: ``{"path": ...}`` (the native Open
        dialog hands us a path) or ``multipart/form-data`` with a ``file``
        field (the browser fallback)."""
        ctype = (request.headers.get("content-type") or "").lower()
        text: str = ""
        if "application/json" in ctype:
            try:
                body = await request.json()
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
            raw_path = str((body or {}).get("path") or "").strip()
            if not raw_path:
                raise HTTPException(status_code=400, detail="Нужен path или файл")
            source = Path(raw_path)
            if not source.exists():
                raise HTTPException(status_code=404, detail="Файл не найден")
            text = source.read_text(encoding="utf-8")
        else:
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise HTTPException(status_code=400, detail="Нужен файл словаря")
            raw = await upload.read()
            if len(raw) > 5 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Файл слишком большой")
            text = raw.decode("utf-8")

        try:
            incoming = Glossary.parse(text)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный словарь: {exc}")
        if not incoming:
            raise HTTPException(status_code=400, detail="В файле нет терминов")

        target = ensure_glossary(lib.root)
        glossary = load_glossary(target)
        added = merge_glossary(glossary, incoming)
        save_glossary(glossary, target)
        lib.log(
            f"glossary imported: +{added} new, {len(incoming) - added} replaced, "
            f"{len(glossary)} total"
        )
        return {
            "added": added,
            "replaced": len(incoming) - added,
            "count": len(glossary),
        }

    @app.get("/api/glossary/preview")
    def preview_glossary(text: str = "") -> Dict[str, object]:
        """Dry-run: what the current glossary would change in ``text``."""
        alias_map = build_alias_map(load_glossary(resolve_glossary_path(lib.root)))
        from ..glossary import postcorrect

        corrected, changes = postcorrect(text, alias_map)
        return {
            "text": corrected,
            "changes": [{"from": a, "to": b} for a, b in changes],
        }

    # ------------------------------------------------------------- voiceprints
    @app.get("/api/voiceprints")
    def list_voiceprints() -> Dict[str, object]:
        items = []
        for meta in voices.list():
            items.append(
                {
                    "slug": meta.slug,
                    "name": meta.display_name,
                    "comment": meta.comment,
                    "num_utterances": meta.num_utterances,
                    "total_duration_sec": round(meta.total_duration_sec, 1),
                    "created_at": meta.created_at,
                    "updated_at": meta.updated_at,
                }
            )
        return {"voiceprints": items}

    @app.post("/api/voiceprints")
    async def add_voiceprint(request: Request) -> Dict[str, object]:
        """Save a speaker's voice from a processed recording.

        Body: ``{name, recording_id, speaker}``. Uses the cached embeddings
        (``embeddings/<id>.npz``); without a cache it falls back to decoding
        the source audio and re-embedding only that speaker's segments - a
        partial result (never the whole recording), so saving stays quick
        even for hour-long files. decoding happens off the event loop.
        """
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}

        name = str(body.get("name") or "").strip()
        record_id = str(body.get("recording_id") or "").strip()
        speaker = str(body.get("speaker") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Укажите имя спикера")
        if not record_id or not speaker:
            raise HTTPException(
                status_code=400, detail="Нужны recording_id и speaker"
            )
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        if recording.status in ("queued", "running"):
            raise HTTPException(
                status_code=409,
                detail="Идёт обработка — голос сейчас запомнить нельзя",
            )
        display = recording.speaker_names.get(speaker) or speaker

        duration = _voiceprint_duration(recording, speaker)
        # decode + embedding are heavy CPU work; keep them off the event
        # loop so the UI stays responsive while the vector is being built
        vector, reason = await run_in_threadpool(
            _voiceprint_vector, lib, record_id, recording, speaker
        )
        if vector is None:
            if reason == "too_short":
                min_sec = f"{VP_MIN_SEGMENT_SEC:.1f}".replace(".", ",")
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"У спикера «{display}» нет реплик длиннее "
                        f"{min_sec} с — голос запомнить не из чего"
                    ),
                )
            if reason == "no_audio":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Аудиофайл записи недоступен — голос «{display}» "
                        "пересчитать не из чего"
                    ),
                )
            raise HTTPException(
                status_code=404,
                detail=f"В записи нет спикера {speaker}",
            )

        meta = voices.add_or_update(
            name,
            vector,
            source_recording_id=record_id,
            duration_sec=duration,
        )
        lib.log(f"voiceprint saved: {meta.display_name} (from {recording.filename}, {speaker})")
        return {"saved": meta.display_name, "slug": meta.slug}

    @app.delete("/api/voiceprints/{slug}")
    def delete_voiceprint(slug: str) -> Dict[str, object]:
        meta = voices.find_by_slug(slug)
        name = meta.display_name if meta else slug
        if not voices.delete(slug):
            raise HTTPException(status_code=404, detail="Голос не найден")
        lib.log(f"voiceprint deleted: {name}")
        return {"deleted": slug}

    @app.patch("/api/voiceprints/{slug}")
    def update_voiceprint(slug: str, body: Dict = Body(...)) -> Dict[str, object]:
        """Update a voiceprint's note (comment)."""
        meta = voices.find_by_slug(slug)
        if meta is None:
            raise HTTPException(status_code=404, detail="Голос не найден")
        comment = str((body or {}).get("comment") or "").strip()
        updated = voices.set_comment(slug, comment)
        if updated is None:
            raise HTTPException(status_code=404, detail="Голос не найден")
        lib.log(f"voiceprint comment updated: {updated.display_name}")
        return {"slug": updated.slug, "comment": updated.comment}

    @app.post("/api/voiceprints/rename")
    async def rename_voiceprint_everywhere(request: Request) -> Dict[str, object]:
        """Rename a speaker everywhere: voice library + all recordings.

        Body: ``{old, new}`` (both display names). Every recording that had
        ``old`` among its speaker names gets it rewritten (the speaker_names
        layer and the voice-match entries); the voice library entry is
        renamed too, or folded into an existing voice of the same name.
        """
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}
        old = str(body.get("old") or "").strip()
        new = str(body.get("new") or "").strip()
        if not old or not new:
            raise HTTPException(status_code=400, detail="Нужны old и new")
        if voices.find(old) is None:
            raise HTTPException(status_code=404, detail=f"Голос «{old}» не найден")
        meta = voices.rename(old, new)
        if meta is None:  # pragma: no cover - guarded above
            raise HTTPException(status_code=404, detail=f"Голос «{old}» не найден")
        changed = 0
        for recording in lib.list():
            hit = [
                label for label, name in recording.speaker_names.items()
                if name == old
            ]
            matched = [
                label for label, match in (recording.speaker_match or {}).items()
                if match.get("assigned_name") == old
            ]
            if not hit and not matched:
                continue
            for label in hit:
                recording.speaker_names[label] = new
            for label in matched:
                match = dict(recording.speaker_match[label])
                match["assigned_name"] = new
                recording.speaker_match[label] = match
            try:
                lib.save(recording, create=False)
                changed += 1
            except OSError:
                pass
        lib.log(f"voiceprint renamed: {old} -> {meta.display_name} ({changed} recordings)")
        return {
            "saved": meta.display_name,
            "slug": meta.slug,
            "merged": meta.display_name != new,
            "changed": changed,
        }

    @app.get("/api/voiceprints/export")
    def export_voiceprints(slugs: Optional[str] = None) -> Response:
        """Voices as one portable JSON file.

        ``?slug=a&slug=b`` exports only the chosen voices (the panel's
        checkbox selection); no query = the whole library (full backup).
        """
        chosen = slugs.split(",") if slugs else None
        payload = voices.export_data(chosen)
        stamp = datetime.now().strftime("%Y-%m-%d")
        count = payload.get("count")
        suffix = f"-{count}" if chosen else ""
        filename = f"nestor-voices{suffix}-{stamp}.json"
        lib.log(
            f"voiceprints exported: {payload['count']} voices"
            + (" (selection)" if chosen else "")
        )
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=1),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/voiceprints/import")
    async def import_voiceprints(request: Request) -> Dict[str, object]:
        """Merge voices from a portable JSON file into the library.

        Two bodies: ``{"path": ...}`` (the native Open dialog hands us a
        path) or ``multipart/form-data`` with a ``file`` field (the browser
        fallback)."""
        ctype = (request.headers.get("content-type") or "").lower()
        payload: object = None
        if "application/json" in ctype:
            try:
                body = await request.json()
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
            path = str((body or {}).get("path") or "").strip()
            if not path:
                raise HTTPException(status_code=400, detail="Нужен path или файл")
            source = Path(path)
            if not source.exists():
                raise HTTPException(status_code=404, detail="Файл не найден")
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        else:
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise HTTPException(status_code=400, detail="Нужен файл голосов")
            raw = await upload.read()
            if len(raw) > 50 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Файл слишком большой")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        try:
            report = voices.import_data(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        lib.log(
            f"voiceprints imported: +{len(report['added'])} new, "
            f"~{len(report['merged'])} merged, {len(report['skipped'])} skipped"
        )
        return report

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def run_server(
    port: int = 8000,
    work_dir: Optional[Path] = None,
    reload: bool = False,
) -> None:
    """Run the web service with uvicorn (loopback only, with an access token)."""
    import uvicorn

    token = generate_session_token()
    if reload:
        # dev mode: no token guard, the factory creates the app itself
        uvicorn.run(
            "nestor.web.app:create_app", factory=True,
            host="127.0.0.1", port=port, reload=True,
        )
        return

    app = create_app(work_dir=work_dir, token=token)
    print(
        f"Nestor is running on http://127.0.0.1:{port}/?token={token}",
        flush=True,
    )
    uvicorn.run(app, host="127.0.0.1", port=port)
