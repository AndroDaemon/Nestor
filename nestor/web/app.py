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
    Query,
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
from ..audio_io import decode_to_wav_16k
from ..constants import SAMPLE_RATE
from ..decode_worker import decode_source_to_cache
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
from ..library import Library, Recording, Segment, WordStamp
from ..pipeline import JobCancelled, process_file
from ..settings import SettingsStore
from ..types import Turn
from ..video_names import names_from_video, probe_video
from ..voiceprints import VoiceprintLibrary
from ..people import PeopleLibrary
from ..organizer import ProjectLibrary

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
# Types for the picture element: a .webm served as audio/webm is rejected by
# Chromium even when the file carries a video track.
VIDEO_MEDIA_TYPES = {
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".3gp": "video/3gpp",
}
EXPORT_FORMATS = ("json", "srt", "vtt", "txt", "md", "docx")

#: Env vars used to pass the dev-reload factory its session token and work dir.
_TOKEN_ENV = "NESTOR_SESSION_TOKEN"
_WORKDIR_ENV = "NESTOR_WORK_DIR"


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
# --- waveform points for the player -----------------------------------------
# A fixed bucket count smeared long takes: one hour of audio got 3.2 s per
# point, so a zoomed-in view was blocky and short turns sat inside a single
# pixel. Tying the count to the duration keeps the shape honest at any zoom.
# The ceiling is a safety net for absurd input only: capping earlier (100 min)
# would quietly bring the old coarseness back on exactly the long files that
# prompted the change.
PEAKS_PER_SECOND = 4.0
PEAKS_MIN = 1200
PEAKS_MAX = 288_000


def peaks_bucket_count(duration_sec: float) -> int:
    """How many waveform points a recording of this length should have."""
    if duration_sec <= 0:
        return PEAKS_MIN
    return int(min(PEAKS_MAX, max(PEAKS_MIN, round(duration_sec * PEAKS_PER_SECOND))))


# 4 M samples per pass: the temporary is ~16 MB whatever the recording length,
# so a five-hour take does not need a second full-size copy just for |x|.
_PEAKS_BLOCK = 1 << 22


def compute_peaks(waveform: np.ndarray, buckets: Optional[int] = None) -> List[float]:
    """Downsample a waveform into normalised peak values for the player."""
    if waveform.size == 0:
        return []
    if buckets is None:
        buckets = peaks_bucket_count(waveform.size / float(SAMPLE_RATE))
    buckets = max(1, min(buckets, waveform.size))
    edges = np.linspace(0, waveform.size, buckets + 1).astype(np.int64)
    # max(|waveform|) per bucket, in bounded passes. buckets <= waveform.size
    # keeps every bucket non-empty, so reduceat is safe here.
    peaks = np.empty(buckets, dtype=np.float32)
    first = 0
    while first < buckets:
        low = int(edges[first])
        last = int(np.searchsorted(edges, low + _PEAKS_BLOCK, side="right")) - 1
        last = min(buckets, max(first + 1, last))
        segment = np.abs(waveform[low : int(edges[last])])
        peaks[first:last] = np.maximum.reduceat(segment, edges[first:last] - low)
        first = last
    maximum = float(peaks.max()) if peaks.size else 1.0
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
    transcript = transcript_from_recording(recording)
    if fmt == "json":
        from ..exporters import json_export

        return json_export.export(transcript, target_dir / "transcript.json")
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

    def active(self) -> List[str]:
        """Ids of jobs this process still owns — queued or running.

        Used by the desktop window to warn before closing while a job is
        unfinished: the executor threads are not daemons, so the process stays
        alive (without a window) until the job ends.
        """
        with self._lock:
            return sorted(set(self._events) | set(self._futures) | set(self._started))

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
            "cpu_name": cpuinfo.cpu_name(),
            "ram_total_bytes": cpuinfo.total_ram_bytes(),
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
        # A fresh attempt starts from zero: without this the bar would briefly
        # show the previous run's fraction (a repeat pass of a finished record
        # showed "100%" while the new run was still queued).
        recording.progress = 0.0
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
        peaks: Optional[List[float]] = None
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
            # Take the envelope while the buffer is still the raw decoded audio:
            # process_file() below gets ownership of it and may filter it in
            # place. Keeping a second full-length copy around is exactly what
            # made multi-hour recordings run out of memory.
            peaks = compute_peaks(waveform)

            # Diarization is the other long CPU stage; run it in a child process
            # so the window keeps answering while sherpa-onnx holds the GIL.
            # Falls back to the in-process pass (turns=None) if it is disabled
            # or unavailable.
            precomputed_turns = None
            if config.diarize:
                def on_diarize(fraction: float) -> None:
                    on_stage(
                        "diarization", 0.07 + 0.18 * max(0.0, min(1.0, fraction))
                    )

                precomputed_turns = diarize_via_child(
                    self.library,
                    recording,
                    config,
                    on_progress=on_diarize,
                    cancel=event.is_set,
                )

            transcript = process_file(
                config,
                self.library.recording_source(recording),
                on_stage=on_stage,
                waveform=waveform,
                cancel=event.is_set,
                reuse_waveform=True,
                turns=precomputed_turns,
            )
            # ownership was handed to the pipeline: release our reference now so
            # the buffer can be freed while the transcript is being written
            waveform = None

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
                    # exact word clock: the editor cuts a segment between words
                    # and must not guess the boundary from the character count
                    words=[
                        WordStamp(text=word.text, start=word.start, end=word.end)
                        for word in (getattr(segment, "words", None) or ())
                    ],
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
                if peaks:
                    self.library.save_peaks(recording.id, peaks)
                self._mark_cancelled(recording)
                write_timings("cancelled", "cancelled")
                self.library.log(
                    f"stopped {recording.filename}: "
                    f"kept {len(recording.segments)} segments"
                )
                return

            # Names burnt into the video frames (conference name plates). A few
            # seconds of work that gives the user a ready roster when renaming
            # speakers — it never guesses who said what.
            try:
                on_stage("names", 0.97)
                recording.video_names = names_from_video(
                    self.library.recording_source(recording)
                )
            except Exception as exc:  # noqa: BLE001 - a bonus, never a failure
                self.library.log(f"names skipped for {recording.filename}: {exc}")

            recording.status = "done"
            recording.stage = "done"
            recording.progress = 1.0
            recording.error = None

            if peaks:
                self.library.save_peaks(recording.id, peaks)
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
            if peaks:
                self.library.save_peaks(recording.id, peaks)
            self._mark_cancelled(recording)
            write_timings("cancelled", "cancelled")
            self.library.log(f"stopped {recording.filename}: no transcript kept")
        except Exception as exc:  # noqa: BLE001 - reported in the UI
            recording.status = "error"
            recording.stage = "error"
            recording.error = str(exc)
            # Keep the last fraction instead of zeroing it: "ошибка · 78%"
            # tells the user how far the run got before it died, which is what
            # matters when a job fails hours in.
            self.library.save(recording)
            write_timings("error", "error")
            self.library.log(f"error {recording.filename}: {exc}")
        finally:
            if recording.status == "done":
                recording.progress = 1.0

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
        # a fresh attempt starts from zero (see _run_job)
        recording.progress = 0.0
        # snapshot for the diff shown after the run («12 → 9 · перенесено 143 реплики»)
        before_counts = Counter(s.speaker for s in recording.segments if s.speaker)
        self._moved_segments = 0
        self._dropped_names: List[str] = []
        # Before labels are touched, remember the old map.  A speakers-only run
        # changes only `segment.speaker`, ids stay stable, so this sidecar is
        # enough for a one-click rollback and for marking changed rows.
        backup_path = _speakers_backup_path(self.library, recording.id)
        backup: Dict[str, object] = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "before": {seg.id: seg.speaker for seg in recording.segments},
            "segments": recording.to_dict()["segments"],
            "structure": _structure_signature(recording.segments),
            "names": dict(recording.speaker_names),
            "keep": list(recording.speaker_keep),
            "match": dict(recording.speaker_match),
            "before_speakers": len(before_counts),
        }
        try:
            _write_speakers_backup(backup_path, backup)
        except OSError as exc:
            self.library.log(f"speakers backup skipped ({recording.id}): {exc}")

        def finish_backup(failed: bool) -> None:
            """Fill in the result half of the rollback sidecar (best effort)."""
            before_map = backup.get("before")
            before_map = before_map if isinstance(before_map, dict) else {}
            changed = [
                seg.id
                for seg in recording.segments
                if before_map.get(seg.id) != seg.speaker
            ]
            backup.update(
                {
                    "after": {seg.id: seg.speaker for seg in recording.segments},
                    "structure_after": _structure_signature(recording.segments),
                    "changed": changed,
                    "after_speakers": len(recording.speakers),
                    "moved": self._moved_segments,
                    "dropped_names": list(self._dropped_names),
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "failed": failed,
                }
            )
            try:
                _write_speakers_backup(backup_path, backup)
            except OSError as exc:
                self.library.log(f"speakers backup update failed ({recording.id}): {exc}")

        try:
            self.library.save(recording)
            self.library.log(f"start speakers-only {recording.filename}")
        except OSError as exc:
            recording.status = "error"
            recording.stage = "error"
            recording.error = f"Не удалось сохранить статус записи: {exc}"
            with contextlib.suppress(OSError):
                self.library.save(recording)
            with contextlib.suppress(OSError):
                backup_path.unlink(missing_ok=True)
            return

        def on_progress(fraction: float) -> bool:
            # sherpa-onnx reports the fraction of processed chunks (0..1).
            # The truthy return value aborts the diarization run (see
            # LocalDiarizer.diarize) when the job was asked to stop.
            # Here diarization *is* the job, so it owns the bar from the end of
            # the decode band to the end of the run (0.06..0.95); the full pass
            # reserves 0.06..0.25 for the same stage because ASR follows it.
            nonlocal last_save
            recording.stage = "diarization"
            recording.progress = 0.06 + max(0.0, min(1.0, fraction)) * 0.89
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
                recording.progress = 0.02 + 0.04 * max(0.0, min(1.0, fraction))
                now = time.monotonic()
                if now - last_save >= 0.4:
                    with contextlib.suppress(OSError):
                        self.library.save(recording)
                    last_save = now

            waveform, from_cache = load_or_decode(self.library, recording, on_decode)
            if from_cache:
                on_decode(1.0)  # cache hit: decode band done instantly

            recording.stage = "diarization"
            self.library.save(recording)

            # Diarization is the long CPU stage; run it in a child process so
            # the window keeps answering while sherpa-onnx holds the GIL. This
            # mode always diarized the raw signal (no speech enhancement), so
            # ask the child for the same: enhance=False.
            turns = diarize_via_child(
                self.library,
                recording,
                config,
                on_progress=on_progress,
                cancel=event.is_set,
                enhance=False,
            )
            if turns is None:
                # Child unavailable or disabled: fall back to the in-process
                # pass. The waveform is still needed below (voiceprints read
                # it). Only this fallback builds the diarizer in this process.
                from ..pipeline import (
                    _accepts_blocking,
                    _accepts_progress,
                    _diarizer_for,
                )

                diarizer, diarizer_lock, _loaded = _diarizer_for(
                    config, lambda _m: None
                )
                turns = []
                with diarizer_lock:
                    if event.is_set():
                        raise JobCancelled("stopped before diarization")
                    if _accepts_blocking(diarizer):
                        # Same block-wise path the full run uses: without it a
                        # multi-hour record is diarized in one call, which both
                        # blows up memory (``bad allocation``) and feeds hours
                        # of silence to the segmenter.
                        turns = diarizer.diarize(
                            waveform,
                            SAMPLE_RATE,
                            progress=on_progress,
                            block_seconds=float(getattr(config, "diar_block_seconds", 0.0) or 0.0),
                            block_overlap=float(getattr(config, "diar_block_overlap", 60.0) or 0.0),
                        )
                    elif _accepts_progress(diarizer):
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

            # Preserve a manual name only when the cluster kept its identity.
            # Diarization may hand the same person a fresh SPEAKER_NN, so we do
            # not match labels to labels: we match *the speech itself*. A name
            # survives only if the pairing dominates both the old speaker's
            # speech and the new cluster's speech (a tiny reshuffle of segment
            # boundaries is fine). If the speaker fell apart into several
            # clusters or was merged with somebody else, the name is dropped
            # (and reported) instead of being carried to a different voice.
            NAME_PURITY = 0.9
            durations: Dict[tuple, float] = {}
            for seg in recording.segments:
                pair = (seg.speaker, new_labels[id(seg)])
                durations[pair] = durations.get(pair, 0.0) + max(0.0, seg.end - seg.start)
            old_total: Dict[str, float] = {}
            new_total: Dict[str, float] = {}
            for (old_label, new_label), seconds in durations.items():
                old_total[old_label] = old_total.get(old_label, 0.0) + seconds
                new_total[new_label] = new_total.get(new_label, 0.0) + seconds

            names: Dict[str, str] = {}
            for (old_label, new_label), seconds in durations.items():
                existing = recording.speaker_names.get(old_label, "")
                if not existing:
                    continue
                if old_total.get(old_label, 0.0) <= 0.0 or new_total.get(new_label, 0.0) <= 0.0:
                    continue
                share_old = seconds / old_total[old_label]
                share_new = seconds / new_total[new_label]
                if share_old >= NAME_PURITY and share_new >= NAME_PURITY:
                    names[new_label] = existing

            for seg in recording.segments:
                new_label = new_labels[id(seg)]
                if new_label != seg.speaker:
                    self._moved_segments += 1
                    # the row was doubtful because of the old decision; the
                    # review it may have had no longer applies.
                    seg.reviewed = False
                seg.speaker = new_label
            previous_names = dict(recording.speaker_names)
            recording.speaker_names = names
            self._dropped_names = sorted(
                {value for value in previous_names.values()}
                - {value for value in names.values()}
                - {""}
            )

            # voiceprints: re-match the fresh clusters against the library
            if event.is_set():
                raise JobCancelled("stopped before voiceprint matching")
            recording.speaker_match = {}
            try:
                from .. import speaker_embed, speaker_match as sm
                from ..people import PeopleLibrary

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
                        min_coh=config.merge_min_coh,
                        report=lambda line: self.library.log(line.strip()),
                    )
                    # the merged labels must also update the manual names
                    recording.speaker_names = {
                        label: value
                        for label, value in recording.speaker_names.items()
                        if label in {s.speaker for s in recording.segments}
                    }
                    recording.speaker_keep = [
                        label
                        for label in recording.speaker_keep
                        if label in {s.speaker for s in recording.segments}
                    ]
                if config.merge_attach_window > 0:
                    # fragments too short to embed have no centroid and would
                    # stay as one-utterance labels; attach them to the voice
                    # speaking right next to them (see speaker_merge)
                    clusters, per_segment = speaker_merge.attach_untimed_clusters(
                        recording.segments,
                        clusters,
                        per_segment,
                        window=config.merge_attach_window,
                        report=lambda line: self.library.log(line.strip()),
                    )
                # Auto-split: undoes a chained mega-cluster when it is really two
                # voices (same high barrier as the full pipeline; see speaker_merge)
                if config.merge_split_min_segments > 0:
                    from .. import speaker_merge as _speaker_merge_split
                    clusters, per_segment = _speaker_merge_split.split_inhomogeneous_clusters(
                        recording.segments,
                        clusters,
                        per_segment,
                        report=lambda line: self.library.log(line.strip()),
                        min_segments=config.merge_split_min_segments,
                        min_segment_sec=config.vp_min_segment_sec,
                    )
                # tidy the raw cluster ids (0, 71, 296...) into sequential
                # SPEAKER_00.. by first appearance; every label-keyed structure
                # must follow, otherwise stale keys survive in the record and in
                # the saved embeddings (the full pipeline remaps them too)
                label_map = align.renumber_speakers(recording.segments)
                if label_map:
                    recording.speaker_names = {
                        label_map.get(label, label): value
                        for label, value in recording.speaker_names.items()
                    }
                    recording.speaker_keep = [
                        label_map.get(label, label) for label in recording.speaker_keep
                    ]
                    recording.speaker_match = {
                        label_map.get(label, label): value
                        for label, value in recording.speaker_match.items()
                    }
                    clusters = {
                        label_map.get(label, label): vector
                        for label, vector in clusters.items()
                    }
                    per_segment = {
                        label_map.get(label, label): rows
                        for label, rows in per_segment.items()
                    }
                speaker_embed.save_record_embeddings(
                    config.embedding_dir, recording.id, clusters, per_segment
                )
                if config.match_voiceprints and clusters:
                    library = PeopleLibrary(config.vp_library_root)
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
            # do NOT set recording.model here: the speakers-only pass does not
            # run ASR, and the transcript may have been produced by a different
            # model — keeping the previous value keeps the header honest
            recording.speakers_diff = (
                f"{len(before_counts)} → {len(recording.speakers)}"
                f" · перенесено {self._moved_segments} "
                + _plural_ru(self._moved_segments, "реплика", "реплики", "реплик")
            )
            if self._dropped_names:
                shown = ", ".join(f"«{name}»" for name in self._dropped_names[:3])
                if len(self._dropped_names) > 3:
                    shown += f" и ещё {len(self._dropped_names) - 3}"
                recording.speakers_diff += (
                    f" · сброшены имена: {shown} — разбивка изменилась, назначьте заново"
                )
                self.library.log(
                    "speakers-only dropped names (разбивка изменилась): "
                    + ", ".join(self._dropped_names)
                )
            finish_backup(False)
            self.library.save(recording)
            self.library.log(
                f"done speakers-only {recording.filename}: "
                f"{len(recording.segments)} segments, {len(recording.speakers)} speakers"
            )
            self._maybe_notify(recording, "Готово: разбивка по спикерам")
        except JobCancelled:
            finish_backup(True)
            self._mark_cancelled(recording)
            self.library.log(f"stopped {recording.filename} (speakers-only)")
        except Exception as exc:  # noqa: BLE001 - reported in the UI
            finish_backup(True)
            recording.status = "error"
            recording.stage = "error"
            recording.error = str(exc)
            # keep the last fraction (see _run_job): "ошибка · N%" is more
            # useful than a blank bar
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
    merge_min_coh: Optional[float] = None
    merge_attach_window: Optional[float] = None
    merge_split_min_segments: Optional[int] = None
    audio_enhance: Optional[bool] = None
    diar_threshold: Optional[float] = None
    diar_min_duration_on: Optional[float] = None
    diar_min_duration_off: Optional[float] = None
    diar_block_seconds: Optional[int] = None
    diar_block_overlap: Optional[int] = None
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
    if chosen.get("merge_min_coh") is not None:
        config.merge_min_coh = float(chosen["merge_min_coh"])
    if chosen.get("merge_attach_window") is not None:
        config.merge_attach_window = float(chosen["merge_attach_window"])
    if chosen.get("merge_split_min_segments") is not None:
        config.merge_split_min_segments = int(chosen["merge_split_min_segments"])
    if chosen.get("audio_enhance") is not None:
        config.audio_enhance = as_bool(chosen.get("audio_enhance"), True)
    if chosen.get("diar_threshold") is not None:
        config.diar_threshold = float(chosen["diar_threshold"])
    if chosen.get("diar_min_duration_on") is not None:
        config.diar_min_duration_on = float(chosen["diar_min_duration_on"])
    if chosen.get("diar_min_duration_off") is not None:
        config.diar_min_duration_off = float(chosen["diar_min_duration_off"])
    if chosen.get("diar_block_seconds") is not None:
        config.diar_block_seconds = float(chosen["diar_block_seconds"])
    if chosen.get("diar_block_overlap") is not None:
        config.diar_block_overlap = float(chosen["diar_block_overlap"])
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


SPEAKERS_BACKUP_NAME = "speakers-backup.json"


def _speakers_backup_path(library: Library, record_id: str) -> Path:
    """Where a speakers-only run keeps the map needed for a one-click rollback."""
    return library.output_dir / record_id / SPEAKERS_BACKUP_NAME


def _structure_signature(segments) -> list:
    """Identity of the transcript *without* speaker labels.

    A rollback restores only ``segment.speaker``; text, ids and times must still
    be the ones the reprocessed transcript had, otherwise the old map no longer
    describes it (the user split, merged or edited rows).
    """
    return [
        [seg.id, seg.text, round(float(seg.start), 3), round(float(seg.end), 3)]
        for seg in segments
    ]


def _write_speakers_backup(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _load_speakers_backup(path: Path) -> Optional[Dict[str, object]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _notify_done(title: str, body: str) -> None:
    """Windows toast notification, best-effort (no external dependencies).

    Uses the WinRT toast API via an inline PowerShell script; any failure
    (old PowerShell, missing WinRT, policies) is silently ignored. Fired
    after a recording finishes when the user has not turned notifications
    off in settings."""
    try:
        # the values are dropped into single-quoted PowerShell literals: a
        # filename with a quote would otherwise break out and inject commands
        safe_title = str(title).replace("'", "''")
        safe_body = str(body).replace("'", "''")
        # ToastText02 ships with two *empty* <text> nodes, so the old
        # placeholders-must-be-replaced trick produced a blank toast (Windows
        # filled in its own "Новое сообщение" title). Filling the nodes by
        # index makes the text appear, and InnerText escapes XML for us.
        script = (
            "[Windows.UI.Notifications.ToastNotificationManager,"
            " Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
            "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
            "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
            "$nodes=$t.GetElementsByTagName('text');"
            f"$nodes.Item(0).InnerText='{safe_title}';"
            f"$nodes.Item(1).InnerText='{safe_body}';"
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
    pin = False   # matches SettingsStore.DEFAULTS; pinning is opt-in
    try:
        data = SettingsStore(library.root).as_public()
        priority = bool(data.get("process_priority", False))
        pin = bool(data.get("pin_performance", False))
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
        "diar_block_seconds": config.diar_block_seconds,
        "diar_block_overlap": config.diar_block_overlap,
        "audio_enhance": bool(config.audio_enhance),
        "rescue_holes": bool(config.rescue_holes),
        "glossary": bool(config.glossary),
        "match_voiceprints": bool(config.match_voiceprints),
        "merge_speakers": bool(config.merge_speakers),
        "merge_threshold": config.merge_threshold,
        "merge_min_coh": config.merge_min_coh,
        "merge_attach_window": config.merge_attach_window,
        "merge_split_min_segments": config.merge_split_min_segments,
        "chunk_seconds": config.chunk_seconds,
    }


# --- decoded waveform cache -------------------------------------------------

WAVEFORM_CACHE_NAME = "audio16k.wav"

# Building the cache in a child process keeps the window responsive: the
# decoder no longer competes with the HTTP server for the same GIL. Set
# NESTOR_DECODE_SUBPROCESS=0 to force the old in-process path.
DECODE_SUBPROCESS_ENV = "NESTOR_DECODE_SUBPROCESS"


def waveform_cache_path(library: Library, record_id: str) -> Path:
    """Where the decoded 16 kHz mono waveform of a recording is cached."""
    return library.output_dir / record_id / WAVEFORM_CACHE_NAME


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _read_progress(path: Path) -> Optional[float]:
    """Fraction written by the decoder child, or None while it is still silent."""
    try:
        text = path.read_text(encoding="ascii").strip()
        return float(text) if text else None
    except (OSError, ValueError):
        return None


def _decode_in_subprocess_enabled() -> bool:
    value = os.environ.get(DECODE_SUBPROCESS_ENV, "").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _decode_via_child(
    source: Path,
    dest: Path,
    on_progress=None,
) -> bool:
    """Decode ``source`` into ``dest`` in a separate process.

    Returns True when a non-empty file was written. Every failure — the
    platform cannot spawn, the child crashed, the file came out empty — returns
    False so the caller decodes in-process instead. Losing the responsive
    window is annoying; losing the job is not acceptable.

    The child reports progress through a small text file rather than a pipe:
    the result is a file anyway, so the only thing crossing the process border
    is a number between 0 and 1.
    """
    import multiprocessing

    progress = dest.with_name(dest.name + ".progress")
    _unlink(progress)
    try:
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=decode_source_to_cache,
            args=(str(source), str(dest), str(progress)),
            name="nestor-decode",
            daemon=True,
        )
        proc.start()
    except Exception:  # noqa: BLE001 - frozen/spawn may be unavailable
        _unlink(progress)
        return False

    reported = -1
    try:
        while proc.is_alive():
            proc.join(0.25)
            if on_progress is None:
                continue
            done = _read_progress(progress)
            if done is not None:
                percent = int(done * 100)
                if percent > reported:
                    reported = percent
                    on_progress(min(0.999, done))
    except BaseException:
        # The job was cancelled or the server is shutting down.
        proc.terminate()
        proc.join(5)
        _unlink(progress)
        raise

    code = proc.exitcode
    _unlink(progress)
    if code != 0 or not dest.exists() or dest.stat().st_size <= 44:
        _unlink(dest)
        return False
    if on_progress is not None:
        on_progress(1.0)
    return True


# Diarization runs in a child process for the same reason decoding does: while
# sherpa-onnx works it holds the GIL for long stretches (measured: single waits
# of up to 106 s), and the HTTP server the window talks to lives in this very
# process. Progress callbacks do not help - they run on the thread that holds
# the lock. Set NESTOR_DIARIZE_SUBPROCESS=0 to force the old in-process path.
DIARIZE_SUBPROCESS_ENV = "NESTOR_DIARIZE_SUBPROCESS"


def _diarize_in_subprocess_enabled() -> bool:
    value = os.environ.get(DIARIZE_SUBPROCESS_ENV, "").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _read_turns(path: Path) -> Optional[List]:
    """Speaker turns written by the diarization child, or None if unusable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, list):
        return None
    turns = []
    for item in payload:
        try:
            turns.append(
                Turn(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    speaker=str(item["speaker"]),
                )
            )
        except (TypeError, KeyError, ValueError):
            continue
    return turns


def diarize_via_child(
    library: Library,
    recording: Recording,
    config: Config,
    on_progress=None,
    cancel=None,
    enhance: Optional[bool] = None,
) -> Optional[List]:
    """Run diarization in a separate process; return turns or None.

    The child reads the cached 16 kHz wav and re-applies the same speech
    enhancement the pipeline would, so the parent does not have to hand over a
    multi-hundred-megabyte buffer across the process border - only the small
    list of speaker turns comes back.

    ``enhance`` overrides whether the child lifts the speech level first. The
    full pass enhances the audio before diarizing, the speakers-only pass
    always diarized the raw signal - so that mode asks for ``enhance=False``.
    ``None`` means "do what the pipeline config says".

    Every failure returns None so the caller falls back to the in-process path:
    a frozen window is annoying, a lost job is not acceptable. A cancellation
    request terminates the child and re-raises :class:`JobCancelled`.
    """
    if not _diarize_in_subprocess_enabled():
        return None
    try:
        from ..models import has_local_diarization

        if not has_local_diarization():
            return None
    except Exception:  # noqa: BLE001 - probe only, never fail the job
        return None

    cache = waveform_cache_path(library, recording.id)
    if not cache.exists():
        return None

    import multiprocessing

    from ..diarize_worker import diarize_cache_to_turns
    from ..cpuinfo import resolve_diar_threads

    turns_path = cache.with_name("diarize.turns.json")
    progress = cache.with_name("diarize.turns.json.progress")
    _unlink(turns_path)
    _unlink(progress)
    try:
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=diarize_cache_to_turns,
            args=(str(cache), str(turns_path), str(progress)),
            kwargs={
                "enhance": (
                    bool(config.audio_enhance) if enhance is None else bool(enhance)
                ),
                "num_speakers": int(config.num_speakers or 0),
                "num_threads": int(resolve_diar_threads(config.threads)),
                "threshold": float(getattr(config, "diar_threshold", 0.5)),
                "min_duration_on": float(getattr(config, "diar_min_duration_on", 0.5)),
                "min_duration_off": float(getattr(config, "diar_min_duration_off", 0.0)),
                "block_seconds": float(getattr(config, "diar_block_seconds", 0.0) or 0.0),
                "block_overlap": float(getattr(config, "diar_block_overlap", 60.0) or 0.0),
            },
            name="nestor-diarize",
            daemon=True,
        )
        proc.start()
    except Exception:  # noqa: BLE001 - frozen/spawn may be unavailable
        _unlink(progress)
        return None

    reported = -1
    cancelled = False
    try:
        while proc.is_alive():
            proc.join(0.25)
            if cancel is not None and cancel():
                cancelled = True
                break
            if on_progress is None:
                continue
            done = _read_progress(progress)
            if done is not None:
                percent = int(done * 100)
                if percent > reported:
                    reported = percent
                    on_progress(min(0.999, done))
    except BaseException:
        proc.terminate()
        proc.join(5)
        _unlink(progress)
        _unlink(turns_path)
        raise

    if cancelled:
        proc.terminate()
        proc.join(5)
        _unlink(progress)
        _unlink(turns_path)
        raise JobCancelled("stopped while diarizing")

    code = proc.exitcode
    _unlink(progress)
    if code != 0:
        _unlink(turns_path)
        return None
    turns = _read_turns(turns_path)
    _unlink(turns_path)
    if turns is None:
        return None
    if on_progress is not None:
        on_progress(1.0)
    return turns


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

    Decoding writes the wav itself, in a child process, and the waveform is
    read back from that file. Two things follow: the decoded audio never exists
    as one giant float32 array built up from hundreds of thousands of chunks,
    and the decode cannot freeze the window (own process, below normal
    priority). The first run also sees exactly the same samples as every later
    one, because both come from the cache.
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
    partial = cache.with_name(cache.name + ".part")
    built = False
    child_used = False
    try:
        child_used = _decode_in_subprocess_enabled() and _decode_via_child(
            source, partial, on_decode
        )
        built = child_used
    except Exception:  # noqa: BLE001 - the fast path must never fail a job
        built = False
        child_used = False

    if not built:
        try:
            decode_to_wav_16k(source, partial, on_progress=on_decode)
            built = True
        except Exception:  # noqa: BLE001 - last resort: decode into memory
            _unlink(partial)
            waveform = decode(source, on_progress=on_decode)
            try:
                write_wav(cache, waveform)
            except OSError:
                pass  # cache is an optimization, never a requirement
            return waveform, False

    try:
        partial.replace(cache)
        built_path = cache
    except OSError:
        # Could not move the file into place: read it where it lies.
        built_path = partial

    try:
        size_kb = built_path.stat().st_size // 1024
        library.log(
            f"decode {recording.id}: "
            + ("child process" if child_used else "in-process")
            + f", {size_kb} KB"
        )
    except OSError:
        pass

    try:
        import soundfile as sf

        waveform, sample_rate = sf.read(
            str(built_path), dtype="float32", always_2d=False
        )
        if sample_rate == SAMPLE_RATE and waveform.ndim == 1:
            if built_path != cache:
                _unlink(built_path)
            return waveform, False
    except Exception:  # noqa: BLE001 - fall back to the PyAV read
        pass
    waveform = decode(built_path)
    if built_path != cache:
        _unlink(built_path)
    return waveform, False


def rebuild_peaks(library: "Library", recording: "Recording") -> List[float]:
    """Recompute a recording's waveform points from its cached 16 kHz wav.

    Recordings made before the adaptive bucket count carry peaks from the old
    fixed 1200-point scheme; those make the zoomed waveform blocky, so a stale
    cache is replaced as soon as it is read. Returns ``[]`` when the wav cache
    is missing — peaks are cosmetic and must never fail a request.
    """
    cache = waveform_cache_path(library, recording.id)
    if not cache.exists():
        return []
    try:
        import soundfile as sf

        waveform, sample_rate = sf.read(cache, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - peaks are cosmetic
        return []
    if sample_rate != SAMPLE_RATE or waveform.ndim != 1 or waveform.size == 0:
        return []
    peaks = compute_peaks(waveform)
    try:
        library.save_peaks(recording.id, peaks)
    except OSError:
        pass
    return peaks


VP_CORE_SHARE = 0.7  # keep this share of the utterances closest to the draft
VP_CORE_MIN = 2      # ... but never fewer than this many


def _voiceprint_long_labels(recording: Recording) -> set:
    """Labels owning at least one segment long enough to build a voice."""
    return {
        seg.speaker for seg in recording.segments
        if float(seg.end) - float(seg.start) >= VP_MIN_SEGMENT_SEC
    }


def _robust_core_mean(vectors) -> tuple:
    """Mean of the utterance embeddings closest to their own centroid.

    A voiceprint built as the plain average of *all* of a speaker's segments is
    dragged towards whoever else got caught in the cluster: one overlapped, noisy
    or mis-assigned line is enough to move the portrait. So we take a draft
    centroid from everything, keep only the ``VP_CORE_SHARE`` closest utterances
    (at least ``VP_CORE_MIN`` of them) and average that core. Returns
    ``(vector, kept, total)``; ``vector`` is ``None`` when there is nothing to
    average, and ``total`` is the number of usable utterances.
    """
    rows = [
        np.asarray(vector, dtype=np.float64)
        for vector in vectors or []
        if vector is not None
    ]
    if not rows:
        return None, 0, 0
    total = len(rows)
    matrix = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 0.0] = 1.0
    matrix = matrix / norms
    if total == 1:
        return matrix[0], 1, 1
    draft = matrix.mean(axis=0)
    draft_norm = np.linalg.norm(draft)
    draft = draft / (draft_norm if draft_norm > 0 else 1.0)
    similarities = matrix @ draft
    keep = min(total, max(VP_CORE_MIN, int(round(total * VP_CORE_SHARE))))
    core = matrix[np.argsort(-similarities)[:keep]]
    centroid = core.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    centroid = centroid / (centroid_norm if centroid_norm > 0 else 1.0)
    return centroid, int(keep), total


def _voiceprint_vector(
    lib: Library,
    record_id: str,
    recording: Recording,
    speaker: str,
    segment_ids: Optional[list] = None,
) -> tuple:
    """Cluster embedding of one speaker, ``(vector, reason)``.

    Order of trust: the segment labels first, then the npz cache, then a
    fresh re-embedding from the source audio. The cache is used only while
    its labels cover every current long segment - after a speakers re-run
    the same label can belong to a different person, and a stale cache must
    never hand out that person's vector. ``reason`` explains a failure:
    "missing" (no such speaker), "too_short" (all his segments are shorter
    than the embedding minimum), "no_audio" (nothing to recompute from).

    The portrait is not the plain mean of the whole cluster: a robust core is
    kept (see :func:`_robust_core_mean`), so a noisy or mis-assigned line does
    not drag the saved voice. When the caller passes ``segment_ids`` (the user
    explicitly picked the samples), only those segments are used and no core
    is applied - an explicit choice is trusted as is. When a ``segment_ids``
    list leaves no usable segment the whole cluster is used instead, so a bad
    request never loses the voice.

    A rebuild embeds only the segments of the requested speaker. Recomputing
    the whole recording is quadratic to its length (a one-hour file spent
    ~45 s in decode + embedding) while the vector of one speaker needs only
    his own segments (~2 s), independent of the others. Such a partial
    result is not written back into the cache: the cache stays valid only
    when every long label is present.
    """
    from ..speaker_embed import (
        cluster_vectors,
        embed_segments,
        load_record_embeddings,
    )

    if not any(seg.speaker == speaker for seg in recording.segments):
        return None, "missing"
    long_labels = _voiceprint_long_labels(recording)
    if speaker not in long_labels:
        return None, "too_short"

    wanted = {str(sid) for sid in segment_ids} if segment_ids else None
    own = [seg for seg in recording.segments if seg.speaker == speaker]
    if wanted is not None:
        own = [seg for seg in own if str(getattr(seg, "id", "")) in wanted]

    def _explicit() -> bool:
        return wanted is not None and bool(own)

    # explicit samples from the cache: the cache stores per-utterance vectors in
    # segment order (short segments dropped), so we can map them back onto the
    # chosen segments directly
    clusters, per_segment = load_record_embeddings(lib.embeddings_dir, record_id)
    if clusters and long_labels <= set(clusters) and _explicit():
        rows = per_segment.get(speaker) or []
        chosen = _explicit_vectors(recording, speaker, wanted, rows)
        if chosen:
            mean = np.asarray(chosen, dtype=np.float64).mean(axis=0)
            norm = np.linalg.norm(mean)
            return (mean / (norm if norm > 0 else 1.0)), None
    if clusters and long_labels <= set(clusters) and not _explicit():
        # a trusting cache always has the label (it covers every long label)
        if speaker not in clusters:
            return None, "no_audio"
        core, _kept, _total = _robust_core_mean(per_segment.get(speaker))
        if core is not None:
            return core, None
        return clusters[speaker], None
    # no usable cache (old recording, missing, or a pass whose labels no
    # longer match the segments): rebuild just this speaker from the audio
    source = lib.recording_source(recording)
    if not source.exists():
        return None, "no_audio"
    try:
        waveform = decode(source)
        if _explicit():
            # embed the chosen samples one by one: a single pathological region
            # must not sink the whole portrait (onnxruntime throws on some)
            vectors = []
            for seg in own:
                try:
                    one = embed_segments(
                        waveform,
                        SAMPLE_RATE,
                        [(seg.start, seg.end)],
                        min_duration=VP_MIN_SEGMENT_SEC,
                    )
                except Exception:  # noqa: BLE001 - skip an unembeddable sample
                    continue
                if one and one[0] is not None:
                    vectors.append(one[0])
            if vectors:
                mean = np.asarray(vectors, dtype=np.float64).mean(axis=0)
                norm = np.linalg.norm(mean)
                return (mean / (norm if norm > 0 else 1.0)), None
            own = [seg for seg in recording.segments if seg.speaker == speaker]
        own_clusters, own_per_segment = cluster_vectors(
            waveform, SAMPLE_RATE, own, min_segment_sec=VP_MIN_SEGMENT_SEC,
        )
    except Exception:  # noqa: BLE001 - broken audio / model failure
        return None, "no_audio"
    core, _kept, _total = _robust_core_mean(own_per_segment.get(speaker))
    if core is not None:
        return core, None
    return own_clusters.get(speaker), None


def _explicit_vectors(recording: Recording, speaker: str, wanted: set, rows) -> list:
    """Cached per-utterance vectors of the chosen segments, in segment order.

    The cache stores one vector per segment *long enough to embed*, in the same
    order the segments appear, so walking the speaker's long segments and
    ticking the chosen ones rebuilds the mapping. Segments the cache skipped
    (too short) have no row and are simply not selectable.
    """
    chosen: list = []
    index = 0
    for seg in recording.segments:
        if seg.speaker != speaker:
            continue
        if float(seg.end) - float(seg.start) < VP_MIN_SEGMENT_SEC:
            continue
        if index >= len(rows):
            break
        if str(getattr(seg, "id", "")) in wanted:
            chosen.append(rows[index])
        index += 1
    return chosen


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
    people = PeopleLibrary(lib.root, voices)
    projects = ProjectLibrary(lib.root)

    if token:
        install_access_guard(app, token)

    app.state.library = lib
    app.state.processor = processor
    app.state.voiceprints = voices
    app.state.people = people
    app.state.projects = projects
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

    def ensure_video_probe(recording: Recording) -> None:
        """Learn once whether the original has a picture and where its audio starts.

        Opening the container costs a few milliseconds and the answer never
        changes, so it is cached on the record.  A running job writes the record
        itself; skip the save there and let the next view fill the fields in.
        """
        if recording.video_checked:
            return
        source = lib.recording_source(recording)
        has_video, offset = probe_video(source) if source.exists() else (False, 0.0)
        recording.has_video = has_video
        recording.media_offset = offset
        recording.video_checked = True
        if recording.status not in ("queued", "running"):
            lib.save(recording, create=False)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/watch.html", response_class=HTMLResponse)
    def watch_page() -> HTMLResponse:
        """Presenter screen: a separate window with the picture and subtitles."""
        return HTMLResponse((STATIC_DIR / "watch.html").read_text(encoding="utf-8"))

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
            "cpu_name": cpuinfo.cpu_name(),
            "ram_total_bytes": cpuinfo.total_ram_bytes(),
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
                "project": recording.project,
                "project_color": projects.color_for(recording.project),
                "edited_at": recording.edited_at,
                "done_at": recording.done_at,
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
        merge_attach_window: Optional[float] = Form(None),
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
        # seed from the global settings (like the import path does), then let
        # the form values override: otherwise an uploaded file silently ran
        # with Config() defaults (glossary off, merge off) while an imported
        # one honoured the user's settings — same file, two behaviours
        settings = app.state.settings.as_public()
        recording.job_settings = {
            "model": DEFAULT_ASR_MODEL,
            "diarize": bool(diarize),
            "backend": backend or "auto",
            "num_speakers": num_speakers if num_speakers is not None
            else (settings.get("num_speakers") or None),
            "match_voiceprints": match_voiceprints,
            "vp_abs_threshold": settings.get("vp_abs_threshold"),
            "vp_margin": settings.get("vp_margin"),
            "vp_segment_agreement": settings.get("vp_segment_agreement"),
            "merge_speakers": settings.get("merge_speakers", True),
            "merge_threshold": settings.get("merge_threshold"),
            "merge_min_weight": settings.get("merge_min_weight"),
            "merge_segment_floor": settings.get("merge_segment_floor"),
            "merge_min_coh": settings.get("merge_min_coh", 0.0),
            "merge_attach_window": settings.get("merge_attach_window", 2.0),
            "merge_split_min_segments": settings.get("merge_split_min_segments", 50),
            "audio_enhance": settings.get("audio_enhance", True),
            "diar_threshold": settings.get("diar_threshold"),
            "diar_min_duration_on": settings.get("diar_min_duration_on"),
            "diar_min_duration_off": settings.get("diar_min_duration_off"),
            "diar_block_seconds": settings.get("diar_block_seconds", 1200),
            "diar_block_overlap": settings.get("diar_block_overlap", 60),
            "rescue_holes": settings.get("rescue_holes", True),
            "glossary": settings.get("glossary", False),
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
        if isinstance(merge_attach_window, (int, float)) and not isinstance(
            merge_attach_window, bool
        ):
            recording.job_settings["merge_attach_window"] = float(merge_attach_window)
            config.merge_attach_window = float(merge_attach_window)
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
            "merge_min_coh": settings.get("merge_min_coh", 0.0),
            "merge_attach_window": settings.get("merge_attach_window", 2.0),
            "merge_split_min_segments": settings.get("merge_split_min_segments", 50),
            "audio_enhance": settings.get("audio_enhance", True),
            "diar_threshold": settings.get("diar_threshold"),
            "diar_min_duration_on": settings.get("diar_min_duration_on"),
            "diar_min_duration_off": settings.get("diar_min_duration_off"),
            "diar_block_seconds": settings.get("diar_block_seconds", 1200),
            "diar_block_overlap": settings.get("diar_block_overlap", 60),
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
        if mode != "speakers":
            # a full/transcribe pass rewrites the text, so an old speaker-only
            # rollback map no longer describes the transcript
            with contextlib.suppress(OSError):
                _speakers_backup_path(lib, record_id).unlink(missing_ok=True)

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
        ensure_video_probe(recording)
        payload = recording.to_dict()
        if recording.status == "running":
            elapsed = processor.elapsed_sec(recording.id)
            if elapsed is not None:
                payload["elapsed_sec"] = round(elapsed, 1)
        backup = _load_speakers_backup(_speakers_backup_path(lib, record_id))
        if backup and backup.get("after"):
            # a speakers-only run left a sidecar: the UI can mark changed rows
            # and offer a one-click return to the previous breakdown
            payload["reprocess_undo"] = {
                "available": bool(backup.get("before")),
                "failed": bool(backup.get("failed")),
                "created_at": backup.get("created_at"),
                "finished_at": backup.get("finished_at"),
                "before_speakers": backup.get("before_speakers"),
                "after_speakers": backup.get("after_speakers"),
                "moved": backup.get("moved"),
                "changed": list(backup.get("changed") or []),
                "dropped_names": list(backup.get("dropped_names") or []),
            }
        return payload

    @app.patch("/api/recordings/{record_id}/meta")
    def patch_recording_meta(
        record_id: str, body: Dict = Body(...)
    ) -> Dict[str, object]:
        """Title alias, notes, project and the review state of a recording.

        Allowed while a job runs: none of these fields touch segments, so a
        concurrent save from the worker cannot clobber them."""
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        title = str(body.get("title") or "").strip() if "title" in body else None
        notes = str(body.get("notes") or "").strip() if "notes" in body else None
        # merge onto the freshest copy: the worker writes segments every ~0.4 s
        # while a job runs, and saving our stale copy would drop its progress
        fresh = lib.load(record_id) or recording
        if "title" in body:
            fresh.title = title or None
        if "notes" in body:
            fresh.notes = notes or None
        if "project" in body:
            # empty string (or null) moves the recording out of any project
            raw = body.get("project")
            project_id = str(raw).strip() if raw else ""
            if project_id and projects.get(project_id) is None:
                raise HTTPException(status_code=404, detail="Проект не найден")
            fresh.project = project_id or None
        if "done" in body:
            # the manual "finished" mark; "not finished" drops it entirely, so
            # the state falls back to whether the transcript was ever edited
            if body.get("done"):
                fresh.done_at = fresh.done_at or _now_iso()
            else:
                fresh.done_at = None
        lib.save(fresh, create=False)
        return {
            "id": fresh.id,
            "title": fresh.title,
            "notes": fresh.notes,
            "project": fresh.project,
            "project_color": projects.color_for(fresh.project),
            "edited_at": fresh.edited_at,
            "done_at": fresh.done_at,
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
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="Ожидался объект JSON с полями записи"
            )

        old_composition = [(seg.id, seg.speaker) for seg in recording.segments]
        segments = payload.get("segments")
        if segments is not None:
            if not isinstance(segments, list):
                raise HTTPException(status_code=400, detail="«segments» должен быть списком")
            try:
                recording.segments = [Segment.from_dict(item) for item in segments]
            except (AttributeError, TypeError, ValueError) as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400, detail=f"Некорректная реплика: {exc}"
                )
        names = payload.get("speaker_names")
        if names is not None:
            if not isinstance(names, dict):
                raise HTTPException(
                    status_code=400, detail="«speaker_names» должен быть объектом"
                )
            recording.speaker_names = {str(k): str(v) for k, v in names.items()}
        keep = payload.get("speaker_keep")
        if keep is not None:
            if not isinstance(keep, list):
                raise HTTPException(
                    status_code=400, detail="«speaker_keep» должен быть списком"
                )
            recording.speaker_keep = [str(label) for label in keep]
        # the records table sends title/notes here too: accept them, otherwise
        # a rename via PUT silently did nothing
        if "title" in payload:
            title = str(payload.get("title") or "").strip()
            recording.title = title or None
        if "notes" in payload:
            notes = str(payload.get("notes") or "").strip()
            recording.notes = notes or None

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
        _mark_edited(recording, payload)
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
        recording.speaker_keep = [
            label for label in recording.speaker_keep if label != from_label
        ]
        # the embedding cache holds the old labels' means - drop it so the
        # next "remember voice" rebuilds the target's voice from all segments
        (lib.embeddings_dir / f"{record_id}.npz").unlink(missing_ok=True)
        lib.save(recording, create=False)
        lib.log(f"merge speakers {from_label} -> {into} ({moved} segments)")
        return recording.to_dict()

    @app.post("/api/recordings/{record_id}/speakers/undo")
    def undo_speakers(record_id: str) -> Dict[str, object]:
        """Restore the speaker map saved before the last speakers-only run.

        Only ``segment.speaker`` and the label-keyed metadata are restored;
        text, ids and times are never touched.  If the transcript itself was
        edited after the run (split/merged rows, changed text), the old map no
        longer matches and the request is refused instead of guessing.
        """
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        if recording.status in ("queued", "running"):
            raise HTTPException(
                status_code=409, detail="Идёт обработка — откатить разбивку нельзя"
            )
        backup_path = _speakers_backup_path(lib, record_id)
        backup = _load_speakers_backup(backup_path)
        before = backup.get("before") if backup else None
        if not isinstance(before, dict):
            raise HTTPException(
                status_code=404, detail="Нет сохранённой разбивки для отката"
            )
        expected = backup.get("structure_after") or backup.get("structure")
        if expected is not None and _structure_signature(recording.segments) != expected:
            raise HTTPException(
                status_code=409,
                detail=(
                    "После переобработки текст или границы реплик менялись — "
                    "автоматический откат перезапишет эти правки. "
                    "Отмените их через Ctrl+Z и попробуйте снова."
                ),
            )

        # Prefer the full snapshot: a speakers-only run can also split a
        # segment, and restoring only the speaker map would leave such rows
        # behind.  The structure check above guarantees no hand edits are lost.
        raw_segments = backup.get("segments")
        if isinstance(raw_segments, list) and raw_segments:
            recording.segments = [
                Segment.from_dict(item)
                for item in raw_segments
                if isinstance(item, dict)
            ]
        else:
            for segment in recording.segments:
                if segment.id in before:
                    segment.speaker = str(before[segment.id])
        live = {segment.speaker for segment in recording.segments}
        names = backup.get("names") if isinstance(backup.get("names"), dict) else {}
        keep = backup.get("keep") if isinstance(backup.get("keep"), list) else []
        match = backup.get("match") if isinstance(backup.get("match"), dict) else {}
        recording.speaker_names = {
            str(label): str(value) for label, value in names.items() if label in live
        }
        recording.speaker_keep = [str(label) for label in keep if label in live]
        recording.speaker_match = {
            str(label): value for label, value in match.items() if label in live
        }
        # the npz cache describes the discarded composition
        (lib.embeddings_dir / f"{record_id}.npz").unlink(missing_ok=True)
        recording.speakers_diff = "прежняя разбивка возвращена"
        recording.edited_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        recording.done_at = None
        lib.save(recording, create=False)
        with contextlib.suppress(OSError):
            backup_path.unlink(missing_ok=True)
        lib.log(f"speakers rollback {recording.filename}")
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
    def recording_media(record_id: str, request: Request, video: int = 0) -> Response:
        recording = lib.load(record_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        source = lib.recording_source(recording)
        if not source.exists():
            raise HTTPException(status_code=404, detail="Файл записи не найден")
        suffix = source.suffix.lower()
        media_type = MEDIA_TYPES.get(suffix, "application/octet-stream")
        if video:
            # The <video> element needs a video type even when the same file is
            # normally served as audio: Chromium rejects audio/webm + picture.
            ensure_video_probe(recording)
            if recording.has_video:
                media_type = VIDEO_MEDIA_TYPES.get(suffix, media_type)
            else:
                raise HTTPException(status_code=404, detail="В записи нет видео")
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
        peaks = lib.load_peaks(record_id)
        recording = lib.load(record_id)
        if recording is not None:
            expected = peaks_bucket_count(recording.duration or 0.0)
            if len(peaks) != expected:
                peaks = rebuild_peaks(lib, recording) or peaks
        return {"peaks": peaks}

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
        raw_aliases = body.get("aliases")
        if raw_aliases is None:
            raw_aliases = []
        elif isinstance(raw_aliases, str):
            raw_aliases = re.split(r"[,\n;]+", raw_aliases)
        elif isinstance(raw_aliases, dict):
            raw_aliases = list(raw_aliases.keys())   # a stray object: keep its keys
        elif not isinstance(raw_aliases, list):
            raise HTTPException(
                status_code=400,
                detail="«aliases» должен быть списком вариантов (или строкой через запятую)",
            )
        aliases = [str(a).strip() for a in raw_aliases if str(a).strip()]
        if not term:
            raise HTTPException(status_code=400, detail="Укажите термин")
        if len(term) > 120:
            raise HTTPException(status_code=400, detail="Слишком длинный термин (макс. 120 символов)")
        if any(len(a) > 120 for a in aliases):
            raise HTTPException(status_code=400, detail="Слишком длинный вариант (макс. 120 символов)")
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
            person = people.person_of_voice(meta.slug)
            items.append(
                {
                    "slug": meta.slug,
                    # the name shown everywhere is the person's, not the one the
                    # voice was once saved under
                    "name": person.display_name if person else meta.display_name,
                    "person_id": person.id if person else "",
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

        Body: ``{name, recording_id, speaker}``, optionally ``segment_ids``
        (the samples the user explicitly selected). Uses the cached embeddings
        (``embeddings/<id>.npz``); without a cache it falls back to decoding
        the source audio and re-embedding only that speaker's segments - a
        partial result (never the whole recording), so saving stays quick
        even for hour-long files. The saved portrait is the robust core of the
        speaker's utterances unless explicit samples were given. Decoding
        happens off the event loop.
        """
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}

        name = str(body.get("name") or "").strip()
        record_id = str(body.get("recording_id") or "").strip()
        speaker = str(body.get("speaker") or "").strip()
        raw_ids = body.get("segment_ids")
        segment_ids = (
            [str(value) for value in raw_ids]
            if isinstance(raw_ids, (list, tuple))
            else None
        )
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
            _voiceprint_vector, lib, record_id, recording, speaker, segment_ids
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
        chosen = f", {len(segment_ids)} sample(s)" if segment_ids else ""
        lib.log(
            f"voiceprint saved: {meta.display_name} "
            f"(from {recording.filename}, {speaker}{chosen})"
        )
        # every voice belongs to a person — a brand-new voice gets one right
        # away, so the library stays a list of people rather than of voices
        person = people.person_of_voice(meta.slug)
        if person is None:
            person = people.create(name, [meta.slug])
            lib.log(f"person added: {person.display_name} (from a saved voice)")
        return {
            "saved": person.display_name,
            "slug": meta.slug,
            "person_id": person.id,
        }

    @app.delete("/api/voiceprints/{slug}")
    def delete_voiceprint(slug: str) -> Dict[str, object]:
        meta = voices.find_by_slug(slug)
        name = meta.display_name if meta else slug
        if not voices.delete(slug):
            raise HTTPException(status_code=404, detail="Голос не найден")
        # the person who owned this voice may be left with nothing behind
        people.reconcile()
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
    def export_voiceprints(slug: Optional[List[str]] = Query(None)) -> Response:
        """Voices as one portable JSON file.

        ``?slug=a&slug=b`` exports only the chosen voices (the panel's
        checkbox selection); no query = the whole library (full backup).
        """
        chosen = [s for s in (slug or []) if s] or None
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
        # Names may have arrived with the voices: rebuild the people layer so a
        # person exists behind every voice.
        people.reconcile()
        return report

    # ----------------------------------------------------------------- people
    # People sit on top of the voice library: one displayed name plus the
    # spellings it is known by, and one or more voices behind it.  The voice
    # routes above stay as they are - matching works on voices, and this layer
    # only decides what the names look like.
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _mark_edited(recording: Recording, payload: Dict) -> None:
        """Remember that the transcript was hand-edited.

        A save carrying segments or speaker names is a real edit — the review
        state becomes "editing".  An already-finished recording silently
        returns to "editing", so marking it done is not a promise that can
        never be broken.  Saves that only carry title/notes/keep do not count.
        """
        if "segments" not in payload and "speaker_names" not in payload:
            return
        recording.edited_at = _now_iso()
        recording.done_at = None

    def _person_payload(person) -> Dict[str, object]:
        items = []
        utterances = 0
        total = 0.0
        for slug in person.voices:
            meta = people.voices.find_by_slug(slug)
            if meta is None:
                continue
            items.append(
                {
                    "slug": meta.slug,
                    "name": meta.display_name,
                    "comment": meta.comment,
                    "num_utterances": meta.num_utterances,
                    "total_duration_sec": round(meta.total_duration_sec, 1),
                }
            )
            utterances += meta.num_utterances
            total += meta.total_duration_sec
        return {
            "id": person.id,
            "name": person.display_name,
            "aliases": list(person.aliases),
            "voices": items,
            "comment": person.comment,
            "num_utterances": utterances,
            "total_duration_sec": round(total, 1),
            "created_at": person.created_at,
            "updated_at": person.updated_at,
        }

    def _resolve_voice(value: object) -> str:
        """A slug from either a slug or the name a voice is stored under."""
        text = str(value or "").strip()
        if not text:
            return ""
        meta = people.voices.find_by_slug(text) or people.voices.find(text)
        return meta.slug if meta is not None else ""

    def _rename_in_recordings(spellings: List[str], new_name: str) -> int:
        """Rewrite the recordings that were named by the voice library.

        Only labels whose name came from voice matching are touched, so a name
        typed by hand into a recording stays exactly as the user left it.
        """
        wanted = {name.casefold() for name in spellings if name}
        if not wanted:
            return 0
        changed = 0
        for recording in lib.list():
            touched = False
            for label, match in list((recording.speaker_match or {}).items()):
                assigned = str((match or {}).get("assigned_name") or "")
                if not assigned or assigned.casefold() not in wanted:
                    continue
                updated = dict(match)
                updated["assigned_name"] = new_name
                recording.speaker_match[label] = updated
                if recording.speaker_names.get(label):
                    recording.speaker_names[label] = new_name
                touched = True
            if not touched:
                continue
            try:
                lib.save(recording, create=False)
                changed += 1
            except OSError:
                pass
        return changed

    @app.get("/api/people")
    def list_people() -> Dict[str, object]:
        return {"people": [_person_payload(person) for person in people.list()]}

    @app.get("/api/people/similar")
    def similar_person(name: str = "") -> Dict[str, object]:
        """Who a name read off the video probably belongs to, if anyone."""
        found = people.similar_person(name.strip())
        if found is None:
            return {"person": None, "spelling": ""}
        person, spelling = found
        return {"person": _person_payload(person), "spelling": spelling}

    @app.post("/api/people")
    async def add_person(request: Request) -> Dict[str, object]:
        """Create a person: a name, optionally with voices attached at once."""
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Укажите имя человека")
        raw = body.get("voices")
        slugs = [_resolve_voice(item) for item in raw] if isinstance(raw, (list, tuple)) else []
        try:
            person = people.create(name, [slug for slug in slugs if slug])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        lib.log(f"person added: {person.display_name}")
        return _person_payload(person)

    @app.patch("/api/people/{person_id}")
    async def edit_person(person_id: str, request: Request) -> Dict[str, object]:
        """Rename a person, pick which spelling is shown, or write a note.

        ``name`` renames and carries the new name into the recordings the voice
        library had named.  ``display`` only chooses among spellings already
        known, so it never touches the recordings.
        """
        person = people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Человек не найден")
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}

        changed = 0
        if "name" in body:
            name = str(body.get("name") or "").strip()
            if not name:
                raise HTTPException(status_code=400, detail="Укажите имя человека")
            spellings = person.spellings()
            person = people.rename(person_id, name)
            if person is None:
                raise HTTPException(status_code=404, detail="Человек не найден")
            changed = _rename_in_recordings(spellings, person.display_name)
            lib.log(f"person renamed: {name} ({changed} recordings)")
        if "display" in body:
            display = str(body.get("display") or "").strip()
            if display and people.set_display(person_id, display) is None:
                raise HTTPException(status_code=400, detail="Такое написание не известно")
            person = people.get(person_id)
        if "comment" in body:
            people.set_comment(person_id, str(body.get("comment") or "").strip())
            person = people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Человек не найден")
        payload = _person_payload(person)
        payload["changed"] = changed
        return payload

    @app.post("/api/people/{person_id}/aliases")
    async def edit_person_aliases(person_id: str, request: Request) -> Dict[str, object]:
        """Add another spelling of the name, or drop one."""
        if people.get(person_id) is None:
            raise HTTPException(status_code=404, detail="Человек не найден")
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}
        alias = str(body.get("alias") or "").strip()
        if not alias:
            raise HTTPException(status_code=400, detail="Пустое написание")
        if body.get("remove"):
            people.remove_alias(person_id, alias)
        else:
            people.add_alias(person_id, alias)
        person = people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Человек не найден")
        return _person_payload(person)

    @app.post("/api/people/{person_id}/voices")
    async def edit_person_voices(person_id: str, request: Request) -> Dict[str, object]:
        """Attach a voice to a person, detach one, or merge two of them.

        ``merge`` folds the voice into the person's own portrait (the one story
        per person the matcher wants); ``detach`` unhooks it again; otherwise
        the voice is just attached.
        """
        person = people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Человек не найден")
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}
        slug = _resolve_voice(body.get("voice"))
        if not slug:
            raise HTTPException(status_code=404, detail="Голос не найден")
        if body.get("detach"):
            people.detach_voice(person_id, slug)
            # the freed voice belongs to somebody: reconcile gives it a person
            # of its own again, so nothing is silently lost
            people.reconcile()
            person = people.get(person_id)
            if person is None:
                return {"id": person_id, "name": "", "aliases": [], "voices": [],
                        "comment": "", "num_utterances": 0, "total_duration_sec": 0.0,
                        "created_at": "", "updated_at": "", "gone": True}
            return _person_payload(person)
        if body.get("merge"):
            source = people.voices.find_by_slug(slug)
            target_slug = next(
                (item for item in person.voices if item != slug), ""
            )
            target = people.voices.find_by_slug(target_slug) if target_slug else None
            if source is None or target is None:
                raise HTTPException(
                    status_code=409, detail="Не с чем склеивать — у человека нет другого голоса"
                )
            people.voices.rename(source.display_name, target.display_name)
            lib.log(f"voice merged: {source.display_name} -> {target.display_name}")
            people.reconcile()
        else:
            if people.attach_voice(person_id, slug) is None:
                raise HTTPException(status_code=404, detail="Человек не найден")
            people.reconcile()
            people.save()
        person = people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Человек не найден")
        return _person_payload(person)

    @app.post("/api/people/merge")
    async def merge_people(request: Request) -> Dict[str, object]:
        """Fold one person into another: spellings, voices and recording names."""
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}
        source = people.get(str(body.get("source") or "").strip())
        target = people.get(str(body.get("target") or "").strip())
        if source is None or target is None:
            raise HTTPException(status_code=404, detail="Человек не найден")
        if source.id == target.id:
            raise HTTPException(status_code=400, detail="Это один и тот же человек")
        spellings = source.spellings() + target.spellings()
        merged = people.merge(source.id, target.id)
        if merged is None:
            raise HTTPException(status_code=404, detail="Человек не найден")
        changed = _rename_in_recordings(spellings, merged.display_name)
        lib.log(
            f"person merged: {source.display_name} -> {merged.display_name} "
            f"({changed} recordings)"
        )
        payload = _person_payload(merged)
        payload["changed"] = changed
        return payload

    @app.delete("/api/people/{person_id}")
    def remove_person(person_id: str) -> Dict[str, object]:
        """Forget a person together with the voices only they carried."""
        if not people.delete(person_id):
            raise HTTPException(status_code=404, detail="Человек не найден")
        lib.log(f"person deleted: {person_id}")
        return {"deleted": person_id}

    # ------------------------------------------------------------- projects
    def _project_payload(project) -> Dict[str, object]:
        count = sum(1 for rec in lib.list() if rec.project == project.id)
        return {
            "id": project.id,
            "name": project.name,
            "color": project.color,
            "count": count,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
        }

    @app.get("/api/projects")
    def list_projects() -> Dict[str, object]:
        """All projects with how many recordings each one holds."""
        return {"projects": [_project_payload(project) for project in projects.list()]}

    @app.post("/api/projects")
    async def add_project(request: Request) -> Dict[str, object]:
        """Create a project (folder).  The same name returns the existing one."""
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Укажите название проекта")
        try:
            project = projects.create(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        lib.log(f"project created: {project.name}")
        return _project_payload(project)

    @app.patch("/api/projects/{project_id}")
    async def edit_project(project_id: str, request: Request) -> Dict[str, object]:
        """Rename a project or change its colour; recordings stay in it."""
        if projects.get(project_id) is None:
            raise HTTPException(status_code=404, detail="Проект не найден")
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}")
        body: Dict[str, object] = body if isinstance(body, dict) else {}
        project = projects.get(project_id)
        if "name" in body:
            name = str(body.get("name") or "").strip()
            if not name:
                raise HTTPException(status_code=400, detail="Укажите название проекта")
            project = projects.rename(project_id, name)
            lib.log(f"project renamed: {name}")
        if "color" in body and project is not None:
            project = projects.set_color(project_id, str(body.get("color") or ""))
        if project is None:
            raise HTTPException(status_code=404, detail="Проект не найден")
        return _project_payload(project)

    @app.delete("/api/projects/{project_id}")
    def remove_project(project_id: str, move_to: str = "") -> Dict[str, object]:
        """Delete a project.  Its recordings are unlinked (never deleted).

        With ``move_to`` they are re-homed to another project instead of
        landing in "uncategorized"."""
        if projects.get(project_id) is None:
            raise HTTPException(status_code=404, detail="Проект не найден")
        target = str(move_to or "").strip()
        if target and projects.get(target) is None:
            raise HTTPException(status_code=404, detail="Проект для переноса не найден")
        moved = 0
        for recording in lib.list():
            if recording.project == project_id:
                recording.project = target or None
                lib.save(recording, create=False)
                moved += 1
        projects.delete(project_id)
        lib.log(f"project deleted: {project_id} ({moved} recordings moved)")
        return {"deleted": project_id, "moved": moved}

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def _reload_app() -> FastAPI:
    """Factory used by uvicorn ``--reload``: keep the access guard on.

    ``uvicorn --reload`` cannot take an app instance (it must import a factory
    string), so the session token generated by :func:`run_server` is handed over
    through the environment. Without this the dev reload path served the app
    with no token check at all.
    """
    token = os.environ.get(_TOKEN_ENV) or None
    work_dir = os.environ.get(_WORKDIR_ENV) or None
    return create_app(work_dir=Path(work_dir) if work_dir else None, token=token)


def run_server(
    port: int = 8000,
    work_dir: Optional[Path] = None,
    reload: bool = False,
) -> None:
    """Run the web service with uvicorn (loopback only, with an access token)."""
    import uvicorn

    token = generate_session_token()
    if reload:
        # dev mode: uvicorn reloads via a factory import string, so pass the
        # token/work dir through the environment — the guard stays on
        os.environ[_TOKEN_ENV] = token
        if work_dir:
            os.environ[_WORKDIR_ENV] = str(work_dir)
        print(
            f"Nestor is running on http://127.0.0.1:{port}/?token={token}",
            flush=True,
        )
        uvicorn.run(
            "nestor.web.app:_reload_app", factory=True,
            host="127.0.0.1", port=port, reload=True,
        )
        return

    app = create_app(work_dir=work_dir, token=token)
    print(
        f"Nestor is running on http://127.0.0.1:{port}/?token={token}",
        flush=True,
    )
    uvicorn.run(app, host="127.0.0.1", port=port)
