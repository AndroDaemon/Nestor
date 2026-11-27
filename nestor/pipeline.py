"""End-to-end pipeline: audio -> diarization -> ASR -> speaker labelled transcript."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from . import align, audio
from . import speaker_embed, speaker_match, speaker_merge
from .asr_onnx import SherpaOnnxAsr
from .config import Config
from .constants import SAMPLE_RATE
from .cpuinfo import resolve_diar_threads, resolve_threads
from .diarize import SingleSpeakerDiarizer, _accepts_blocking, _accepts_progress, build_diarizer
from .models import ensure_gigaam_punct
from .types import Segment, Transcript, Turn, Word

ProgressFn = Callable[[str], None]
StageFn = Callable[[str, float], None]
CancelFn = Callable[[], bool]

# Rescue-pass constants (Sloveya-tuned): holes shorter than RESCUE_MIN_GAP are
# normal pauses, longer than RESCUE_MAX_HOLE are true silence, and each hole is
# re-decoded with RESCUE_CONTEXT seconds of speech on both sides.
RESCUE_MIN_GAP = 2.0
RESCUE_MAX_HOLE = 20.0
RESCUE_CONTEXT = 0.7

# A hole is worth re-decoding only when something is audible in it. Measurements
# on hour-long meeting recordings showed that the loudest 20 ms frame of every
# hole sits 6-20 dB above the room's own noise floor, and that the floor itself
# is what the first ASR pass leaves unclaimed; a stretch that stays within a few
# dB of the floor is not missed speech but breathing and paper rustle. The
# threshold is deliberately generous - a distant participant who stepped away
# from the microphone still clears it - so the pass keeps finding quiet speech
# while dropping the pure-silence calls that cost seconds each.
RESCUE_FLOOR_DB = 6.0
RESCUE_FRAME_SECONDS = 0.02


class JobCancelled(Exception):
    """The user stopped the job before anything had been transcribed."""


def _noop(_message: str) -> None:
    return None


# Heavy models are cached here and reused across jobs so a batch of files pays
# the load cost only once. The Processor runs one job at a time (max_workers=1);
# the per-key locks keep shared access safe if that ever changes.
_ASR_CACHE: dict = {}
_DIARIZER_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _asr_for(config: Config, report: ProgressFn, on_load: Optional[StageFn] = None):
    """Return ``(asr, lock, loaded)`` for this config, loading on first use."""
    threads = resolve_threads(config.threads)
    key = threads
    with _CACHE_LOCK:
        entry = _ASR_CACHE.get(key)
        if entry is not None:
            return entry[0], entry[1], False
        if on_load is not None:
            # the first transcription loads the model (10-20 s); let the UI say
            # so instead of showing a stalled "transcription" percentage
            on_load()
        report(f"Loading GigaAM punct model ({config.model_name}) ...")
        model_files = ensure_gigaam_punct()
        entry = (
            SherpaOnnxAsr(model_files, threads=threads),
            threading.Lock(),
        )
        _ASR_CACHE[key] = entry
        return entry[0], entry[1], True


def _frame_db(waveform: "np.ndarray") -> "np.ndarray":
    """Per-frame level of the recording in dBFS (one value per 20 ms frame).

    Reduced in blocks straight to a small per-frame vector, so a multi-hour
    recording does not need a second full-length array.
    """
    frame = max(1, int(RESCUE_FRAME_SECONDS * SAMPLE_RATE))
    n_frames = waveform.shape[0] // frame
    if n_frames == 0:
        return np.empty(0, dtype=np.float32)
    db = np.empty(n_frames, dtype=np.float32)
    block_frames = max(1, (1 << 16) // frame)
    for first in range(0, n_frames, block_frames):
        last = min(n_frames, first + block_frames)
        rows = np.asarray(
            waveform[first * frame:last * frame], dtype=np.float32
        ).reshape(last - first, frame)
        energy = np.einsum("ij,ij->i", rows, rows, dtype=np.float64)
        db[first:last] = np.sqrt(energy / frame)
    np.log10(np.maximum(db, 1e-6), out=db)
    db *= 20.0
    return db


def _noise_floor_db(db: "np.ndarray") -> float:
    """Level of the room's own noise: a low percentile of frame levels."""
    if db.size == 0:
        return -120.0
    return float(np.percentile(db, 10))


def _hole_is_audible(db: "np.ndarray", floor_db: float,
                     gap_start: float, gap_end: float) -> bool:
    """True when the loudest frame inside the hole rises above the room floor.

    The edges are trimmed by ``RESCUE_CONTEXT``, otherwise the tail of the word
    that borders the hole would answer for the hole itself.
    """
    frame = RESCUE_FRAME_SECONDS
    first = int((gap_start + RESCUE_CONTEXT) / frame)
    last = int((gap_end - RESCUE_CONTEXT) / frame)
    first = max(0, min(first, db.size - 1))
    last = max(first + 1, min(last, db.size))
    if last <= first:
        return False
    peak = float(np.max(db[first:last]))
    return peak - floor_db >= RESCUE_FLOOR_DB


def _rescue_holes(
    waveform: "np.ndarray",
    words: List[Word],
    asr,
    report: ProgressFn,
    cancelled: CancelFn,
    stage: StageFn,
) -> List[Word]:
    """Re-transcribe silent holes between recognised words.

    Mirrors Sloveya's rescue pass: gaps longer than ``rescue_min_gap`` but
    shorter than ``rescue_max_hole`` seconds between consecutive words may
    hide quiet speech the first pass missed. Each hole is re-decoded with a
    bit of context on both sides; anything new found there is spliced in.

    Holes that stay at the recording's own noise floor are skipped: they are
    pauses, not missed speech, and re-decoding them only invites the model to
    invent words (and to pull in background talk from deep in the room).
    """
    holes: List[tuple] = []
    for prev_word, next_word in zip(words, words[1:]):
        gap_start = prev_word.end
        gap_end = next_word.start
        if gap_end - gap_start < RESCUE_MIN_GAP:
            continue
        if gap_end - gap_start > RESCUE_MAX_HOLE:
            continue
        holes.append((gap_start, gap_end))
    if not holes:
        return words

    db = _frame_db(waveform)
    floor_db = _noise_floor_db(db)
    audible = [
        (gap_start, gap_end)
        for gap_start, gap_end in holes
        if _hole_is_audible(db, floor_db, gap_start, gap_end)
    ]
    dropped = len(holes) - len(audible)
    if dropped:
        report(
            f"  noise floor {floor_db:.1f} dB: skipped {dropped} hole(s) "
            f"as pure silence"
        )
    holes = audible
    if not holes:
        return words

    duration = waveform.shape[0] / float(SAMPLE_RATE)
    rescued: List[Word] = []
    report(f"  rescuing {len(holes)} silent holes ...")
    for index, (gap_start, gap_end) in enumerate(holes):
        if cancelled():
            break
        start = max(0.0, gap_start - RESCUE_CONTEXT)
        end = min(duration, gap_end + RESCUE_CONTEXT)
        extra = asr.transcribe(waveform[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)], offset=start)
        kept = [
            word
            for word in extra
            if word.start >= gap_start - 0.05 and word.end <= gap_end + 0.05
        ]
        if kept:
            rescued.extend(kept)
            report(
                f"    hole {index + 1}/{len(holes)} "
                f"[{gap_start:.1f}-{gap_end:.1f}s]: +{len(kept)} words"
            )
        stage("transcription", 0.9 + 0.05 * (index + 1) / len(holes))

    if not rescued:
        return words

    merged = sorted(words + rescued, key=lambda w: (w.start, w.end))
    deduped: List[Word] = []
    for word in merged:
        if deduped and word.start < deduped[-1].end - 0.05 and word.text == deduped[-1].text:
            continue  # context re-decode repeated a neighbouring word
        deduped.append(word)
    return deduped


def _diarizer_for(config: Config, report: ProgressFn, on_load: Optional[StageFn] = None):
    key = (
        bool(config.diarize),
        (config.diarization_backend or "auto").strip().lower(),
        config.num_speakers,
        resolve_diar_threads(config.threads),
    )
    with _CACHE_LOCK:
        entry = _DIARIZER_CACHE.get(key)
        if entry is not None:
            return entry[0], entry[1], False
        if on_load is not None:
            on_load()
        report("Preparing the diarization model ...")
        entry = (build_diarizer(config, report=report), threading.Lock())
        _DIARIZER_CACHE[key] = entry
        return entry[0], entry[1], True


def process_file(
    config: Config,
    input_path: Union[str, Path],
    progress: Optional[ProgressFn] = None,
    on_stage: Optional[StageFn] = None,
    waveform: Optional["np.ndarray"] = None,
    cancel: Optional[CancelFn] = None,
    reuse_waveform: bool = False,
    turns: Optional[List[Turn]] = None,
) -> Transcript:
    """Transcribe and diarize a single audio file.

    Parameters
    ----------
    config:
        Runtime configuration.
    input_path:
        Path to the audio file.
    progress:
        Optional callback used for status messages.
    on_stage:
        Optional callback receiving ``(stage, fraction)`` where fraction is the
        overall progress in the ``0..1`` range. Used by the web UI.
    waveform:
        Already decoded 16 kHz mono waveform. When given, the file is not
        decoded again.
    cancel:
        Optional predicate polled between checkpoints. When it turns true the
        job stops: before the first chunk this raises :class:`JobCancelled`,
        during transcription the already transcribed chunks are kept and the
        transcript comes back partial.
    reuse_waveform:
        The caller hands over ownership of ``waveform`` and promises not to
        read it afterwards. Audio enhancement may then filter the buffer in
        place instead of allocating a second full-length copy — the difference
        between one and two copies of a multi-hour recording in memory.

    Raises
    ------
    JobCancelled
        The job was stopped before any chunk was transcribed.
    """
    report: ProgressFn = progress or _noop
    input_path = Path(input_path)
    started = time.perf_counter()

    def stage(name: str, fraction: float) -> None:
        if on_stage is not None:
            on_stage(name, fraction)

    def cancelled() -> bool:
        return bool(cancel and cancel())

    report(f"Decoding {input_path.name} ...")
    stage("decode", 0.02)
    if waveform is None:
        waveform = audio.decode(input_path)
        decoded_here = True
    else:
        decoded_here = False
    total_duration = audio.duration(waveform)
    report(f"  {total_duration:.1f}s of audio at {SAMPLE_RATE} Hz mono")

    if config.audio_enhance and waveform.size:
        from .audio_io import enhance_speech

        # Filter in place only for a waveform this run owns: decoded here, or
        # handed over by a caller that gave up ownership. A buffer that the
        # caller still needs must survive the call. Saves one full-length copy,
        # which is what made multi-hour files run out of memory.
        waveform, level_db, mode = enhance_speech(
            waveform, SAMPLE_RATE, inplace=decoded_here or reuse_waveform
        )
        report(f"  audio enhanced: speech level {level_db:.1f} dB ({mode})")

    if cancelled():
        raise JobCancelled("stopped while decoding")

    # ``turns=None`` means "the caller did not diarize"; an empty list means
    # "the caller did and got nothing" - keep the difference, or a silent
    # recording would build the heavy model here and freeze the window again.
    precomputed = turns is not None
    turns = list(turns) if precomputed else []
    diarization_model = None
    if config.diarize and precomputed:
        # Caller already ran diarization (in a child process, to keep the window
        # responsive) and handed the turns over. Skip the local pass entirely -
        # and do not build the diarizer here, the whole point is to keep the
        # heavy model out of this process.
        from .diarize_local import LOCAL_DIARIZATION_MODEL_ID

        stage("diarization", 0.07)
        diarization_model = LOCAL_DIARIZATION_MODEL_ID
        report(f"  {len(turns)} speaker turns (precomputed)")
    elif config.diarize:
        stage("diarization", 0.06)
        diarizer, diarizer_lock, _loaded = _diarizer_for(
            config, report, on_load=lambda: stage("loading", 0.06)
        )
        diarization_model = diarizer.model_id
        report("  running diarization ...")
        stage("diarization", 0.07)

        def diarization_progress(fraction: float):
            # sherpa-onnx reports chunk progress; map it into the 0.07..0.25
            # band so the indicator keeps moving during long files. Diarization
            # is roughly a fifth of a full run, so it owns that much of the bar
            # instead of the sliver it used to (0.12..0.19). Returning a truthy
            # value aborts the diarization run itself.
            stage("diarization", 0.07 + 0.18 * max(0.0, min(1.0, fraction)))
            return cancelled()

        with diarizer_lock:
            if _accepts_blocking(diarizer):
                turns = diarizer.diarize(
                    waveform,
                    SAMPLE_RATE,
                    progress=diarization_progress,
                    block_seconds=float(getattr(config, "diar_block_seconds", 0.0) or 0.0),
                    block_overlap=float(getattr(config, "diar_block_overlap", 60.0) or 0.0),
                )
            elif _accepts_progress(diarizer):
                turns = diarizer.diarize(
                    waveform, SAMPLE_RATE, progress=diarization_progress
                )
            else:
                turns = diarizer.diarize(waveform, SAMPLE_RATE)
        report(f"  {len(turns)} speaker turns")
    else:
        diarizer = SingleSpeakerDiarizer()
        turns = diarizer.diarize(waveform, SAMPLE_RATE)
    # diarization is one long call; the stop flag is honoured either inside it
    # (the progress callback aborts the run) or right after it returns
    if cancelled():
        raise JobCancelled("stopped while diarizing")

    chunks = align.build_chunks(
        turns,
        max_chunk=config.chunk_seconds,
        max_gap=config.chunk_max_gap,
        waveform=waveform,
        sample_rate=SAMPLE_RATE,
    )
    report(f"  {len(chunks)} ASR chunks")

    words: List[Word] = []
    stopped = False
    last_index = 0
    if chunks:
        stage("transcription", 0.25)
        asr, asr_lock, _loaded = _asr_for(
            config, report, on_load=lambda: stage("loading", 0.25)
        )
        with asr_lock:
            for index, (start, end) in enumerate(chunks, start=1):
                if cancelled():
                    stopped = True
                    break
                chunk_words = asr.transcribe_chunk(
                    waveform,
                    start,
                    end,
                    word_timestamps=config.word_timestamps,
                )
                words.extend(chunk_words)
                last_index = index
                stage("transcription", 0.25 + 0.65 * index / len(chunks))
                report(
                    f"  chunk {index}/{len(chunks)} "
                    f"[{start:.1f}-{end:.1f}s]: {len(chunk_words)} words"
                )

    if stopped:
        report(f"  stopped by the user after chunk {last_index}/{len(chunks)}")
    elif config.rescue_holes and words:
        words = _rescue_holes(waveform, words, asr, report, cancelled, stage)

    words = align.assign_speakers(words, turns)
    segments: List[Segment] = align.merge_words(words)

    # Glossary post-correction: rewrite heard forms ("докер", "днс") to the
    # canonical term ("Docker", "DNS"). Pure text, offline, no decoder involved.
    if config.glossary and segments:
        from .glossary import (
            apply_to_segments,
            load_glossary,
            resolve_glossary_path,
        )
        from .library import default_root

        path = (
            config.glossary_path
            if config.glossary_path is not None
            else resolve_glossary_path(default_root())
        )
        changed = apply_to_segments(segments, load_glossary(path))
        if changed:
            report(f"  glossary: {changed} term(s) corrected")

    align.relabel_speakers(segments, config.speaker_labels)

    # Voiceprints: cache the embeddings and (optionally) match the library.
    # Segments keep their raw SPEAKER_NN labels - the matcher only maps them
    # to names via the speaker_names layer, never into segment.speaker.
    match_info: dict = {}
    match_enabled = bool(config.match_voiceprints) and bool(config.diarize)
    embeddings_wanted = (
        match_enabled
        or bool(config.save_embeddings)
        or bool(config.merge_speakers)
    ) and bool(config.diarize)
    if segments and (embeddings_wanted or match_enabled):
        record_id = config.record_id or Path(config.output_dir).name
        embedding_dir = config.embedding_dir or Path(config.output_dir) / "embeddings"
        try:
            stage("voiceprints", 0.95)
            clusters, per_segment = speaker_embed.cluster_vectors(
                waveform,
                SAMPLE_RATE,
                segments,
                min_segment_sec=config.vp_min_segment_sec,
            )
            if config.merge_speakers and len(clusters) > 1:
                clusters, per_segment = speaker_merge.apply_to_segments(
                    segments,
                    clusters,
                    per_segment,
                    threshold=config.merge_threshold,
                    min_weight=config.merge_min_weight,
                    segment_floor=config.merge_segment_floor,
                    min_coh=config.merge_min_coh,
                    report=report,
                )
            # Phantom attach: a label whose every utterance is shorter than the
            # embedding minimum has no centroid, so the similarity merge cannot
            # see it at all. Attach those fragments to the voice speaking right
            # next to them (nothing to compare by cosine - see speaker_merge).
            if config.merge_attach_window > 0:
                clusters, per_segment = speaker_merge.attach_untimed_clusters(
                    segments,
                    clusters,
                    per_segment,
                    window=config.merge_attach_window,
                    report=report,
                )
            # Auto-split: the merge pass above can chain a continuum of voices
            # into one huge cluster. Re-check the (now merged) clusters and cut
            # back the ones that are really two voices - the barrier is high, so
            # a normal speaker is never carved up (see speaker_merge).
            if config.merge_split_min_segments > 0:
                clusters, per_segment = speaker_merge.split_inhomogeneous_clusters(
                    segments,
                    clusters,
                    per_segment,
                    report=report,
                    min_segments=config.merge_split_min_segments,
                    min_segment_sec=config.vp_min_segment_sec,
                )
            # tidy the raw cluster ids (0, 71, 296...) into sequential
            # SPEAKER_00.. by first appearance, before names/matching/embeddings
            label_map = align.renumber_speakers(segments)
            if label_map:
                clusters = {
                    label_map.get(label, label): vector
                    for label, vector in clusters.items()
                }
                per_segment = {
                    label_map.get(label, label): rows
                    for label, rows in per_segment.items()
                }
            if embeddings_wanted and record_id:
                speaker_embed.save_record_embeddings(
                    embedding_dir,
                    record_id,
                    clusters,
                    per_segment,
                )
            if match_enabled and clusters:
                stage("voiceprints", 0.96)
                from .people import PeopleLibrary

                library = PeopleLibrary(config.vp_library_root)
                thresholds = speaker_match.MatchThresholds(
                    abs_threshold=config.vp_abs_threshold,
                    margin=config.vp_margin,
                    segment_agreement=config.vp_segment_agreement,
                )
                matches = speaker_match.match_clusters(
                    clusters,
                    library,
                    thresholds=thresholds,
                    segment_embeddings=per_segment,
                )
                match_info = {
                    label: result.to_dict() for label, result in matches.items()
                }
                applied = ", ".join(
                    f"{label} -> {result['assigned_name']} ({result['confidence']:.2f})"
                    if result["assigned_name"]
                    else f"{label} -> (не узнан)"
                    for label, result in sorted(match_info.items())
                )
                report(f"  [voiceprints] {applied}")
        except JobCancelled:
            raise  # a stop request must not be swallowed by the vp guard
        except Exception as exc:  # noqa: BLE001 - never fail the job here
            report(f"  voiceprints skipped: {exc}")

    elapsed = time.perf_counter() - started
    rtf = elapsed / total_duration if total_duration > 0 else None
    if not stopped:
        stage("done", 1.0)

    return Transcript(
        source=str(input_path),
        duration=round(total_duration, 3),
        model=config.model_name,
        diarization_model=diarization_model,
        language=config.language,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        segments=segments,
        rtf=round(rtf, 3) if rtf is not None else None,
        processing_seconds=round(elapsed, 2),
        speaker_match=match_info,
    )
