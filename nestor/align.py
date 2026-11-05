"""Turning diarization turns + recognised words into speaker-labelled segments."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .types import DEFAULT_SPEAKER, Segment, Turn, Word

Chunk = Tuple[float, float]


def find_quiet_point(
    waveform: "np.ndarray",
    sample_rate: int,
    t: float,
    radius: float = 1.5,
) -> float:
    """Return the quietest moment near ``t`` (within ±``radius`` seconds).

    Frame RMS is scanned in a small window around ``t`` and the centre of the
    quietest frame is returned, clamped to the search window. Falls back to
    ``t`` for empty waveforms.
    """
    import numpy as np

    n = waveform.shape[0]
    if n == 0:
        return t
    lo = max(0.0, t - radius)
    hi = min(n / float(sample_rate), t + radius)
    if hi <= lo:
        return t

    frame = max(1, int(0.02 * sample_rate))
    start_sample = int(lo * sample_rate)
    end_sample = min(n, int(hi * sample_rate) + frame)
    segment = waveform[start_sample:end_sample]
    n_frames = max(1, len(segment) // frame)
    frames = segment[: n_frames * frame].reshape(n_frames, frame)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    quiet_index = int(np.argmin(frame_rms))
    quiet_t = (start_sample + quiet_index * frame + frame / 2.0) / float(sample_rate)
    return min(hi, max(lo, quiet_t))


def build_chunks(
    turns: Sequence[Turn],
    max_chunk: float = 20.0,
    max_gap: float = 0.5,
    min_chunk: float = 0.2,
    waveform: Optional["np.ndarray"] = None,
    sample_rate: int = 16000,
) -> List[Chunk]:
    """Group speaker turns into chunks that fit into the ASR model.

    Turns that follow each other closely are merged into a single chunk (which
    keeps the number of ASR calls low), while a single long turn is cut into
    ``max_chunk`` pieces.

    Parameters
    ----------
    turns:
        Speaker turns, sorted or not.
    max_chunk:
        Maximum chunk length in seconds (GigaAM rejects anything above 25 s).
    max_gap:
        Silence between two turns that still allows merging them.
    min_chunk:
        Chunks shorter than this are dropped as noise. Keep it below the length
        of a shortest possible word, otherwise whole utterances get lost.
    waveform:
        Optional 16 kHz mono waveform. When given, oversized chunks are split
        at quiet points (the least energetic moment near each boundary) instead
        of arbitrary equal intervals, so words are not cut in half.
    sample_rate:
        Sample rate of ``waveform``.
    """
    if not turns:
        return []

    ordered = sorted(turns, key=lambda t: t.start)
    chunks: List[Chunk] = []

    cur_start, cur_end = ordered[0].start, ordered[0].end
    for turn in ordered[1:]:
        gap = turn.start - cur_end
        if gap <= max_gap and (turn.end - cur_start) <= max_chunk:
            cur_end = max(cur_end, turn.end)
            continue

        if cur_end - cur_start > 0:
            chunks.append((cur_start, cur_end))
        cur_start, cur_end = turn.start, turn.end

    if cur_end - cur_start > 0:
        chunks.append((cur_start, cur_end))

    # Split oversized chunks
    result: List[Chunk] = []
    for start, end in chunks:
        length = end - start
        if length <= max_chunk:
            result.append((start, end))
            continue
        pieces = int(length // max_chunk) + 1
        step = length / pieces
        previous = start
        for index in range(1, pieces + 1):
            boundary = start + index * step
            if index == pieces:
                piece_end = end
            elif waveform is not None:
                # Move the boundary to the quietest nearby moment so the cut
                # lands in a pause, not in the middle of a word.
                boundary = find_quiet_point(waveform, sample_rate, boundary)
            piece_end = min(boundary, end)
            if piece_end - previous > 0:
                result.append((previous, piece_end))
            previous = piece_end

    return [(s, e) for s, e in result if e - s >= min_chunk]


def assign_speakers(words: Sequence[Word], turns: Sequence[Turn]) -> List[Word]:
    """Attribute every word to the speaker that overlaps it the most.

    Words that do not overlap any turn (speech detected by the ASR but missed by
    the diarizer) are attributed to the closest turn in time, so that no text is
    lost.
    """
    if not turns:
        return [Word(w.text, w.start, w.end, DEFAULT_SPEAKER) for w in words]

    result: List[Word] = []
    for word in words:
        best_speaker = None
        best_overlap = 0.0
        best_distance = float("inf")

        for turn in turns:
            overlap = min(word.end, turn.end) - max(word.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
                best_distance = 0.0
            elif overlap <= 0:
                midpoint = (word.start + word.end) / 2.0
                if word.end < turn.start:
                    distance = turn.start - word.end
                elif word.start > turn.end:
                    distance = word.start - turn.end
                else:
                    distance = 0.0
                if distance < best_distance:
                    best_distance = distance
                    if best_speaker is None:
                        best_speaker = turn.speaker

        result.append(
            Word(
                text=word.text,
                start=word.start,
                end=word.end,
                speaker=best_speaker or DEFAULT_SPEAKER,
            )
        )
    return result


def merge_words(
    words: Sequence[Word],
    max_gap: float = 0.8,
    min_duration: float = 0.0,
) -> List[Segment]:
    """Merge consecutive words of the same speaker into segments."""
    segments: List[Segment] = []

    for word in words:
        speaker = word.speaker or DEFAULT_SPEAKER
        if segments:
            last = segments[-1]
            if last.speaker == speaker and word.start - last.end <= max_gap:
                last.words.append(word)
                last.end = word.end
                last.text = " ".join(w.text for w in last.words)
                continue
        segments.append(
            Segment(
                start=word.start,
                end=word.end,
                speaker=speaker,
                text=word.text,
                words=[word],
            )
        )

    if min_duration > 0:
        segments = [s for s in segments if (s.end - s.start) >= min_duration]
    return segments


def speaker_order(segments: Sequence[Segment]) -> List[str]:
    """Speaker labels ordered by their first appearance."""
    order: List[str] = []
    for segment in segments:
        if segment.speaker not in order:
            order.append(segment.speaker)
    return order


def renumber_speakers(segments: Sequence[Segment]) -> Dict[str, str]:
    """Renumber SPEAKER_NN labels 00, 01, 02... by first appearance, in place.

    The diarizer's raw cluster ids are not sequential (0, 71, 296...), which
    confuses users. Internal labels produced by the auto-split carry a
    ``__SPLIT<k>`` suffix (``SPEAKER_05__SPLIT1``) and are treated the same way.
    Returns the old->new mapping; other structures keyed by labels
    (speaker_names, speaker_match) must be remapped by the caller.
    """
    mapping: Dict[str, str] = {}
    for index, label in enumerate(speaker_order(segments)):
        base = label.split("__SPLIT", 1)[0]
        if base.startswith("SPEAKER_") and base[8:].isdigit():
            mapping[label] = f"SPEAKER_{index:02d}"
    if not mapping:
        return mapping
    # avoid collisions when e.g. SPEAKER_02 already exists among old labels:
    # two-phase rename through temporary ids
    tokens = {old: f"__RSWAP_{i}" for i, old in enumerate(mapping)}
    inverse = {token: mapping[old] for old, token in tokens.items()}
    # pipeline segments carry words; library segments do not - handle both
    for segment in segments:
        segment.speaker = tokens.get(segment.speaker, segment.speaker)
        for word in getattr(segment, "words", ()) or ():
            word.speaker = segment.speaker
    for segment in segments:
        segment.speaker = inverse.get(segment.speaker, segment.speaker)
        for word in getattr(segment, "words", ()) or ():
            word.speaker = inverse.get(word.speaker, word.speaker)
    return mapping


def relabel_speakers(segments: Sequence[Segment], names: Optional[List[str]]) -> None:
    """Rename ``SPEAKER_00``-style labels to human names, in place."""
    if not names:
        return
    mapping: dict[str, str] = {}
    for index, label in enumerate(speaker_order(segments)):
        if index < len(names):
            mapping[label] = names[index]
    for segment in segments:
        segment.speaker = mapping.get(segment.speaker, segment.speaker)
        for word in segment.words:
            word.speaker = segment.speaker
