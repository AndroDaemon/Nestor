"""Participant names read from a video conference recording.

TrueConf can burn the participant names into the recording (the client setting
"Отображать имена участников на записи").  This module samples a handful of
frames, reads the text with :mod:`nestor.ocr` and keeps the strings that look
like person names.

The result is only a *roster*: who took part, not who said what.  Mapping a
name to a speaker is still the user's decision — the names are offered as
suggestions when a speaker is renamed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .ocr import OcrEngine, TextLine

DEFAULT_FRAMES = 16
"""How many frames are sampled when nothing else is requested."""

MIN_WHOLE_LINE_CONFIDENCE = 0.70
MIN_CANDIDATE_CONFIDENCE = 0.55

# A screen-share frame carries the whole desktop: hundreds of small lines.  Its
# bottom strip of participant tiles is clipped and unreadable, so only the large
# caption over the shared screen (the presenter's name) is trusted there.
SCREEN_SHARE_LINES = 40
LARGE_CAPTION_HEIGHT = 0.03
MAX_BOXES = 80

# Words that show up in screen shares and slide decks and can look like names.
_IGNORED_WORDS = {
    "заполнить",
    "автосумма",
    "калибри",
    "файл",
    "правка",
    "вид",
    "вставка",
    "вставить",
    "удалить",
    "формат",
    "данные",
    "рецензирование",
    "разметка",
    "формулы",
    "вкладка",
    "главная",
    "сортировка",
    "найти",
    "ссылки",
    "рассылки",
    "надстройки",
    "разработчик",
    "справка",
    "excel",
    "word",
    "powerpoint",
    "outlook",
    "acrobat",
    "adobe",
}

_NAME_PART = re.compile(r"^[А-ЯЁA-Z][а-яёa-z\-']{1,30}$")
_SPLIT_CAMEL = re.compile(r"(?<=[а-яёa-z])(?=[А-ЯЁA-Z])")
_NOT_LETTERS = re.compile(r"[^А-ЯЁа-яёA-Za-z'\-]+")
_PARENTHESIS = re.compile(r"\s*[\(\[].*?[\)\]]\s*")
_CYRILLIC = re.compile(r"[А-ЯЁа-яё]")


@dataclass
class NameCandidate:
    """A name and how often it was seen."""

    name: str
    count: int
    confidence: float


# A video stream with one of these codecs is a still picture: MP3 files carry
# their cover art that way.  Such a stream must not turn the video pane on.
_STILL_CODECS = {"mjpeg", "png", "bmp", "gif", "webp"}


def video_stream_index(container) -> Optional[int]:
    """Index of the first moving-picture stream, or ``None`` when there is none.

    Cover art in an audio file lives in a video stream and is skipped, so the
    player is never offered a frozen album cover as "video".
    """
    for index, stream in enumerate(container.streams):
        if stream.type != "video":
            continue
        codec = getattr(getattr(stream, "codec_context", None), "name", "") or ""
        if codec.lower() in _STILL_CODECS:
            continue
        return index
    return None


def has_video(path: Path) -> bool:
    """True when the file carries a video stream with a moving picture."""
    import av

    try:
        with av.open(str(path)) as container:
            return video_stream_index(container) is not None
    except Exception:
        return False


def probe_video(path: Path) -> Tuple[bool, float]:
    """Whether the file has a picture, and where its audio begins.

    Returns ``(has_video, audio_offset)``.  ``audio_offset`` is the position of
    the first audio frame relative to the container start, in seconds.  The
    extracted 16 kHz track starts at that frame while a player counts from the
    container start, so the picture must be shifted by this offset to match the
    transcript clock.  Unknown or negative offsets are reported as ``0.0``.
    """
    import av

    try:
        with av.open(str(path)) as container:
            if video_stream_index(container) is None:
                return False, 0.0
            offset = 0.0
            try:
                audio = next(
                    (stream for stream in container.streams if stream.type == "audio"),
                    None,
                )
                if audio is not None:
                    base = float(container.start_time or 0) / 1_000_000
                    start = audio.start_time
                    if start is not None and audio.time_base:
                        offset = float(start * audio.time_base) - base
                    else:
                        for frame in container.decode(audio):
                            if frame.pts is not None and frame.time_base:
                                offset = float(frame.pts * frame.time_base) - base
                            break
            except Exception:
                offset = 0.0
            return True, max(0.0, round(offset, 3))
    except Exception:
        return False, 0.0


def _container_seconds(container, video_stream) -> float:
    if video_stream.duration is not None and video_stream.time_base:
        return float(video_stream.duration * video_stream.time_base)
    if container.duration:
        return float(container.duration / 1_000_000)
    return 0.0


def sample_frames(path: Path, count: int) -> Iterable[Tuple[float, "object"]]:
    """Yield ``(timestamp, bgr_array)`` pairs spread evenly over the video."""
    import av

    count = max(1, int(count))
    with av.open(str(path)) as container:
        index = video_stream_index(container)
        if index is None:
            return
        stream = container.streams[index]
        total = _container_seconds(container, stream)
        # Skip the very beginning and the very end: intros and goodbyes are
        # where the layout is least typical.
        fractions = [(step + 0.5) / count for step in range(count)]
        for fraction in fractions:
            timestamp = total * (0.05 + 0.9 * fraction) if total else 0.0
            try:
                if timestamp > 0:
                    container.seek(int(timestamp * 1_000_000), stream=None)
                frame = next(iter(container.decode(stream)))
            except (StopIteration, ValueError):
                continue
            array = frame.to_ndarray(format="rgb24")
            yield timestamp, array[:, :, ::-1].copy()


def _squash(text: str) -> str:
    return " ".join(text.split())


def _looks_like_name_part(word: str) -> bool:
    return bool(_NAME_PART.match(word)) and word.lower() not in _IGNORED_WORDS


def _name_candidates(text: str, allow_tail: bool = False) -> List[Tuple[str, bool]]:
    """Return ``(candidate, is_whole_line)`` pairs found in a recognized line.

    Lines produced by the OCR are sometimes glued together (two name plates on
    one line).  Splitting on capital letters recovers the tail of such a line.

    ``allow_tail`` is for lines known to be captions or window titles, where the
    name sits at the end: ``"Формулы… Шаталов Дмитрий Олегович"``.  There the
    last three tokens are a name even though the line is longer.
    """
    cleaned = _PARENTHESIS.sub(" ", text or "")
    cleaned = _NOT_LETTERS.sub(" ", cleaned)
    tokens: List[str] = []
    for token in _squash(cleaned).split(" "):
        if not token:
            continue
        tokens.extend(part for part in _SPLIT_CAMEL.split(token) if part)
    if not tokens:
        return []

    if len(tokens) in (2, 3) and all(_looks_like_name_part(t) for t in tokens):
        # A clean plate: the whole line is one name.  Sub-windows of it would
        # only produce truncated duplicates such as "Шаталов Дмитрий".
        return [(" ".join(tokens), True)]

    candidates: List[Tuple[str, bool]] = []
    if allow_tail and len(tokens) > 3:
        tail = tokens[-3:]
        if all(_looks_like_name_part(part) for part in tail):
            candidates.append((" ".join(tail), True))
    for size in (3, 2):
        for start in range(0, len(tokens) - size + 1):
            window = tokens[start : start + size]
            if not all(_looks_like_name_part(part) for part in window):
                continue
            candidates.append((" ".join(window), False))
    return candidates


def collect_names(
    lines: Sequence[TextLine],
    whole: Counter,
    window: Counter,
    confidences: Counter,
    allow_tail: bool = False,
) -> None:
    """Fold the lines of one frame into the running tallies.

    ``whole`` counts frames where a line was cleanly one name, ``window`` counts
    names recovered from a glued line (several plates read as one string).
    """
    seen_this_frame = set()
    for line in lines:
        if line.confidence < MIN_CANDIDATE_CONFIDENCE:
            continue
        for candidate, is_whole in _name_candidates(line.text, allow_tail):
            if is_whole and line.confidence < MIN_WHOLE_LINE_CONFIDENCE:
                continue
            key = candidate.casefold()
            if key in seen_this_frame:
                continue
            seen_this_frame.add(key)
            (whole if is_whole else window)[candidate] += 1
            confidences[candidate] = max(confidences[candidate], line.confidence)


def _same_person(left: str, right: str) -> bool:
    if len(left.split()) != len(right.split()):
        return False
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio() >= 0.93


def _drop_fragments(items: List[NameCandidate]) -> List[NameCandidate]:
    """Drop a name that is a contiguous part of a longer, at least as common one."""
    kept: List[NameCandidate] = []
    for item in items:
        tokens = [token.casefold() for token in item.name.split()]
        is_fragment = False
        for other in items:
            if other is item or other.count < item.count:
                continue
            other_tokens = [token.casefold() for token in other.name.split()]
            if len(other_tokens) <= len(tokens):
                continue
            span = len(tokens)
            for start in range(0, len(other_tokens) - span + 1):
                if other_tokens[start : start + span] == tokens:
                    is_fragment = True
                    break
            if is_fragment:
                break
        if not is_fragment:
            kept.append(item)
    return kept


def _drop_noise_around(items: List[NameCandidate]) -> List[NameCandidate]:
    """Drop a name with junk around it: a longer variant of a stronger one.

    Clipped tile captions often come back as the same name with a stray prefix
    ("Ренирс Шаталов Дмитрий"). When a shorter candidate is at least as common,
    the longer one is noise.
    """
    kept: List[NameCandidate] = []
    for item in items:
        tokens = {token.casefold() for token in item.name.split()}
        noisy = False
        for other in items:
            if other is item or other.count < item.count * 2:
                continue
            other_tokens = {token.casefold() for token in other.name.split()}
            if len(tokens) < 2 or len(other_tokens) < 2:
                continue
            if tokens <= other_tokens or other_tokens <= tokens:
                continue
            if len(tokens & other_tokens) >= 2:
                noisy = True
                break
        if not noisy:
            kept.append(item)
    return kept


def rank_names(
    whole: Counter, window: Counter, confidences: Counter, frames: int
) -> List[NameCandidate]:
    """Merge near-duplicates, drop weak entries and sort by strength."""
    scores: Counter = Counter()
    for name, count in whole.items():
        scores[name] += count * 3
    for name, count in window.items():
        scores[name] += count

    whole_min = 2 if frames >= 4 else 1
    window_min = max(3, int(round(frames * 0.25)))

    merged: List[NameCandidate] = []
    for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        strong = whole.get(name, 0) >= whole_min
        repeated = window.get(name, 0) >= window_min
        if not (strong or repeated):
            continue
        for existing in merged:
            if _same_person(existing.name, name):
                existing.count += score
                existing.confidence = max(
                    existing.confidence, confidences.get(name, 0.0)
                )
                break
        else:
            merged.append(NameCandidate(name, score, confidences.get(name, 0.0)))

    merged = _drop_fragments(merged)
    merged = _drop_noise_around(merged)
    # When at least one name was read cleanly, drop everything that only ever
    # appeared inside a glued line.  Those are window slices of a title bar or a
    # clipped strip ("Дднтые Рценирс Шаталов") and never a real plate.
    clean = [name for name, count in whole.items() if count >= whole_min]
    if clean:
        merged = [
            item
            for item in merged
            if any(_same_person(item.name, name) or item.name == name for name in clean)
        ]
    # Latin-only lines are usually interface labels ("Location Request",
    # "Protel So"). Keep them only when several sampled frames agree.
    merged = [
        item
        for item in merged
        if _CYRILLIC.search(item.name) or whole.get(item.name, 0) >= 3
    ]
    merged.sort(key=lambda item: (-item.count, -item.confidence, item.name))
    return merged


def _boxes_confined_to_bottom(boxes: List[TextLine], frame_height: int, tall: float) -> bool:
    """True when every small line sits in the lower part of the frame.

    Screen sharing puts the participant strip under the content, so all small
    captions hug the bottom edge.  A gallery spreads its captions over the whole
    height, which keeps them readable.
    """
    small = [box for box in boxes if box.height < tall]
    if not small:
        return False
    edge = frame_height * 0.62
    return all(box.y + box.height >= edge for box in small)


def extract_participant_names(
    path: Path,
    engine: OcrEngine,
    frames: int = DEFAULT_FRAMES,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """Read the participant names burned into ``path``.

    Returns the names ordered by how often they were seen.  An empty list means
    either the names are not burned into the recording or the file has no video.
    """
    whole: Counter = Counter()
    window: Counter = Counter()
    confidences: Counter = Counter()

    plans = []
    for _, frame in sample_frames(Path(path), frames):
        height, width = frame.shape[0], frame.shape[1]
        tall = height * LARGE_CAPTION_HEIGHT
        wide = width * 0.15
        boxes = engine.detect(frame)
        captions = [box for box in boxes if box.height >= tall and box.width >= wide]
        share = len(boxes) >= SCREEN_SHARE_LINES or _boxes_confined_to_bottom(boxes, height, tall)
        plans.append((frame, boxes, captions, share))

    sampled = len(plans)
    if not sampled:
        return []

    for index, (frame, boxes, captions, share) in enumerate(plans, start=1):
        if share:
            # Only the presenter caption carries a readable name here.  The tile
            # strip is clipped, and detection often glues neighbouring tiles into
            # one tall box that looks like a caption but reads as garbage.
            boxes = captions
        if boxes:
            boxes.sort(key=lambda box: box.width * box.height, reverse=True)
            lines = engine.read_boxes(
                frame,
                [(box.x, box.y, box.width, box.height) for box in boxes[:MAX_BOXES]],
            )
            collect_names(lines, whole, window, confidences, allow_tail=share)
        if on_progress is not None:
            on_progress(index, sampled)

    ranked = rank_names(whole, window, confidences, sampled)
    return [item.name for item in ranked]


def names_from_video(
    path: Path,
    frames: int = DEFAULT_FRAMES,
    threads: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """Roster of a video file, ready to be offered in the UI.

    Returns an empty list when the file has no video, when the names are not
    burned into the frames, or when the OCR models are not installed.  Silence
    is deliberate: the roster is a bonus and must never turn a finished
    transcription into a failure.
    """
    path = Path(path)
    if not path.exists():
        return []
    if not has_video(path):
        return []

    from .models import local_ocr_models

    models = local_ocr_models()
    if models is None:
        # Offline by design: no model downloads in the middle of a job.
        return []

    engine = OcrEngine(models["det"], models["rec"], models["yaml"], threads=threads)
    return extract_participant_names(path, engine, frames=frames, on_progress=on_progress)
