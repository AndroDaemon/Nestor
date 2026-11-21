"""Minimal ONNX text recognition for on-screen names.

Two PaddleOCR models are used:

* ``PP-OCRv5_mobile_det`` — DB text detector (language agnostic);
* ``cyrillic_PP-OCRv5_mobile_rec`` — CTC recognizer with a dictionary that
  covers Russian and Latin, so ``Elizaveta Khabaeva`` and ``Ирина (Протей)``
  are read by the same model.

The implementation deliberately avoids OpenCV and SciPy (the frozen build
excludes them): connected components are labelled with a small run-length
union-find, image resampling comes from Pillow.  It is not a general-purpose
OCR engine — it reads short, high-contrast lines such as the participant name
plates of a video conference recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml

_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class TextLine:
    """One recognized line of text and where it sits on the frame."""

    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


def load_vocabulary(rec_yaml: Path) -> List[str]:
    """Return the recognizer's character list in output-index order.

    Row 0 of the CTC output is the blank symbol and the last row is the space
    character, so ``characters[i]`` corresponds to output index ``i + 1``.
    """
    config = yaml.safe_load(Path(rec_yaml).read_text(encoding="utf-8"))
    characters = config["PostProcess"]["character_dict"]
    if not isinstance(characters, list) or not characters:
        raise ValueError(f"no character_dict in {rec_yaml}")
    return [str(ch) for ch in characters]


def _row_runs(row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Half-open column intervals of the True values in one row."""
    padded = np.empty(row.shape[0] + 2, dtype=bool)
    padded[0] = False
    padded[-1] = False
    padded[1:-1] = row
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


def _components(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Connected components of a boolean mask as ``(y0, y1, x0, x1)`` boxes.

    Each row becomes a list of runs; runs of neighbouring rows that overlap are
    merged with a union-find.  The detector only ever labels text blobs, so the
    run count stays small and no SciPy is needed.
    """
    parent: List[int] = []
    bounds: List[List[int]] = []

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        first, second = find(left), find(right)
        if first == second:
            return
        parent[second] = first
        target, source = bounds[first], bounds[second]
        target[0] = min(target[0], source[0])
        target[1] = max(target[1], source[1])
        target[2] = min(target[2], source[2])
        target[3] = max(target[3], source[3])

    previous: List[Tuple[int, int, int]] = []
    for y in range(mask.shape[0]):
        starts, stops = _row_runs(mask[y])
        current: List[Tuple[int, int, int]] = []
        cursor = 0
        for start, stop in zip(starts.tolist(), stops.tolist()):
            index = len(parent)
            parent.append(index)
            bounds.append([y, y + 1, start, stop])
            while cursor < len(previous) and previous[cursor][1] <= start:
                cursor += 1
            probe = cursor
            while probe < len(previous) and previous[probe][0] < stop:
                union(index, previous[probe][2])
                probe += 1
            current.append((start, stop, index))
        previous = current

    return [
        (bounds[i][0], bounds[i][1], bounds[i][2], bounds[i][3])
        for i in range(len(parent))
        if find(i) == i
    ]


class OcrEngine:
    """Detector + recognizer sharing a pair of ONNX sessions."""

    def __init__(
        self,
        det_model: Path,
        rec_model: Path,
        rec_yaml: Path,
        threads: Optional[int] = None,
    ) -> None:
        import onnxruntime as ort

        # Warnings about unused initializers are harmless but noisy.
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        if threads and threads > 0:
            options.intra_op_num_threads = int(threads)
        self._det = ort.InferenceSession(
            str(det_model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._rec = ort.InferenceSession(
            str(rec_model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._det_input = self._det.get_inputs()[0].name
        self._det_output = self._det.get_outputs()[0].name
        self._rec_input = self._rec.get_inputs()[0].name
        self._rec_output = self._rec.get_outputs()[0].name
        self._characters = load_vocabulary(rec_yaml)
        # blank + characters + space
        self._classes = len(self._characters) + 2

    # ------------------------------------------------------------- detector
    def _detect_boxes(
        self,
        bgr: np.ndarray,
        max_side: int = 960,
        threshold: float = 0.3,
        box_threshold: float = 0.6,
        unclip: float = 1.7,
    ) -> List[Tuple[int, int, int, int]]:
        from PIL import Image

        height, width = bgr.shape[:2]
        scale = 1.0
        if max(height, width) > max_side:
            scale = max_side / max(height, width)
        # The network needs both sides to be multiples of 32.
        net_h = max(32, int(round(height * scale / 32.0) * 32))
        net_w = max(32, int(round(width * scale / 32.0) * 32))
        resized = np.asarray(
            Image.fromarray(bgr).resize((net_w, net_h), Image.BILINEAR), dtype=np.float32
        )
        resized = (resized / 255.0 - _DET_MEAN) / _DET_STD
        tensor = resized.transpose(2, 0, 1)[None]

        probability = self._det.run([self._det_output], {self._det_input: tensor})[0][0, 0]
        regions = _components(probability > threshold)
        if not regions:
            return []

        scale_y = height / net_h
        scale_x = width / net_w
        boxes: List[Tuple[int, int, int, int]] = []
        for y0, y1, x0, x1 in regions:
            patch = probability[y0:y1, x0:x1]
            if patch.size == 0 or patch.mean() < box_threshold:
                continue
            box_h = (y1 - y0) * scale_y
            box_w = (x1 - x0) * scale_x
            if box_h < 4 or box_w < 4:
                continue
            center_y = (y0 + y1) / 2.0 * scale_y
            center_x = (x0 + x1) / 2.0 * scale_x
            box_h *= unclip
            box_w *= unclip
            boxes.append(
                (
                    int(center_x - box_w / 2),
                    int(center_y - box_h / 2),
                    int(box_w),
                    int(box_h),
                )
            )
        return boxes

    # ---------------------------------------------------------- recognizer
    def _recognize_crop(self, bgr: np.ndarray) -> Tuple[str, float]:
        from PIL import Image

        height, width = bgr.shape[:2]
        if height < 2 or width < 2:
            return "", 0.0
        target_w = max(4, int(round(48.0 * width / height)))
        resized = np.asarray(
            Image.fromarray(bgr).resize((target_w, 48), Image.BILINEAR), dtype=np.float32
        )
        tensor = ((resized / 255.0) - 0.5) / 0.5
        logits = self._rec.run(
            [self._rec_output], {self._rec_input: tensor.transpose(2, 0, 1)[None]}
        )[0][0]

        best = logits.argmax(axis=1)
        scores = logits.max(axis=1)
        characters = self._characters
        space_index = len(characters) + 1
        pieces: List[str] = []
        kept: List[float] = []
        previous = -1
        for position, index in enumerate(best):
            index = int(index)
            if index == previous:
                continue
            previous = index
            if index == 0:
                continue
            if index == space_index:
                pieces.append(" ")
            elif 1 <= index <= len(characters):
                pieces.append(characters[index - 1])
            else:
                continue
            kept.append(float(scores[position]))
        text = "".join(pieces).strip()
        confidence = float(np.mean(kept)) if kept else 0.0
        return text, confidence

    # -------------------------------------------------------------- public
    def detect(
        self,
        bgr: np.ndarray,
        min_height: int = 5,
        max_relative_height: float = 0.5,
    ) -> List[TextLine]:
        """Text boxes only, without recognition (cheap layout probe)."""
        height, width = bgr.shape[:2]
        lines: List[TextLine] = []
        for x, y, box_w, box_h in self._detect_boxes(bgr):
            if box_h < min_height or box_h > height * max_relative_height:
                continue
            if box_w > width * 0.95:
                continue
            left, top = max(0, x), max(0, y)
            right = min(width, x + box_w)
            bottom = min(height, y + box_h)
            if right - left < 6 or bottom - top < 6:
                continue
            lines.append(
                TextLine(
                    text="",
                    confidence=0.0,
                    x=left,
                    y=top,
                    width=right - left,
                    height=bottom - top,
                )
            )
        return lines

    def read(
        self,
        bgr: np.ndarray,
        max_boxes: int = 120,
        min_confidence: float = 0.5,
        min_height: int = 5,
        max_relative_height: float = 0.5,
    ) -> List[TextLine]:
        """Detect and recognize text lines on a single frame (BGR array)."""
        height, width = bgr.shape[:2]
        boxes = self._detect_boxes(bgr)
        if not boxes:
            return []

        # Largest boxes first: names are usually among the bigger lines, and a
        # busy desktop screenshot can produce hundreds of tiny fragments.
        boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        limit = max(1, int(max_boxes))
        lines: List[TextLine] = []
        for x, y, box_w, box_h in boxes[:limit]:
            if box_h < min_height or box_h > height * max_relative_height:
                continue
            if box_w > width * 0.95:
                continue
            left, top = max(0, x), max(0, y)
            right = min(width, x + box_w)
            bottom = min(height, y + box_h)
            if right - left < 6 or bottom - top < 6:
                continue
            text, confidence = self._recognize_crop(bgr[top:bottom, left:right])
            if not text or confidence < min_confidence:
                continue
            lines.append(
                TextLine(
                    text=text,
                    confidence=confidence,
                    x=left,
                    y=top,
                    width=right - left,
                    height=bottom - top,
                )
            )
        return lines

    def read_boxes(self, bgr: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]]):
        """Recognize a caller-supplied list of boxes (used by tests)."""
        height, width = bgr.shape[:2]
        out: List[TextLine] = []
        for x, y, box_w, box_h in boxes:
            left, top = max(0, x), max(0, y)
            right = min(width, x + box_w)
            bottom = min(height, y + box_h)
            if right - left < 6 or bottom - top < 6:
                continue
            text, confidence = self._recognize_crop(bgr[top:bottom, left:right])
            if text:
                out.append(
                    TextLine(
                        text=text,
                        confidence=confidence,
                        x=left,
                        y=top,
                        width=right - left,
                        height=bottom - top,
                    )
                )
        return out
