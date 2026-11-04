"""GigaAM v3 punct as an ONNX transducer, run by sherpa-onnx.

This is the same engine Sloveya uses: the ``nemo_transducer`` port of
``gigaam-v3-punct`` published by the k2-fsa project. It produces punctuated
Russian text out of the box and decodes much faster than the PyTorch CTC
models, so it is the recommended ASR backend.

sherpa-onnx quirk: the model must be created with
``model_type="nemo_transducer"`` — the plain transducer loader reads
``vocab_size``/``context_size`` from the decoder metadata, which this model
does not carry (they live in the encoder metadata instead).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

from .constants import SAMPLE_RATE
from .types import Word

# Gap (seconds) below which tokens are glued into one word. Sloveya uses the
# same value (word merge gap in _tokens_to_words).
WORD_MERGE_GAP = 0.24

# Extra length given to the last token of a word so neighbouring words do not
# overlap visually in exports.
WORD_END_PAD = 0.12


def _token_text(token: str) -> str:
    """Strip the SentencePiece word-start marker from a token.

    sherpa-onnx hands out tokens with a leading space (``" То"``) or the raw
    ``▁`` marker depending on version; punctuation tokens have neither.
    """
    return token.replace("▁", " ").strip()


def _starts_word(token: str) -> bool:
    """Whether this token begins a new word (word markers: space or ▁)."""
    return token.startswith((" ", "▁"))


class SherpaOnnxAsr:
    """ASR backend backed by sherpa-onnx (ONNX transducer, CPU)."""

    def __init__(
        self,
        model_files: Dict[str, Path],
        threads: int = 1,
        decoding_method: str = "greedy_search",
    ) -> None:
        import sherpa_onnx

        self.model_id = "gigaam-v3-punct"
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_files["encoder.int8.onnx"]),
            decoder=str(model_files["decoder.onnx"]),
            joiner=str(model_files["joiner.onnx"]),
            tokens=str(model_files["tokens.txt"]),
            num_threads=max(1, int(threads)),
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method=decoding_method,
            model_type="nemo_transducer",
        )
        # sherpa-onnx streams are decoded on C++ worker threads; a lock keeps
        # a single recognizer safe if the pipeline is ever called concurrently.
        self._lock = threading.Lock()

    def transcribe(
        self,
        waveform,
        offset: float = 0.0,
        word_timestamps: bool = True,
    ) -> List[Word]:
        """Transcribe a mono float32 waveform at 16 kHz."""
        if waveform.size == 0:
            return []

        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, waveform)
            self._recognizer.decode_stream(stream)

        result = stream.result
        tokens = list(result.tokens)
        timestamps = list(result.timestamps)
        if not tokens:
            return []

        if not word_timestamps:
            return [Word(text=result.text.strip(), start=offset, end=offset)]

        return self._words_from_tokens(tokens, timestamps, offset)

    def _words_from_tokens(self, tokens: List[str], timestamps: List[float], offset: float) -> List[Word]:
        """Group subword tokens into words.

        SentencePiece marks word starts (``▁``, surfaced by sherpa-onnx as a
        leading space); punctuation tokens carry no marker and are appended to
        the preceding word without a space.
        """
        words: List[Word] = []
        current_text: Optional[str] = None
        current_start = 0.0
        current_end = 0.0

        for token, ts in zip(tokens, timestamps):
            text = _token_text(token)
            if _starts_word(token):
                if current_text is not None:
                    start = current_start + offset
                    end = current_end + offset + WORD_END_PAD
                    words.append(Word(text=current_text, start=round(start, 3), end=round(end, 3)))
                current_text = text
                current_start = ts
                current_end = ts
            elif current_text is not None:
                current_text += text
                current_end = ts
            else:
                # A word that does not start with the marker (leading
                # punctuation): treat it as a word start so nothing is lost.
                current_text = text
                current_start = ts
                current_end = ts

        if current_text is not None:
            words.append(
                Word(
                    text=current_text,
                    start=round(current_start + offset, 3),
                    end=round(current_end + offset + WORD_END_PAD, 3),
                )
            )
        return words

    def transcribe_chunk(
        self,
        waveform,
        start: float,
        end: float,
        word_timestamps: bool = True,
    ) -> List[Word]:
        """Slice ``[start, end)`` out of the waveform and transcribe it."""
        from .audio import slice_waveform

        chunk = slice_waveform(waveform, start, end, SAMPLE_RATE)
        return self.transcribe(chunk, offset=start, word_timestamps=word_timestamps)

    def close(self) -> None:
        return None

    def __enter__(self) -> "SherpaOnnxAsr":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
