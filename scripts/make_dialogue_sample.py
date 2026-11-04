"""Generate a two-speaker Russian dialogue for testing diarization.

Uses Microsoft Edge TTS voices (needs internet) and ffmpeg to stitch the
turns into one file with short pauses.

Usage: python scripts/make_dialogue_sample.py samples/dialogue.wav
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import edge_tts

DIALOGUE = [
    ("ru-RU-DmitryNeural", "Привет! Меня зовут Дмитрий. Сегодня мы обсуждаем наш новый проект."),
    ("ru-RU-SvetlanaNeural", "Здравствуйте, Дмитрий. Я Светлана. Расскажите, что вы планируете сделать."),
    ("ru-RU-DmitryNeural", "Мы хотим запустить приложение для распознавания речи на процессоре."),
    ("ru-RU-SvetlanaNeural", "Звучит интересно. А когда можно будет увидеть первые результаты?"),
    ("ru-RU-DmitryNeural", "Через неделю покажем прототип и сравним качество распознавания."),
    ("ru-RU-SvetlanaNeural", "Договорились. Я запишу всё это в протокол встречи и пришлю его завтра."),
]


async def synthesize(tmp_dir: Path) -> list[Path]:
    parts = []
    for index, (voice, text) in enumerate(DIALOGUE):
        target = tmp_dir / f"part_{index:02d}.mp3"
        # the endpoint is flaky behind some proxies: retry with a delay
        for attempt in range(10):
            try:
                await edge_tts.Communicate(text, voice).save(str(target))
                break
            except Exception as exc:  # noqa: BLE001 - network flakiness
                target.unlink(missing_ok=True)
                if attempt == 9:
                    raise
                print(f"  retry {attempt + 1} ({type(exc).__name__})", flush=True)
                await asyncio.sleep(3)
        parts.append(target)
        print(f"  {voice}: {text}")
    return parts


def concat(parts: list[Path], target: Path, gap: float = 0.6) -> None:
    """Join parts with a silence gap between them."""
    silence = target.parent / "_silence.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"anullsrc=r=24000:cl=mono", "-t", str(gap), str(silence),
        ],
        check=True,
        capture_output=True,
    )

    concat_list = target.parent / "_concat.txt"
    lines = []
    for index, part in enumerate(parts):
        lines.append(f"file '{part.resolve().as_posix()}'")
        if index != len(parts) - 1:
            lines.append(f"file '{silence.resolve().as_posix()}'")
    concat_list.write_text("\n".join(lines), encoding="utf-8")

    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c:a", "libmp3lame", "-b:a", "128k", str(target),
        ],
        check=True,
        capture_output=True,
    )
    silence.unlink(missing_ok=True)
    concat_list.unlink(missing_ok=True)


async def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/dialogue.mp3")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = target.parent / "_tts_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Synthesizing ...")
    parts = await synthesize(tmp_dir)
    print("Concatenating ...")
    concat(parts, target)
    for part in parts:
        part.unlink(missing_ok=True)
    tmp_dir.rmdir()

    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(target)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"Saved {target} ({float(duration):.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
