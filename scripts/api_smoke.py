"""Smoke test of the HTTP API: upload, wait, read, edit, export.

Usage: python scripts/api_smoke.py [audio_file] [base_url]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests


def main() -> int:
    audio = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/dialogue.mp3")
    base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

    print("status:", requests.get(base + "/api/status", timeout=10).json())

    with audio.open("rb") as handle:
        response = requests.post(
            base + "/api/recordings",
            files={"file": (audio.name, handle, "audio/mpeg")},
            data={"diarize": "true"},
            timeout=120,
        )
    record = response.json()
    print("created:", record["id"], record["filename"])
    record_id = record["id"]

    started = time.time()
    while time.time() - started < 1800:
        current = requests.get(f"{base}/api/recordings/{record_id}", timeout=10).json()
        if current.get("status") in ("done", "error"):
            break
        time.sleep(1)

    print(
        "status:", current["status"],
        "| error:", current["error"],
        "| segments:", len(current["segments"]),
        "| rtf:", current.get("rtf"),
    )
    if current["status"] != "done":
        return 1

    peaks = requests.get(f"{base}/api/recordings/{record_id}/peaks", timeout=10).json()
    print("peaks:", len(peaks["peaks"]))

    ranged = requests.get(
        f"{base}/api/recordings/{record_id}/media",
        headers={"Range": "bytes=0-99"},
        timeout=10,
    )
    print("media range:", ranged.status_code, ranged.headers.get("Content-Range"))

    # edit: rename the first speaker and append a bookmark
    segments = current["segments"]
    if segments:
        segments[0]["bookmark"] = True
    saved = requests.put(
        f"{base}/api/recordings/{record_id}",
        json={"segments": segments, "speaker_names": {"SPEAKER_00": "Первый"}},
        timeout=30,
    ).json()
    print("saved:", saved["segments"][0]["bookmark"], saved["speaker_names"])

    for fmt in ("json", "srt", "txt", "md", "docx"):
        export = requests.get(
            f"{base}/api/recordings/{record_id}/export/{fmt}", timeout=60
        )
        print(f"  export {fmt}: {export.status_code}, {len(export.content)} bytes")

    print("rail:", requests.get(base + "/api/recordings", timeout=10).json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
