"""Probing the original file for a picture, and serving it to the video player."""

from __future__ import annotations

import shutil
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from nestor.library import Library, Recording
from nestor.video_names import _STILL_CODECS, has_video, probe_video, video_stream_index
from nestor.web.app import create_app

SAMPLE_RATE = 48000


def _require_encoders() -> None:
    missing = {"libvpx", "libopus"} - set(av.codecs_available)
    if missing:
        pytest.skip(f"PyAV build lacks encoders: {sorted(missing)}")


def _write_silence(path: Path, seconds: float = 1.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _write_webm(path: Path, audio_delay: float = 0.0, seconds: float = 1.0) -> Path:
    """A tiny VP8+Opus file; ``audio_delay`` starts the audio track later."""
    _require_encoders()
    with av.open(str(path), "w", format="webm") as container:
        video = container.add_stream("libvpx", rate=15)
        video.width = 160
        video.height = 90
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("libopus", rate=SAMPLE_RATE)
        audio.layout = "mono"

        frames = []
        for index in range(int(15 * seconds)):
            image = np.zeros((90, 160, 3), dtype=np.uint8)
            image[:, :, 0] = (index * 7) % 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 15)
            frames.append(("video", frame))

        for index in range(0, int(SAMPLE_RATE * seconds), 960):
            samples = np.zeros((1, 960), dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(samples, format="flt", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            frame.pts = int(audio_delay * SAMPLE_RATE) + index
            frame.time_base = Fraction(1, SAMPLE_RATE)
            frames.append(("audio", frame))

        for kind, frame in frames:
            stream = video if kind == "video" else audio
            for packet in stream.encode(frame):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode(None):
                container.mux(packet)
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("media")
    return {
        "wav": _write_silence(root / "call.wav"),
        "webm": _write_webm(root / "call.webm"),
        "delayed": _write_webm(root / "delayed.webm", audio_delay=0.5),
    }


def test_audio_file_has_no_video(media):
    assert has_video(media["wav"]) is False
    assert probe_video(media["wav"]) == (False, 0.0)


def test_video_file_is_detected(media):
    assert has_video(media["webm"]) is True
    found, offset = probe_video(media["webm"])
    assert found is True
    assert offset == pytest.approx(0.0, abs=0.2)


def test_audio_offset_points_at_the_first_audio_frame(media):
    found, offset = probe_video(media["delayed"])
    assert found is True
    assert 0.35 < offset < 0.65


def test_still_image_streams_are_not_video():
    class Stream:
        def __init__(self, kind, codec):
            self.type = kind
            self.codec_context = SimpleNamespace(name=codec)

    container = SimpleNamespace(
        streams=[
            Stream("audio", "opus"),
            Stream("video", "mjpeg"),
            Stream("video", "vp8"),
        ]
    )
    assert "mjpeg" in _STILL_CODECS
    assert video_stream_index(container) == 2


def _client_with(work: Path, filename: str, source: Path) -> TestClient:
    library = Library(work)
    recording = Recording(id="rec1", filename=filename, status="done", duration=1.0)
    library.save(recording)
    target = library.source_path(recording.id, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return TestClient(create_app(work_dir=work))


def test_detail_probes_video_and_persists_it(tmp_path, media):
    client = _client_with(tmp_path, "call.webm", media["webm"])

    payload = client.get("/api/recordings/rec1").json()
    assert payload["has_video"] is True
    assert payload["video_checked"] is True
    assert payload["media_offset"] == pytest.approx(0.0, abs=0.2)

    stored = Library(tmp_path).load("rec1")
    assert stored is not None
    assert stored.has_video is True
    assert stored.video_checked is True


def test_media_serves_video_mime_on_request(tmp_path, media):
    client = _client_with(tmp_path, "call.webm", media["webm"])

    plain = client.get("/api/recordings/rec1/media")
    assert plain.status_code == 200
    assert plain.headers["content-type"].startswith("audio/webm")

    picture = client.get("/api/recordings/rec1/media", params={"video": 1})
    assert picture.status_code == 200
    assert picture.headers["content-type"].startswith("video/webm")
    assert picture.content


def test_video_request_for_audio_recording_is_refused(tmp_path, media):
    client = _client_with(tmp_path, "call.wav", media["wav"])

    payload = client.get("/api/recordings/rec1").json()
    assert payload["has_video"] is False

    response = client.get("/api/recordings/rec1/media", params={"video": 1})
    assert response.status_code == 404
    assert "видео" in response.json()["detail"].lower()
