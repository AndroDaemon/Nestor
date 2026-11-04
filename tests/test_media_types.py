"""Every accepted upload format must be playable in the built-in player.

Regression: ``/api/recordings/<id>/media`` fell back to
``application/octet-stream`` for ``.webm`` (and the video formats), so the
browser refused the file with ``MEDIA_ERR_SRC_NOT_SUPPORTED`` — the player
stuck at "00:00 / 00:00" even though the transcript opened fine.
"""

from __future__ import annotations

from nestor.web.app import ALLOWED_SUFFIXES, MEDIA_TYPES


def test_every_allowed_suffix_has_a_media_type():
    missing = sorted(ALLOWED_SUFFIXES - set(MEDIA_TYPES))
    assert not missing, f"no Content-Type for {missing}"


def test_media_types_are_concrete():
    for suffix, media_type in MEDIA_TYPES.items():
        assert media_type.startswith(("audio/", "video/")), (suffix, media_type)
