"""Head scale ratio metrics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from headswap.metrics.head_scale import head_scale_metrics
from headswap.preprocess import FaceBox


def _fake_faces(_rgb, _cache, allow_prior=True):
    return []


def test_head_scale_ratio_near_one_when_unchanged():
    w, h = 300, 400
    body = Image.new("RGB", (w, h), (200, 180, 160))
    draw = ImageDraw.Draw(body)
    draw.ellipse([110, 80, 190, 160], fill=(150, 120, 100))
    result = body.copy()
    selected = FaceBox(110, 80, 190, 160, 0.99)
    cache = Path("/tmp/headswap_test_cache")
    cache.mkdir(exist_ok=True)
    with patch("headswap.metrics.head_scale.detect_faces", return_value=[selected]):
        hs = head_scale_metrics(body, result, selected, cache)
    ratio = hs.get("head_to_body_scale_ratio")
    assert ratio is not None
    assert 0.95 <= float(ratio) <= 1.05


def test_head_scale_ratio_detects_oversized_head():
    w, h = 300, 400
    body = Image.new("RGB", (w, h), (200, 180, 160))
    draw = ImageDraw.Draw(body)
    draw.ellipse([110, 80, 190, 160], fill=(150, 120, 100))
    result = Image.new("RGB", (w, h), (200, 180, 160))
    draw2 = ImageDraw.Draw(result)
    draw2.ellipse([90, 60, 210, 190], fill=(150, 120, 100))
    selected = FaceBox(110, 80, 190, 160, 0.99)
    big = FaceBox(90, 60, 210, 190, 0.99)
    cache = Path("/tmp/headswap_test_cache")
    cache.mkdir(exist_ok=True)

    calls = [0]

    def faces_for(rgb, cache, allow_prior=True):
        calls[0] += 1
        return [selected] if calls[0] == 1 else [big]

    with patch("headswap.metrics.head_scale.detect_faces", side_effect=faces_for):
        hs = head_scale_metrics(body, result, selected, cache)
    ratio = float(hs["head_to_body_scale_ratio"])
    assert ratio > 1.1
