"""A small donor must be UPSCALED before the single-image edit.

Upstream's Colab prepares every edit input with
``if max(W, H) < min_side: scale = min_side / max(W, H)`` -- i.e. edits
always run at >=1024 on the long side. The first port of that graph brought
over the workflow but not the prep, and this repo's resize_max_keep_ar is
`min(1.0, ...)`, a CAP that never upscales. A small donor therefore reached
the sampler at native size: measured on GPU at 192x176, a thumbnail with
nowhere near enough pixels to carry a likeness. The resulting identity loss
was initially blamed on denoise.

_fit_body_dim already documents the identical trap for the body image.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def _method_body() -> str:
    i = KREA2.find("def edit_single_image(")
    j = KREA2.find("\n    def probe_expression_transfer(", i)
    assert i > 0 and j > i
    return KREA2[i:j]


def test_small_donor_is_upscaled_not_just_capped():
    body = _method_body()
    assert "single_edit_min_long_side" in body
    assert "fit_long_side_keep_ar(" in body, (
        "resize_max_keep_ar only caps (min(1.0, ...)) and cannot upscale; "
        "the small-input branch must use fit_long_side_keep_ar"
    )


def test_upscale_branch_is_conditional_on_being_small():
    """Large donors must still be CAPPED, not blown up to the minimum."""
    body = _method_body()
    assert "if max(_native) < min_long:" in body
    assert "resize_max_keep_ar(" in body, "the large-input path must still cap"


def test_upscale_is_logged():
    """Silent resizing is what made the first run unreadable -- the actual
    sampled size must appear in the run log."""
    body = _method_body()
    assert "upscaled small donor" in body


def test_resize_max_keep_ar_really_cannot_upscale():
    """Pins the premise of this whole test file, so it fails loudly rather
    than silently becoming vacuous if that helper ever gains upscaling."""
    from PIL import Image

    from headswap.preprocess import resize_max_keep_ar

    small = Image.new("RGB", (192, 176))
    out = resize_max_keep_ar(small, 1024, div_by=16)
    assert out.size == (192, 176), (
        "resize_max_keep_ar now upscales; edit_single_image's separate "
        "small-input branch should be re-examined rather than deleted"
    )
