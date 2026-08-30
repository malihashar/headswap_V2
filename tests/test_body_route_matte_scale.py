"""The routing matte must be computed on a downscaled probe.

Measured: body_route took 12.9s of a 13.4s pre-dispatch stage, running the
~1GB rembg matte model at full 1024px -- on CPU, because onnxruntime
advertises CUDAExecutionProvider and then silently falls back to CPU at
session creation (requested=[CUDA, CPU] -> ACTUAL=[CPU]).

The matte here feeds ONLY row extents -> person_height_frac /
below_face_frac. Those are ratios, so they are scale-invariant, and none of
it reaches the output image. Segmenting at full resolution to compute a
fraction is pure waste even on a GPU.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def _detect_full_body_src() -> str:
    i = KREA2.find("def _detect_full_body(")
    assert i > 0
    return KREA2[i:i + 6000]


def test_matte_is_computed_on_a_downscaled_probe():
    body = _detect_full_body_src()
    assert "body_route_matte_max_dim" in body
    assert "_try_birefnet_mask(_probe" in body, (
        "the full-resolution image must not be handed to the matte model"
    )


def test_downscale_is_conditional():
    """An already-small input must not be upscaled into extra work."""
    assert "if max(body.size) > _mp:" in _detect_full_body_src()


def test_ratio_divides_by_the_matte_height_not_the_full_frame():
    """The bug this pins was introduced BY the downscale and nearly shipped.

    person_h_frac measures matte rows, so dividing by the full-frame h would
    shrink it by the probe scale factor (~2.7x at 384 from 1024) and bias
    every routing decision toward "not a full body". Both terms must come
    from the same frame.
    """
    body = _detect_full_body_src()
    assert "person_h_frac = float(rows.max() - rows.min()) / _mask_h" in body
    assert "_mask_h = float(alpha.shape[0])" in body
    assert "rows.min()) / float(h)" not in body


def test_below_face_frac_is_independent_of_the_matte():
    """below_face_frac comes from the face box and the full-size frame, so
    the downscale must not touch it -- pinned so a later refactor does not
    quietly route it through the probe."""
    body = _detect_full_body_src()
    assert "below_face_frac = max(0.0, h - selected_face.y1) / float(h)" in body
