"""Downscaling the routing matte's input does NOT speed it up. Do not retry.

Reasoning that looked sound: the matte feeds only row extents ->
person_height_frac, a scale-invariant ratio that never reaches the output
image, so segmenting a 384px probe instead of 1024px should cost ~7x less.

Measured on GPU: body_route went 12.9s -> 16.1s. SLOWER. bria-rmbg-2.0 has a
FIXED input resolution, so rembg resizes whatever it is handed to the
model's native size -- input size does not drive its cost, and the extra
resize is pure overhead. Reverted.

The real levers for this stage, both untested:
  1. A working CUDA provider. onnxruntime advertises CUDAExecutionProvider
     and then silently falls back to CPU at session creation
     (requested=[CUDA, CPU] -> ACTUAL=[CPU]), and the install is fragile:
     a later pip install replaced onnxruntime-gpu with the CPU build and
     the requested list collapsed to ['CPUExecutionProvider'].
  2. body_route_use_segmentation=false. The routing decision that actually
     fires here is `below_face_frac >= 0.38`, computed from the face box and
     the full-size frame -- it never reads the matte. Only person_height_frac
     does, and the code's own comment notes that metric rates a bust crop
     (~0.99) as MORE full-body than a real full-body shot.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def _detect_full_body_src() -> str:
    """Source of _detect_full_body, bounded by the NEXT method definition.

    Not a fixed character window: a 6000-char slice silently truncated when
    a comment block was added, failing on code that was present and correct.
    """
    i = KREA2.find("def _detect_full_body(")
    assert i > 0
    j = KREA2.find("\n    def ", i + 1)
    return KREA2[i:j if j > i else len(KREA2)]


def test_matte_gets_the_full_image_not_a_downscaled_probe():
    """The downscale is reverted; this fails if someone reintroduces it."""
    body = _detect_full_body_src()
    assert "_try_birefnet_mask(body, None, blur_px=0)" in body
    assert "body_route_matte_max_dim" not in body, (
        "downscaling was measured SLOWER (12.9s -> 16.1s); see this "
        "module's docstring before trying it again"
    )


def test_the_negative_result_is_recorded_at_the_call_site():
    """A future reader must hit the finding before re-deriving the idea."""
    body = _detect_full_body_src()
    assert "FIXED input" in body and "16.1s" in body


def test_ratio_still_divides_by_the_matte_frame():
    """Kept from the reverted change: it is correct either way, and it makes
    person_h_frac robust if the matte's frame ever diverges from the body's
    again."""
    body = _detect_full_body_src()
    assert "_mask_h = float(alpha.shape[0])" in body
    assert "person_h_frac = float(rows.max() - rows.min()) / _mask_h" in body


def test_below_face_frac_is_independent_of_the_matte():
    """This is the value that actually routes. It comes from the face box and
    the full-size frame, which is why disabling the matte entirely is the
    promising lever."""
    body = _detect_full_body_src()
    assert "below_face_frac = max(0.0, h - selected_face.y1) / float(h)" in body
