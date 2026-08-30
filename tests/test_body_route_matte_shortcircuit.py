"""The routing matte must not run when it cannot change the answer.

_detect_full_body's segmentation branch computes

    full_body_detected = person_h_frac >= min AND face_h_frac <= max

so once face_h_frac exceeds face_h_frac_max the result is False whatever the
matte says -- and the non-segmentation default, bool(heuristic_full_body),
carries the same face_h_frac condition, so it is False on that path too.
Identical outcome, so skipping the matte there is a short-circuit rather
than a behaviour change.

It was costing 16.7s of a 16.9s pre-dispatch stage: the ~1GB rembg model on
CPU (onnxruntime advertises CUDAExecutionProvider, then silently falls back
at session creation). On the athlete bust shot face_h_frac=0.3958 against a
0.30 default max, so the entire stage computed a term already overruled.

NOT the same as body_route_use_segmentation=false, which would change real
decisions: for multi-person photos full_body_detected gates the full_frame
route and its LoRA/ref_boost overrides.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox


class _Pipe(Krea2IdentityEditPipeline):
    def __init__(self, cfg=None):
        self.cfg = dict(cfg or {})
        self.cache_dir = ROOT / "results" / "_cache"


def _run(face_h_frac: float, frame_h: int = 1000):
    """Build a face box occupying face_h_frac of the frame."""
    pipe = _Pipe()
    body = Image.new("RGB", (1000, frame_h))
    fh = int(face_h_frac * frame_h)
    # Placed near the top so below_face_frac stays large (a "body visible"
    # shot), isolating face size as the variable under test.
    box = FaceBox(x0=400, y0=10, x1=600, y1=10 + fh, conf=0.99)
    return pipe._detect_full_body(body, box, [box])


def test_big_face_skips_the_matte_entirely():
    res = _run(0.3958)  # the athlete bust shot
    assert res["full_body_detected"] is False
    assert "short_circuit" in str(res.get("segmentation_skip_reason", ""))
    assert res.get("method") != "person_segmentation", (
        "the matte ran even though it could not change the answer"
    )


def test_short_circuit_result_matches_the_segmentation_path_result():
    """The property that makes this safe: on a big face BOTH paths say
    False, so skipping is not a behaviour change."""
    assert _run(0.3958)["full_body_detected"] is False
    assert _run(0.50)["full_body_detected"] is False


def test_small_face_still_consults_the_matte():
    """A real full-body shot (small face) must NOT be short-circuited -- the
    matte still gates full_body_detected there, and for multi-person photos
    that gates the full_frame route."""
    res = _run(0.08)
    assert "short_circuit" not in str(res.get("segmentation_skip_reason", "")), (
        "a small face must still reach the segmentation branch"
    )


def test_threshold_is_the_configured_one_not_a_literal():
    src = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()
    i = src.find("def _detect_full_body(")
    body = src[i:i + 8000]
    assert "_seg_can_matter = face_h_frac <= face_h_frac_max" in body
