"""Decision-gate regression: multi post-crop conditioning matches single recipe."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox


def _load_harness():
    path = ROOT / "scripts" / "compare_single_vs_multi.py"
    spec = importlib.util.spec_from_file_location("compare_single_vs_multi", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _Pipe(Krea2IdentityEditPipeline):
    def __init__(self, **cfg):
        self.cfg = {
            "single_person_parity": True,
            "clamp_crop_away_neighbors": True,
            "mask_top_extend": 1.55,
            "mask_side_extend": 0.60,
            "mask_bot_extend": 0.40,
            "mask_expand_px": 18,
            "mask_blur_px": 4,
            "crop_long_side": 256,
            "person_match_crop_size": True,
            "identity_scale_match": False,
            "square_crop": False,
            "head_mask_backend": "ellipse",
            "div_by": 8,
            "prompt": "replace the face with the identity from the second image",
            "steps": 8,
            "cfg": 1.0,
            "ref_boost": 3.5,
            "ref_boost_a": 1.6,
            "grounding_px": 768,
            "fit_mode": "fit",
            "seed": 46,
            **cfg,
        }
        self.cache_dir = ROOT / "results" / "_cache"


def test_multi_recipe_matches_solo_fingerprint():
    """SPP multi must share exact conditioning recipe keys with solo."""
    h = _load_harness()
    pipe = _Pipe()
    donor = Image.new("RGB", (64, 80), (20, 20, 20))
    ImageDraw.Draw(donor).ellipse([8, 8, 56, 70], fill=(90, 50, 30))

    solo = Image.new("RGB", (200, 260), (30, 30, 30))
    ImageDraw.Draw(solo).ellipse([60, 40, 140, 140], fill=(200, 160, 140))
    solo_face = FaceBox(60, 40, 140, 140, 0.9)

    multi = Image.new("RGB", (480, 320), (30, 30, 30))
    ImageDraw.Draw(multi).ellipse([40, 60, 120, 160], fill=(200, 160, 140))
    ImageDraw.Draw(multi).ellipse([300, 60, 380, 160], fill=(190, 150, 130))
    faces = [FaceBox(40, 60, 120, 160, 0.9), FaceBox(300, 60, 380, 160, 0.9)]

    a = h._build_crop_case(
        pipe, solo, donor, solo_face, [solo_face], case_id="A", cache=pipe.cache_dir
    )
    b = h._build_crop_case(
        pipe, multi, donor, faces[1], faces, case_id="B", cache=pipe.cache_dir
    )
    for key in h.PARITY_KEYS:
        assert a["metrics"][key] == b["metrics"][key], key


def test_neighbor_clamp_preserves_face_fill_vs_selected_only():
    """A_ctrl (selected-only) vs B (neighbors) geometry stays within tolerance."""
    h = _load_harness()
    pipe = _Pipe()
    donor = Image.new("RGB", (64, 80), (20, 20, 20))
    # Wide canvas so both faces land in the same unclamped crop window.
    body = Image.new("RGB", (400, 240), (30, 30, 30))
    ImageDraw.Draw(body).ellipse([40, 50, 120, 150], fill=(200, 160, 140))
    ImageDraw.Draw(body).ellipse([200, 50, 280, 150], fill=(190, 150, 130))
    faces = [FaceBox(40, 50, 120, 150, 0.9), FaceBox(200, 50, 280, 150, 0.9)]
    selected = faces[1]

    a_ctrl = h._build_crop_case(
        pipe, body, donor, selected, [selected], case_id="Actrl", cache=pipe.cache_dir
    )
    b = h._build_crop_case(
        pipe, body, donor, selected, faces, case_id="B", cache=pipe.cache_dir
    )
    parity = h.evaluate_conditioning_parity(a_ctrl["metrics"], b["metrics"])
    assert parity["parity_ok"], parity["failed_keys"]
    # Neighbor center must not remain inside B crop when clamp can exclude it.
    box = b["box"]
    ncx = 0.5 * (faces[0].x0 + faces[0].x1)
    ncy = 0.5 * (faces[0].y0 + faces[0].y1)
    assert not (box[0] <= ncx < box[2] and box[1] <= ncy < box[3])


def test_align_paste_diverges_from_spp_person_prep():
    h = _load_harness()
    pipe = _Pipe()
    donor = Image.new("RGB", (64, 80), (20, 20, 20))
    body = Image.new("RGB", (400, 240), (30, 30, 30))
    ImageDraw.Draw(body).ellipse([40, 50, 120, 150], fill=(200, 160, 140))
    ImageDraw.Draw(body).ellipse([200, 50, 280, 150], fill=(190, 150, 130))
    faces = [FaceBox(40, 50, 120, 150, 0.9), FaceBox(200, 50, 280, 150, 0.9)]
    spp = h._build_crop_case(
        pipe, body, donor, faces[1], faces, case_id="B", cache=pipe.cache_dir
    )
    ap = h._build_align_paste_case(
        pipe.cfg, body, donor, faces[1], faces, pipe.cache_dir
    )
    assert spp["metrics"]["person_prep"] == "resize_contain"
    assert ap["metrics"]["person_prep"] == "identity_face_only_matte"
    assert spp["metrics"]["scene_size"] != ap["metrics"]["scene_size"]
