"""SPP-Strict architecture contracts for production multi-person swaps."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox, clamp_crop_away_neighbors


CFG_PATH = ROOT / "configs" / "krea2_identity_edit.yaml"


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
            **cfg,
        }
        self.cache_dir = ROOT / "results" / "_cache"


def test_production_config_is_spp_strict():
    cfg = yaml.safe_load(CFG_PATH.read_text())
    assert cfg["multi_person_swap_mode"] == "krea2_crop"
    assert cfg["single_person_parity"] is True
    assert cfg["clamp_crop_away_neighbors"] is True
    assert cfg["multi_crop_hard_freeze_neighbors"] is False
    assert cfg["identity_scale_match"] is False
    assert cfg["face_white_bg"] is False
    assert cfg["multi_extra_prompt"] is False


def test_multi_build_matches_single_conditioning_recipe():
    """Multi and single must share person prep + mask extents under SPP-Strict."""
    pipe = _Pipe()
    body = Image.new("RGB", (480, 320), (25, 25, 25))
    d = ImageDraw.Draw(body)
    d.ellipse([40, 60, 120, 160], fill=(200, 160, 140))
    d.ellipse([300, 60, 380, 160], fill=(190, 150, 130))
    faces = [
        FaceBox(40, 60, 120, 160, 0.9),
        FaceBox(300, 60, 380, 160, 0.9),
    ]
    donor = Image.new("RGB", (96, 120), (10, 10, 10))
    ImageDraw.Draw(donor).ellipse([10, 10, 86, 100], fill=(80, 40, 20))

    multi = pipe._build_scene_person(
        body,
        donor,
        faces[1],
        div_by=8,
        use_tight=False,
        top_ext=1.55,
        side_ext=0.60,
        bot_ext=0.40,
        expand_px=18,
        crop_pad=12,
        all_faces=faces,
        isolate_selected=False,
    )
    single_body = Image.new("RGB", (200, 260), (25, 25, 25))
    ImageDraw.Draw(single_body).ellipse([60, 40, 140, 140], fill=(190, 150, 130))
    single_face = FaceBox(60, 40, 140, 140, 0.9)
    single = pipe._build_scene_person(
        single_body,
        donor,
        single_face,
        div_by=8,
        use_tight=False,
        top_ext=1.55,
        side_ext=0.60,
        bot_ext=0.40,
        expand_px=18,
        crop_pad=12,
        all_faces=[single_face],
        isolate_selected=False,
    )
    assert multi["diag"]["person_prep"] == single["diag"]["person_prep"] == "resize_contain"
    assert multi["diag"]["identity_scale_match"] is False
    assert multi["diag"]["single_person_parity"] is True
    assert multi["diag"]["use_tight"] is False
    assert multi["diag"]["isolate_selected"] is False
    # Neighbor center must not remain inside the multi crop window.
    box = multi["box"]
    ncx = 0.5 * (faces[0].x0 + faces[0].x1)
    ncy = 0.5 * (faces[0].y0 + faces[0].y1)
    assert not (box[0] <= ncx < box[2] and box[1] <= ncy < box[3])


def test_clamp_ignores_selected_duplicates_but_excludes_real_neighbor():
    selected = FaceBox(150, 40, 220, 120, 0.9)
    nested = FaceBox(160, 50, 210, 110, 0.4)  # duplicate detection of selected
    far = FaceBox(20, 40, 80, 120, 0.9)
    box, info = clamp_crop_away_neighbors(
        (300, 200),
        (0, 0, 300, 200),
        selected,
        [selected, nested, far],
    )
    # Nested duplicate must not force a clamp by itself; far neighbor should.
    assert info["neighbors_excluded"] == 1.0
    assert info["neighbors_unexcludable"] == 0.0
    fcx = 0.5 * (far.x0 + far.x1)
    assert not (box[0] <= fcx < box[2])
    # Selected face plus a margin must still fit inside the clamped crop.
    assert box[0] <= selected.x0 and box[2] >= selected.x1
