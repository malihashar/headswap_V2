from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.pipelines import PIPELINES, create_pipeline
from headswap.preprocess import (
    crop_face_reference,
    crop_identity_head,
    draw_identity_head_debug,
    identity_content_frac,
    prepare_krea2_identity_person,
    resize_contain,
)


def test_full_image_synth_registered_and_isolated_from_localized():
    assert "krea2_full_image_synth" in PIPELINES
    loc = load_config(ROOT / "configs" / "krea2_identity_edit.yaml")
    full = load_config(ROOT / "configs" / "krea2_full_image_synth.yaml")
    assert loc["pipeline"] in {"krea2", "krea2_identity_edit"}
    assert full["pipeline"] == "krea2_full_image_synth"
    assert bool(full.get("identity_use_head_crop", False)) is True
    assert float(full["ref_boost_a"]) <= 1.0 + 1e-6


def test_head_crop_larger_than_tight_face_pads(tmp_path: Path):
    # Synthetic portrait with room above/below the face for hair + beard.
    im = Image.new("RGB", (400, 560), (10, 10, 12))
    d = ImageDraw.Draw(im)
    # Face oval in the middle third.
    d.ellipse([120, 180, 280, 380], fill=(200, 160, 140))
    d.ellipse([155, 240, 185, 270], fill=(30, 30, 30))
    d.ellipse([215, 240, 245, 270], fill=(30, 30, 30))
    # Dark hair above face.
    d.ellipse([110, 80, 290, 210], fill=(20, 15, 10))
    # Beard below mouth.
    d.ellipse([140, 340, 260, 430], fill=(40, 30, 25))

    cache = tmp_path / "cache"
    tight = crop_face_reference(
        im, cache, top=0.70, bot=0.12, side=0.22, include_shoulders=False
    )
    head, info = crop_identity_head(im, cache, square=True, head_fill=0.88)
    assert info.get("detector_box") is not None or info.get("policy")
    # Head crop must be larger than the legacy tight face crop.
    assert head.size[0] * head.size[1] > tight.size[0] * tight.size[1]
    # Detector box area should be a minority of the head crop (hair/beard margin).
    det = info.get("detector_box_in_crop")
    if det is not None:
        dw = max(1, det[2] - det[0])
        dh = max(1, det[3] - det[1])
        det_frac = (dw * dh) / float(head.size[0] * head.size[1])
        assert det_frac < 0.85

    overlay = draw_identity_head_debug(im, info)
    assert overlay.size == im.size


def test_prepare_identity_uses_head_crop_by_default(tmp_path: Path):
    face = Image.new("RGB", (400, 560), (0, 0, 0))
    d = ImageDraw.Draw(face)
    d.ellipse([80, 60, 320, 420], fill=(200, 160, 140))
    person, crop, info = prepare_krea2_identity_person(
        face, tmp_path / "cache", long_side=512, use_head_crop=True
    )
    assert info["use_head_crop"] is True
    assert info["letterboxed_to_scene"] is False
    assert person.size[0] == person.size[1]
    assert "head_crop" in info


def test_identity_person_not_letterboxed_into_wide_scene(tmp_path: Path):
    face = Image.new("RGB", (400, 560), (0, 0, 0))
    d = ImageDraw.Draw(face)
    d.ellipse([80, 60, 320, 420], fill=(200, 160, 140))
    d.ellipse([140, 160, 180, 200], fill=(40, 40, 40))
    d.ellipse([220, 160, 260, 200], fill=(40, 40, 40))

    scene_size = (1024, 576)
    letterboxed = resize_contain(face, scene_size, fill=(0, 0, 0))
    starved = identity_content_frac(letterboxed)

    person, _crop, info = prepare_krea2_identity_person(
        face,
        tmp_path / "cache",
        long_side=512,
        white_bg=True,
        square_fill=True,
        use_head_crop=True,
    )
    assert info["letterboxed_to_scene"] is False
    assert person.size[0] == person.size[1]
    fixed = identity_content_frac(person)
    assert fixed > starved
    assert fixed >= 0.35


def test_full_image_synth_mock_runs_without_comfy(tmp_path: Path):
    cfg = load_config(ROOT / "configs" / "krea2_full_image_synth.yaml")
    pipe = create_pipeline(cfg, force_mock=True)
    body = Image.new("RGB", (256, 320), (30, 40, 50))
    face = Image.new("RGB", (200, 280), (200, 160, 140))
    d = ImageDraw.Draw(body)
    d.ellipse([90, 60, 160, 150], fill=(190, 150, 130))
    d2 = ImageDraw.Draw(face)
    d2.ellipse([40, 50, 160, 200], fill=(190, 150, 130))
    out_dir = tmp_path / "run"
    result = pipe.run(body, face, out_dir=out_dir)
    assert result.meta.get("postprocess") == "none"
    assert result.meta.get("identity_prep", {}).get("use_head_crop") is True
    assert (out_dir / "prompt.txt").is_file()
