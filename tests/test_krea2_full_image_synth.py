from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.pipelines import PIPELINES, create_pipeline
from headswap.preprocess import (
    identity_content_frac,
    prepare_krea2_identity_person,
    resize_contain,
)
from headswap.prompting.scene_describe import (
    build_identity_edit_prompt,
    describe_scene,
)


def test_full_image_synth_registered_and_isolated_from_localized():
    assert "krea2_full_image_synth" in PIPELINES
    loc = load_config(ROOT / "configs" / "krea2_identity_edit.yaml")
    full = load_config(ROOT / "configs" / "krea2_full_image_synth.yaml")
    assert loc["pipeline"] in {"krea2", "krea2_identity_edit"}
    assert full["pipeline"] == "krea2_full_image_synth"
    assert bool(loc.get("mask_crop_stitch", True)) is True
    # Identity-max conditioning (not scene-boosted).
    assert float(full["ref_boost"]) >= 4.0
    assert float(full["ref_boost_a"]) <= 1.0 + 1e-6
    assert int(full["grounding_px"]) <= 768
    assert int(full["steps"]) >= 10
    assert bool(full.get("identity_square_fill", False)) is True


def test_auto_prompt_demands_head_face_hair_replacement():
    import numpy as np

    arr = np.zeros((240, 480, 3), dtype=np.uint8)
    arr[:] = (20, 22, 35)
    for cx in (80, 240, 400):
        yy, xx = np.ogrid[:240, :480]
        mask = ((xx - cx) / 28) ** 2 + ((yy - 90) / 36) ** 2 <= 1.0
        arr[mask] = (180, 150, 130)
    body = Image.fromarray(arr)
    desc, selected, faces = describe_scene(
        body, ROOT / ".cache" / "test_full_synth", face_policy="largest"
    )
    prompt = build_identity_edit_prompt(desc).lower()
    assert "replace the face, hair, and head" in prompt
    assert "image 2" in prompt
    assert "jawline" in prompt or "bone structure" in prompt
    # Must not ask to preserve target hairstyle (locks scene identity).
    assert "hairstyle when possible" not in prompt
    assert "preserve that silhouette" not in prompt
    assert desc.n_faces >= 1
    assert selected is not None


def test_identity_person_not_letterboxed_into_wide_scene(tmp_path: Path):
    # Wide scene-like canvas vs portrait face — old path starved identity (~0.23).
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
    )
    assert info["letterboxed_to_scene"] is False
    assert person.size[0] == person.size[1]
    # Face-filled square must beat letterboxed wide canvas occupancy.
    fixed = identity_content_frac(person)
    assert fixed > starved
    assert fixed >= 0.35
    assert float(info["identity_face_area_frac"]) > 0.10


def test_full_image_synth_mock_runs_without_comfy(tmp_path: Path):
    cfg = load_config(ROOT / "configs" / "krea2_full_image_synth.yaml")
    pipe = create_pipeline(cfg, force_mock=True)
    body = Image.new("RGB", (256, 320), (30, 40, 50))
    face = Image.new("RGB", (128, 128), (200, 160, 140))
    d = ImageDraw.Draw(body)
    d.ellipse([90, 60, 160, 150], fill=(190, 150, 130))
    out_dir = tmp_path / "run"
    result = pipe.run(body, face, out_dir=out_dir)
    assert result.image.size[0] > 0
    assert result.meta.get("mode") == "mock_full_image_synth_raw"
    assert result.meta.get("postprocess") == "none"
    assert "replace the face, hair, and head" in str(result.meta.get("prompt", "")).lower()
    assert result.meta.get("identity_prep", {}).get("letterboxed_to_scene") is False
    assert (out_dir / "prompt.txt").is_file()
