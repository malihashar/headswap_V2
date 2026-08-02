from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.pipelines import PIPELINES, create_pipeline
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
    # Localized production path remains mask/crop oriented by default.
    assert bool(loc.get("mask_crop_stitch", True)) is True
    # Full synth uses identity-max dials distinct from crop defaults.
    assert float(full["ref_boost"]) >= 4.0
    assert int(full["grounding_px"]) >= 1024
    assert int(full["steps"]) >= 10


def test_auto_prompt_mentions_identity_and_preservation():
    # Synthetic dark group-ish image with three bright face blobs.
    import numpy as np

    arr = np.zeros((240, 480, 3), dtype=np.uint8)
    arr[:] = (20, 22, 35)
    # three face-like bright ellipses
    for cx in (80, 240, 400):
        yy, xx = np.ogrid[:240, :480]
        mask = ((xx - cx) / 28) ** 2 + ((yy - 90) / 36) ** 2 <= 1.0
        arr[mask] = (180, 150, 130)
    body = Image.fromarray(arr)
    desc, selected, faces = describe_scene(
        body, ROOT / ".cache" / "test_full_synth", face_policy="largest"
    )
    prompt = build_identity_edit_prompt(desc)
    assert "identity" in prompt.lower()
    assert "expression" in prompt.lower()
    assert "clothing" in prompt.lower()
    assert "image 2" in prompt.lower()
    assert desc.n_faces >= 1
    assert selected is not None


def test_full_image_synth_mock_runs_without_comfy(tmp_path: Path):
    cfg = load_config(ROOT / "configs" / "krea2_full_image_synth.yaml")
    pipe = create_pipeline(cfg, force_mock=True)
    body = Image.new("RGB", (256, 320), (30, 40, 50))
    face = Image.new("RGB", (128, 128), (200, 160, 140))
    # Draw a crude face on body so detector/heuristics have something.
    from PIL import ImageDraw

    d = ImageDraw.Draw(body)
    d.ellipse([90, 60, 160, 150], fill=(190, 150, 130))
    out_dir = tmp_path / "run"
    result = pipe.run(body, face, out_dir=out_dir)
    assert result.image.size[0] > 0
    assert result.meta.get("mode") == "mock_full_image_synth_raw"
    assert result.meta.get("postprocess") == "none"
    assert "prompt" in result.meta
    assert (out_dir / "prompt.txt").is_file()
