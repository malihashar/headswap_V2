from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.eval.dataset import generate_synthetic_eval_set, load_pairs
from headswap.pipelines import create_pipeline_from_config
from headswap.preprocess import head_hair_mask_from_face, soft_composite, evenify


def test_evenify():
    assert evenify(15, 8) == 8
    assert evenify(16, 8) == 16


def test_synthetic_eval_and_mock_klein():
    generate_synthetic_eval_set(n_pairs=4)
    pairs = load_pairs()
    assert len(pairs) >= 4
    pipe = create_pipeline_from_config(ROOT / "configs" / "klein4b.yaml", force_mock=True)
    body = Image.open(pairs[0]["body_path"])
    face = Image.open(pairs[0]["face_path"])
    out = pipe.run(body, face, out_dir=ROOT / "results" / "_test_mock")
    assert out.image.size[0] > 0
    assert out.latency_s >= 0
    assert "debug_mask" in out.debug_paths


def test_soft_composite_preserves_outside_mask():
    base = Image.new("RGB", (64, 64), (10, 20, 30))
    edit = Image.new("RGB", (32, 32), (200, 0, 0))
    mask = Image.new("L", (64, 64), 0)
    # white square in crop region mapped via soft_composite box
    from PIL import ImageDraw

    m = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(m).rectangle([16, 16, 47, 47], fill=255)
    out = soft_composite(base, edit, m, (16, 16, 48, 48))
    arr = np.asarray(out)
    # Corner should stay base color
    assert tuple(arr[0, 0].tolist()) == (10, 20, 30)
