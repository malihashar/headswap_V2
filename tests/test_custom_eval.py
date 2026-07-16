"""Tests for custom real-photo eval set preparation."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from headswap.eval.dataset import load_pairs, prepare_custom_eval_set


def test_prepare_custom_eval_set_one_pair(tmp_path: Path):
    custom = tmp_path / "custom"
    custom.mkdir()
    Image.new("RGB", (64, 80), (200, 100, 50)).save(custom / "body.png")
    Image.new("RGB", (48, 48), (50, 100, 200)).save(custom / "face.png")

    eval_root = tmp_path / "eval"
    manifest = prepare_custom_eval_set(custom_dir=custom, root=eval_root)
    assert manifest.exists()

    data = json.loads(manifest.read_text())
    assert data["n_pairs"] == 1
    assert data["pairs"][0]["id"] == "custom_001"
    assert (eval_root / "bodies" / "custom_001.png").is_file()
    assert (eval_root / "faces" / "custom_001.png").is_file()

    pairs = load_pairs(eval_root)
    assert len(pairs) == 1
    assert pairs[0]["body_path"].is_file()
    assert pairs[0]["face_path"].is_file()
    assert pairs[0]["tags"] == ["custom", "real_photo"]


def test_prepare_custom_missing_files(tmp_path: Path):
    custom = tmp_path / "custom"
    custom.mkdir()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(custom / "body.png")
    # face.png missing
    try:
        prepare_custom_eval_set(custom_dir=custom, root=tmp_path / "eval")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "face.png" in str(exc)
