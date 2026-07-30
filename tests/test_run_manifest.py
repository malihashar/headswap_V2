from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.profiling.run_manifest import (  # noqa: E402
    build_manifest,
    sha256_image,
    write_manifest,
)


def test_manifest_hashes_decoded_inputs_and_selected_target(tmp_path: Path):
    body_path = tmp_path / "body.png"
    face_path = tmp_path / "face.png"
    body = Image.new("RGB", (20, 10), (10, 20, 30))
    face = Image.new("RGB", (10, 20), (40, 50, 60))
    body.save(body_path)
    face.save(face_path)

    manifest = build_manifest(
        repo=ROOT,
        body_path=body_path,
        face_path=face_path,
        body=body,
        face=face,
        config={"seed": 46, "multi_person_swap_mode": "krea2_crop"},
        detections=[{"index": 0, "box": [1, 2, 8, 9], "confidence": 0.9}],
        selected_index=0,
        selected_box=[1, 2, 8, 9],
        runtime={"gpu": "test"},
    )

    assert manifest["inputs"]["body"]["decoded_rgb_sha256"] == sha256_image(body)
    assert manifest["inputs"]["face"]["decoded_rgb_sha256"] == sha256_image(face)
    assert manifest["target_selection"]["selected_box"] == [1, 2, 8, 9]
    assert manifest["config_sha256"]

    path = write_manifest(tmp_path, manifest)
    saved = json.loads(path.read_text())
    assert saved["inputs"]["body"]["file_sha256"] == manifest["inputs"]["body"]["file_sha256"]


def test_manifest_writer_never_overwrites(tmp_path: Path):
    manifest = {"schema_version": 1}
    write_manifest(tmp_path, manifest)
    try:
        write_manifest(tmp_path, manifest)
    except FileExistsError:
        pass
    else:
        raise AssertionError("run manifest must be immutable")
