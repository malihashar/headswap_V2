"""Unit tests for head-scale geometry tracer (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.preprocess import FaceBox
from headswap.profiling.head_scale_trace import (
    HeadScaleTrace,
    first_enlargement_stage,
    measure_stage,
)


CACHE = ROOT / "results" / "_cache"


def test_first_enlargement_detects_s3_local_growth():
    s0 = measure_stage(
        "S0_original",
        Image.new("RGB", (200, 200), (30, 30, 30)),
        CACHE,
        face=FaceBox(80, 60, 120, 110, 1.0),
    )
    # Fake S2/S3 with local heights showing growth inside crop.
    from headswap.profiling.head_scale_trace import StageGeom

    s2 = StageGeom(
        name="S2_scene",
        image_size=[100, 100],
        face_box=[30, 20, 70, 60],
        face_w=40,
        face_h=40,
        face_h_body=50.0,
    )
    s3 = StageGeom(
        name="S3_edited",
        image_size=[100, 100],
        face_box=[20, 10, 80, 80],
        face_w=60,
        face_h=70,  # 70/40 = 1.75 > 1.12
        face_h_body=87.5,
    )
    s4 = StageGeom(
        name="S4_stitched",
        image_size=[200, 200],
        face_box=[70, 40, 140, 130],
        face_w=70,
        face_h=90,
        face_h_body=90.0,
    )
    verdict = first_enlargement_stage(
        {
            "S0_original": s0,
            "S2_scene": s2,
            "S3_edited": s3,
            "S4_stitched": s4,
        },
        tol=1.12,
    )
    assert verdict["FIRST_ENLARGEMENT_STAGE"] == "S3_edited"
    assert verdict["s3_vs_s2_local"] > 1.12


def test_trace_records_prep_edited_stitched(tmp_path: Path):
    body = Image.new("RGB", (320, 240), (25, 25, 25))
    d = ImageDraw.Draw(body)
    d.ellipse([40, 50, 110, 140], fill=(200, 160, 140))
    d.ellipse([200, 50, 270, 140], fill=(190, 150, 130))
    selected = FaceBox(200, 50, 270, 140, 0.9)
    mask = Image.new("L", body.size, 0)
    ImageDraw.Draw(mask).ellipse([180, 20, 290, 170], fill=255)
    crop_box = (160, 0, 320, 200)
    scene = body.crop(crop_box).resize((128, 160), Image.Resampling.LANCZOS)
    # Oversized face in edited crop
    edited = scene.copy()
    ImageDraw.Draw(edited).ellipse([10, 5, 118, 150], fill=(80, 40, 20))
    result = body.copy()
    ImageDraw.Draw(result).ellipse([170, 10, 300, 180], fill=(80, 40, 20))

    trace = HeadScaleTrace(
        tmp_path,
        cache_dir=CACHE,
        enabled=True,
        body_full=body,
        selected=selected,
        crop_box=crop_box,
    )
    trace.record_prep(
        body_full=body,
        selected=selected,
        mask=mask,
        crop_box=crop_box,
        scene=scene,
    )
    trace.record_edited(edited)
    trace.record_stitched(result)
    report = trace.finalize()
    assert report["enabled"] is True
    assert "S0_original" in report["stages"]
    assert "S3_edited" in report["stages"]
    assert "S4_stitched" in report["stages"]
    assert (tmp_path / "REPORT.md").is_file()
    assert (tmp_path / "SCALE_DRIFT.png").is_file()
    assert report["verdict"]["FIRST_ENLARGEMENT_STAGE"] in {
        "S3_edited",
        "S4_stitched",
        "S1_crop",
        "S2_scene",
        "none",
    }
