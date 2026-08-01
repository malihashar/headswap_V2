"""Identity stage tracer saves intermediates and diagnoses Krea2 bypass."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.align_paste_swap import run_align_paste_swap
from headswap.preprocess import FaceBox
from headswap.profiling.identity_stage_trace import IdentityStageTrace


def test_identity_trace_marks_krea2_absent(tmp_path: Path):
    body = Image.new("RGB", (320, 200), (30, 30, 30))
    d = ImageDraw.Draw(body)
    d.ellipse([200, 40, 270, 130], fill=(190, 150, 130))
    faces = [FaceBox(200, 40, 270, 130, 0.9)]
    donor = Image.new("RGB", (120, 140), (255, 255, 255))
    ImageDraw.Draw(donor).ellipse([20, 20, 100, 110], fill=(20, 80, 200))

    trace = IdentityStageTrace(
        tmp_path / "stages",
        donor=donor,
        body=body,
        selected=faces[0],
        selected_index=0,
        cache_dir=ROOT / "results" / "_cache",
        enabled=True,
    )
    out = run_align_paste_swap(
        body,
        donor,
        ROOT / "results" / "_cache",
        selected_face=faces[0],
        all_faces=faces,
        cfg={
            "align_paste_krea2_refine": False,
            "align_paste_seamless_clone": False,
            "pre_color_match_strength": 0.0,
            "align_paste_post_color_match": 0.0,
            "div_by": 8,
        },
        refine_fn=None,
        identity_trace=trace,
    )
    report = out["identity_stage_report"]
    assert report is not None
    diag = report["diagnosis"]
    assert diag["krea2_ran"] is False
    assert "krea2" in diag["classification"] or "geometry_lock" in diag["classification"]
    names = {s["name"] for s in report["stages"]}
    assert "06_raw_krea2_output" in names
    assert "10_final" in names
    assert (tmp_path / "stages" / "IDENTITY_STAGE_REPORT.json").is_file()
    raw = next(s for s in report["stages"] if s["name"] == "06_raw_krea2_output")
    assert raw["present"] is False
