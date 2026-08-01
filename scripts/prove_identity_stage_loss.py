#!/usr/bin/env python3
"""Prove which geometry-lock stage kills donor identity (no Krea2/GPU required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from headswap.align_paste_swap import run_align_paste_swap
from headswap.config import load_config
from headswap.preprocess import detect_faces, ensure_insightface_app
from headswap.profiling.identity_stage_trace import IdentityStageTrace


def main() -> None:
    cache = ROOT / "results" / "_cache"
    ensure_insightface_app(cache)
    cfg = load_config(ROOT / "configs" / "krea2_identity_edit.yaml")

    face = Image.open(ROOT / "data" / "custom" / "face.png").convert("RGB")
    body_path = ROOT / "data" / "eval" / "bodies" / "custom_001.png"
    body = Image.open(body_path).convert("RGB")
    # Synthesize 3-up multi so we exercise the multi path locally.
    w, h = body.size
    canvas = Image.new("RGB", (w * 3, h), (0, 0, 0))
    for i in range(3):
        canvas.paste(body, (i * w, 0))
    body = canvas
    import numpy as np

    faces = detect_faces(np.asarray(body), cache, allow_prior=False)
    selected = max(faces, key=lambda f: f.width * f.height) if faces else None
    sel_idx = faces.index(selected) if selected in faces else None

    out_dir = ROOT / "results" / "_identity_stage_trace"
    if out_dir.exists():
        for p in out_dir.glob("*"):
            if p.is_file():
                p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = IdentityStageTrace(
        out_dir,
        donor=face,
        body=body,
        selected=selected,
        selected_index=sel_idx,
        cache_dir=cache,
        enabled=True,
    )
    ap = run_align_paste_swap(
        body,
        face,
        cache,
        selected_face=selected,
        all_faces=faces,
        cfg=cfg,
        refine_fn=None,
        identity_trace=trace,
    )
    ap["image"].save(out_dir / "RESULT.png")
    report = ap.get("identity_stage_report") or {}
    print(json.dumps(report.get("diagnosis"), indent=2))
    print(f"stages → {out_dir}")


if __name__ == "__main__":
    main()
