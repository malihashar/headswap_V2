#!/usr/bin/env python3
"""Confirm single-person path is unchanged after multi-person full_frame promote.

Runs one single-person swap with production yaml defaults and asserts:
  - edit_mode / multi_person_edit_mode do not force full_frame on 1-face bodies
  - identity_lora_name stays the global (r64) unless explicitly overridden
  - ref_boost stays the global yaml default (not full_frame_ref_boost)

Usage:
  PYTHONPATH=src python scripts/ab_full_frame_spp_check.py
  PYTHONPATH=src python scripts/ab_full_frame_spp_check.py --mock
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline, reset_shared_krea2_runtime
from headswap.preprocess import detect_faces, pil_to_rgb_np, resize_max_keep_ar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--body",
        type=Path,
        default=ROOT / "data" / "custom" / "body.png",
    )
    ap.add_argument(
        "--face",
        type=Path,
        default=ROOT / "data" / "custom" / "face.png",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "krea2_identity_edit.yaml",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "_ab_full_frame_spp_check",
    )
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    # Explicitly leave multi defaults as-is (crop_stitch production).
    assert str(cfg.get("multi_person_edit_mode") or "crop_stitch") == "crop_stitch", (
        "yaml multi_person_edit_mode must remain crop_stitch until promote gate passes"
    )

    body = Image.open(args.body).convert("RGB")
    face = Image.open(args.face).convert("RGB")
    probe = resize_max_keep_ar(body, int(cfg.get("max_body_dim", 1024)), div_by=16)
    faces = detect_faces(pil_to_rgb_np(probe), cache, allow_prior=False)
    if len(faces) >= 2:
        print(
            f"[warn] body has {len(faces)} faces — SPP check prefers a single-person "
            "photo; continuing anyway (pipeline should still use crop_stitch when "
            "multi_person_edit_mode=crop_stitch)."
        )

    reset_shared_krea2_runtime()
    if args.mock:
        from headswap.pipelines import create_pipeline

        pipe = create_pipeline(dict(cfg), force_mock=True)
    else:
        pipe = Krea2IdentityEditPipeline(cfg=dict(cfg), cache_dir=cache)

    t0 = time.perf_counter()
    result = pipe.run(body, face, out_dir=out)
    latency = time.perf_counter() - t0
    meta = dict(result.meta or {})
    result.image.save(out / "result.png")

    checks: dict[str, Any] = {
        "faces_on_body": len(faces),
        "edit_mode": meta.get("edit_mode"),
        "multi_person_edit_mode": meta.get("multi_person_edit_mode"),
        "multi_person": meta.get("multi_person"),
        "loras_loaded": meta.get("loras_loaded"),
        "ref_boost": meta.get("ref_boost"),
        "yaml_identity_lora": cfg.get("identity_lora_name"),
        "yaml_ref_boost": cfg.get("ref_boost"),
        "yaml_full_frame_lora": cfg.get("full_frame_identity_lora_name"),
        "yaml_full_frame_ref_boost": cfg.get("full_frame_ref_boost"),
        "latency_s": round(latency, 3),
        "mock": bool(args.mock),
    }

    ok = True
    failures: list[str] = []
    # Single-person: must not enter full_frame edit mode.
    if meta.get("multi_person") is False or len(faces) < 2:
        if meta.get("edit_mode") == "full_frame":
            ok = False
            failures.append("single-person leaked into edit_mode=full_frame")
    # LoRA must be global r64, not full_frame override (override is null in yaml).
    loras = list(meta.get("loras_loaded") or [])
    expected_lora = str(cfg.get("identity_lora_name") or "")
    if loras and expected_lora and expected_lora not in str(loras[0]):
        # mock may omit
        if not args.mock:
            ok = False
            failures.append(f"unexpected LoRA {loras} (expected {expected_lora})")
    # ref_boost must match global yaml, not a full_frame override.
    rb = meta.get("ref_boost")
    yaml_rb = float(cfg.get("ref_boost", 3.5))
    if rb is not None and abs(float(rb) - yaml_rb) > 1e-6 and not args.mock:
        ok = False
        failures.append(f"ref_boost={rb} != yaml {yaml_rb}")

    report = {
        "ok": ok,
        "failures": failures,
        "checks": checks,
        "summary": (
            "SPP OK — single-person path unchanged."
            if ok
            else "SPP FAIL — " + "; ".join(failures)
        ),
    }
    (out / "SPP_CHECK.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "SPP_CHECK.md").write_text(
        "# SPP regression check\n\n"
        f"**{report['summary']}**\n\n"
        "```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(report["summary"])
    print(json.dumps(checks, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
