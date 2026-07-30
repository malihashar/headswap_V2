#!/usr/bin/env python3
"""Classify the first pipeline stage that loses a multi-person identity edit.

Usage:
  python scripts/audit_identity_stages.py \
    --manifest /content/headswap_outputs/run_x/run_manifest.json \
    --debug-dir /content/headswap_outputs/run_x/debug \
    --out /content/headswap_outputs/run_x/stage_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.metrics.scoring import identity_cosine, psnr


def _load(path: Path) -> Image.Image | None:
    return Image.open(path).convert("RGB") if path.is_file() else None


def _region(image: Image.Image, box: list[int] | None) -> Image.Image:
    if not box:
        return image
    x0, y0, x1, y1 = box
    return image.crop((max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)))


def _delta(reference: Image.Image, image: Image.Image, box: list[int] | None) -> dict[str, float]:
    ref = _region(reference, box)
    candidate = _region(image.resize(reference.size, Image.Resampling.LANCZOS), box)
    a = np.asarray(ref, dtype=np.float32)
    b = np.asarray(candidate.resize(ref.size, Image.Resampling.LANCZOS), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return {"mse": round(mse, 4), "psnr": round(psnr(a, b), 4)}


def _stage_report(
    *,
    label: str,
    image: Image.Image | None,
    body: Image.Image,
    donor: Image.Image,
    selected_box: list[int] | None,
) -> dict[str, Any]:
    if image is None:
        return {"label": label, "present": False}
    target = _region(image, selected_box)
    return {
        "label": label,
        "present": True,
        "size": list(image.size),
        "target_delta_from_body": _delta(body, image, selected_box),
        "target_identity_cosine": identity_cosine(donor, target),
    }


def classify(stages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Heuristic diagnosis with conservative language.

    Identity cosine is optional because InsightFace may not be installed. When
    unavailable, the audit still flags stages that reintroduce body pixels.
    """
    present = [s for s in stages if s.get("present")]
    if not present:
        return {"classification": "insufficient_artifacts"}
    raw = next((s for s in present if s["label"] == "raw_krea2"), None)
    final = next((s for s in present if s["label"] == "final"), None)
    if raw is None:
        return {"classification": "raw_generation_missing"}

    raw_id = raw.get("target_identity_cosine")
    final_id = final.get("target_identity_cosine") if final else None
    raw_mse = raw["target_delta_from_body"]["mse"]
    final_mse = final["target_delta_from_body"]["mse"] if final else None
    if raw_id is not None and final_id is not None and raw_id - final_id > 0.08:
        return {
            "classification": "postprocess_identity_loss",
            "evidence": "raw donor similarity exceeds final donor similarity",
        }
    if raw_mse < 20.0:
        return {
            "classification": "conditioning_or_diffusion_identity_loss",
            "evidence": "raw decoded crop is close to the original target",
        }
    if final_mse is not None and final_mse < raw_mse * 0.5:
        return {
            "classification": "stitch_or_neighbor_freeze_overwrite",
            "evidence": "final target delta collapses after raw generation",
        }
    return {
        "classification": "geometry_or_mask_integration_review",
        "evidence": "raw edit survives, but identity/geometry requires visual review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    body = _load(Path(manifest["inputs"]["body"]["path"]))
    donor = _load(Path(manifest["inputs"]["face"]["path"]))
    if body is None or donor is None:
        raise SystemExit("Manifest input files are unavailable; cannot audit stages.")
    selected = (manifest.get("target_selection") or {}).get("selected_box")
    stages = [
        _stage_report(label="raw_krea2", image=_load(args.debug_dir / "debug_edited_crop.png"), body=body, donor=donor, selected_box=selected),
        _stage_report(label="composite_before_lab", image=_load(args.debug_dir / "debug_composite_before_lab.png"), body=body, donor=donor, selected_box=selected),
        _stage_report(label="composite_after_lab", image=_load(args.debug_dir / "debug_composite_after_lab.png"), body=body, donor=donor, selected_box=selected),
        _stage_report(label="after_neighbor_freeze", image=_load(args.debug_dir / "debug_after_neighbor_freeze.png"), body=body, donor=donor, selected_box=selected),
        _stage_report(label="final", image=_load(args.debug_dir / "debug_final.png"), body=body, donor=donor, selected_box=selected),
    ]
    report = {"manifest": str(args.manifest), "stages": stages, "diagnosis": classify(stages)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
