#!/usr/bin/env python3
"""Run fixed multi-person architecture variants against an annotated suite.

The suite must conform to ``data/benchmarks/multi_person_v1.schema.json`` and
contain at least 30 consented pairs for a release decision. Variants never
share a warm pipeline/config, preventing one route's settings from leaking into
another route.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.metrics.scoring import score_pair
from headswap.pipelines import create_pipeline
from headswap.preprocess import head_hair_mask_from_face, select_face_box, pil_to_rgb_np
from headswap.profiling.run_manifest import build_manifest, write_manifest


VARIANTS = {
    "geometry_lock": {
        "multi_person_swap_mode": "align_paste",
        "align_paste_krea2_refine": False,
        "align_paste_seamless_clone": True,
        "align_paste_full_affine": True,
    },
    "krea2_crop_spp_strict": {
        "multi_person_swap_mode": "krea2_crop",
        "clamp_crop_away_neighbors": True,
        "multi_crop_hard_freeze_neighbors": False,
    },
    "krea2_crop_neighbor_freeze": {
        "multi_person_swap_mode": "krea2_crop",
        "clamp_crop_away_neighbors": True,
        "multi_crop_hard_freeze_neighbors": True,
    },
    "deterministic_paste": {
        "multi_person_swap_mode": "align_paste",
        "align_paste_krea2_refine": False,
        "align_paste_pose_relock": False,
        "pre_color_match_strength": 0.0,
        "align_paste_post_color_match": 0.0,
    },
    "align_paste_refine_no_relock": {
        "multi_person_swap_mode": "align_paste",
        "align_paste_krea2_refine": True,
        "align_paste_pose_relock": False,
    },
    "align_paste_complete": {"multi_person_swap_mode": "align_paste"},
}


def _path(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", default=ROOT / "configs" / "krea2_identity_edit.yaml", type=Path)
    parser.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-mock", action="store_true")
    args = parser.parse_args()

    suite = json.loads(args.benchmark.read_text())
    pairs = list(suite.get("pairs") or [])
    if len(pairs) < 30:
        raise SystemExit(f"Frozen suite requires >=30 pairs; found {len(pairs)}.")
    if args.limit:
        pairs = pairs[: args.limit]
    variants = VARIANTS if args.variant == "all" else {args.variant: VARIANTS[args.variant]}
    cfg_base = load_config(args.config)
    root = args.benchmark.parent
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for variant_name, override in variants.items():
        for pair in pairs:
            body_path, face_path = _path(root, pair["body"]), _path(root, pair["face"])
            body, face = Image.open(body_path).convert("RGB"), Image.open(face_path).convert("RGB")
            cfg = {
                **cfg_base,
                **override,
                "save_debug": True,
                "body_face_policy": "index",
                "body_face_index": int(pair["target_index"]),
            }
            run_dir = args.out / variant_name / pair["id"]
            run_dir.mkdir(parents=True, exist_ok=False)
            selected, detected = select_face_box(
                pil_to_rgb_np(body), cfg.get("cache_dir", ROOT / ".cache"), policy="index", index=int(pair["target_index"])
            )
            detections = [
                {"index": i, "box": [f.x0, f.y0, f.x1, f.y1], "confidence": f.conf}
                for i, f in enumerate(detected)
            ]
            manifest = build_manifest(
                repo=ROOT, body_path=body_path, face_path=face_path, body=body, face=face,
                config=cfg, detections=detections, selected_index=int(pair["target_index"]),
                selected_box=[selected.x0, selected.y0, selected.x1, selected.y1] if selected else None,
                runtime={"variant": variant_name, "benchmark_version": suite.get("version")},
            )
            write_manifest(run_dir, manifest)
            pipe = create_pipeline(cfg, force_mock=args.force_mock)
            result = pipe.run(body, face, out_dir=run_dir)
            result.image.save(run_dir / "result.png")
            mask = head_hair_mask_from_face(body, pipe.cache_dir, face_box=selected)
            metric = score_pair(
                pair_id=pair["id"], pipeline=variant_name, body=body, face=face,
                result=result.image, latency_s=result.latency_s, head_mask=mask,
                cache_dir=pipe.cache_dir,
            ).to_dict()
            row = {**metric, "variant": variant_name, "tags": pair.get("tags", []), "run_dir": str(run_dir)}
            (run_dir / "metrics.json").write_text(json.dumps(row, indent=2, default=str))
            rows.append(row)
            print(f"{variant_name} {pair['id']}: identity={row['identity_cosine']} success={row['success']}")

    report = {"benchmark": str(args.benchmark), "variants": list(variants), "rows": rows}
    (args.out / "benchmark_report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
