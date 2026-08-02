#!/usr/bin/env python3
"""Compare Krea2 full-image synth vs localized production pipeline.

Experimental only — does not modify the localized pipeline.

Examples:
  # Mock smoke (no GPU):
  PYTHONPATH=src python scripts/compare_krea2_full_vs_localized.py --mock \\
    --body data/custom/body.png --face data/custom/face.png

  # GPU (Colab/Kaggle with Comfy + Krea2 weights):
  PYTHONPATH=src python scripts/compare_krea2_full_vs_localized.py \\
    --body data/custom/body.png --face data/custom/face.png \\
    --out results/_krea2_full_vs_localized
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from headswap.config import load_config
from headswap.metrics.full_synth_eval import make_side_by_side, score_full_synth_case
from headswap.pipelines import create_pipeline
from headswap.prompting.scene_describe import describe_scene


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _default_cases() -> list[dict]:
    cases: list[dict] = []
    custom_body = ROOT / "data" / "custom" / "body.png"
    custom_face = ROOT / "data" / "custom" / "face.png"
    if custom_body.is_file() and custom_face.is_file():
        cases.append(
            {
                "id": "custom_001",
                "body": str(custom_body),
                "face": str(custom_face),
            }
        )
    # Night group photo from chat assets (if present).
    assets = Path.home() / ".cursor" / "projects" / "Users-ali-Repos-headswap-V2" / "assets"
    night = assets / "image-c5c00ab0-5981-49b6-bb56-a9b08ecd1273.png"
    # Prefer a face identity from custom if available.
    if night.is_file() and custom_face.is_file():
        cases.append(
            {
                "id": "night_group_001",
                "body": str(night),
                "face": str(custom_face),
            }
        )
    eval_body = ROOT / "data" / "eval" / "bodies" / "custom_001.png"
    eval_face = ROOT / "data" / "eval" / "faces" / "custom_001.png"
    if eval_body.is_file() and eval_face.is_file():
        # Avoid duplicate if same as data/custom
        if not any(c["id"] == "custom_001" for c in cases):
            cases.append(
                {
                    "id": "eval_custom_001",
                    "body": str(eval_body),
                    "face": str(eval_face),
                }
            )
    return cases


def _run_one(pipe, body: Image.Image, face: Image.Image, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = pipe.run(body, face, out_dir=out_dir)
    result.image.save(out_dir / "result.png")
    (out_dir / "meta.json").write_text(
        json.dumps(result.meta, indent=2, default=str), encoding="utf-8"
    )
    return result


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--body", type=Path, default=None)
    ap.add_argument("--face", type=Path, default=None)
    ap.add_argument("--cases", type=Path, default=None, help="Optional JSON list of cases")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "_krea2_full_vs_localized",
    )
    ap.add_argument(
        "--localized-config",
        type=Path,
        default=ROOT / "configs" / "krea2_identity_edit.yaml",
    )
    ap.add_argument(
        "--full-config",
        type=Path,
        default=ROOT / "configs" / "krea2_full_image_synth.yaml",
    )
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--face-policy", default=None, help="Override face_select_policy")
    ap.add_argument("--face-index", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.body and args.face:
        cases = [{"id": "case_0", "body": str(args.body), "face": str(args.face)}]
    elif args.cases:
        cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    else:
        cases = _default_cases()
    if not cases:
        print("No cases found. Pass --body/--face or --cases.", file=sys.stderr)
        return 2
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    loc_cfg = load_config(args.localized_config)
    full_cfg = load_config(args.full_config)
    if args.face_policy:
        full_cfg["face_select_policy"] = args.face_policy
        loc_cfg["face_select_policy"] = args.face_policy
    if args.face_index is not None:
        full_cfg["face_index"] = int(args.face_index)
        loc_cfg["face_index"] = int(args.face_index)

    loc_pipe = create_pipeline(loc_cfg, force_mock=bool(args.mock))
    full_pipe = create_pipeline(full_cfg, force_mock=bool(args.mock))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "localized_config": str(args.localized_config),
        "full_config": str(args.full_config),
        "mock": bool(args.mock),
        "cases": [],
    }

    md_lines = [
        "# Krea2 full-image synth vs localized",
        "",
        f"- mock: `{bool(args.mock)}`",
        f"- localized: `{args.localized_config.name}`",
        f"- full-image: `{args.full_config.name}`",
        "",
    ]

    for case in cases:
        cid = str(case["id"])
        body = _load_rgb(Path(case["body"]))
        face = _load_rgb(Path(case["face"]))
        case_dir = out_root / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        body.save(case_dir / "00_body.png")
        face.save(case_dir / "00_face.png")

        desc, selected, all_faces = describe_scene(
            body,
            full_pipe.cache_dir,
            face_index=int(full_cfg.get("face_index", 0)),
            face_policy=str(full_cfg.get("face_select_policy", "largest")),
            cfg=full_cfg,
        )

        print(f"\n=== {cid}: localized ===", flush=True)
        loc_res = _run_one(loc_pipe, body, face, case_dir / "localized")
        print(f"\n=== {cid}: full_image_synth ===", flush=True)
        full_res = _run_one(full_pipe, body, face, case_dir / "full_image_synth")

        loc_metrics = score_full_synth_case(
            cid,
            "localized",
            body,
            face,
            loc_res.image,
            selected=selected,
            all_faces=all_faces,
            cache_dir=full_pipe.cache_dir,
            latency_s=loc_res.latency_s,
            prompt=str(loc_res.meta.get("prompt") or ""),
        )
        full_metrics = score_full_synth_case(
            cid,
            "full_image_synth",
            body,
            face,
            full_res.image,
            selected=selected,
            all_faces=all_faces,
            cache_dir=full_pipe.cache_dir,
            latency_s=full_res.latency_s,
            prompt=str(full_res.meta.get("prompt") or ""),
        )

        sbs = make_side_by_side(body, loc_res.image, full_res.image)
        sbs_path = case_dir / "side_by_side.png"
        sbs.save(sbs_path)

        case_entry = {
            "id": cid,
            "body": case["body"],
            "face": case["face"],
            "faces_detected": len(all_faces),
            "selected_role": desc.selected_role,
            "scene_description": desc.to_dict(),
            "localized": loc_metrics.to_dict(),
            "full_image_synth": full_metrics.to_dict(),
            "side_by_side": str(sbs_path),
        }
        report["cases"].append(case_entry)
        (case_dir / "metrics.json").write_text(
            json.dumps(case_entry, indent=2), encoding="utf-8"
        )

        md_lines.extend(
            [
                f"## {cid}",
                "",
                f"- faces detected: {len(all_faces)}",
                f"- selected: {desc.selected_role}",
                f"- side-by-side: `{sbs_path}`",
                "",
                "| metric | localized | full_image_synth |",
                "|---|---:|---:|",
                f"| identity_cosine | {_fmt(loc_metrics.identity_cosine)} | {_fmt(full_metrics.identity_cosine)} |",
                f"| expression_landmark_l2 (↓) | {_fmt(loc_metrics.expression_landmark_l2)} | {_fmt(full_metrics.expression_landmark_l2)} |",
                f"| pose_landmark_l2 (↓) | {_fmt(loc_metrics.pose_landmark_l2)} | {_fmt(full_metrics.pose_landmark_l2)} |",
                f"| head_size_ratio (~1) | {_fmt(loc_metrics.head_size_ratio)} | {_fmt(full_metrics.head_size_ratio)} |",
                f"| clothing_psnr (↑) | {_fmt(loc_metrics.clothing_psnr)} | {_fmt(full_metrics.clothing_psnr)} |",
                f"| background_psnr (↑) | {_fmt(loc_metrics.background_psnr)} | {_fmt(full_metrics.background_psnr)} |",
                f"| neighbor_identity_mean (↑) | {_fmt(loc_metrics.neighbor_identity_mean)} | {_fmt(full_metrics.neighbor_identity_mean)} |",
                f"| latency_s | {_fmt(loc_metrics.latency_s)} | {_fmt(full_metrics.latency_s)} |",
                "",
            ]
        )

    # Rough winner by identity + neighbor preservation when both present.
    wins = {"localized": 0, "full_image_synth": 0, "tie": 0}
    for c in report["cases"]:
        a = c["localized"].get("identity_cosine")
        b = c["full_image_synth"].get("identity_cosine")
        na = c["localized"].get("neighbor_identity_mean")
        nb = c["full_image_synth"].get("neighbor_identity_mean")
        score_a = (a or 0) + 0.25 * (na or 0)
        score_b = (b or 0) + 0.25 * (nb or 0)
        if a is None and b is None:
            wins["tie"] += 1
        elif score_b > score_a + 0.02:
            wins["full_image_synth"] += 1
        elif score_a > score_b + 0.02:
            wins["localized"] += 1
        else:
            wins["tie"] += 1
    report["wins"] = wins
    md_lines.extend(
        [
            "## Summary",
            "",
            f"- localized wins: {wins['localized']}",
            f"- full_image_synth wins: {wins['full_image_synth']}",
            f"- ties: {wins['tie']}",
            "",
            "Heuristic win = higher identity_cosine + 0.25× neighbor_identity_mean.",
            "Inspect side-by-side images for expression / realism judgments.",
            "",
        ]
    )

    (out_root / "AB_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_root / "AB_REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {out_root / 'AB_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
