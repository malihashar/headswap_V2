#!/usr/bin/env python3
"""A/B: multi_person_edit_mode crop_stitch vs full_frame on the same cases.

Default product mode stays crop_stitch. This script runs BOTH methods and
writes a comparison report. Promote full_frame only if it consistently wins.

Usage (GPU + ComfyUI required for real scores):
  PYTHONPATH=src python scripts/ab_multi_edit_modes.py \\
    --body path/to/group.png --face path/to/id.png \\
    --out results/_ab_multi_edit_modes

  # Multiple cases from a JSON list:
  # [{"id":"t1","body":"...","face":"..."}, ...]
  PYTHONPATH=src python scripts/ab_multi_edit_modes.py --cases cases.json

Metrics per mode:
  - identity_cosine (InsightFace when available)
  - body_preserve_psnr (outside selected-head mask)
  - neighbor_psnr (mean PSNR inside other face boxes — higher = less collateral)
  - seam_edge_delta (crop_stitch only; lower often better)
  - latency_s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.metrics.scoring import (
    body_preserve_score,
    identity_cosine,
    psnr,
    seam_edge_delta,
)
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline, reset_shared_krea2_runtime
from headswap.preprocess import (
    FaceBox,
    head_hair_mask_from_face,
    pil_to_rgb_np,
    resize_max_keep_ar,
    select_face_box,
)
import numpy as np
import cv2


def _side_by_side(a: Image.Image, b: Image.Image, la: str, lb: str) -> Image.Image:
    def fit(im: Image.Image, h: int = 512) -> Image.Image:
        im = im.convert("RGB")
        s = h / max(1, im.size[1])
        return im.resize((max(1, int(im.size[0] * s)), h), Image.Resampling.LANCZOS)

    aa, bb = fit(a), fit(b)
    canvas = Image.new("RGB", (aa.size[0] + bb.size[0] + 16, aa.size[1] + 28), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    d.text((4, 6), la, fill=(160, 255, 160))
    d.text((aa.size[0] + 20, 6), lb, fill=(255, 200, 120))
    canvas.paste(aa, (0, 28))
    canvas.paste(bb, (aa.size[0] + 16, 28))
    return canvas


def _neighbor_psnr(
    body: Image.Image,
    result: Image.Image,
    selected: FaceBox,
    all_faces: list[FaceBox],
) -> float | None:
    """Mean PSNR inside non-selected face boxes (untouched neighbors)."""
    b = np.asarray(body.convert("RGB"))
    r = np.asarray(result.convert("RGB"))
    if r.shape != b.shape:
        r = cv2.resize(r, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    scores: list[float] = []
    for f in all_faces:
        if (
            f.x0 == selected.x0
            and f.y0 == selected.y0
            and f.x1 == selected.x1
            and f.y1 == selected.y1
        ):
            continue
        x0, y0, x1, y1 = f.x0, f.y0, f.x1, f.y1
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            continue
        scores.append(psnr(b[y0:y1, x0:x1], r[y0:y1, x0:x1]))
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _score_mode(
    *,
    body: Image.Image,
    face: Image.Image,
    result: Image.Image,
    latency_s: float,
    selected: FaceBox,
    all_faces: list[FaceBox],
    cache_dir: Path,
    mode: str,
) -> dict[str, Any]:
    mask = head_hair_mask_from_face(
        body, cache_dir, face_box=selected
    ).resize(result.size)
    body_r = body.resize(result.size, Image.Resampling.LANCZOS)
    id_cos = identity_cosine(face, result)
    body_psnr = body_preserve_score(body_r, result, mask)
    neighbor = _neighbor_psnr(body_r, result, selected, all_faces)
    seam = seam_edge_delta(result, mask) if mode == "crop_stitch" else None
    return {
        "mode": mode,
        "latency_s": round(float(latency_s), 3),
        "identity_cosine": None if id_cos is None else round(float(id_cos), 4),
        "body_preserve_psnr": None if body_psnr is None else round(float(body_psnr), 2),
        "neighbor_face_psnr": None if neighbor is None else round(float(neighbor), 2),
        "seam_edge_delta": None if seam is None else round(float(seam), 2),
        "result_size": list(result.size),
    }


def _run_mode(
    body: Image.Image,
    face: Image.Image,
    base_cfg: dict,
    mode: str,
    cache_dir: Path,
    out_dir: Path,
) -> tuple[Image.Image, dict[str, Any], float]:
    cfg = deepcopy(base_cfg)
    cfg["multi_person_edit_mode"] = mode
    cfg["save_debug"] = True
    cfg["verbose"] = False
    # Fresh pipe per mode so cfg is applied cleanly.
    reset_shared_krea2_runtime()
    pipe = Krea2IdentityEditPipeline(cfg=cfg, cache_dir=cache_dir)
    t0 = time.perf_counter()
    result = pipe.run(body, face, out_dir=out_dir / mode)
    latency = time.perf_counter() - t0
    result.image.save(out_dir / mode / "result.png")
    return result.image, result.meta, latency


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.cases:
        data = json.loads(Path(args.cases).read_text())
        if isinstance(data, dict) and "cases" in data:
            data = data["cases"]
        return list(data)
    if not args.body or not args.face:
        raise SystemExit("Provide --body/--face or --cases JSON")
    return [{"id": "case_0", "body": str(args.body), "face": str(args.face)}]


def _plant_faces(image_size: tuple[int, int], n: int = 3) -> list[FaceBox]:
    """Deterministic face boxes for dry-run / CI when OpenCV cannot detect."""
    w, h = image_size
    boxes: list[FaceBox] = []
    fw, fh = max(40, w // (n + 1)), max(50, h // 4)
    y0 = max(10, h // 8)
    for i in range(n):
        cx = int((i + 0.5) * w / n)
        x0 = max(0, cx - fw // 2)
        x1 = min(w, x0 + fw)
        y1 = min(h, y0 + fh)
        boxes.append(FaceBox(x0, y0, x1, y1, conf=0.99 - 0.01 * i))
    return boxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=Path)
    ap.add_argument("--face", type=Path)
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "_ab_multi_edit_modes")
    ap.add_argument("--dry-run", action="store_true", help="Prep only; skip Krea2 sample")
    ap.add_argument(
        "--plant-faces",
        type=int,
        default=0,
        help="If >0, plant N face boxes (for dry-run when detector finds <2 faces)",
    )
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(ROOT / "configs" / "krea2_identity_edit.yaml")

    cases = _load_cases(args)
    summary: list[dict[str, Any]] = []
    wins = {"crop_stitch": 0, "full_frame": 0, "tie": 0}

    for case in cases:
        cid = str(case.get("id") or Path(case["body"]).stem)
        case_dir = out / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        body = Image.open(case["body"]).convert("RGB")
        face = Image.open(case["face"]).convert("RGB")

        body_full = resize_max_keep_ar(
            body, int(base_cfg.get("max_body_dim", 1024)), div_by=16
        )
        selected, all_faces = select_face_box(
            pil_to_rgb_np(body_full),
            cache,
            policy=str(base_cfg.get("body_face_policy", "largest")),
        )
        if (selected is None or len(all_faces) < 2) and args.plant_faces >= 2:
            all_faces = _plant_faces(body_full.size, n=int(args.plant_faces))
            # Prefer rightmost as selected (matches common group-shot intent).
            selected = max(all_faces, key=lambda f: 0.5 * (f.x0 + f.x1))
            print(f"[warn] {cid}: planted {len(all_faces)} face boxes for A/B")
        if selected is None or len(all_faces) < 2:
            print(f"[skip] {cid}: need >=2 faces (got {0 if selected is None else len(all_faces)})")
            summary.append({"id": cid, "skipped": True, "reason": "not_multi_person"})
            continue

        print(f"\n=== {cid}: {len(all_faces)} faces, selected={ [selected.x0,selected.y0,selected.x1,selected.y1] } ===")
        row: dict[str, Any] = {
            "id": cid,
            "faces": len(all_faces),
            "selected_box": [selected.x0, selected.y0, selected.x1, selected.y1],
            "modes": {},
        }

        if args.dry_run:
            # Build inputs only for both modes via pipeline helpers.
            reset_shared_krea2_runtime()
            pipe = Krea2IdentityEditPipeline(cfg=dict(base_cfg), cache_dir=cache)
            from headswap.preprocess import crop_face_reference

            face_crop = crop_face_reference(face, cache)
            for mode in ("crop_stitch", "full_frame"):
                if mode == "full_frame":
                    built = pipe._build_full_frame_inputs(
                        body_full, face_crop, div_by=16,
                        selected_face=selected, all_faces=all_faces,
                    )
                    built["scene"].save(case_dir / f"{mode}_scene.png")
                    built["person"].save(case_dir / f"{mode}_person.png")
                    row["modes"][mode] = {"dry_run": True, **built["diag"]}
                else:
                    flags = pipe._tight_crop_flags(body_full, selected, all_faces)
                    built = pipe._build_scene_person(
                        body_full, face_crop, selected, div_by=16,
                        use_tight=bool(flags["use_tight"]),
                        top_ext=float(flags["top_ext"]),
                        side_ext=float(flags["side_ext"]),
                        bot_ext=float(flags["bot_ext"]),
                        expand_px=int(flags["expand_px"]),
                        crop_pad=int(flags["crop_pad"]),
                        all_faces=all_faces,
                        isolate_selected=bool(flags.get("isolate_selected")),
                    )
                    built["scene"].save(case_dir / f"{mode}_scene.png")
                    built["person"].save(case_dir / f"{mode}_person.png")
                    row["modes"][mode] = {"dry_run": True, **(built.get("diag") or {})}
            summary.append(row)
            continue

        results_img: dict[str, Image.Image] = {}
        for mode in ("crop_stitch", "full_frame"):
            print(f"  running {mode}…")
            try:
                img, meta, latency = _run_mode(
                    body, face, base_cfg, mode, cache, case_dir
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {mode} FAILED: {type(exc).__name__}: {exc}")
                row["modes"][mode] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            scores = _score_mode(
                body=body_full,
                face=face,
                result=img,
                latency_s=latency,
                selected=selected,
                all_faces=all_faces,
                cache_dir=cache,
                mode=mode,
            )
            scores["edit_mode_meta"] = meta.get("edit_mode")
            scores["scene_size"] = meta.get("scene_size")
            row["modes"][mode] = scores
            results_img[mode] = img
            print(
                f"  {mode}: id={scores['identity_cosine']} "
                f"body_psnr={scores['body_preserve_psnr']} "
                f"neighbor_psnr={scores['neighbor_face_psnr']} "
                f"seam={scores['seam_edge_delta']} "
                f"lat={scores['latency_s']}s"
            )

        if "crop_stitch" in results_img and "full_frame" in results_img:
            _side_by_side(
                results_img["crop_stitch"],
                results_img["full_frame"],
                "crop_stitch",
                "full_frame",
            ).save(case_dir / "side_by_side.png")

            # Lightweight win heuristic (higher better except seam/latency).
            def rank(m: dict) -> float:
                idv = m.get("identity_cosine")
                bp = m.get("body_preserve_psnr")
                np_ = m.get("neighbor_face_psnr")
                seam = m.get("seam_edge_delta")
                lat = float(m.get("latency_s") or 1e9)
                score = 0.0
                if idv is not None:
                    score += 0.35 * float(idv)
                if bp is not None:
                    score += 0.25 * min(float(bp), 45.0) / 45.0
                if np_ is not None:
                    score += 0.25 * min(float(np_), 45.0) / 45.0
                if seam is not None:
                    score += 0.10 * max(0.0, 1.0 - float(seam) / 80.0)
                score += 0.05 * max(0.0, 1.0 - lat / 300.0)
                return score

            cs = row["modes"].get("crop_stitch") or {}
            ff = row["modes"].get("full_frame") or {}
            if "error" not in cs and "error" not in ff:
                scs, sff = rank(cs), rank(ff)
                if abs(scs - sff) < 0.02:
                    winner = "tie"
                elif sff > scs:
                    winner = "full_frame"
                else:
                    winner = "crop_stitch"
                wins[winner] += 1
                row["winner"] = winner
                row["rank_scores"] = {"crop_stitch": round(scs, 4), "full_frame": round(sff, 4)}
                print(f"  winner={winner} ranks={row['rank_scores']}")

        summary.append(row)
        (case_dir / "scores.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    report = {
        "default_product_mode": "crop_stitch",
        "note": (
            "Do not flip multi_person_edit_mode to full_frame in yaml unless "
            "full_frame consistently wins across cases."
        ),
        "wins": wins,
        "cases": summary,
    }
    (out / "AB_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Multi-person edit mode A/B",
        "",
        f"Product default remains **crop_stitch**.",
        f"Wins: crop_stitch={wins['crop_stitch']} full_frame={wins['full_frame']} tie={wins['tie']}",
        "",
    ]
    for row in summary:
        if row.get("skipped"):
            lines.append(f"- {row['id']}: skipped ({row.get('reason')})")
            continue
        lines.append(f"- {row['id']}: winner={row.get('winner', 'n/a')} modes={list((row.get('modes') or {}).keys())}")
    if wins["full_frame"] > wins["crop_stitch"] and wins["full_frame"] >= 2:
        lines.append(
            "\nRecommendation: full_frame is ahead — consider setting "
            "`multi_person_edit_mode: full_frame` after manual visual review."
        )
    else:
        lines.append(
            "\nRecommendation: keep `multi_person_edit_mode: crop_stitch` "
            "(default) until full_frame consistently outperforms."
        )
    (out / "AB_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
