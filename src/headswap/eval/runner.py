from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from headswap.config import load_config, project_root, resolve_out_dir
from headswap.eval.dataset import load_pairs
from headswap.metrics.scoring import PairMetrics, score_pair
from headswap.pipelines import create_pipeline
from headswap.preprocess import head_hair_mask_from_face


def run_eval(
    config_path: str | Path,
    out_dir: str | Path | None = None,
    force_mock: bool = False,
    limit: int | None = None,
    pair_ids: list[str] | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    pipe = create_pipeline(cfg, force_mock=force_mock)
    results_dir = resolve_out_dir(cfg, out_dir)
    images_dir = results_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs()
    if pair_ids:
        want = set(pair_ids)
        pairs = [p for p in pairs if p["id"] in want]
    if limit is not None:
        pairs = pairs[:limit]

    rows: list[dict[str, Any]] = []
    for p in pairs:
        body = Image.open(p["body_path"]).convert("RGB")
        face = Image.open(p["face_path"]).convert("RGB")
        pair_out = images_dir / p["id"]
        pair_out.mkdir(parents=True, exist_ok=True)
        result = pipe.run(body, face, out_dir=pair_out)
        out_path = pair_out / "result.png"
        result.image.save(out_path)

        mask = head_hair_mask_from_face(body, pipe.cache_dir)
        # Align mask to result size
        mask_r = mask.resize(result.image.size, Image.Resampling.NEAREST)
        body_r = body.resize(result.image.size, Image.Resampling.LANCZOS)
        metrics = score_pair(
            pair_id=p["id"],
            pipeline=cfg.get("name", pipe.name),
            body=body_r,
            face=face,
            result=result.image,
            latency_s=result.latency_s,
            head_mask=mask_r,
            cache_dir=pipe.cache_dir,
        )
        row = {
            **metrics.to_dict(),
            "difficulty": p.get("difficulty"),
            "tags": p.get("tags"),
            "result_path": str(out_path),
            "meta": result.meta,
        }
        rows.append(row)
        print(
            f"[{row['pipeline']}] {p['id']} success={row['success']} "
            f"id={row['identity_cosine']} lat={row['latency_s']:.2f}s reasons={row['fail_reasons']}"
        )

    summary = summarize(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config_path).resolve()),
        "pipeline": cfg.get("name", pipe.name),
        "force_mock": force_mock or bool(cfg.get("force_mock")),
        "n_pairs": len(rows),
        "summary": summary,
        "pairs": rows,
    }
    report_path = results_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    return report


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"success_rate": 0.0, "latency_p50": None, "latency_p95": None}
    succ = sum(1 for r in rows if r["success"])
    lats = sorted(r["latency_s"] for r in rows)

    def pct(vals, p):
        if not vals:
            return None
        idx = min(len(vals) - 1, max(0, int(round((p / 100) * (len(vals) - 1)))))
        return vals[idx]

    by_diff: dict[str, list] = defaultdict(list)
    for r in rows:
        by_diff[str(r.get("difficulty", "unknown"))].append(r)

    id_vals = [r["identity_cosine"] for r in rows if r.get("identity_cosine") is not None]
    body_vals = [r["body_preserve_psnr"] for r in rows if r.get("body_preserve_psnr") is not None]

    return {
        "success_rate": succ / len(rows),
        "n_success": succ,
        "latency_p50": pct(lats, 50),
        "latency_p95": pct(lats, 95),
        "latency_mean": sum(lats) / len(lats),
        "identity_mean": (sum(id_vals) / len(id_vals)) if id_vals else None,
        "body_psnr_mean": (sum(body_vals) / len(body_vals)) if body_vals else None,
        "by_difficulty": {
            k: {
                "n": len(v),
                "success_rate": sum(1 for x in v if x["success"]) / len(v),
            }
            for k, v in by_diff.items()
        },
    }


def compare_reports(
    report_paths: list[Path],
    latency_budget_s: float = 30.0,
    out_path: Path | None = None,
) -> dict[str, Any]:
    reports = []
    for p in report_paths:
        reports.append(json.loads(Path(p).read_text()))

    scored = []
    for r in reports:
        s = r["summary"]
        success = float(s.get("success_rate") or 0)
        lat_p95 = s.get("latency_p95")
        within = lat_p95 is not None and lat_p95 <= latency_budget_s
        # Primary: success rate; secondary: body PSNR; tertiary: latency
        body = s.get("body_psnr_mean") or 0.0
        identity = s.get("identity_mean") or 0.0
        # Soft penalty if over budget
        penalty = 0.0 if within else 0.15
        score = success * 0.6 + min(body / 40.0, 1.0) * 0.2 + min(max(identity, 0), 1.0) * 0.2 - penalty
        scored.append(
            {
                "pipeline": r.get("pipeline"),
                "path": str(p),
                "success_rate": success,
                "latency_p95": lat_p95,
                "within_latency_budget": within,
                "body_psnr_mean": s.get("body_psnr_mean"),
                "identity_mean": s.get("identity_mean"),
                "composite_score": score,
                "force_mock": r.get("force_mock", False),
                "n_pairs": r.get("n_pairs"),
            }
        )

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    winner = scored[0] if scored else None

    # Research-backed recommendation for when only mock ran / ties
    recommendation = {
        "promote": winner["pipeline"] if winner else None,
        "fallback": scored[1]["pipeline"] if len(scored) > 1 else None,
        "rationale": (
            "Winner selected by composite score = 0.6*success + 0.2*body_preserve + 0.2*identity "
            f"with -0.15 penalty if p95 latency > {latency_budget_s}s."
        ),
        "production_note": (
            "If runs are force_mock=true, promote FLUX.2 Klein 4B mask-crop-stitch as the "
            "primary production candidate per research; re-confirm with GPU metrics before cutover."
        ),
    }

    # Prefer naming Klein when mock-only comparison
    if winner and all(x.get("force_mock") for x in scored):
        recommendation["promote"] = "klein4b_mask_crop_stitch"
        recommendation["fallback"] = "qwen_improved_mask_crop"
        recommendation["empirical_mock_leader"] = winner["pipeline"]
        recommendation["rationale"] += (
            " Mock-only run: empirical ranking is harness-smoke only; "
            "recommended production promote is Klein 4B with Qwen-improved as fallback."
        )

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latency_budget_s": latency_budget_s,
        "ranking": scored,
        "recommendation": recommendation,
    }
    out_path = out_path or (project_root() / "results" / "comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    md = project_root() / "results" / "COMPARISON.md"
    lines = [
        "# Head-swap V2 comparison",
        "",
        f"Generated: {out['generated_at']}",
        f"Latency budget (p95): {latency_budget_s}s",
        "",
        "## Ranking",
        "",
        "| Pipeline | Score | Success | ID mean | Body PSNR | p95 lat | Budget OK | Mock |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for x in scored:
        lines.append(
            f"| {x['pipeline']} | {x['composite_score']:.3f} | {x['success_rate']:.1%} | "
            f"{x['identity_mean']} | {x['body_psnr_mean']} | {x['latency_p95']} | "
            f"{'Y' if x['within_latency_budget'] else 'N'} | {'Y' if x['force_mock'] else 'N'} |"
        )
    rec = recommendation
    lines += [
        "",
        "## Recommendation",
        "",
        f"- **Promote:** `{rec['promote']}`",
        f"- **Fallback:** `{rec.get('fallback')}`",
        f"- {rec['rationale']}",
        f"- {rec['production_note']}",
        "",
    ]
    md.write_text("\n".join(lines))
    print(md.read_text())
    return out
