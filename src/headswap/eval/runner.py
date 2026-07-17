from __future__ import annotations

import json
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from headswap.config import load_config, project_root, resolve_out_dir
from headswap.eval.dataset import load_pairs
from headswap.metrics.scoring import PairMetrics, score_pair
from headswap.pipelines import create_pipeline
from headswap.pipelines.base import PipelineResult
from headswap.pipelines.errors import PipelineRunError
from headswap.preprocess import head_hair_mask_from_face
from headswap.profiling.reporting import emit_run_finished, flush_stdio


def _write_metrics_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))


def _error_row(
    *,
    pair: dict[str, Any],
    pipeline: str,
    message: str,
    meta: dict[str, Any] | None = None,
    latency_s: float | None = None,
) -> dict[str, Any]:
    return {
        "pair_id": pair["id"],
        "pipeline": pipeline,
        "latency_s": latency_s,
        "identity_cosine": None,
        "body_preserve_psnr": None,
        "seam_edge_delta": None,
        "face_detected": False,
        "success": False,
        "fail_reasons": [f"run_error:{message[:200]}"],
        "difficulty": pair.get("difficulty"),
        "tags": pair.get("tags"),
        "result_path": None,
        "meta": {**(meta or {}), "run_error": message},
    }


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
    report_path = results_dir / "metrics.json"
    pipeline_name = str(cfg.get("name", pipe.name))

    pairs = load_pairs()
    if pair_ids:
        want = set(pair_ids)
        pairs = [p for p in pairs if p["id"] in want]
    if limit is not None:
        pairs = pairs[:limit]

    rows: list[dict[str, Any]] = []
    report_base = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config_path).resolve()),
        "pipeline": pipeline_name,
        "force_mock": force_mock or bool(cfg.get("force_mock")),
    }

    for p in pairs:
        body = Image.open(p["body_path"]).convert("RGB")
        face = Image.open(p["face_path"]).convert("RGB")
        pair_out = images_dir / p["id"]
        pair_out.mkdir(parents=True, exist_ok=True)

        result: PipelineResult | None = None
        pair_error: str | None = None

        try:
            try:
                result = pipe.run(body, face, out_dir=pair_out)
            except PipelineRunError as exc:
                pair_error = str(exc)
                result = exc.to_partial_result()
                if result is None:
                    emit_run_finished(
                        pipeline=pipeline_name,
                        pair_id=p["id"],
                        result_meta=exc.meta,
                        latency_s=exc.latency_s,
                        had_error=True,
                    )
                    rows.append(
                        _error_row(
                            pair=p,
                            pipeline=pipeline_name,
                            message=pair_error,
                            meta=exc.meta,
                            latency_s=exc.latency_s,
                        )
                    )
                    _write_metrics_report(
                        report_path,
                        {**report_base, "n_pairs": len(rows), "summary": summarize(rows), "pairs": rows},
                    )
                    flush_stdio()
                    print(
                        f"[{pipeline_name}] {p['id']} FAILED lat={exc.latency_s:.2f}s "
                        f"(profile saved to metrics.json)"
                    )
                    continue
                result.meta.setdefault("run_error", pair_error)

            emit_run_finished(
                pipeline=pipeline_name,
                pair_id=p["id"],
                result_meta=result.meta,
                latency_s=result.latency_s,
                had_error=pair_error is not None,
            )

            out_path: str | None = None
            metrics: PairMetrics | None = None
            post_error: str | None = None
            try:
                out_file = pair_out / "result.png"
                result.image.save(out_file)
                out_path = str(out_file)

                mask = head_hair_mask_from_face(body, pipe.cache_dir)
                mask_r = mask.resize(result.image.size, Image.Resampling.NEAREST)
                body_r = body.resize(result.image.size, Image.Resampling.LANCZOS)
                metrics = score_pair(
                    pair_id=p["id"],
                    pipeline=pipeline_name,
                    body=body_r,
                    face=face,
                    result=result.image,
                    latency_s=result.latency_s,
                    head_mask=mask_r,
                    cache_dir=pipe.cache_dir,
                )
            except Exception as post_exc:
                post_error = str(post_exc)
                traceback.print_exc()

            if metrics is not None:
                row = {
                    **metrics.to_dict(),
                    "difficulty": p.get("difficulty"),
                    "tags": p.get("tags"),
                    "result_path": out_path,
                    "meta": result.meta,
                }
            else:
                row = _error_row(
                    pair=p,
                    pipeline=pipeline_name,
                    message=post_error or pair_error or "post_pipeline_failed",
                    meta=result.meta,
                    latency_s=result.latency_s,
                )
                row["result_path"] = out_path

            if post_error:
                row["meta"] = {**row.get("meta", {}), "post_run_error": post_error}

            rows.append(row)
            lat_val = row.get("latency_s")
            lat_txt = f"{float(lat_val):.2f}s" if lat_val is not None else "n/a"
            print(
                f"[{row['pipeline']}] {p['id']} success={row['success']} "
                f"id={row.get('identity_cosine')} lat={lat_txt} "
                f"reasons={row.get('fail_reasons')}"
            )
        except Exception as exc:
            traceback.print_exc()
            rows.append(
                _error_row(
                    pair=p,
                    pipeline=pipeline_name,
                    message=str(exc),
                    latency_s=None,
                )
            )
            print(f"[{pipeline_name}] {p['id']} FAILED before metrics: {exc}")

        _write_metrics_report(
            report_path,
            {**report_base, "n_pairs": len(rows), "summary": summarize(rows), "pairs": rows},
        )
        flush_stdio()

    report = {**report_base, "n_pairs": len(rows), "summary": summarize(rows), "pairs": rows}
    print(f"Wrote {report_path}")
    flush_stdio()
    return report


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"success_rate": 0.0, "latency_p50": None, "latency_p95": None}
    succ = sum(1 for r in rows if r["success"])
    lats = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)

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
        "latency_mean": (sum(lats) / len(lats)) if lats else None,
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
