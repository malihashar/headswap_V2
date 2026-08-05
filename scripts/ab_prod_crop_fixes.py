#!/usr/bin/env python3
"""A/B: production crop_stitch vs Procrustes vs wide-crop (config-gated).

Arms (yaml production defaults stay crop_stitch / tight / procrustes OFF):

  prod             — current production (crop_stitch, tight, no Procrustes)
  prod_procrustes  — production + enable_procrustes_correction=true
  prod_wide_crop   — production + crop_margin_mode=wide (~1MP scene)

Usage (GPU + ComfyUI):
  PYTHONPATH=src python scripts/ab_prod_crop_fixes.py \\
    --out results/_ab_prod_crop_fixes

  PYTHONPATH=src python scripts/ab_prod_crop_fixes.py --prep-only
  PYTHONPATH=src python scripts/ab_prod_crop_fixes.py --arms prod,prod_procrustes
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline, reset_shared_krea2_runtime
from headswap.preprocess import pil_to_rgb_np, resize_max_keep_ar, select_face_box

_spec = importlib.util.spec_from_file_location(
    "ab_full_frame_author_parity",
    ROOT / "scripts" / "ab_full_frame_author_parity.py",
)
assert _spec and _spec.loader
_parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parity)

score_arm = _parity.score_arm
_side_by_side = _parity._side_by_side
_assert_not_mock = _parity._assert_not_mock
_mp = _parity._mp

LORA_R64 = "krea2_identity_edit_v1_2_r64.safetensors"
DEFAULT_OUT = ROOT / "results" / "_ab_prod_crop_fixes"
DEFAULT_FACE = ROOT / "data" / "custom" / "face.png"

ARMS: list[dict[str, Any]] = [
    {
        "id": "prod",
        "label": "production crop_stitch (tight, no Procrustes)",
        "multi_person_edit_mode": "crop_stitch",
        "identity_lora_name": LORA_R64,
        "ref_boost": 3.5,
        "crop_margin_mode": "tight",
        "enable_procrustes_correction": False,
    },
    {
        "id": "prod_procrustes",
        "label": "production + Procrustes correction",
        "multi_person_edit_mode": "crop_stitch",
        "identity_lora_name": LORA_R64,
        "ref_boost": 3.5,
        "crop_margin_mode": "tight",
        "enable_procrustes_correction": True,
    },
    {
        "id": "prod_wide_crop",
        "label": "production + wide crop (~1MP)",
        "multi_person_edit_mode": "crop_stitch",
        "identity_lora_name": LORA_R64,
        "ref_boost": 3.5,
        "crop_margin_mode": "wide",
        "enable_procrustes_correction": False,
    },
]


def _cfg_for_arm(base: dict, arm: dict[str, Any]) -> dict:
    cfg = deepcopy(base)
    # Keep production crop_stitch path; only flip the gated experiment knobs.
    cfg["multi_person_edit_mode"] = "crop_stitch"
    cfg["multi_person_swap_mode"] = "krea2_crop"
    cfg["identity_lora_name"] = arm["identity_lora_name"]
    cfg["ref_boost"] = float(arm["ref_boost"])
    cfg["crop_margin_mode"] = str(arm.get("crop_margin_mode", "tight"))
    cfg["enable_procrustes_correction"] = bool(
        arm.get("enable_procrustes_correction", False)
    )
    # Explicitly leave full_frame overrides alone / null.
    cfg["full_frame_identity_lora_name"] = None
    cfg["full_frame_ref_boost"] = None
    cfg["full_frame_procrustes_align"] = False
    cfg["single_person_parity"] = True
    cfg["mask_crop_stitch"] = True
    cfg["save_debug"] = True
    cfg["verbose"] = False
    return cfg


def write_report(path: Path, payload: dict[str, Any]) -> None:
    arms = payload["arms"]
    lines = [
        "# Production crop fixes A/B (Procrustes vs wide crop)",
        "",
        f"_Generated {payload.get('generated_at', '')}_",
        "",
        "Production yaml defaults remain `crop_stitch` / `crop_margin_mode: tight` /",
        "`enable_procrustes_correction: false`. This harness only overrides per-arm.",
        "",
        "## Arms",
        "",
        "| id | crop_margin_mode | enable_procrustes_correction | LoRA | ref_boost |",
        "|---|---|:---:|---|---:|",
    ]
    for a in arms:
        lines.append(
            f"| `{a['id']}` | {a.get('crop_margin_mode')} | "
            f"{a.get('enable_procrustes_correction')} | "
            f"`{a['identity_lora_name']}` | {a['ref_boost']} |"
        )
    lines += [
        "",
        "## Key metrics (unchanged)",
        "",
        "`head_to_body_scale_ratio`, `identity_cosine`, `expression_landmark_l2`,",
        "`gaze_eye_line_delta_deg`, `neighbor_identity_mean`, `latency_s`.",
        "",
    ]
    for case in payload.get("cases") or []:
        cid = case["id"]
        lines += [f"## Case `{cid}`", ""]
        if case.get("skipped"):
            lines += [f"_Skipped: {case.get('reason')}_", ""]
            continue
        lines += [
            f"- body: `{case.get('body')}`",
            f"- faces: **{case.get('faces_detected')}**",
            f"- selected: `{case.get('selected_box')}`",
            "",
            f"![side_by_side]({cid}/side_by_side.png)",
            "",
            "| arm | mode | scene MP | head_scale | id_cos | expr_l2 | "
            "gaze_Δ° | neighbor_id | latency_s | procrustes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm_id, sc in (case.get("scores") or {}).items():
            if sc.get("error"):
                lines.append(f"| `{arm_id}` | ERR | — | — | — | — | — | — | — | — |")
                continue
            meta = (case.get("metas") or {}).get(arm_id) or {}
            proc = meta.get("procrustes_correction") or {}
            proc_s = (
                f"ok scale={proc.get('scale')}"
                if proc.get("procrustes")
                else (proc.get("procrustes_reason") or sc.get("crop_margin_mode") or "—")
            )
            ex = sc.get("expression_landmark_l2")
            lines.append(
                "| `{arm}` | {mode} | {mp} | **{hs}** | {idc} | {ex} | "
                "{gz} | {nb} | {lat} | {proc} |".format(
                    arm=arm_id,
                    mode=sc.get("edit_mode") or meta.get("crop_margin_mode"),
                    mp=sc.get("scene_megapixels"),
                    hs=sc.get("head_to_body_scale_ratio"),
                    idc=sc.get("identity_cosine"),
                    ex=None if ex is None else round(float(ex), 4),
                    gz=sc.get("gaze_eye_line_delta_deg"),
                    nb=sc.get("neighbor_identity_mean"),
                    lat=sc.get("latency_s"),
                    proc=proc_s,
                )
            )
        lines.append("")
    j = payload.get("judgment") or {}
    lines += [
        "## Notes",
        "",
        j.get(
            "summary",
            "_Run on Colab GPU to fill metrics. Do not flip yaml defaults from one case._",
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _default_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    face = DEFAULT_FACE
    for cid, body, policy in (
        (
            "night_group_001",
            ROOT / "results" / "_krea2_full_vs_localized" / "night_group_001" / "00_body.png",
            "rightmost",
        ),
        (
            "geometry_lock_multi",
            ROOT / "results" / "_geometry_lock_smoke" / "multi_body.png",
            "largest",
        ),
        (
            "compare_multi_body",
            ROOT / "results" / "_compare_single_vs_multi" / "00_body_full.png",
            "largest",
        ),
    ):
        if body.is_file() and face.is_file():
            cases.append(
                {"id": cid, "body": str(body), "face": str(face), "policy": policy}
            )
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--body", type=Path)
    ap.add_argument("--face", type=Path, default=DEFAULT_FACE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "krea2_identity_edit.yaml"
    )
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument(
        "--arms",
        default="prod,prod_procrustes,prod_wide_crop",
        help="Subset, e.g. prod,prod_procrustes",
    )
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(args.config)

    # Safety: refuse if someone flipped production defaults by mistake.
    if str(base_cfg.get("multi_person_edit_mode") or "") not in ("crop_stitch",):
        raise SystemExit(
            "Refusing to run: yaml multi_person_edit_mode must stay crop_stitch "
            f"(got {base_cfg.get('multi_person_edit_mode')})"
        )
    if bool(base_cfg.get("enable_procrustes_correction", False)):
        raise SystemExit(
            "Refusing to run: yaml enable_procrustes_correction must stay false "
            "(arm overrides only)"
        )
    if str(base_cfg.get("crop_margin_mode") or "tight") != "tight":
        raise SystemExit(
            "Refusing to run: yaml crop_margin_mode must stay tight "
            "(arm overrides only)"
        )

    if args.cases:
        raw = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        cases_in = list(raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw)
    elif args.body:
        cases_in = [
            {
                "id": Path(args.body).stem,
                "body": str(args.body),
                "face": str(args.face),
                "policy": "largest",
            }
        ]
    else:
        cases_in = _default_cases()
        if not cases_in:
            raise SystemExit("No cases; pass --body/--face or --cases")

    want = {x.strip().lower() for x in str(args.arms).split(",") if x.strip()}
    arms = [a for a in ARMS if a["id"].lower() in want]
    if not arms:
        arms = list(ARMS)

    summary: list[dict[str, Any]] = []
    for case in cases_in:
        cid = str(case.get("id") or Path(case["body"]).stem)
        case_dir = out / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        body = Image.open(case["body"]).convert("RGB")
        face = Image.open(case["face"]).convert("RGB")
        policy = str(case.get("policy") or "largest")
        probe = resize_max_keep_ar(
            body, int(base_cfg.get("max_body_dim", 1024)), div_by=16
        )
        selected, all_faces = select_face_box(
            pil_to_rgb_np(probe), cache, policy=policy
        )
        row: dict[str, Any] = {
            "id": cid,
            "body": str(case["body"]),
            "face": str(case["face"]),
            "policy": policy,
            "faces_detected": len(all_faces),
            "selected_box": (
                None
                if selected is None
                else [selected.x0, selected.y0, selected.x1, selected.y1]
            ),
            "body_native_mp": round(_mp(body.size), 4),
            "scores": {},
            "metas": {},
        }
        # Allow single-person bodies too (Procrustes/wide are crop_stitch features).
        if selected is None:
            row["skipped"] = True
            row["reason"] = "no face detected"
            summary.append(row)
            continue

        body.save(case_dir / "00_body.png")
        face.save(case_dir / "00_face.png")
        print(
            f"\n=== {cid}: faces={len(all_faces)} selected={row['selected_box']} ==="
        )

        if args.prep_only:
            reset_shared_krea2_runtime()
            pipe = Krea2IdentityEditPipeline(cfg=dict(base_cfg), cache_dir=cache)
            from headswap.preprocess import crop_face_reference

            face_crop = crop_face_reference(face, cache)
            for arm in arms:
                cfg = _cfg_for_arm(base_cfg, arm)
                pipe.cfg = cfg
                flags = pipe._tight_crop_flags(probe, selected, all_faces)
                built = pipe._build_scene_person(
                    probe,
                    face_crop,
                    selected,
                    div_by=int(cfg.get("div_by", 16)),
                    use_tight=bool(flags["use_tight"]),
                    top_ext=float(flags["top_ext"]),
                    side_ext=float(flags["side_ext"]),
                    bot_ext=float(flags["bot_ext"]),
                    expand_px=int(flags["expand_px"]),
                    crop_pad=int(flags["crop_pad"]),
                    all_faces=all_faces,
                    isolate_selected=bool(flags.get("isolate_selected")),
                )
                arm_dir = case_dir / arm["id"]
                arm_dir.mkdir(parents=True, exist_ok=True)
                built["scene"].save(arm_dir / "scene.png")
                built["person"].save(arm_dir / "person.png")
                diag = built.get("diag") or {}
                row["scores"][arm["id"]] = {
                    "arm": arm["id"],
                    "prep_only": True,
                    "edit_mode": "crop_stitch",
                    "crop_margin_mode": arm["crop_margin_mode"],
                    "enable_procrustes_correction": arm["enable_procrustes_correction"],
                    "scene_size": list(built["scene"].size),
                    "scene_megapixels": diag.get("scene_megapixels")
                    or round(_mp(built["scene"].size), 4),
                    "crop_box": diag.get("crop_box"),
                }
                print(
                    f"  prep {arm['id']}: margin={arm['crop_margin_mode']} "
                    f"scene={built['scene'].size} "
                    f"mp={row['scores'][arm['id']]['scene_megapixels']}"
                )
            summary.append(row)
            continue

        result_imgs: list[tuple[Image.Image, str]] = [(body, "original")]
        for arm in arms:
            arm_id = arm["id"]
            arm_dir = case_dir / arm_id
            arm_dir.mkdir(parents=True, exist_ok=True)
            cfg = _cfg_for_arm(base_cfg, arm)
            print(
                f"  running {arm_id} margin={arm['crop_margin_mode']} "
                f"procrustes={arm['enable_procrustes_correction']}…"
            )
            try:
                reset_shared_krea2_runtime()
                if args.mock:
                    from headswap.pipelines import create_pipeline

                    pipe = create_pipeline(cfg, force_mock=True)
                else:
                    pipe = Krea2IdentityEditPipeline(cfg=cfg, cache_dir=cache)
                pipe.cfg["body_face_policy"] = policy
                t0 = time.perf_counter()
                result = pipe.run(body, face, out_dir=arm_dir)
                latency = time.perf_counter() - t0
                meta = dict(result.meta or {})
                meta["mock"] = bool(args.mock)
                if not args.mock:
                    _assert_not_mock(meta, latency, arm_id)
                result.image.save(arm_dir / "result.png")
                result.image.save(arm_dir / "final_output.png")
                body.save(arm_dir / "input_scene.png")
                sel_score, faces_score = select_face_box(
                    pil_to_rgb_np(
                        resize_max_keep_ar(body, max(result.image.size), div_by=16)
                    ),
                    cache,
                    policy=policy,
                )
                if sel_score is None:
                    sel_score, faces_score = selected, all_faces
                sc = score_arm(
                    arm_id=arm_id,
                    body=body,
                    face=face,
                    result=result.image,
                    selected=sel_score,
                    all_faces=faces_score if faces_score else all_faces,
                    cache=cache,
                    latency_s=latency,
                    meta=meta,
                )
                sc["crop_margin_mode"] = arm["crop_margin_mode"]
                sc["enable_procrustes_correction"] = arm[
                    "enable_procrustes_correction"
                ]
                row["scores"][arm_id] = sc
                diag = meta.get("face_prep_diag") or {}
                row["metas"][arm_id] = {
                    "edit_mode": meta.get("edit_mode"),
                    "crop_margin_mode": meta.get("crop_margin_mode"),
                    "enable_procrustes_correction": meta.get(
                        "enable_procrustes_correction"
                    ),
                    "scene_size": meta.get("scene_size"),
                    "procrustes_correction": diag.get("procrustes_correction"),
                    "timing_s": meta.get("timing_s"),
                }
                result_imgs.append((result.image, arm_id))
                print(
                    f"    id={sc.get('identity_cosine')} "
                    f"hs={sc.get('head_to_body_scale_ratio')} "
                    f"scene_mp={sc.get('scene_megapixels')} lat={sc.get('latency_s')}s"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {arm_id}: {exc}")
                row["scores"][arm_id] = {"arm": arm_id, "error": str(exc)}

        if len(result_imgs) > 1:
            _side_by_side(result_imgs, max_per_row=4).save(case_dir / "side_by_side.png")
        summary.append(row)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
        "cases": summary,
        "prep_only": bool(args.prep_only),
        "mock": bool(args.mock),
        "config": str(args.config),
        "judgment": {
            "summary": (
                "Incomplete — prep/mock only."
                if args.prep_only or args.mock
                else "Compare head_scale + identity vs prod; promote only if clear win."
            )
        },
        "note": (
            "Do not flip enable_procrustes_correction / crop_margin_mode in yaml "
            "until this REPORT has real GPU aggregates."
        ),
    }
    (out / "REPORT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out / "REPORT.md", payload)
    print(f"\nWrote {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
