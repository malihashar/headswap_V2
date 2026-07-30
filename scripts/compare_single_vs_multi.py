#!/usr/bin/env python3
"""End-to-end single-person vs multi-person prep / input comparison for Krea2.

Does NOT tune mask/stitch. Forces both code profiles on the same selected face
so differences are pipeline divergences, not content differences.

Usage:
  PYTHONPATH=src python scripts/compare_single_vs_multi.py \\
    --body data/custom/body.png --face data/custom/face.png \\
    --out results/_compare_single_vs_multi

Optional:
  --run-krea2   full sample both ways (needs ComfyUI + GPU)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.preprocess import (
    FaceBox,
    crop_face_reference,
    crop_with_mask,
    detect_faces,
    expand_crop_box_for_face_fill,
    face_on_white_background,
    get_face_landmarks5,
    head_hair_mask_from_face,
    pil_to_rgb_np,
    resize_long_side,
    resize_max_keep_ar,
    select_face_box,
    suppress_neighbor_faces_in_mask,
)


def _sha(im: Image.Image) -> str:
    return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()[:16]


def _save(im: Image.Image | None, path: Path) -> str | None:
    if im is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return str(path)


def _overlay_faces(im: Image.Image, faces: list[FaceBox], selected: FaceBox | None) -> Image.Image:
    out = im.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, f in enumerate(faces):
        color = (0, 255, 80) if selected and (
            f.x0 == selected.x0 and f.y0 == selected.y0 and f.x1 == selected.x1 and f.y1 == selected.y1
        ) else (255, 200, 0)
        draw.rectangle([f.x0, f.y0, f.x1, f.y1], outline=color, width=3)
        draw.text((f.x0 + 2, max(0, f.y0 - 12)), f"F{i+1} {f.conf:.2f}", fill=color)
    return out


def _make_group_from_portrait(portrait: Image.Image, n: int = 3) -> Image.Image:
    """Synthesize a multi-person body when only a single portrait is available."""
    p = portrait.convert("RGB")
    # Head-ish upper crop repeated side-by-side on a tall canvas.
    w, h = p.size
    head = p.crop((0, 0, w, int(h * 0.55))).resize((256, 320), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (256 * n + 40, 640), (30, 30, 35))
    for i in range(n):
        canvas.paste(head, (20 + i * 256, 80))
    return canvas


def _profile_params(cfg: dict, *, multi: bool) -> dict[str, Any]:
    """Profiles mirror current krea2.py product behavior after the compare fix."""
    if multi:
        # Current product: isolate selected face spatially, but keep single-person
        # identity / prompt / resolution (no white-bg sticker, no extra prompt).
        return {
            "label": "multi",
            "use_tight": False,
            "multi_person": True,
            "top_ext": float(cfg.get("mask_top_extend", 1.25)),
            "side_ext": float(cfg.get("isolate_mask_side_extend", 0.50)),
            "bot_ext": float(cfg.get("mask_bot_extend", 0.40)),
            "expand_px": int(cfg.get("mask_expand_px", 18)),
            "blur_px": int(cfg.get("mask_blur_px", 12)),
            "crop_pad": int(cfg.get("isolate_crop_pad", 12)),
            "crop_long": int(cfg.get("crop_long_side", 768)),
            "face_white_bg": False,
            "expand_face_fill": True,
            "suppress_neighbors": True,
            "extra_prompt": bool(cfg.get("multi_extra_prompt", False)),
        }
    return {
        "label": "single",
        "use_tight": False,
        "multi_person": False,
        "top_ext": float(cfg.get("mask_top_extend", 1.25)),
        "side_ext": float(cfg.get("mask_side_extend", 0.60)),
        "bot_ext": float(cfg.get("mask_bot_extend", 0.40)),
        "expand_px": int(cfg.get("mask_expand_px", 18)),
        "blur_px": int(cfg.get("mask_blur_px", 12)),
        "crop_pad": 12,
        "crop_long": int(cfg.get("crop_long_side", 768)),
        "face_white_bg": False,
        "expand_face_fill": False,
        "suppress_neighbors": False,
        "extra_prompt": False,
    }


def _build_for_profile(
    body_full: Image.Image,
    face_crop: Image.Image,
    selected: FaceBox,
    all_faces: list[FaceBox],
    cfg: dict,
    *,
    multi: bool,
    div_by: int,
) -> dict[str, Any]:
    p = _profile_params(cfg, multi=multi)
    mask = head_hair_mask_from_face(
        body_full,
        ROOT / ".cache" / "headswap_v2",
        expand_px=p["expand_px"],
        blur_px=p["blur_px"],
        top_extend=p["top_ext"],
        side_extend=p["side_ext"],
        bot_extend=p["bot_ext"],
        face_box=selected,
    )
    if p["suppress_neighbors"]:
        mask = suppress_neighbor_faces_in_mask(mask, selected, all_faces, shrink=0.90)

    crop_native, mask_crop, box = crop_with_mask(
        body_full, mask, pad=p["crop_pad"], div_by=div_by
    )
    expand_info = {}
    if p["expand_face_fill"]:
        box, expand_info = expand_crop_box_for_face_fill(
            body_full.size,
            box,
            selected,
            target_face_area_frac=float(cfg.get("multi_target_face_fill", 0.16)),
            min_long_side=int(cfg.get("multi_min_crop_long", 448)),
            other_faces=all_faces,
            div_by=div_by,
        )
        crop_native = body_full.crop(box)
        mask_in_box = mask.crop(box)
    else:
        mask_in_box = mask_crop

    scene = resize_long_side(crop_native, p["crop_long"], div_by=div_by)
    from headswap.preprocess import resize_contain

    person_raw = resize_contain(face_crop.convert("RGB"), scene.size, fill=(0, 0, 0))
    person = person_raw
    if p["face_white_bg"]:
        person = face_on_white_background(
            person_raw,
            cache_dir=ROOT / ".cache" / "headswap_v2",
            force_ellipse=True,
        )

    face_area = selected.width * selected.height
    crop_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    face_fill = face_area / float(crop_area)
    resize_factor = max(scene.size) / max(1, max(crop_native.size))

    base_prompt = str(cfg.get("prompt", "") or "").strip()
    prompt = base_prompt
    if p["extra_prompt"]:
        prompt = (
            base_prompt
            + " The first image crop shows ONE person only — replace only that "
            "person's head. Match the original head size and neck placement "
            "exactly. Do not change other people, arms, or background. "
            "Use ONLY the face identity from the second image — do not copy "
            "clothing, collar, bowtie, or shoulders from the second image. "
            "Match lighting and skin tone to the neck and jaw in the first image."
        )

    return {
        "profile": p,
        "mask": mask,
        "mask_in_box": mask_in_box,
        "crop_native": crop_native,
        "box": box,
        "scene": scene,
        "person_raw": person_raw,
        "person": person,
        "prompt": prompt,
        "metrics": {
            "crop_box": list(box),
            "crop_native_size": list(crop_native.size),
            "crop_long_native": int(max(crop_native.size)),
            "resize_factor": round(resize_factor, 4),
            "face_occupancy_pct": round(100.0 * face_fill, 2),
            "scene_size": list(scene.size),
            "person_size": list(person.size),
            "mask_size": list(mask.size),
            "face_white_bg_applied": bool(p["face_white_bg"]),
            "person_sha_raw": _sha(person_raw),
            "person_sha_final": _sha(person),
            "scene_sha": _sha(scene),
            "mask_sha": _sha(mask),
            "prompt_len": len(prompt),
            "prompt_extra": bool(p["extra_prompt"]),
            "crop_expand": expand_info,
            "inference_params": {
                "steps": int(cfg.get("steps", 8)),
                "cfg": float(cfg.get("cfg", 1.0)),
                "seed": int(cfg.get("seed", 46)),
                "sampler": str(cfg.get("sampler", "euler")),
                "scheduler": str(cfg.get("scheduler", "simple")),
                "denoise": float(cfg.get("denoise", 1.0)),
                "ref_boost": float(cfg.get("ref_boost", 3.5)),
                "ref_boost_a": float(cfg.get("ref_boost_a", 1.6)),
                "grounding_px": int(cfg.get("grounding_px", 768)),
                "fit_mode": str(cfg.get("fit_mode", "fit")),
                "crop_long": p["crop_long"],
            },
        },
    }


def _side_by_side(a: Image.Image, b: Image.Image, label_a: str, label_b: str) -> Image.Image:
    def _fit(im: Image.Image, h: int = 360) -> Image.Image:
        im = im.convert("RGB")
        scale = h / max(1, im.size[1])
        return im.resize((max(1, int(im.size[0] * scale)), h), Image.Resampling.LANCZOS)

    aa, bb = _fit(a), _fit(b)
    gap = 12
    header = 28
    canvas = Image.new("RGB", (aa.size[0] + bb.size[0] + gap, aa.size[1] + header), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), label_a, fill=(180, 255, 180))
    draw.text((aa.size[0] + gap + 4, 6), label_b, fill=(255, 200, 120))
    canvas.paste(aa, (0, header))
    canvas.paste(bb, (aa.size[0] + gap, header))
    return canvas


def _diff_dict(a: dict, b: dict, prefix: str = "") -> list[dict[str, Any]]:
    diffs = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        pa = f"{prefix}.{k}" if prefix else k
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(_diff_dict(va, vb, pa))
        elif va != vb:
            diffs.append({"key": pa, "single": va, "multi": vb})
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=Path, required=True)
    ap.add_argument("--face", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "_compare_single_vs_multi")
    ap.add_argument("--synthesize-group", action="store_true",
                    help="Tile the body into a 3-person group if <2 faces detected")
    ap.add_argument("--run-krea2", action="store_true")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "krea2_identity_edit.yaml")
    div_by = int(cfg.get("div_by", 16))

    body = Image.open(args.body).convert("RGB")
    face = Image.open(args.face).convert("RGB")
    body_full = resize_max_keep_ar(body, int(cfg.get("max_body_dim", 1024)), div_by=div_by)

    # --- 1. Face detection ---
    selected, all_faces = select_face_box(
        pil_to_rgb_np(body_full), cache, policy=str(cfg.get("body_face_policy", "largest"))
    )
    if len(all_faces) < 2 and args.synthesize_group:
        body_full = _make_group_from_portrait(body_full, n=3)
        selected, all_faces = select_face_box(
            pil_to_rgb_np(body_full), cache, policy="largest"
        )
        # If detector still fails (headless opencv without caffe), plant boxes.
        if len(all_faces) < 2:
            all_faces = [
                FaceBox(30, 90, 250, 380, 0.99),
                FaceBox(286, 90, 506, 380, 0.95),
                FaceBox(542, 90, 762, 380, 0.90),
            ]
            selected = all_faces[0]  # leftmost / first as primary for synth
            # Prefer rightmost to mimic product "interesting" subject? Use largest area = all equal → index0

    if selected is None:
        raise SystemExit("No face detected on body — pass a real photo or --synthesize-group")

    lm, lm_backend, lm_note = get_face_landmarks5(pil_to_rgb_np(body_full), cache)
    det_report = {
        "body_size": list(body_full.size),
        "faces_detected": len(all_faces),
        "faces": [
            {"index": i, "box": [f.x0, f.y0, f.x1, f.y1], "conf": f.conf,
             "area": f.width * f.height}
            for i, f in enumerate(all_faces)
        ],
        "selected_index": next(
            (i for i, f in enumerate(all_faces)
             if f.x0 == selected.x0 and f.y0 == selected.y0 and f.x1 == selected.x1),
            0,
        ),
        "selected_box": [selected.x0, selected.y0, selected.x1, selected.y1],
        "selected_conf": selected.conf,
        "landmarks_backend": lm_backend,
        "landmarks_note": lm_note,
        "landmarks": None if lm is None else lm.tolist(),
    }

    face_crop = crop_face_reference(
        face,
        cache,
        top=float(cfg.get("face_top_pad", 0.55)),
        bot=float(cfg.get("face_bot_pad", 0.20)),
        side=float(cfg.get("face_side_pad", 0.28)),
        include_shoulders=bool(cfg.get("include_shoulders", False)),
    )
    id_report = {
        "face_input_size": list(face.size),
        "face_crop_size": list(face_crop.size),
        "face_crop_sha": _sha(face_crop),
        "note": "Same face_crop object used for both profiles before profile-specific white_bg",
    }

    # --- Build both profiles on THE SAME selected face ---
    single = _build_for_profile(
        body_full, face_crop, selected, all_faces, cfg, multi=False, div_by=div_by
    )
    multi = _build_for_profile(
        body_full, face_crop, selected, all_faces, cfg, multi=True, div_by=div_by
    )

    # Save intermediates
    _save(body_full, out / "00_body_full.png")
    _save(_overlay_faces(body_full, all_faces, selected), out / "01_detected_faces_overlay.png")
    _save(face_crop, out / "02_identity_face_crop_shared.png")

    for name, pack in (("single", single), ("multi", multi)):
        d = out / name
        _save(pack["mask"], d / "03_mask_full.png")
        _save(pack["crop_native"], d / "04_crop_native.png")
        _save(pack["mask_in_box"].resize(pack["crop_native"].size), d / "05_mask_in_crop.png")
        # masked crop preview
        crop_rgba = pack["crop_native"].convert("RGBA")
        alpha = pack["mask_in_box"].convert("L").resize(pack["crop_native"].size)
        crop_rgba.putalpha(alpha)
        _save(crop_rgba, d / "06_masked_crop.png")
        _save(pack["scene"], d / "07_krea2_scene_input.png")
        _save(pack["person_raw"], d / "08_identity_matched_size_before_whitebg.png")
        _save(pack["person"], d / "09_krea2_person_input.png")
        (d / "prompt.txt").write_text(pack["prompt"], encoding="utf-8")
        (d / "metrics.json").write_text(
            json.dumps(pack["metrics"], indent=2), encoding="utf-8"
        )

    # Side-by-side report images
    pairs = [
        ("04_crop_native", "crop_native"),
        ("07_scene", "scene"),
        ("08_person_before_whitebg", "person_raw"),
        ("09_person_final_krea2_input", "person"),
        ("03_mask", "mask"),
    ]
    for label, key in pairs:
        _save(
            _side_by_side(single[key], multi[key], f"SINGLE {label}", f"MULTI {label}"),
            out / "side_by_side" / f"{label}.png",
        )

    metric_diffs = _diff_dict(single["metrics"], multi["metrics"])
    profile_diffs = _diff_dict(single["profile"], multi["profile"])

    # First divergence analysis (ordered by pipeline stage)
    first_divergence = None
    stage_notes = []
    stage_notes.append({
        "stage": "1_face_detection",
        "identical": True,
        "detail": "Same body_full, same select_face_box result used for both profiles",
    })
    stage_notes.append({
        "stage": "2_identity_face_crop",
        "identical": True,
        "detail": f"Shared face_crop sha={id_report['face_crop_sha']} before profile branching",
    })
    mask_same = single["metrics"]["mask_sha"] == multi["metrics"]["mask_sha"]
    stage_notes.append({
        "stage": "3_body_mask",
        "identical": mask_same,
        "detail": (
            "Mask may differ slightly (isolate side_ext) but uses same mask_* family; "
            "legacy multi_mask_* / use_tight no longer auto-enabled"
        ),
    })
    if first_divergence is None and not mask_same:
        first_divergence = {
            "stage": "3_body_mask",
            "why": (
                "Selected-face isolation uses a slightly tighter side_extend and may "
                "carve neighbors from the mask. Vertical coverage matches single-person."
            ),
        }
    crop_same = single["metrics"]["scene_sha"] == multi["metrics"]["scene_sha"] and (
        single["metrics"]["crop_box"] == multi["metrics"]["crop_box"]
    )
    stage_notes.append({
        "stage": "4_body_crop_and_resize",
        "identical": crop_same,
        "detail": (
            f"single crop={single['metrics']['crop_native_size']}→{single['metrics']['scene_size']} "
            f"multi crop={multi['metrics']['crop_native_size']}→{multi['metrics']['scene_size']}"
        ),
    })
    person_pre_same = single["metrics"]["person_sha_raw"] == multi["metrics"]["person_sha_raw"]
    stage_notes.append({
        "stage": "5_identity_resize_to_scene",
        "identical": person_pre_same,
        "detail": (
            "person_raw is face_crop resized to scene.size; differs only if scene sizes differ"
        ),
    })
    white_bg_div = bool(multi["metrics"]["face_white_bg_applied"]) or bool(
        single["metrics"]["face_white_bg_applied"]
    )
    person_final_same = single["metrics"]["person_sha_final"] == multi["metrics"]["person_sha_final"]
    id_preprocess_same = (
        single["metrics"]["person_sha_raw"] == single["metrics"]["person_sha_final"]
        and multi["metrics"]["person_sha_raw"] == multi["metrics"]["person_sha_final"]
        and not white_bg_div
    )
    stage_notes.append({
        "stage": "6_identity_white_bg",
        "identical": id_preprocess_same,
        "detail": (
            "Identity preprocess must match single (no white-ellipse). "
            f"white_bg applied single={single['metrics']['face_white_bg_applied']} "
            f"multi={multi['metrics']['face_white_bg_applied']}; "
            f"person tensors equal={person_final_same}"
        ),
    })
    if first_divergence is None and not id_preprocess_same:
        first_divergence = {
            "stage": "6_identity_white_bg",
            "why": "Identity reference preprocessing diverges (white ellipse sticker vs natural crop).",
        }
    prompt_same = single["prompt"] == multi["prompt"]
    stage_notes.append({
        "stage": "7_prompt",
        "identical": prompt_same,
        "detail": f"multi appends extra locality/lighting instructions ({multi['metrics']['prompt_len']-single['metrics']['prompt_len']} chars)",
    })
    if first_divergence is None and not prompt_same:
        first_divergence = {"stage": "7_prompt", "why": "Prompt text differs for multi/tight path."}

    inf_same = single["metrics"]["inference_params"] == multi["metrics"]["inference_params"]
    stage_notes.append({
        "stage": "8_sampler_settings",
        "identical": inf_same,
        "detail": "steps/cfg/seed/sampler/scheduler/ref_boost compared; crop_long may differ",
    })

    report = {
        "detection": det_report,
        "identity_shared": id_report,
        "stages": stage_notes,
        "first_divergence": first_divergence,
        "profile_diffs": profile_diffs,
        "metric_diffs": metric_diffs,
        "single_metrics": single["metrics"],
        "multi_metrics": multi["metrics"],
        "krea2_inputs": {
            "single": {
                "scene": "single/07_krea2_scene_input.png",
                "person": "single/09_krea2_person_input.png",
                "prompt": "single/prompt.txt",
            },
            "multi": {
                "scene": "multi/07_krea2_scene_input.png",
                "person": "multi/09_krea2_person_input.png",
                "prompt": "multi/prompt.txt",
            },
        },
    }

    # Optional full Krea2 (heavy)
    if args.run_krea2:
        from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

        pipe = Krea2IdentityEditPipeline(cfg=dict(cfg), cache_dir=cache)
        # Natural run (will pick multi path if faces>1)
        natural = pipe.run(body_full, face, out_dir=out / "krea2_natural")
        _save(natural.image, out / "krea2_natural" / "result.png")
        report["krea2_natural_meta"] = {
            k: natural.meta.get(k)
            for k in (
                "multi_person", "tight_crop", "scene_size", "person_size",
                "face_prep_diag", "prompt", "steps", "cfg", "seed",
            )
        }

    (out / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Human-readable summary
    lines = [
        "# Single vs Multi pipeline comparison",
        "",
        f"Faces detected: {det_report['faces_detected']}",
        f"Selected: index={det_report['selected_index']} box={det_report['selected_box']} conf={det_report['selected_conf']}",
        f"Landmarks: backend={det_report['landmarks_backend']} note={det_report['landmarks_note']}",
        "",
        "## First divergence",
        json.dumps(first_divergence, indent=2),
        "",
        "## Stage identity checklist",
    ]
    for s in stage_notes:
        mark = "SAME" if s["identical"] else "DIFF"
        lines.append(f"- [{mark}] {s['stage']}: {s['detail']}")
    lines += ["", "## Metric diffs (single → multi)"]
    for d in metric_diffs:
        lines.append(f"- {d['key']}: {d['single']!r} → {d['multi']!r}")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
