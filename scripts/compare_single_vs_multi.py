#!/usr/bin/env python3
"""Decision-gate harness: Krea2 multi via single-person conditioning.

Answers one question:
  Can Krea2 match single-person quality on a selected multi-person face when
  post-selection conditioning is effectively identical to the single-person path?

Cases (same identity face):
  A  solo body   → live single-person crop helpers (_tight_crop_flags/_build_scene_person)
  B  multi body  → same helpers + SPP (neighbor exclude only)
  C  multi body  → align_paste (divergence control; not a candidate to tune)

Usage:
  PYTHONPATH=src .venv/bin/python scripts/compare_single_vs_multi.py \\
    --solo-body data/custom/body.png \\
    --multi-body data/custom/body.png --synthesize-multi \\
    --face data/custom/face.png \\
    --out results/_conditioning_parity

  # Optional quality gate (needs ComfyUI+GPU unless --force-mock):
  ... --run-krea2 [--force-mock]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import (
    FaceBox,
    crop_around_face_box,
    crop_face_reference,
    get_face_landmarks5,
    identity_face_only_matte,
    pil_to_rgb_np,
    resize_max_keep_ar,
    select_face_box,
)

# Conditioning keys that must match A vs B within tolerance (decision gate Phase 1).
PARITY_KEYS = (
    "person_prep",
    "crop_long_side",
    "mask_top_ext",
    "mask_side_ext",
    "mask_bot_ext",
    "mask_expand_px",
    "prompt_sha1_12",
    "steps",
    "cfg",
    "ref_boost",
    "ref_boost_a",
    "grounding_px",
    "fit_mode",
    "use_tight",
    "isolate_selected",
    "single_person_parity",
)


def _sha(im: Image.Image) -> str:
    return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()[:16]


def _sha1_12(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _save(im: Image.Image | None, path: Path) -> str | None:
    if im is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return str(path)


def _nonblack_frac(im: Image.Image) -> float:
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    return float(np.mean(np.any(a > 8.0, axis=2)))


def _overlay_faces(im: Image.Image, faces: list[FaceBox], selected: FaceBox | None) -> Image.Image:
    out = im.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, f in enumerate(faces):
        same = (
            selected is not None
            and f.x0 == selected.x0
            and f.y0 == selected.y0
            and f.x1 == selected.x1
            and f.y1 == selected.y1
        )
        color = (0, 255, 80) if same else (255, 200, 0)
        draw.rectangle([f.x0, f.y0, f.x1, f.y1], outline=color, width=3)
        draw.text((f.x0 + 2, max(0, f.y0 - 12)), f"F{i + 1} {f.conf:.2f}", fill=color)
    return out


def _overlay_crop_and_mask(
    body: Image.Image,
    box: tuple[int, int, int, int] | list[int] | None,
    mask: Image.Image | None,
    selected: FaceBox | None,
) -> Image.Image:
    out = body.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    if mask is not None:
        m = mask.convert("L").resize(out.size, Image.Resampling.BILINEAR)
        tint = Image.new("RGB", out.size, (0, 80, 255))
        out = Image.composite(Image.blend(out, tint, 0.35), out, m.point(lambda p: 160 if p > 32 else 0))
        draw = ImageDraw.Draw(out)
    if box is not None:
        draw.rectangle(list(box), outline=(255, 60, 60), width=3)
    if selected is not None:
        draw.rectangle(
            [selected.x0, selected.y0, selected.x1, selected.y1],
            outline=(0, 255, 80),
            width=2,
        )
    return out


def _eye_geometry(rgb: np.ndarray, cache: Path, prefer: FaceBox | None) -> dict[str, Any]:
    lm, backend, note = get_face_landmarks5(rgb, cache, prefer_box=prefer)
    out: dict[str, Any] = {
        "landmarks_backend": backend,
        "landmarks_note": note,
        "landmarks": None if lm is None else lm.tolist(),
        "left_eye": None,
        "right_eye": None,
        "iod": None,
        "eye_line_deg": None,
    }
    if lm is None or len(lm) < 2:
        return out
    le, re = lm[0], lm[1]
    out["left_eye"] = [float(le[0]), float(le[1])]
    out["right_eye"] = [float(re[0]), float(re[1])]
    dx, dy = float(re[0] - le[0]), float(re[1] - le[1])
    out["iod"] = round(math.hypot(dx, dy), 3)
    out["eye_line_deg"] = round(math.degrees(math.atan2(dy, dx)), 3)
    return out


def _draw_landmarks(im: Image.Image, eye_info: dict[str, Any]) -> Image.Image:
    out = im.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    le, re = eye_info.get("left_eye"), eye_info.get("right_eye")
    if le and re:
        draw.ellipse([le[0] - 3, le[1] - 3, le[0] + 3, le[1] + 3], fill=(0, 255, 255))
        draw.ellipse([re[0] - 3, re[1] - 3, re[0] + 3, re[1] + 3], fill=(255, 0, 255))
        draw.line([le[0], le[1], re[0], re[1]], fill=(255, 255, 0), width=2)
    return out


def _make_group_from_portrait(portrait: Image.Image, n: int = 3) -> Image.Image:
    p = portrait.convert("RGB")
    w, h = p.size
    head = p.crop((0, 0, w, int(h * 0.55))).resize((256, 320), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (256 * n + 40, 640), (30, 30, 35))
    for i in range(n):
        canvas.paste(head, (20 + i * 256, 80))
    return canvas


def _plant_faces(size: tuple[int, int], n: int = 3) -> list[FaceBox]:
    w, h = size
    faces: list[FaceBox] = []
    for i in range(n):
        x0 = int(20 + i * (w / n))
        x1 = int(x0 + min(220, w / n - 20))
        y0, y1 = int(h * 0.12), int(h * 0.55)
        faces.append(FaceBox(x0, y0, x1, y1, 0.99 - 0.02 * i))
    return faces


def _side_by_side(a: Image.Image, b: Image.Image, la: str, lb: str) -> Image.Image:
    def _fit(im: Image.Image, hh: int = 360) -> Image.Image:
        im = im.convert("RGB")
        scale = hh / max(1, im.size[1])
        return im.resize((max(1, int(im.size[0] * scale)), hh), Image.Resampling.LANCZOS)

    aa, bb = _fit(a), _fit(b)
    gap, header = 12, 28
    canvas = Image.new("RGB", (aa.size[0] + bb.size[0] + gap, aa.size[1] + header), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), la, fill=(180, 255, 180))
    draw.text((aa.size[0] + gap + 4, 6), lb, fill=(255, 200, 120))
    canvas.paste(aa, (0, header))
    canvas.paste(bb, (aa.size[0] + gap, header))
    return canvas


class _Pipe(Krea2IdentityEditPipeline):
    """Lightweight pipe: reuse live crop helpers without loading ComfyUI."""

    def __init__(self, cfg: dict[str, Any], cache_dir: Path):
        self.cfg = dict(cfg)
        self.cache_dir = cache_dir


def _sampler_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": int(cfg.get("steps", 8)),
        "cfg": float(cfg.get("cfg", 1.0)),
        "seed": int(cfg.get("seed", 46)),
        "denoise": float(cfg.get("denoise", 1.0)),
        "ref_boost": float(cfg.get("ref_boost", 4.0)),
        "ref_boost_a": float(cfg.get("ref_boost_a", 1.0)),
        "grounding_px": int(cfg.get("grounding_px", 768)),
        "fit_mode": str(cfg.get("fit_mode", "fit") or "fit"),
        "sampler_name": str(cfg.get("sampler_name", "euler") or "euler"),
        "scheduler": str(cfg.get("scheduler", "simple") or "simple"),
        "crop_long_side": int(cfg.get("crop_long_side", 768)),
    }


def _build_crop_case(
    pipe: _Pipe,
    body_full: Image.Image,
    face_crop: Image.Image,
    selected: FaceBox,
    all_faces: list[FaceBox],
    *,
    case_id: str,
    cache: Path,
) -> dict[str, Any]:
    flags = pipe._tight_crop_flags(body_full, selected, all_faces)
    built = pipe._build_scene_person(
        body_full,
        face_crop,
        selected,
        div_by=int(pipe.cfg.get("div_by", 16)),
        use_tight=bool(flags["use_tight"]),
        top_ext=float(flags["top_ext"]),
        side_ext=float(flags["side_ext"]),
        bot_ext=float(flags["bot_ext"]),
        expand_px=int(flags["expand_px"]),
        crop_pad=int(flags["crop_pad"]),
        all_faces=all_faces,
        isolate_selected=bool(flags.get("isolate_selected")),
    )
    prompt = pipe._prompt_for_edit(
        use_tight=bool(flags["use_tight"]),
        multi_person=bool(flags["multi_person"]),
    )
    scene = built["scene"]
    person = built["person"]
    box = built["box"]
    diag = dict(built["diag"] or {})
    scene_eyes = _eye_geometry(pil_to_rgb_np(scene), cache, prefer=None)
    person_eyes = _eye_geometry(pil_to_rgb_np(person), cache, prefer=None)
    samp = _sampler_fingerprint(pipe.cfg)
    head_to_canvas = float(diag.get("face_height_frac_scene") or 0.0)
    metrics = {
        "case": case_id,
        "path": "krea2_crop_spp",
        "person_prep": diag.get("person_prep"),
        "use_tight": bool(diag.get("use_tight")),
        "isolate_selected": bool(diag.get("isolate_selected")),
        "single_person_parity": bool(diag.get("single_person_parity")),
        "multi_person": bool(diag.get("multi_person")),
        "faces_detected": int(diag.get("faces_detected") or len(all_faces)),
        "crop_box": list(box) if box is not None else None,
        "crop_native_size": diag.get("crop_native_size"),
        "scene_size": list(scene.size),
        "person_size": list(person.size),
        "scene_ar": round(scene.size[0] / max(1, scene.size[1]), 4),
        "face_area_frac_crop": diag.get("face_area_frac_crop"),
        "face_height_frac_scene": diag.get("face_height_frac_scene"),
        "head_to_canvas": head_to_canvas,
        "person_nonblack_frac": round(_nonblack_frac(person), 4),
        "scene_sha": _sha(scene),
        "person_sha": _sha(person),
        "mask_sha": _sha(built["mask"]) if built.get("mask") is not None else None,
        "prompt_sha1_12": _sha1_12(prompt),
        "prompt_len": len(prompt),
        "mask_top_ext": float(flags["top_ext"]),
        "mask_side_ext": float(flags["side_ext"]),
        "mask_bot_ext": float(flags["bot_ext"]),
        "mask_expand_px": int(flags["expand_px"]),
        "neighbor_crop_clamp": diag.get("neighbor_crop_clamp"),
        "scene_eyes": scene_eyes,
        "person_eyes": person_eyes,
        **samp,
    }
    return {
        "scene": scene,
        "person": person,
        "mask": built["mask"],
        "box": box,
        "prompt": prompt,
        "flags": flags,
        "diag": diag,
        "metrics": metrics,
        "overlay": _overlay_crop_and_mask(body_full, box, built["mask"], selected),
        "scene_lm": _draw_landmarks(scene, scene_eyes),
        "person_lm": _draw_landmarks(person, person_eyes),
    }


def _build_align_paste_case(
    cfg: dict[str, Any],
    body_full: Image.Image,
    face: Image.Image,
    selected: FaceBox,
    all_faces: list[FaceBox],
    cache: Path,
) -> dict[str, Any]:
    """Divergence control: dump align_paste work crop + identity matte (no refine)."""
    id_matte, matte_info = identity_face_only_matte(
        face,
        cache,
        top=float(cfg.get("face_matte_top_pad", 0.35)),
        bot=float(cfg.get("face_matte_bot_pad", 0.08)),
        side=float(cfg.get("face_matte_side_pad", 0.12)),
        force_ellipse=bool(cfg.get("face_matte_force_ellipse", True)),
    )
    work, box = crop_around_face_box(
        body_full,
        selected,
        pad_frac=float(cfg.get("align_paste_crop_pad_frac", 1.15)),
        div_by=int(cfg.get("div_by", 16)),
    )
    # Face-only refine mask extents (what Krea2 would see under align_paste).
    top = float(cfg.get("align_paste_mask_top", 0.28))
    side = float(cfg.get("align_paste_mask_side", 0.28))
    bot = float(cfg.get("align_paste_mask_bot", 0.18))
    fw, fh = selected.width, selected.height
    face_area = fw * fh
    crop_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    # Approximate refine region vs crop (face-only ellipse extents).
    refine_w = fw * (1.0 + 2.0 * side)
    refine_h = fh * (1.0 + top + bot)
    refine_frac = (refine_w * refine_h) / float(max(1, crop_area))
    metrics = {
        "case": "C_align_paste",
        "path": "align_paste",
        "person_prep": "identity_face_only_matte",
        "use_tight": False,
        "isolate_selected": False,
        "single_person_parity": False,
        "multi_person": True,
        "faces_detected": len(all_faces),
        "crop_box": list(box),
        "crop_native_size": [box[2] - box[0], box[3] - box[1]],
        "scene_size": list(work.size),
        "person_size": list(id_matte.size),
        "scene_ar": round(work.size[0] / max(1, work.size[1]), 4),
        "face_area_frac_crop": round(face_area / float(crop_area), 4),
        "face_height_frac_scene": round(fh / max(1, work.size[1]), 4),
        "head_to_canvas": round(fh / max(1, work.size[1]), 4),
        "refine_mask_frac_approx": round(refine_frac, 4),
        "align_paste_mask_top": top,
        "align_paste_mask_side": side,
        "align_paste_mask_bot": bot,
        "person_nonblack_frac": round(_nonblack_frac(id_matte), 4),
        "scene_sha": _sha(work),
        "person_sha": _sha(id_matte),
        "matte_info": matte_info,
        "prompt_sha1_12": None,
        "steps": int(cfg.get("align_paste_refine_steps", 6)),
        "ref_boost": float(cfg.get("ref_boost", 4.0)),
        "crop_long_side": None,
    }
    return {
        "scene": work,
        "person": id_matte,
        "mask": None,
        "box": box,
        "prompt": "",
        "metrics": metrics,
        "overlay": _overlay_crop_and_mask(body_full, box, None, selected),
        "scene_lm": work,
        "person_lm": id_matte,
    }


def _rel_close(a: float | None, b: float | None, *, atol: float, rtol: float) -> bool:
    if a is None or b is None:
        return a == b
    return abs(float(a) - float(b)) <= atol + rtol * max(abs(float(a)), abs(float(b)), 1e-9)


def evaluate_conditioning_parity(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    *,
    face_fill_atol: float = 0.04,
    face_fill_rtol: float = 0.25,
    size_atol: int = 16,
) -> dict[str, Any]:
    """Phase-1 gate: A vs B conditioning within tolerance."""
    checks: list[dict[str, Any]] = []
    ok = True

    for key in PARITY_KEYS:
        va, vb = metrics_a.get(key), metrics_b.get(key)
        same = va == vb
        checks.append({"key": key, "A": va, "B": vb, "ok": same, "kind": "exact"})
        ok = ok and same

    for key, atol, rtol in (
        ("face_area_frac_crop", face_fill_atol, face_fill_rtol),
        ("face_height_frac_scene", 0.05, 0.20),
        ("person_nonblack_frac", 0.08, 0.15),
        ("head_to_canvas", 0.05, 0.20),
    ):
        va, vb = metrics_a.get(key), metrics_b.get(key)
        same = _rel_close(
            None if va is None else float(va),
            None if vb is None else float(vb),
            atol=atol,
            rtol=rtol,
        )
        checks.append(
            {
                "key": key,
                "A": va,
                "B": vb,
                "ok": same,
                "kind": "tolerance",
                "atol": atol,
                "rtol": rtol,
            }
        )
        ok = ok and same

    sa, sb = metrics_a.get("scene_size"), metrics_b.get("scene_size")
    size_ok = (
        isinstance(sa, list)
        and isinstance(sb, list)
        and len(sa) == 2
        and len(sb) == 2
        and abs(int(sa[0]) - int(sb[0])) <= size_atol
        and abs(int(sa[1]) - int(sb[1])) <= size_atol
    )
    checks.append({"key": "scene_size", "A": sa, "B": sb, "ok": size_ok, "kind": "tolerance"})
    ok = ok and size_ok

    # Eye geometry: soft check (landmarks may fail on synth); do not fail parity alone.
    ea = (metrics_a.get("scene_eyes") or {}).get("eye_line_deg")
    eb = (metrics_b.get("scene_eyes") or {}).get("eye_line_deg")
    eye_ok = True
    if ea is not None and eb is not None:
        eye_ok = abs(float(ea) - float(eb)) <= 8.0
    checks.append(
        {
            "key": "scene_eye_line_deg",
            "A": ea,
            "B": eb,
            "ok": eye_ok,
            "kind": "soft",
            "note": "informational; not required for Phase-1 pass",
        }
    )

    failed = [c for c in checks if not c["ok"] and c.get("kind") != "soft"]
    return {
        "parity_ok": bool(ok),
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_keys": [c["key"] for c in failed],
        "checks": checks,
        "neighbor_clamp_B": metrics_b.get("neighbor_crop_clamp"),
    }


def _load_body(
    path: Path,
    cfg: dict[str, Any],
    cache: Path,
    *,
    synthesize_multi: bool,
    require_multi: bool,
) -> tuple[Image.Image, FaceBox, list[FaceBox]]:
    div_by = int(cfg.get("div_by", 16))
    body = Image.open(path).convert("RGB")
    body_full = resize_max_keep_ar(body, int(cfg.get("max_body_dim", 1024)), div_by=div_by)
    selected, all_faces = select_face_box(
        pil_to_rgb_np(body_full),
        cache,
        policy=str(cfg.get("body_face_policy", "largest")),
    )
    if require_multi and len(all_faces) < 2 and synthesize_multi:
        body_full = _make_group_from_portrait(body_full, n=3)
        selected, all_faces = select_face_box(
            pil_to_rgb_np(body_full), cache, policy="largest"
        )
        if len(all_faces) < 2:
            all_faces = _plant_faces(body_full.size, n=3)
            selected = all_faces[-1]
    if selected is None:
        raise SystemExit(f"No face detected on {path}")
    if require_multi and len(all_faces) < 2:
        raise SystemExit(
            f"{path}: need >=2 faces for multi case (got {len(all_faces)}); "
            "pass --synthesize-multi"
        )
    if not require_multi and len(all_faces) > 1:
        # Solo baseline: keep only the selected face in the face list so flags
        # match true single-person (multi_person=False).
        all_faces = [selected]
    return body_full, selected, all_faces


def _quality_metrics(
    body: Image.Image,
    face: Image.Image,
    result: Image.Image,
    selected: FaceBox,
    all_faces: list[FaceBox],
    cache: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    from headswap.metrics.scoring import identity_cosine

    id_cos = identity_cosine(face, result)
    body_eyes = _eye_geometry(pil_to_rgb_np(body), cache, prefer=selected)
    res_eyes = _eye_geometry(pil_to_rgb_np(result), cache, prefer=selected)
    gaze_delta = None
    if body_eyes.get("eye_line_deg") is not None and res_eyes.get("eye_line_deg") is not None:
        gaze_delta = round(
            abs(float(res_eyes["eye_line_deg"]) - float(body_eyes["eye_line_deg"])), 3
        )
    head_ratio = None
    if selected is not None and body.size[1] > 0:
        head_ratio = round(float(selected.height) / float(body.size[1]), 4)
    # Neighbor PSNR outside selected face box (rough locality).
    neighbor_psnr = None
    try:
        from headswap.metrics.scoring import psnr

        ba = np.asarray(body.convert("RGB"))
        ra = np.asarray(result.convert("RGB").resize(body.size, Image.Resampling.LANCZOS))
        mask = np.ones(ba.shape[:2], dtype=bool)
        for f in all_faces:
            if (
                f.x0 == selected.x0
                and f.y0 == selected.y0
                and f.x1 == selected.x1
                and f.y1 == selected.y1
            ):
                mask[f.y0 : f.y1, f.x0 : f.x1] = False
                continue
            # Also exclude other faces for "unchanged neighbors" — measure outside all faces
        outside = mask
        if np.any(outside):
            neighbor_psnr = round(float(psnr(ba[outside], ra[outside])), 3)
    except Exception as exc:  # noqa: BLE001
        neighbor_psnr = f"error:{type(exc).__name__}"

    return {
        "identity_cosine": None if id_cos is None else round(float(id_cos), 4),
        "gaze_eye_line_delta_deg": gaze_delta,
        "body_eye_line_deg": body_eyes.get("eye_line_deg"),
        "result_eye_line_deg": res_eyes.get("eye_line_deg"),
        "head_height_ratio_body": head_ratio,
        "neighbor_outside_selected_psnr": neighbor_psnr,
        "edit_mode": meta.get("edit_mode") or meta.get("multi_person_swap_mode"),
        "scene_size": meta.get("scene_size"),
        "mock": bool(meta.get("force_mock") or meta.get("mode", "").startswith("mock")),
    }


def _decide(
    parity: dict[str, Any],
    quality: dict[str, Any] | None,
    *,
    id_gap_tol: float = 0.08,
    gaze_gap_tol: float = 6.0,
) -> dict[str, Any]:
    """Binary architectural decision from Phase-1 + Phase-2 evidence."""
    if not parity.get("parity_ok"):
        return {
            "decision": "BLOCKED",
            "reason": (
                "Conditioning parity A≈B not achieved. Fix neighbor exclusion "
                "geometry only; do not treat this as a Krea2 model limitation yet."
            ),
            "action": "fix_clamp_protect_only",
        }
    if not quality or not quality.get("ran"):
        return {
            "decision": "PARITY_OK_QUALITY_PENDING",
            "reason": (
                "Conditioning parity holds. Quality gate not run (pass --run-krea2 "
                "on GPU). Architecture under test remains select→single-person path."
            ),
            "action": "run_quality_on_gpu",
        }
    if quality.get("mock_only"):
        # Mock cannot answer identity/gaze. Architecture decision proceeds as YES
        # for production path (enable single-path multi); quality remains GPU-gated.
        return {
            "decision": "YES_ARCHITECTURE",
            "reason": (
                "Conditioning parity holds. Quality metrics are mock-only and not "
                "decisive. Production multi must use select→single-person crop path "
                "so a real GPU gate can answer YES/NO on Krea2 capability. "
                "align_paste demoted to A/B control only."
            ),
            "action": "simplify_to_krea2_crop_spp",
            "quality_note": "re-run --run-krea2 without --force-mock on Colab/GPU",
        }

    qa = quality.get("A") or {}
    qb = quality.get("B") or {}
    id_a, id_b = qa.get("identity_cosine"), qb.get("identity_cosine")
    g_a, g_b = qa.get("gaze_eye_line_delta_deg"), qb.get("gaze_eye_line_delta_deg")
    id_ok = (
        id_a is not None
        and id_b is not None
        and float(id_b) >= float(id_a) - id_gap_tol
    )
    gaze_ok = True
    if g_a is not None and g_b is not None:
        gaze_ok = float(g_b) <= float(g_a) + gaze_gap_tol

    if id_ok and gaze_ok:
        return {
            "decision": "YES",
            "reason": (
                "Conditioning parity holds and multi quality approaches single "
                f"(id A={id_a} B={id_b}, gaze_delta A={g_a} B={g_b})."
            ),
            "action": "detect_select_single_path_stitch_delete_multi_forks",
            "identity_cosine_A": id_a,
            "identity_cosine_B": id_b,
            "gaze_delta_A": g_a,
            "gaze_delta_B": g_b,
        }
    return {
        "decision": "NO",
        "reason": (
            "Conditioning parity holds but multi quality remains significantly "
            f"worse than single (id A={id_a} B={id_b}, gaze_delta A={g_a} B={g_b}). "
            "Limitation is inside Krea2 — stop multi architecture investment; "
            "evaluate alternative identity-edit models."
        ),
        "action": "stop_krea2_multi_arch_eval_other_models",
        "identity_cosine_A": id_a,
        "identity_cosine_B": id_b,
        "gaze_delta_A": g_a,
        "gaze_delta_B": g_b,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solo-body", type=Path, required=True)
    ap.add_argument("--multi-body", type=Path, required=True)
    ap.add_argument("--face", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "_conditioning_parity")
    ap.add_argument("--synthesize-multi", action="store_true")
    ap.add_argument("--run-krea2", action="store_true")
    ap.add_argument("--force-mock", action="store_true")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "krea2_identity_edit.yaml")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    # Force SPP + crop path for the B case under test.
    cfg_spp = dict(cfg)
    cfg_spp["single_person_parity"] = True
    cfg_spp["multi_person_swap_mode"] = "krea2_crop"
    cfg_spp["multi_person_edit_mode"] = "crop_stitch"
    cfg_spp["clamp_crop_away_neighbors"] = True
    cfg_spp["identity_scale_match"] = False
    cfg_spp["face_white_bg"] = False
    cfg_spp["multi_extra_prompt"] = False

    pipe = _Pipe(cfg_spp, cache)

    solo_body, solo_sel, solo_faces = _load_body(
        args.solo_body, cfg_spp, cache, synthesize_multi=False, require_multi=False
    )
    multi_body, multi_sel, multi_faces = _load_body(
        args.multi_body,
        cfg_spp,
        cache,
        synthesize_multi=bool(args.synthesize_multi),
        require_multi=True,
    )

    face = Image.open(args.face).convert("RGB")
    face_crop = crop_face_reference(
        face,
        cache,
        top=float(cfg_spp.get("face_top_pad", 0.55)),
        bot=float(cfg_spp.get("face_bot_pad", 0.20)),
        side=float(cfg_spp.get("face_side_pad", 0.28)),
        include_shoulders=bool(cfg_spp.get("include_shoulders", False)),
    )
    _save(face_crop, out / "00_identity_face_crop_shared.png")
    _save(_overlay_faces(solo_body, solo_faces, solo_sel), out / "A_solo" / "01_faces.png")
    _save(_overlay_faces(multi_body, multi_faces, multi_sel), out / "B_multi_spp" / "01_faces.png")

    # A: true solo portrait (quality baseline + recipe fingerprint).
    case_a = _build_crop_case(
        pipe, solo_body, face_crop, solo_sel, solo_faces, case_id="A_solo", cache=cache
    )
    # A_ctrl: same multi pixels as B, but faces=[selected] only — isolates the
    # neighbor-exclusion delta. Geometry parity is A_ctrl vs B, not A vs B
    # (different photos naturally differ in scene AR / face fill).
    case_a_ctrl = _build_crop_case(
        pipe,
        multi_body,
        face_crop,
        multi_sel,
        [multi_sel],
        case_id="A_ctrl_selected_only",
        cache=cache,
    )
    case_b = _build_crop_case(
        pipe,
        multi_body,
        face_crop,
        multi_sel,
        multi_faces,
        case_id="B_multi_spp",
        cache=cache,
    )
    case_c = _build_align_paste_case(cfg, multi_body, face, multi_sel, multi_faces, cache)

    for name, pack in (
        ("A_solo", case_a),
        ("A_ctrl_selected_only", case_a_ctrl),
        ("B_multi_spp", case_b),
        ("C_align_paste", case_c),
    ):
        d = out / name
        _save(pack["overlay"], d / "02_detector_head_crop_overlay.png")
        _save(pack["scene"], d / "03_scene.png")
        _save(pack["person"], d / "04_person.png")
        _save(pack.get("scene_lm"), d / "05_scene_landmarks.png")
        _save(pack.get("person_lm"), d / "06_person_landmarks.png")
        if pack.get("mask") is not None:
            _save(pack["mask"], d / "07_mask_full.png")
        (d / "prompt.txt").write_text(pack.get("prompt") or "", encoding="utf-8")
        (d / "metrics.json").write_text(
            json.dumps(pack["metrics"], indent=2), encoding="utf-8"
        )

    _save(
        _side_by_side(
            case_a_ctrl["scene"], case_b["scene"], "A_ctrl scene", "B scene"
        ),
        out / "side_by_side" / "scene_Actrl_vs_B.png",
    )
    _save(
        _side_by_side(
            case_a_ctrl["person"], case_b["person"], "A_ctrl person", "B person"
        ),
        out / "side_by_side" / "person_Actrl_vs_B.png",
    )
    _save(
        _side_by_side(case_a["scene"], case_b["scene"], "A_solo scene", "B scene"),
        out / "side_by_side" / "scene_Asolo_vs_B.png",
    )
    _save(
        _side_by_side(case_a_ctrl["scene"], case_c["scene"], "A_ctrl scene", "C align_paste"),
        out / "side_by_side" / "scene_Actrl_vs_C.png",
    )
    _save(
        _side_by_side(case_a_ctrl["person"], case_c["person"], "A_ctrl person", "C matte"),
        out / "side_by_side" / "person_Actrl_vs_C.png",
    )

    # Recipe must match solo fingerprint; geometry parity is A_ctrl vs B.
    recipe_parity = evaluate_conditioning_parity(
        case_a["metrics"], case_b["metrics"]
    )
    # Solo vs multi photos differ in scene AR — drop scene_size from recipe gate.
    recipe_checks = [c for c in recipe_parity["checks"] if c["key"] in set(PARITY_KEYS)]
    recipe_ok = all(c["ok"] for c in recipe_checks)
    geom_parity = evaluate_conditioning_parity(
        case_a_ctrl["metrics"], case_b["metrics"]
    )
    parity = {
        "parity_ok": bool(recipe_ok and geom_parity["parity_ok"]),
        "recipe_ok": recipe_ok,
        "geometry_ok": bool(geom_parity["parity_ok"]),
        "recipe_checks": recipe_checks,
        "geometry_checks": geom_parity["checks"],
        "failed_keys": (
            [c["key"] for c in recipe_checks if not c["ok"]]
            + [f"geom:{k}" for k in geom_parity.get("failed_keys") or []]
        ),
        "neighbor_clamp_B": case_b["metrics"].get("neighbor_crop_clamp"),
        "note": (
            "Recipe: A_solo vs B exact keys. Geometry: A_ctrl (same multi body, "
            "selected-only faces) vs B (neighbors) — isolates neighbor exclusion."
        ),
    }
    # C is expected to diverge — record key deltas for the report.
    c_deltas = {
        "person_prep": {
            "A_ctrl": case_a_ctrl["metrics"].get("person_prep"),
            "C": case_c["metrics"].get("person_prep"),
        },
        "face_area_frac_crop": {
            "A_ctrl": case_a_ctrl["metrics"].get("face_area_frac_crop"),
            "C": case_c["metrics"].get("face_area_frac_crop"),
        },
        "scene_size": {
            "A_ctrl": case_a_ctrl["metrics"].get("scene_size"),
            "C": case_c["metrics"].get("scene_size"),
        },
        "path": {"A_ctrl": "krea2_crop_spp", "C": "align_paste"},
    }

    quality_block: dict[str, Any] = {"ran": False}
    if args.run_krea2:
        from headswap.pipelines import create_pipeline

        quality_block["ran"] = True
        quality_block["mock_only"] = bool(args.force_mock)
        for label, body_im, swap_mode, sel, faces in (
            ("A", solo_body, "krea2_crop", solo_sel, solo_faces),
            ("B", multi_body, "krea2_crop", multi_sel, multi_faces),
        ):
            run_cfg = dict(cfg_spp)
            run_cfg["multi_person_swap_mode"] = swap_mode
            run_cfg["body_face_policy"] = "largest"
            if args.force_mock:
                run_cfg["force_mock"] = True
            # Pin selected face via index when possible.
            try:
                idx = next(
                    i
                    for i, f in enumerate(faces)
                    if f.x0 == sel.x0 and f.y0 == sel.y0 and f.x1 == sel.x1
                )
                run_cfg["body_face_index"] = idx
            except StopIteration:
                pass
            pipe_run = create_pipeline(run_cfg, force_mock=bool(args.force_mock))
            result = pipe_run.run(body_im, face, out_dir=out / f"krea2_{label}")
            _save(result.image, out / f"krea2_{label}" / "result.png")
            meta = dict(result.meta or {})
            if args.force_mock:
                meta["force_mock"] = True
            qm = _quality_metrics(body_im, face, result.image, sel, faces, cache, meta)
            quality_block[label] = qm
            (out / f"krea2_{label}" / "quality.json").write_text(
                json.dumps(qm, indent=2), encoding="utf-8"
            )

    decision = _decide(parity, quality_block if quality_block.get("ran") else None)

    report = {
        "question": (
            "Can Krea2 produce single-person quality on a selected face from a "
            "multi-person image if conditioning after face selection is effectively "
            "identical to the single-person pipeline?"
        ),
        "inputs": {
            "solo_body": str(args.solo_body),
            "multi_body": str(args.multi_body),
            "face": str(args.face),
            "synthesize_multi": bool(args.synthesize_multi),
        },
        "detection": {
            "A_faces": len(solo_faces),
            "A_selected": [solo_sel.x0, solo_sel.y0, solo_sel.x1, solo_sel.y1],
            "B_faces": len(multi_faces),
            "B_selected": [multi_sel.x0, multi_sel.y0, multi_sel.x1, multi_sel.y1],
        },
        "metrics": {
            "A_solo": case_a["metrics"],
            "A_ctrl": case_a_ctrl["metrics"],
            "B": case_b["metrics"],
            "C": case_c["metrics"],
        },
        "conditioning_parity": parity,
        "align_paste_divergence_Actrl_vs_C": c_deltas,
        "quality": quality_block,
        "decision": decision,
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    lines = [
        "# Conditioning parity decision gate",
        "",
        report["question"],
        "",
        f"## Detection",
        f"- A (solo): {len(solo_faces)} face(s), selected={report['detection']['A_selected']}",
        f"- B (multi): {len(multi_faces)} faces, selected={report['detection']['B_selected']}",
        "",
        "## Phase 1 — Conditioning parity",
        f"- parity_ok: **{parity['parity_ok']}** (recipe={parity['recipe_ok']}, geometry={parity['geometry_ok']})",
        f"- failed: {parity['failed_keys']}",
        f"- neighbor_clamp_B: {parity.get('neighbor_clamp_B')}",
        f"- note: {parity.get('note')}",
        "",
        "### Recipe (A_solo vs B)",
        "| metric | A_solo | B | ok |",
        "|---|---|---|---|",
    ]
    for c in parity["recipe_checks"]:
        lines.append(f"| {c['key']} | {c['A']!r} | {c['B']!r} | {c['ok']} |")
    lines += [
        "",
        "### Geometry (A_ctrl vs B — same multi body)",
        "| metric | A_ctrl | B | ok |",
        "|---|---|---|---|",
    ]
    for c in parity["geometry_checks"]:
        if c.get("kind") == "soft":
            continue
        lines.append(f"| {c['key']} | {c['A']!r} | {c['B']!r} | {c['ok']} |")
    lines += [
        "",
        "## Align_paste control (A_ctrl vs C) — expected divergence",
        f"- person_prep: {c_deltas['person_prep']}",
        f"- face_area_frac_crop: {c_deltas['face_area_frac_crop']}",
        f"- scene_size: {c_deltas['scene_size']}",
        "",
        "## Decision",
        f"- **{decision['decision']}**",
        f"- {decision['reason']}",
        f"- action: `{decision['action']}`",
    ]
    if quality_block.get("ran"):
        lines += ["", "## Quality (Phase 2)", json.dumps(quality_block, indent=2)]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0 if parity["parity_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
