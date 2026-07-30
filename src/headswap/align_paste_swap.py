"""Geometry-locked align → paste → optional refine (Architecture B).

Locks expression / pose / head scale from the target via full affine landmarks,
pastes a clothing-free identity face matte, optionally runs a masked Krea2
refine, then re-locks looking direction onto the destination landmarks.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PIL import Image

from headswap.preprocess import (
    FaceBox,
    align_face_to_destination,
    color_match_rgba_to_destination,
    crop_around_face_box,
    detect_best_face,
    feathered_soft_composite,
    head_hair_mask_from_face,
    identity_face_only_matte,
    lab_histogram_match_face,
    paste_aligned_face,
    pil_to_rgb_np,
    relock_pose_to_destination,
    suppress_neighbor_faces_in_mask,
)


def _head_height_ratio(
    original: Image.Image, result: Image.Image, cache_dir
) -> float | None:
    fo = detect_best_face(pil_to_rgb_np(original), cache_dir)
    fe = detect_best_face(pil_to_rgb_np(result), cache_dir)
    if fo is None or fe is None or fo.height <= 0:
        return None
    return float(fe.height) / float(fo.height)


def _mouth_open_proxy(rgb: np.ndarray, face: FaceBox | None) -> float | None:
    """Crude mouth openness proxy in [0,1] from mid-face luminance variance."""
    if face is None:
        return None
    h, w = rgb.shape[:2]
    x0, y0 = max(0, face.x0), max(0, face.y0)
    x1, y1 = min(w, face.x1), min(h, face.y1)
    if x1 <= x0 or y1 <= y0:
        return None
    # Lower third of the face box ≈ mouth region.
    my0 = y0 + int(0.55 * (y1 - y0))
    patch = rgb[my0:y1, x0:x1].astype(np.float32)
    if patch.size < 16:
        return None
    # Higher variance ≈ open mouth / teeth contrast.
    return float(min(1.0, patch.std() / 64.0))


def measure_align_paste_gates(
    body: Image.Image,
    result: Image.Image,
    *,
    selected: FaceBox | None,
    cache_dir,
    face_mask: Image.Image | None = None,
) -> dict[str, Any]:
    """Metrics gates: head-scale ratio, mouth proxy, neighbor PSNR outside mask."""
    body_a = np.asarray(body.convert("RGB"))
    res = result.convert("RGB")
    if res.size != body.size:
        res = res.resize(body.size, Image.Resampling.LANCZOS)
    res_a = np.asarray(res)
    gates: dict[str, Any] = {}
    ratio = _head_height_ratio(body, res, cache_dir)
    if ratio is not None:
        gates["head_height_ratio"] = round(ratio, 4)
    mout_b = _mouth_open_proxy(body_a, selected)
    mout_r = _mouth_open_proxy(res_a, selected)
    if mout_b is not None and mout_r is not None:
        gates["mouth_open_body"] = round(mout_b, 4)
        gates["mouth_open_result"] = round(mout_r, 4)
        gates["mouth_open_abs_delta"] = round(abs(mout_b - mout_r), 4)
    if face_mask is not None:
        m = np.asarray(
            face_mask.convert("L").resize(body.size, Image.Resampling.BILINEAR)
        )
        outside = m < 128
        if np.any(outside):
            mse = float(
                np.mean(
                    (
                        body_a[outside].astype(np.float64)
                        - res_a[outside].astype(np.float64)
                    )
                    ** 2
                )
            )
            psnr = 99.0 if mse <= 1e-12 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
            gates["neighbor_psnr_outside_mask"] = round(psnr, 3)
            gates["neighbor_mse_outside_mask"] = round(mse, 3)
    return gates


def run_align_paste_swap(
    body: Image.Image,
    face: Image.Image,
    cache_dir,
    *,
    selected_face: FaceBox | None = None,
    all_faces: list[FaceBox] | None = None,
    cfg: dict | None = None,
    refine_fn: Callable[..., Image.Image] | None = None,
) -> dict[str, Any]:
    """
    Architecture B core: face-only matte → landmark align → paste → optional refine.

    ``refine_fn(composite_crop, identity_matte, face_mask_crop) -> refined_crop``
    when provided (e.g. Krea2 masked refine). Outside the face mask, body pixels
    stay exact.
    """
    cfg = dict(cfg or {})
    all_faces = list(all_faces or [])
    body_full = body.convert("RGB")
    div_by = int(cfg.get("div_by", 16))

    id_matte, matte_info = identity_face_only_matte(
        face,
        cache_dir,
        top=float(cfg.get("face_matte_top_pad", 0.35)),
        bot=float(cfg.get("face_matte_bot_pad", 0.08)),
        side=float(cfg.get("face_matte_side_pad", 0.12)),
        force_ellipse=bool(cfg.get("face_matte_force_ellipse", True)),
    )

    work, box = crop_around_face_box(
        body_full,
        selected_face,
        pad_frac=float(cfg.get("align_paste_crop_pad_frac", 1.15)),
        div_by=div_by,
    )

    aligned_rgba, align_info = align_face_to_destination(
        id_matte,
        work,
        cache_dir,
        core_min_alpha=float(cfg.get("paste_core_min_alpha", 0.90)),
        ellipse_scale_x=float(cfg.get("paste_ellipse_scale_x", 2.05)),
        ellipse_scale_y=float(cfg.get("paste_ellipse_scale_y", 2.55)),
        feather_px=int(cfg.get("paste_feather_px", 21)),
        use_full_affine=bool(cfg.get("align_paste_full_affine", True)),
    )
    pre_match = float(cfg.get("pre_color_match_strength", 0.55) or 0.0)
    paste_info: dict[str, Any] = {"composite_paste": False}
    composite_crop = work
    if aligned_rgba is not None:
        if pre_match > 0:
            aligned_rgba = color_match_rgba_to_destination(
                aligned_rgba, work, strength=pre_match
            )
            align_info["pre_color_match_strength"] = pre_match
        composite_crop, paste_info = paste_aligned_face(work, aligned_rgba)
    else:
        paste_info = {
            "composite_paste": False,
            "composite_paste_skip_reason": align_info.get(
                "face_alignment_skip_reason"
            )
            or "align_failed",
        }

    # Face-local mask on the working crop (neighbors carved out in full-body coords).
    face_in_crop: FaceBox | None = None
    if selected_face is not None:
        face_in_crop = FaceBox(
            selected_face.x0 - box[0],
            selected_face.y0 - box[1],
            selected_face.x1 - box[0],
            selected_face.y1 - box[1],
            selected_face.conf,
        )
    face_mask_crop = head_hair_mask_from_face(
        work,
        cache_dir,
        expand_px=int(cfg.get("align_paste_mask_expand_px", 10)),
        blur_px=int(cfg.get("align_paste_mask_blur_px", 10)),
        top_extend=float(cfg.get("align_paste_mask_top", 0.85)),
        side_extend=float(cfg.get("align_paste_mask_side", 0.35)),
        bot_extend=float(cfg.get("align_paste_mask_bot", 0.25)),
        face_box=face_in_crop,
    )
    # Map other faces into crop space and carve them out.
    others_in_crop: list[FaceBox] = []
    if selected_face is not None:
        for f in all_faces:
            if (
                f.x0 == selected_face.x0
                and f.y0 == selected_face.y0
                and f.x1 == selected_face.x1
                and f.y1 == selected_face.y1
            ):
                continue
            others_in_crop.append(
                FaceBox(
                    f.x0 - box[0],
                    f.y0 - box[1],
                    f.x1 - box[0],
                    f.y1 - box[1],
                    f.conf,
                )
            )
        if others_in_crop:
            face_mask_crop = suppress_neighbor_faces_in_mask(
                face_mask_crop, face_in_crop, others_in_crop, shrink=1.05
            )

    refined_crop = composite_crop
    refine_meta: dict[str, Any] = {"refine_applied": False}
    if refine_fn is not None and bool(paste_info.get("composite_paste")):
        try:
            refined_crop = refine_fn(composite_crop, id_matte, face_mask_crop)
            refine_meta["refine_applied"] = True
        except Exception as exc:  # noqa: BLE001 — fall back to paste
            refine_meta["refine_error"] = str(exc)
            refined_crop = composite_crop

    # Lock looking direction / head yaw back onto the original crop landmarks.
    # Generative refine often front-faces the identity; re-align fixes that.
    pose_meta: dict[str, Any] = {"pose_relock": False}
    if bool(cfg.get("align_paste_pose_relock", True)) and bool(
        paste_info.get("composite_paste")
    ):
        refined_crop, pose_meta = relock_pose_to_destination(
            refined_crop,
            work,
            cache_dir,
            face_mask=face_mask_crop,
            use_full_affine=bool(cfg.get("align_paste_full_affine", True)),
            core_min_alpha=float(cfg.get("paste_core_min_alpha", 0.90)),
            ellipse_scale_x=float(cfg.get("paste_ellipse_scale_x", 2.05)),
            ellipse_scale_y=float(cfg.get("paste_ellipse_scale_y", 2.55)),
            feather_px=int(cfg.get("paste_feather_px", 21)),
            stitch_feather_px=int(cfg.get("align_paste_stitch_feather_px", 10)),
        )

    # Soft-stitch refined crop into full body; outside mask = exact body pixels.
    feather = int(cfg.get("align_paste_stitch_feather_px", 10))
    out = feathered_soft_composite(
        body_full,
        refined_crop,
        face_mask_crop,
        box,
        extra_blur_px=feather,
    )
    # Full-body mask for metrics / debug.
    full_mask = Image.new("L", body_full.size, 0)
    mw, mh = box[2] - box[0], box[3] - box[1]
    full_mask.paste(
        face_mask_crop.convert("L").resize((mw, mh), Image.Resampling.BILINEAR),
        (box[0], box[1]),
    )
    post_match = float(cfg.get("align_paste_post_color_match", 0.40) or 0.0)
    if post_match > 0:
        out = lab_histogram_match_face(out, body_full, full_mask, strength=post_match)

    gates = measure_align_paste_gates(
        body_full,
        out,
        selected=selected_face,
        cache_dir=cache_dir,
        face_mask=full_mask,
    )

    return {
        "image": out,
        "identity_matte": id_matte,
        "work_crop": work,
        "composite_crop": composite_crop,
        "refined_crop": refined_crop,
        "face_mask": full_mask,
        "face_mask_crop": face_mask_crop,
        "box": box,
        "matte_info": matte_info,
        "align_info": align_info,
        "paste_info": paste_info,
        "refine_meta": refine_meta,
        "pose_meta": pose_meta,
        "gates": gates,
        "mode": "align_paste",
    }
