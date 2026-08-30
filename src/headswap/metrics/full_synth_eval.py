"""Extended metrics for full-image vs localized face-swap comparison."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from headswap.metrics.scoring import body_preserve_score, identity_cosine, psnr
from headswap.preprocess import FaceBox, detect_faces, get_face_landmarks5, pil_to_rgb_np


@dataclass
class FullSynthCaseMetrics:
    case_id: str
    pipeline: str
    latency_s: float
    identity_cosine: float | None
    expression_landmark_l2: float | None
    pose_landmark_l2: float | None
    head_size_ratio: float | None
    clothing_psnr: float | None
    background_psnr: float | None
    neighbor_identity_mean: float | None
    neighbor_count: int
    faces_in_result: int
    prompt: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resize_match(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    if other.shape[:2] == ref.shape[:2]:
        return other
    return cv2.resize(other, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)


def _face_area(f: FaceBox) -> float:
    return float(max(1, f.width) * max(1, f.height))


def _match_faces(
    src_faces: list[FaceBox], dst_faces: list[FaceBox]
) -> list[tuple[FaceBox, FaceBox | None]]:
    """Greedy center-distance matching from source faces to result faces."""
    remaining = list(dst_faces)
    pairs: list[tuple[FaceBox, FaceBox | None]] = []
    for s in src_faces:
        scx = (s.x0 + s.x1) / 2.0
        scy = (s.y0 + s.y1) / 2.0
        if not remaining:
            pairs.append((s, None))
            continue
        best_i = 0
        best_d = 1e18
        for i, d in enumerate(remaining):
            dcx = (d.x0 + d.x1) / 2.0
            dcy = (d.y0 + d.y1) / 2.0
            dist = (scx - dcx) ** 2 + (scy - dcy) ** 2
            if dist < best_d:
                best_d, best_i = dist, i
        pairs.append((s, remaining.pop(best_i)))
    return pairs


def _landmark_l2(
    a: np.ndarray,
    b: np.ndarray,
    face_a: FaceBox,
    face_b: FaceBox,
    cache_dir,
) -> float | None:
    lm_a, _, _ = get_face_landmarks5(a, cache_dir, prefer_box=face_a)
    lm_b, _, _ = get_face_landmarks5(b, cache_dir, prefer_box=face_b)
    if lm_a is None or lm_b is None:
        return None
    # Normalize by face diagonal so scores are scale-invariant.
    diag = float(np.hypot(face_a.width, face_a.height))
    if diag < 1:
        return None
    return float(np.linalg.norm(lm_a - lm_b) / diag)


def _region_psnr(
    body: np.ndarray,
    result: np.ndarray,
    keep_mask: np.ndarray,
) -> float | None:
    if keep_mask.sum() < 100:
        return None
    return psnr(body[keep_mask], result[keep_mask])


def _build_person_masks(
    h: int, w: int, selected: FaceBox | None, all_faces: list[FaceBox]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (selected_head, clothing_band, background) boolean masks."""
    selected_m = np.zeros((h, w), dtype=bool)
    clothing_m = np.zeros((h, w), dtype=bool)
    people_m = np.zeros((h, w), dtype=bool)

    for f in all_faces:
        y0 = max(0, f.y0 - int(0.55 * f.height))
        y1 = min(h, f.y1 + int(1.7 * f.height))
        x0 = max(0, f.x0 - int(0.25 * f.width))
        x1 = min(w, f.x1 + int(0.25 * f.width))
        people_m[y0:y1, x0:x1] = True

    if selected is not None:
        y0 = max(0, selected.y0 - int(0.55 * selected.height))
        y1 = min(h, selected.y1 + int(0.35 * selected.height))
        x0 = max(0, selected.x0 - int(0.2 * selected.width))
        x1 = min(w, selected.x1 + int(0.2 * selected.width))
        selected_m[y0:y1, x0:x1] = True

        cy0 = min(h - 1, selected.y1 + int(0.05 * selected.height))
        cy1 = min(h, selected.y1 + int(1.5 * selected.height))
        cx0 = max(0, selected.x0 - int(0.15 * selected.width))
        cx1 = min(w, selected.x1 + int(0.15 * selected.width))
        clothing_m[cy0:cy1, cx0:cx1] = True
        clothing_m &= ~selected_m

    background_m = ~people_m
    return selected_m, clothing_m, background_m


def _arcface_emb(im: Image.Image, box: FaceBox | None = None):
    try:
        from insightface.app import FaceAnalysis
    except Exception:
        return None
    if not hasattr(_arcface_emb, "_app"):
        try:
            from headswap.preprocess import (  # noqa: PLC0415
                preferred_onnx_providers,
            )

            # Was hardcoded to CPU, which would have stayed on CPU even
            # after installing onnxruntime-gpu.
            app = FaceAnalysis(
                name="buffalo_l", providers=preferred_onnx_providers()
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _arcface_emb._app = app  # type: ignore[attr-defined]
        except Exception:
            _arcface_emb._app = None  # type: ignore[attr-defined]
            return None
    app = _arcface_emb._app  # type: ignore[attr-defined]
    if app is None:
        return None
    arr = pil_to_rgb_np(im)[:, :, ::-1]
    faces = app.get(arr)
    if not faces:
        return None
    if box is None:
        faces = sorted(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return faces[-1].normed_embedding
    best, best_iou = None, 0.0
    for f in faces:
        x0, y0, x1, y1 = [int(v) for v in f.bbox]
        ix0, iy0 = max(x0, box.x0), max(y0, box.y0)
        ix1, iy1 = min(x1, box.x1), min(y1, box.y1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = box.width * box.height + (x1 - x0) * (y1 - y0) - inter
        iou = inter / max(union, 1)
        if iou > best_iou:
            best_iou, best = iou, f
    return None if best is None else best.normed_embedding


def score_full_synth_case(
    case_id: str,
    pipeline: str,
    body: Image.Image,
    face: Image.Image,
    result: Image.Image,
    *,
    selected: FaceBox | None,
    all_faces: list[FaceBox],
    cache_dir: Path | str,
    latency_s: float,
    prompt: str | None = None,
) -> FullSynthCaseMetrics:
    body_rgb = body.convert("RGB")
    result_rgb = result.convert("RGB")
    if result_rgb.size != body_rgb.size:
        result_rgb = result_rgb.resize(body_rgb.size, Image.Resampling.LANCZOS)

    b = pil_to_rgb_np(body_rgb)
    r = pil_to_rgb_np(result_rgb)
    h, w = b.shape[:2]

    result_faces = detect_faces(r, cache_dir, allow_prior=False)
    matched = _match_faces(all_faces, result_faces)

    id_cos = identity_cosine(face, result_rgb)

    expr_l2 = None
    pose_l2 = None
    head_ratio = None
    if selected is not None:
        # Find matched result face for selected.
        sel_pair = None
        for s, d in matched:
            if (
                s.x0 == selected.x0
                and s.y0 == selected.y0
                and s.x1 == selected.x1
                and s.y1 == selected.y1
            ):
                sel_pair = d
                break
        if sel_pair is not None:
            expr_l2 = _landmark_l2(b, r, selected, sel_pair, cache_dir)
            pose_l2 = expr_l2  # same 5-pt proxy; reported separately for schema clarity
            head_ratio = _face_area(sel_pair) / max(_face_area(selected), 1.0)

    _, clothing_m, background_m = _build_person_masks(h, w, selected, all_faces)
    clothing_psnr = _region_psnr(b, r, clothing_m)
    background_psnr = _region_psnr(b, r, background_m)

    neighbor_sims: list[float] = []
    for s, d in matched:
        if selected is not None and (
            s.x0 == selected.x0
            and s.y0 == selected.y0
            and s.x1 == selected.x1
            and s.y1 == selected.y1
        ):
            continue
        if d is None:
            continue
        ea = _arcface_emb(body_rgb, s)
        eb = _arcface_emb(result_rgb, d)
        if ea is None or eb is None:
            continue
        neighbor_sims.append(float(np.dot(ea, eb)))

    return FullSynthCaseMetrics(
        case_id=case_id,
        pipeline=pipeline,
        latency_s=float(latency_s),
        identity_cosine=id_cos,
        expression_landmark_l2=expr_l2,
        pose_landmark_l2=pose_l2,
        head_size_ratio=head_ratio,
        clothing_psnr=clothing_psnr,
        background_psnr=background_psnr,
        neighbor_identity_mean=(
            float(np.mean(neighbor_sims)) if neighbor_sims else None
        ),
        neighbor_count=len(neighbor_sims),
        faces_in_result=len(result_faces),
        prompt=prompt,
        extras={
            "body_preserve_psnr_legacy": body_preserve_score(body_rgb, result_rgb),
        },
    )


def make_side_by_side(
    body: Image.Image,
    localized: Image.Image,
    full_synth: Image.Image,
    *,
    labels: tuple[str, str, str] = ("original", "localized", "full_image_synth"),
) -> Image.Image:
    imgs = [body.convert("RGB"), localized.convert("RGB"), full_synth.convert("RGB")]
    h = max(im.height for im in imgs)
    scaled = []
    for im in imgs:
        if im.height != h:
            nh = h
            nw = max(1, int(round(im.width * (h / im.height))))
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        scaled.append(im)
    gap = 8
    label_h = 28
    total_w = sum(im.width for im in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (total_w, h + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, lab in zip(scaled, labels):
        canvas.paste(im, (x, label_h))
        draw.text((x + 6, 6), lab, fill=(240, 240, 240))
        x += im.width + gap
    return canvas
