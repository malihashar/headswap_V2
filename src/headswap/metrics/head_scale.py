"""Head-to-body scale metrics (face height ratio body vs result)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image

from headswap.preprocess import FaceBox, detect_faces, get_face_landmarks5, pil_to_rgb_np


def _eye_line_deg(
    im: Image.Image, cache: Path, prefer: FaceBox | None
) -> float | None:
    lm, _, _ = get_face_landmarks5(pil_to_rgb_np(im), cache, prefer_box=prefer)
    if lm is None or len(lm) < 2:
        return None
    le, re = lm[0], lm[1]
    return float(math.degrees(math.atan2(float(re[1] - le[1]), float(re[0] - le[0]))))


def head_scale_metrics(
    body: Image.Image,
    result: Image.Image,
    selected: FaceBox,
    cache: Path,
) -> dict[str, Any]:
    """Head-to-body scale: (result face_h / image_h) / (body face_h / image_h)."""
    body_rgb = body.convert("RGB")
    res = result.convert("RGB")
    if res.size != body_rgb.size:
        res = res.resize(body_rgb.size, Image.Resampling.LANCZOS)

    body_faces = detect_faces(pil_to_rgb_np(body_rgb), cache, allow_prior=False)
    res_faces = detect_faces(pil_to_rgb_np(res), cache, allow_prior=False)

    def _nearest(target: FaceBox, faces: list[FaceBox]) -> FaceBox | None:
        if not faces:
            return None
        tcx = 0.5 * (target.x0 + target.x1)
        tcy = 0.5 * (target.y0 + target.y1)

        def dist(f: FaceBox) -> float:
            return (0.5 * (f.x0 + f.x1) - tcx) ** 2 + (0.5 * (f.y0 + f.y1) - tcy) ** 2

        return min(faces, key=dist)

    b_face = _nearest(selected, body_faces) or selected
    r_face = _nearest(selected, res_faces)

    ih = max(1, body_rgb.size[1])
    body_frac = float(b_face.height) / float(ih)
    result_frac = None if r_face is None else float(r_face.height) / float(ih)
    ratio = None if result_frac is None else float(result_frac) / max(1e-6, body_frac)
    area_ratio = None
    if r_face is not None:
        area_ratio = (r_face.width * r_face.height) / max(
            1.0, float(b_face.width * b_face.height)
        )

    return {
        "body_face_h_px": int(b_face.height),
        "result_face_h_px": None if r_face is None else int(r_face.height),
        "body_face_h_frac": round(body_frac, 4),
        "result_face_h_frac": None if result_frac is None else round(result_frac, 4),
        "head_to_body_scale_ratio": None if ratio is None else round(ratio, 4),
        "head_area_ratio": None if area_ratio is None else round(float(area_ratio), 4),
        "body_eye_line_deg": _eye_line_deg(body_rgb, cache, b_face),
        "result_eye_line_deg": (
            None if r_face is None else _eye_line_deg(res, cache, r_face)
        ),
    }
