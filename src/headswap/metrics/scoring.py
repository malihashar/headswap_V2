from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class PairMetrics:
    pair_id: str
    pipeline: str
    latency_s: float
    identity_cosine: float | None
    body_preserve_psnr: float | None
    seam_edge_delta: float | None
    face_detected: bool
    success: bool
    fail_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_rgb(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-10:
        return 99.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def body_preserve_score(
    body: Image.Image, result: Image.Image, head_mask: Image.Image | None = None
) -> float | None:
    """PSNR outside the head mask — higher means body wasn't rewritten."""
    b = _to_rgb(body)
    r = _to_rgb(result)
    if r.shape != b.shape:
        r = cv2.resize(r, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    if head_mask is None:
        # Compare border band only
        h, w = b.shape[:2]
        m = np.ones((h, w), dtype=bool)
        m[h // 8 : 7 * h // 8, w // 8 : 7 * w // 8] = False
    else:
        m = np.asarray(head_mask.convert("L").resize((b.shape[1], b.shape[0]))) < 32
    if m.sum() < 100:
        return None
    return psnr(b[m], r[m])


def seam_edge_delta(result: Image.Image, head_mask: Image.Image) -> float | None:
    """Mean gradient magnitude along the mask boundary — high can mean harsh seams."""
    rgb = _to_rgb(result).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = np.asarray(head_mask.convert("L").resize((gray.shape[1], gray.shape[0])))
    edge = cv2.Canny(m, 50, 150)
    if edge.sum() == 0:
        return None
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    vals = mag[edge > 0]
    return float(vals.mean()) if vals.size else None


_INSIGHT = None


def identity_cosine(face_ref: Image.Image, result: Image.Image) -> float | None:
    """
    ArcFace cosine similarity when insightface is installed.
    Returns None if unavailable — evaluation still proceeds with other metrics.
    """
    global _INSIGHT
    try:
        if _INSIGHT is None:
            from insightface.app import FaceAnalysis

            _INSIGHT = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            _INSIGHT.prepare(ctx_id=-1, det_size=(640, 640))
        app = _INSIGHT
    except Exception:
        return None

    def emb(im: Image.Image):
        arr = _to_rgb(im)[:, :, ::-1]  # BGR
        faces = app.get(arr)
        if not faces:
            return None
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        e = faces[-1].normed_embedding
        return e

    a, b = emb(face_ref), emb(result)
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def face_present(result: Image.Image, cache_dir) -> bool:
    from headswap.preprocess import detect_best_face, pil_to_rgb_np

    return detect_best_face(pil_to_rgb_np(result), cache_dir) is not None


def score_pair(
    pair_id: str,
    pipeline: str,
    body: Image.Image,
    face: Image.Image,
    result: Image.Image,
    latency_s: float,
    head_mask: Image.Image | None,
    cache_dir,
    identity_thresh: float = 0.35,
    body_psnr_thresh: float = 28.0,
) -> PairMetrics:
    reasons: list[str] = []
    id_cos = identity_cosine(face, result)
    body_psnr = body_preserve_score(body, result, head_mask)
    seam = seam_edge_delta(result, head_mask) if head_mask is not None else None
    detected = face_present(result, cache_dir)

    if not detected:
        reasons.append("no_face_detected")
    if id_cos is not None and id_cos < identity_thresh:
        reasons.append(f"low_identity:{id_cos:.3f}")
    if body_psnr is not None and body_psnr < body_psnr_thresh and head_mask is not None:
        # Only enforce body preserve when we intentionally masked
        reasons.append(f"body_drift:{body_psnr:.1f}")

    # Heuristic plastic-skin proxy: unusually low high-frequency energy in face crop
    rgb = _to_rgb(result)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 15.0:
        reasons.append(f"over_smooth:{lap_var:.1f}")

    success = len(reasons) == 0 or (detected and id_cos is not None and id_cos >= identity_thresh and "no_face_detected" not in reasons and not any(r.startswith("low_identity") for r in reasons))
    # Relax: if identity unavailable, require face + not over_smooth only
    if id_cos is None:
        success = detected and not any(r.startswith("over_smooth") for r in reasons)

    return PairMetrics(
        pair_id=pair_id,
        pipeline=pipeline,
        latency_s=latency_s,
        identity_cosine=id_cos,
        body_preserve_psnr=body_psnr,
        seam_edge_delta=seam,
        face_detected=detected,
        success=success,
        fail_reasons=reasons,
    )
