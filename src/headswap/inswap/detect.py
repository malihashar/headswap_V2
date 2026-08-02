"""InsightFace face detection + ArcFace identity extraction."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from headswap.inswap.engines.base import DetectedFace

FacePolicy = Literal["largest", "rightmost", "leftmost", "index"]


class InsightFaceDetector:
    """RetinaFace/SCRFD detection + ArcFace embeddings via buffalo_l."""

    def __init__(self) -> None:
        self._app: Any = None
        self._device = "cpu"

    def load(self, cache_dir: Path, *, device: str = "cuda") -> None:
        from insightface.app import FaceAnalysis  # type: ignore

        self._device = device
        root = Path(cache_dir) / "insightface"
        root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("INSIGHTFACE_HOME", str(root))

        providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort  # type: ignore

            avail = set(ort.get_available_providers())
            if device.startswith("cuda") and "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

        app = FaceAnalysis(name="buffalo_l", providers=providers)
        # ctx_id >= 0 → GPU when CUDA provider is active
        ctx_id = 0 if device.startswith("cuda") else -1
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._app = app
        print(f"[detect] InsightFace buffalo_l ready providers={providers}", flush=True)

    @property
    def ready(self) -> bool:
        return self._app is not None

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        if self._app is None:
            raise RuntimeError("InsightFaceDetector.load() must be called first")
        img = np.ascontiguousarray(image_bgr)
        faces = self._app.get(img) or []
        out: list[DetectedFace] = []
        for f in faces:
            bbox = tuple(float(x) for x in f.bbox.tolist())
            kps = None
            if getattr(f, "kps", None) is not None:
                kps = np.asarray(f.kps, dtype=np.float32)
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)
            out.append(
                DetectedFace(
                    bbox=bbox,  # type: ignore[arg-type]
                    kps=kps,
                    det_score=float(getattr(f, "det_score", 0.0) or 0.0),
                    embedding=emb,
                    raw=f,
                )
            )
        # Largest-first for stable indexing
        out.sort(key=lambda d: d.area, reverse=True)
        return out

    def get_source_identity(
        self, source_bgr: np.ndarray, *, policy: FacePolicy = "largest", index: int = 0
    ) -> DetectedFace:
        faces = self.detect(source_bgr)
        if not faces:
            raise RuntimeError("No face detected in source/identity image")
        return select_face(faces, policy=policy, index=index)


def select_face(
    faces: list[DetectedFace],
    *,
    policy: FacePolicy | str = "largest",
    index: int = 0,
) -> DetectedFace:
    if not faces:
        raise RuntimeError("No faces to select from")
    pol = (policy or "largest").strip().lower()
    if pol == "rightmost":
        ordered = sorted(faces, key=lambda d: d.center_x, reverse=True)
        return ordered[0]
    if pol == "leftmost":
        ordered = sorted(faces, key=lambda d: d.center_x)
        return ordered[0]
    if pol == "index":
        i = int(index)
        if i < 0 or i >= len(faces):
            raise RuntimeError(f"face index {i} out of range (n={len(faces)})")
        return faces[i]
    # largest (faces already largest-first from detect, but be safe)
    return max(faces, key=lambda d: d.area)


def aligned_face_crop(
    image_bgr: np.ndarray,
    face: DetectedFace,
    *,
    size: int = 256,
) -> np.ndarray:
    """5-point similarity-align crop for debug / intermediate export."""
    if face.kps is None or face.kps.shape != (5, 2):
        x0, y0, x1, y1 = [int(round(v)) for v in face.bbox]
        h, w = image_bgr.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    # Standard ArcFace template (112x112), scaled to ``size``.
    dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    scale = float(size) / 112.0
    dst = dst * scale
    src = face.kps.astype(np.float32)
    m = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    if m is None:
        x0, y0, x1, y1 = [int(round(v)) for v in face.bbox]
        crop = image_bgr[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(image_bgr, m, (size, size), borderValue=0.0)
