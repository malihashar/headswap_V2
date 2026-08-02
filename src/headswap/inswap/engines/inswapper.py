"""InsightFace InSwapper-128 engine."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from headswap.inswap.engines.base import DetectedFace, SwapEngine, SwapEngineResult

# Public mirror used by FaceFusion / many OSS face-swap projects.
INSWAPPER_URL = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models/"
    "inswapper_128.onnx"
)
INSWAPPER_FILENAME = "inswapper_128.onnx"


def download_inswapper_model(cache_dir: Path, *, force: bool = False) -> Path:
    """Idempotently download ``inswapper_128.onnx`` into ``cache_dir/models``."""
    models = Path(cache_dir) / "models"
    models.mkdir(parents=True, exist_ok=True)
    dest = models / INSWAPPER_FILENAME
    if dest.is_file() and dest.stat().st_size > 1_000_000 and not force:
        return dest
    tmp = dest.with_suffix(".onnx.partial")
    print(f"[inswapper] downloading {INSWAPPER_FILENAME} → {dest}", flush=True)
    urllib.request.urlretrieve(INSWAPPER_URL, tmp)
    tmp.replace(dest)
    return dest


def _onnx_providers(device: str) -> list[str]:
    providers = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort  # type: ignore

        avail = set(ort.get_available_providers())
        want_cuda = device.startswith("cuda") and "CUDAExecutionProvider" in avail
        if want_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CoreMLExecutionProvider" in avail and device == "mps":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return providers


class InSwapperEngine(SwapEngine):
    """Deterministic ArcFace-conditioned face replacement (inswapper_128)."""

    name = "inswapper"

    def __init__(self) -> None:
        self._swapper: Any = None
        self._model_path: Path | None = None
        self._device = "cpu"

    def load(self, cache_dir: Path, *, device: str = "cuda") -> None:
        self._device = device
        model_path = download_inswapper_model(cache_dir)
        self._model_path = model_path

        # Ensure insightface can resolve relative model paths.
        os.environ.setdefault("INSIGHTFACE_HOME", str(Path(cache_dir) / "insightface"))

        import insightface  # type: ignore

        providers = _onnx_providers(device)
        swapper = insightface.model_zoo.get_model(str(model_path), providers=providers)
        # Some insightface builds need an explicit session prepare.
        if hasattr(swapper, "prepare"):
            try:
                swapper.prepare(ctx_id=0 if device.startswith("cuda") else -1)
            except TypeError:
                swapper.prepare(ctx_id=0)
        self._swapper = swapper
        print(
            f"[inswapper] loaded {model_path.name} providers={providers}",
            flush=True,
        )

    def swap(
        self,
        target_bgr: np.ndarray,
        target_face: DetectedFace,
        source_face: DetectedFace,
        *,
        paste_back: bool = True,
    ) -> SwapEngineResult:
        if self._swapper is None:
            raise RuntimeError("InSwapperEngine.load() must be called first")
        if source_face.raw is None or target_face.raw is None:
            raise RuntimeError(
                "InSwapperEngine requires InsightFace Face objects on "
                "DetectedFace.raw (run detect via InsightFaceDetector)."
            )
        if source_face.embedding is None and getattr(source_face.raw, "normed_embedding", None) is None:
            raise RuntimeError("Source face missing ArcFace embedding")

        img = np.ascontiguousarray(target_bgr)
        # paste_back=True: engine blends into full image at original pose/scale.
        out = self._swapper.get(
            img,
            target_face.raw,
            source_face.raw,
            paste_back=paste_back,
        )
        swapped_crop = None
        paste_mask = None
        if not paste_back:
            # Returns aligned 128x128 BGR face; caller must paste.
            swapped_crop = np.asarray(out)
            out_img = img.copy()
        else:
            out_img = np.asarray(out)
            # Approximate paste region for downstream blend / restore.
            x0, y0, x1, y1 = [int(round(v)) for v in target_face.bbox]
            h, w = out_img.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 > x0 and y1 > y0:
                paste_mask = np.zeros((h, w), dtype=np.uint8)
                paste_mask[y0:y1, x0:x1] = 255
                # Soften edges a bit for optional re-blend.
                k = max(3, int(round(0.08 * max(x1 - x0, y1 - y0))) | 1)
                paste_mask = cv2.GaussianBlur(paste_mask, (k, k), 0)
                swapped_crop = out_img[y0:y1, x0:x1].copy()

        return SwapEngineResult(
            image_bgr=out_img,
            swapped_face_bgr=swapped_crop,
            paste_mask=paste_mask,
            meta={
                "engine": self.name,
                "model": str(self._model_path) if self._model_path else None,
                "paste_back": paste_back,
                "target_bbox": list(target_face.bbox),
            },
        )

    def unload(self) -> None:
        self._swapper = None
