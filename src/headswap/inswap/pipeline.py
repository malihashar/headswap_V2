"""End-to-end deterministic face-swap pipeline (engine-agnostic)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from headswap.inswap.blend import color_match_face, face_ellipse_mask, soft_blend
from headswap.inswap.detect import InsightFaceDetector, aligned_face_crop, select_face
from headswap.inswap.engines import create_engine
from headswap.inswap.engines.base import DetectedFace, SwapEngine
from headswap.inswap.restore import FaceRestorer, RestorerName
from headswap.inswap.viz import (
    bgr_to_pil,
    difference_map,
    draw_detections,
    pil_to_bgr,
    side_by_side,
)

FacePolicy = Literal["largest", "rightmost", "leftmost", "index"]


@dataclass
class InSwapResult:
    image: Image.Image
    latency_s: float
    meta: dict[str, Any] = field(default_factory=dict)
    debug_paths: dict[str, str] = field(default_factory=dict)
    selected_face: DetectedFace | None = None
    all_faces: list[DetectedFace] = field(default_factory=list)


class InSwapPipeline:
    """
    Detect → select → ArcFace ID → local swap → blend → optional restore.

    Preserves full-resolution body pixels outside the selected face. Never
    regenerates the scene.
    """

    def __init__(
        self,
        *,
        cache_dir: Path | str,
        engine: str | SwapEngine = "inswapper",
        restorer: RestorerName | str = "none",
        device: str = "cuda",
        blend_strength: float = 1.0,
        color_match_strength: float = 0.25,
        restore_fidelity: float = 0.5,
        reblend_after_engine: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.blend_strength = float(blend_strength)
        self.color_match_strength = float(color_match_strength)
        self.restore_fidelity = float(restore_fidelity)
        self.reblend_after_engine = bool(reblend_after_engine)

        self.detector = InsightFaceDetector()
        if isinstance(engine, SwapEngine):
            self.engine = engine
            self.engine_name = engine.name
        else:
            self.engine_name = str(engine)
            self.engine = create_engine(self.engine_name)
        self.restorer = FaceRestorer(name=restorer)  # type: ignore[arg-type]
        self._loaded = False

    def load(self) -> None:
        stages = [
            ("detector", lambda: self.detector.load(self.cache_dir, device=self.device)),
            ("engine", lambda: self.engine.load(self.cache_dir, device=self.device)),
            ("restorer", lambda: self.restorer.load(self.cache_dir, device=self.device)),
        ]
        for _name, fn in tqdm(stages, desc="Load models", leave=False):
            fn()
        self._loaded = True

    def run(
        self,
        body: Image.Image,
        face: Image.Image,
        *,
        out_dir: Path | str | None = None,
        face_policy: FacePolicy | str = "largest",
        face_index: int = 0,
        source_face_policy: FacePolicy | str = "largest",
        source_face_index: int = 0,
        save_intermediates: bool = True,
    ) -> InSwapResult:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        out_path = Path(out_dir) if out_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "intermediates").mkdir(parents=True, exist_ok=True)

        timing: dict[str, float] = {}
        body_rgb = body.convert("RGB")
        # Preserve native resolution — never downscale the body canvas.
        body_bgr = pil_to_bgr(body_rgb)
        source_bgr = pil_to_bgr(face.convert("RGB"))

        def _save(name: str, img_bgr: np.ndarray) -> str | None:
            if out_path is None or not save_intermediates:
                return None
            p = out_path / "intermediates" / name
            cv2.imwrite(str(p), img_bgr)
            return str(p)

        dbg: dict[str, str] = {}

        # --- 1. Detect all faces ---
        t = time.perf_counter()
        with tqdm(total=1, desc="1/7 Detect faces", leave=False) as bar:
            faces = self.detector.detect(body_bgr)
            bar.update(1)
        timing["detect_s"] = time.perf_counter() - t
        if not faces:
            raise RuntimeError("No faces detected in body/scene image")

        # --- 2. Select target face ---
        t = time.perf_counter()
        with tqdm(total=1, desc="2/7 Select face", leave=False) as bar:
            selected = select_face(faces, policy=face_policy, index=face_index)
            bar.update(1)
        timing["select_s"] = time.perf_counter() - t

        det_overlay = draw_detections(body_bgr, faces, selected=selected)
        p = _save("01_detection.png", det_overlay)
        if p:
            dbg["detection"] = p

        # --- 3. Align selected face (debug crop) ---
        t = time.perf_counter()
        with tqdm(total=1, desc="3/7 Align face", leave=False) as bar:
            aligned = aligned_face_crop(body_bgr, selected, size=256)
            bar.update(1)
        timing["align_s"] = time.perf_counter() - t
        p = _save("02_aligned_face.png", aligned)
        if p:
            dbg["aligned_face"] = p

        # --- 4. Extract source identity (ArcFace via buffalo_l) ---
        t = time.perf_counter()
        with tqdm(total=1, desc="4/7 Extract identity", leave=False) as bar:
            source_face = self.detector.get_source_identity(
                source_bgr, policy=source_face_policy, index=source_face_index
            )
            bar.update(1)
        timing["identity_s"] = time.perf_counter() - t
        src_aligned = aligned_face_crop(source_bgr, source_face, size=256)
        p = _save("03_source_identity.png", src_aligned)
        if p:
            dbg["source_identity"] = p

        # --- 5. Local swap (InSwapper / pluggable engine) ---
        t = time.perf_counter()
        with tqdm(total=1, desc=f"5/7 Swap ({self.engine_name})", leave=False) as bar:
            engine_out = self.engine.swap(
                body_bgr, selected, source_face, paste_back=True
            )
            bar.update(1)
        timing["swap_s"] = time.perf_counter() - t
        swapped_raw = engine_out.image_bgr
        if engine_out.swapped_face_bgr is not None:
            p = _save("04_swapped_face.png", engine_out.swapped_face_bgr)
            if p:
                dbg["swapped_face"] = p
        p = _save("05_swapped_full.png", swapped_raw)
        if p:
            dbg["swapped_full"] = p

        # --- 6. Blend back (optional re-feather + color match) ---
        t = time.perf_counter()
        with tqdm(total=1, desc="6/7 Blend", leave=False) as bar:
            mask = engine_out.paste_mask
            if mask is None or self.reblend_after_engine:
                mask = face_ellipse_mask(
                    body_bgr.shape[:2], selected, expand=0.18, blur_frac=0.14
                )
            matched = color_match_face(
                swapped_raw,
                body_bgr,
                mask,
                strength=self.color_match_strength,
            )
            if self.reblend_after_engine:
                blended = soft_blend(
                    body_bgr, matched, mask, strength=self.blend_strength
                )
            else:
                blended = matched
            bar.update(1)
        timing["blend_s"] = time.perf_counter() - t
        p = _save("06_blend_mask.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        if p:
            dbg["blend_mask"] = p
        p = _save("07_blended.png", blended)
        if p:
            dbg["blended"] = p

        # --- 7. Optional face restoration ---
        t = time.perf_counter()
        with tqdm(total=1, desc="7/7 Restore", leave=False) as bar:
            final_bgr = self.restorer.enhance(
                blended, selected, fidelity=self.restore_fidelity
            )
            bar.update(1)
        timing["restore_s"] = time.perf_counter() - t
        p = _save("08_final.png", final_bgr)
        if p:
            dbg["final"] = p

        # Comparison assets
        diff = difference_map(body_bgr, final_bgr, amplify=4.0)
        sbs = side_by_side(body_bgr, final_bgr, diff)
        p = _save("09_difference.png", diff)
        if p:
            dbg["difference"] = p
        p = _save("10_comparison.png", sbs)
        if p:
            dbg["comparison"] = p

        final_pil = bgr_to_pil(final_bgr)
        if out_path is not None:
            result_path = out_path / "result.png"
            final_pil.save(result_path)
            dbg["result"] = str(result_path)
            body_rgb.save(out_path / "original.png")
            meta_path = out_path / "meta.json"
            meta_body = {
                "pipeline": "inswap",
                "engine": self.engine_name,
                "restorer": self.restorer.name,
                "device": self.device,
                "faces_detected": len(faces),
                "face_policy": face_policy,
                "face_index": face_index,
                "selected_bbox": list(selected.bbox),
                "selected_score": selected.det_score,
                "body_size": list(body_rgb.size),
                "source_size": list(face.size),
                "blend_strength": self.blend_strength,
                "color_match_strength": self.color_match_strength,
                "restore_fidelity": self.restore_fidelity,
                "timing_s": timing,
                "engine_meta": engine_out.meta,
            }
            meta_path.write_text(json.dumps(meta_body, indent=2), encoding="utf-8")
            dbg["meta"] = str(meta_path)

        latency = time.perf_counter() - t0
        timing["total_s"] = latency
        return InSwapResult(
            image=final_pil,
            latency_s=latency,
            meta={
                "pipeline": "inswap",
                "engine": self.engine_name,
                "restorer": self.restorer.name,
                "faces_detected": len(faces),
                "face_policy": face_policy,
                "face_index": int(face_index),
                "selected_bbox": list(selected.bbox),
                "timing_s": timing,
                "preserved_resolution": True,
                "full_scene_regenerated": False,
            },
            debug_paths=dbg,
            selected_face=selected,
            all_faces=faces,
        )
