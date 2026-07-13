from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from headswap.preprocess import (
    crop_face_reference,
    crop_with_mask,
    describe_hair_length_hint,
    head_hair_mask_from_face,
    lab_histogram_match_face,
    resize_long_side,
    resize_max_keep_ar,
    soft_composite,
)


@dataclass
class PipelineResult:
    image: Image.Image
    latency_s: float
    meta: dict[str, Any] = field(default_factory=dict)
    debug_paths: dict[str, str] = field(default_factory=dict)


class BasePipeline:
    name = "base"

    def __init__(self, cfg: dict[str, Any], runtime=None, cache_dir: Path | None = None):
        self.cfg = cfg
        self.runtime = runtime
        if cache_dir is None:
            from headswap.config import project_root

            cache_dir = project_root() / ".cache" / "headswap_v2"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, body: Image.Image, face: Image.Image, out_dir: Path | None = None) -> PipelineResult:
        raise NotImplementedError

    def _save_debug(self, out_dir: Path | None, name: str, im: Image.Image) -> str | None:
        if out_dir is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        im.save(path)
        return str(path)


class MockHeadSwapPipeline(BasePipeline):
    """
    CPU mock used for harness validation / offline CI.
    - Masked configs: composite face into head region (preserves body).
    - Baseline configs: rewrite full frame (simulates denoise=1.0 body drift).
    """

    name = "mock"

    def run(self, body: Image.Image, face: Image.Image, out_dir: Path | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        body = body.convert("RGB")
        face_crop = crop_face_reference(
            face,
            self.cache_dir,
            top=float(self.cfg.get("face_top_pad", 0.65)),
            bot=float(self.cfg.get("face_bot_pad", 0.25)),
            side=float(self.cfg.get("face_side_pad", 0.35)),
            include_shoulders=bool(self.cfg.get("include_shoulders", True)),
        )
        mask = head_hair_mask_from_face(
            body,
            self.cache_dir,
            expand_px=int(self.cfg.get("mask_expand_px", 18)),
            blur_px=int(self.cfg.get("mask_blur_px", 12)),
        )
        pipeline_key = str(self.cfg.get("pipeline", self.cfg.get("name", "")))
        full_frame = pipeline_key in {"qwen_baseline"} or bool(self.cfg.get("mock_full_frame"))

        import numpy as np

        if full_frame:
            # Simulate full-image regeneration: mild global recolor + face paste (body drifts)
            arr = np.asarray(body).astype("float32")
            arr = np.clip(arr * 0.92 + 12.0, 0, 255)
            drifted = Image.fromarray(arr.astype("uint8"))
            crop_img, crop_mask, box = crop_with_mask(
                drifted, mask, pad=8, div_by=int(self.cfg.get("div_by", 16))
            )
            donor = face_crop.resize(crop_img.size, Image.Resampling.LANCZOS)
            a = np.asarray(crop_mask.convert("L")).astype("float32") / 255.0
            a = a[..., None]
            out_crop = np.asarray(crop_img).astype("float32") * (1 - a) + np.asarray(donor).astype(
                "float32"
            ) * a
            edited = Image.fromarray(out_crop.clip(0, 255).astype("uint8"))
            stitched = soft_composite(drifted, edited, mask, box)
            mode = "mock_full_frame_drift"
        else:
            crop_img, crop_mask, box = crop_with_mask(
                body, mask, pad=8, div_by=int(self.cfg.get("div_by", 16))
            )
            donor = face_crop.resize(crop_img.size, Image.Resampling.LANCZOS)
            a = np.asarray(crop_mask.convert("L")).astype("float32") / 255.0
            a = a[..., None]
            out_crop = np.asarray(crop_img).astype("float32") * (1 - a) + np.asarray(donor).astype(
                "float32"
            ) * a
            edited = Image.fromarray(out_crop.clip(0, 255).astype("uint8"))
            stitched = soft_composite(body, edited, mask, box)
            stitched = lab_histogram_match_face(stitched, body, mask, strength=0.35)
            mode = "mock_mask_composite"

        dbg = {
            k: v
            for k, v in {
                "debug_face_crop": self._save_debug(out_dir, "debug_face_crop.png", face_crop),
                "debug_mask": self._save_debug(out_dir, "debug_mask.png", mask),
                "debug_crop": self._save_debug(out_dir, "debug_crop.png", crop_img),
                "debug_edited_crop": self._save_debug(out_dir, "debug_edited_crop.png", edited),
            }.items()
            if v
        }
        return PipelineResult(
            image=stitched,
            latency_s=time.perf_counter() - t0,
            meta={"pipeline": self.cfg.get("name", self.name), "mode": mode},
            debug_paths=dbg,
        )


def build_prompt(cfg: dict[str, Any], body: Image.Image, face: Image.Image, cache_dir: Path) -> str:
    base = str(cfg.get("prompt", "")).strip()
    hint = describe_hair_length_hint(body, face, cache_dir)
    return (base + hint).strip()
