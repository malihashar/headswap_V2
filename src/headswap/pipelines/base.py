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
    - flux_kontext: Align → Paste path + feature instrumentation (no diffusion).
    """

    name = "mock"

    def run(self, body: Image.Image, face: Image.Image, out_dir: Path | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        body = body.convert("RGB")
        pipeline_key = str(self.cfg.get("pipeline", self.cfg.get("name", "")))

        if pipeline_key == "flux_kontext":
            return self._run_kontext_mock(body, face, out_dir, t0)
        if pipeline_key == "step1x_edit":
            return self._run_step1x_mock(body, face, out_dir, t0)
        if pipeline_key == "omnigen2":
            return self._run_omnigen2_mock(body, face, out_dir, t0)

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

    def _run_kontext_mock(
        self,
        body: Image.Image,
        face: Image.Image,
        out_dir: Path | None,
        t0: float,
    ) -> PipelineResult:
        """Exercise Align→Paste + record which community features config requests."""
        from headswap.preprocess import (
            align_face_to_destination,
            color_match_rgba_to_destination,
            paste_aligned_face,
            resize_max_keep_ar,
        )

        def _cfg_bool(*keys: str, default: bool = True) -> bool:
            for k in keys:
                if k in self.cfg and self.cfg[k] is not None:
                    return bool(self.cfg[k])
            return default

        max_dim = int(self.cfg.get("max_dim", 768))
        div_by = int(self.cfg.get("div_by", 8))
        body_pil = resize_max_keep_ar(body.convert("RGB"), max_dim, div_by)
        face_crop = crop_face_reference(
            face,
            self.cache_dir,
            top=float(self.cfg.get("face_top_pad", 0.75)),
            bot=float(self.cfg.get("face_bot_pad", 0.18)),
            side=float(self.cfg.get("face_side_pad", 0.40)),
            include_shoulders=False,
        )
        aligned_rgba, align_info = align_face_to_destination(
            face_crop, body_pil, self.cache_dir
        )
        pre_match = float(self.cfg.get("pre_color_match_strength", 0.55) or 0.0)
        if aligned_rgba is not None:
            if pre_match > 0:
                aligned_rgba = color_match_rgba_to_destination(
                    aligned_rgba, body_pil, strength=pre_match
                )
                align_info["pre_color_match_strength"] = pre_match
            composite, paste_info = paste_aligned_face(body_pil, aligned_rgba)
        else:
            paste_info = {
                "composite_paste": False,
                "composite_paste_skip_reason": (
                    align_info.get("face_alignment_skip_reason")
                    or "alignment_failed_no_paste"
                ),
            }
            composite = body_pil.copy()
            donor = face_crop.resize(
                (max(32, body_pil.width // 3), max(32, body_pil.height // 3)),
                Image.Resampling.LANCZOS,
            )
            composite.paste(
                donor,
                ((body_pil.width - donor.width) // 2, max(0, body_pil.height // 8)),
            )
            paste_info["composite_paste"] = True
            paste_info["composite_paste_fallback"] = "center_resize_paste"

        mask = head_hair_mask_from_face(
            body_pil,
            self.cache_dir,
            expand_px=int(self.cfg.get("mask_expand_px", 22)),
            blur_px=int(self.cfg.get("mask_blur_px", 14)),
        )
        box = (0, 0, body_pil.width, body_pil.height)
        # Mild mock refine: slightly soften composite inside head mask
        import numpy as np

        arr = np.asarray(composite).astype("float32")
        m = np.asarray(mask.convert("L")).astype("float32")[..., None] / 255.0
        refined = Image.fromarray(
            np.clip(arr * (1 - 0.08 * m) + 8.0 * m, 0, 255).astype("uint8")
        )
        stitched = soft_composite(body_pil, refined, mask, box)
        post_match = float(self.cfg.get("post_color_match_strength", 0.35) or 0.0)
        stitched = lab_histogram_match_face(
            stitched, body_pil, mask, strength=post_match
        )

        guidance = float(self.cfg.get("flux_guidance", 4.0))
        ref_en = _cfg_bool("reference_latent", default=True)
        id_en = _cfg_bool("identity_reference", default=True)
        zero_en = _cfg_bool("conditioning_zero_out", default=True)
        scale_en = _cfg_bool("image_scale", "flux_kontext_image_scale", default=True)
        lora_name = self.cfg.get("placement_lora")
        # Mock cannot load Comfy nodes / LoRAs — record intent + skip reasons.
        mock_skip = "force_mock_no_comfy_runtime"
        meta = {
            "pipeline": "flux_kontext",
            "mode": "mock_align_paste_refine",
            "face_alignment": bool(align_info.get("face_alignment")),
            "face_alignment_backend": align_info.get("face_alignment_backend"),
            "face_alignment_skip_reason": align_info.get("face_alignment_skip_reason"),
            "composite_paste": bool(paste_info.get("composite_paste")),
            "composite_paste_skip_reason": paste_info.get("composite_paste_skip_reason"),
            "composite_paste_fallback": paste_info.get("composite_paste_fallback"),
            "pre_color_match_strength": align_info.get(
                "pre_color_match_strength", pre_match
            ),
            "reference_latent_enabled": ref_en,
            "reference_latent_used": False,
            "reference_latent_skip_reason": (
                mock_skip if ref_en else "disabled_by_config"
            ),
            "identity_reference_enabled": id_en,
            "identity_reference_used": False,
            "identity_reference_skip_reason": (
                mock_skip if id_en else "disabled_by_config"
            ),
            "conditioning_zero_out_enabled": zero_en,
            "conditioning_zero_out_applied": False,
            "conditioning_zero_out_skip_reason": (
                mock_skip if zero_en else "disabled_by_config"
            ),
            "flux_kontext_image_scale_enabled": scale_en,
            "flux_kontext_image_scale_applied": False,
            "flux_kontext_image_scale_skip_reason": (
                mock_skip if scale_en else "disabled_by_config"
            ),
            "placement_lora_loaded": False,
            "placement_lora_name": lora_name,
            "placement_lora_strength": float(
                self.cfg.get("placement_lora_strength", 1.0) or 0.0
            ),
            "placement_lora_skip_reason": mock_skip,
            "flux_guidance_value": guidance,
            "flux_guidance_applied": False,
            "flux_guidance_skip_reason": mock_skip,
            "denoise": float(self.cfg.get("denoise", 0.72)),
            "steps": int(self.cfg.get("steps", 32)),
            "features": {
                "face_alignment": bool(align_info.get("face_alignment")),
                "composite_paste": bool(paste_info.get("composite_paste")),
                "reference_latent": False,
                "identity_reference": False,
                "conditioning_zero_out": False,
                "flux_kontext_image_scale": False,
                "placement_lora_loaded": False,
                "flux_guidance_value": guidance,
                "denoise": float(self.cfg.get("denoise", 0.72)),
            },
        }
        dbg = {
            k: v
            for k, v in {
                "debug_body": self._save_debug(out_dir, "debug_body.png", body_pil),
                "debug_face_crop": self._save_debug(
                    out_dir, "debug_face_crop.png", face_crop
                ),
                "debug_composite": self._save_debug(
                    out_dir, "debug_composite.png", composite
                ),
                "debug_mask": self._save_debug(out_dir, "debug_mask.png", mask),
                "debug_aligned_face": (
                    self._save_debug(
                        out_dir, "debug_aligned_face.png", aligned_rgba.convert("RGBA")
                    )
                    if aligned_rgba is not None
                    else None
                ),
            }.items()
            if v
        }
        return PipelineResult(
            image=stitched,
            latency_s=time.perf_counter() - t0,
            meta=meta,
            debug_paths=dbg,
        )

    def _run_step1x_mock(
        self,
        body: Image.Image,
        face: Image.Image,
        out_dir: Path | None,
        t0: float,
    ) -> PipelineResult:
        """CPU stand-in: dual-panel pack + soft face paste (no Diffusers)."""
        from headswap.pipelines.step1x_edit import build_dual_panel
        from headswap.preprocess import resize_max_keep_ar

        size_level = int(self.cfg.get("crop_size", 1024))
        body_work = resize_max_keep_ar(body.convert("RGB"), min(size_level, 768), div_by=8)
        face_crop = crop_face_reference(
            face,
            self.cache_dir,
            top=float(self.cfg.get("face_top_pad", 0.65)),
            bot=float(self.cfg.get("face_bot_pad", 0.15)),
            side=float(self.cfg.get("face_side_pad", 0.35)),
            include_shoulders=False,
        )
        panel, layout = build_dual_panel(
            body_work, face_crop, size_level=min(size_level, 512), label=True
        )
        # Mock "edit": paste resized face onto left head region via existing mask path.
        mask = head_hair_mask_from_face(
            body_work,
            self.cache_dir,
            expand_px=18,
            blur_px=12,
        )
        crop_img, crop_mask, box = crop_with_mask(
            body_work, mask, pad=8, div_by=8
        )
        import numpy as np

        donor = face_crop.resize(crop_img.size, Image.Resampling.LANCZOS)
        a = np.asarray(crop_mask.convert("L")).astype("float32") / 255.0
        a = a[..., None]
        out_crop = np.asarray(crop_img).astype("float32") * (1 - a) + np.asarray(
            donor
        ).astype("float32") * a
        edited_crop = Image.fromarray(out_crop.clip(0, 255).astype("uint8"))
        stitched = soft_composite(body_work, edited_crop, mask, box)
        stitched = lab_histogram_match_face(stitched, body_work, mask, strength=0.3)

        guidance = float(self.cfg.get("guidance", 6.0))
        steps = int(self.cfg.get("steps", 50))
        meta = {
            "pipeline": "step1x_edit",
            "mode": "mock_dual_panel",
            "model_version": "stepfun-ai/Step1X-Edit-v1p2",
            "inference_settings": {
                "steps": steps,
                "guidance": guidance,
                "seed": int(self.cfg.get("seed", 42)),
                "strength": float(self.cfg.get("strength", 1.0)),
                "crop_size": size_level,
            },
            "timing_s": {
                "load": 0.0,
                "sampling": 0.0,
                "total": round(time.perf_counter() - t0, 4),
            },
            "load_time_s": 0.0,
            "encode_time_s": None,
            "sampling_time_s": 0.0,
            "decode_time_s": None,
            "total_latency_s": round(time.perf_counter() - t0, 4),
            "note_encode_decode": (
                "encode/decode bundled inside Diffusers __call__; null outside mock"
            ),
            "strength_applied": False,
            "strength_skip_reason": "force_mock_no_diffusers",
            "dual_panel_layout": layout,
            "features": {
                "dual_panel": True,
                "diffusers": False,
            },
        }
        dbg = {
            k: v
            for k, v in {
                "debug_body": self._save_debug(out_dir, "debug_body.png", body_work),
                "debug_face_crop": self._save_debug(
                    out_dir, "debug_face_crop.png", face_crop
                ),
                "debug_dual_panel": self._save_debug(
                    out_dir, "debug_dual_panel.png", panel
                ),
                "debug_mask": self._save_debug(out_dir, "debug_mask.png", mask),
            }.items()
            if v
        }
        return PipelineResult(
            image=stitched,
            latency_s=time.perf_counter() - t0,
            meta=meta,
            debug_paths=dbg,
        )

    def _run_omnigen2_mock(
        self,
        body: Image.Image,
        face: Image.Image,
        out_dir: Path | None,
        t0: float,
    ) -> PipelineResult:
        """CPU stand-in: multi-image inputs + soft face paste (no OmniGen2 package)."""
        from headswap.preprocess import resize_max_keep_ar

        crop_size = int(self.cfg.get("crop_size", 1024))
        body_work = resize_max_keep_ar(body.convert("RGB"), min(crop_size, 768), div_by=16)
        face_crop = crop_face_reference(
            face,
            self.cache_dir,
            top=float(self.cfg.get("face_top_pad", 0.65)),
            bot=float(self.cfg.get("face_bot_pad", 0.15)),
            side=float(self.cfg.get("face_side_pad", 0.35)),
            include_shoulders=False,
        )
        mask = head_hair_mask_from_face(
            body_work, self.cache_dir, expand_px=18, blur_px=12
        )
        crop_img, crop_mask, box = crop_with_mask(body_work, mask, pad=8, div_by=8)
        import numpy as np

        donor = face_crop.resize(crop_img.size, Image.Resampling.LANCZOS)
        a = np.asarray(crop_mask.convert("L")).astype("float32") / 255.0
        a = a[..., None]
        out_crop = np.asarray(crop_img).astype("float32") * (1 - a) + np.asarray(
            donor
        ).astype("float32") * a
        edited_crop = Image.fromarray(out_crop.clip(0, 255).astype("uint8"))
        stitched = soft_composite(body_work, edited_crop, mask, box)
        stitched = lab_histogram_match_face(stitched, body_work, mask, strength=0.3)

        text_g = float(self.cfg.get("guidance", 5.0))
        image_g = float(self.cfg.get("image_guidance", 2.5))
        steps = int(self.cfg.get("steps", 50))
        meta = {
            "pipeline": "omnigen2",
            "mode": "mock_multi_image_in_context",
            "model_version": "OmniGen2/OmniGen2",
            "inference_settings": {
                "steps": steps,
                "guidance": text_g,
                "image_guidance": image_g,
                "seed": int(self.cfg.get("seed", 0)),
                "strength": float(self.cfg.get("strength", 1.0)),
                "crop_size": crop_size,
            },
            "timing_s": {
                "load": 0.0,
                "sampling": 0.0,
                "total": round(time.perf_counter() - t0, 4),
            },
            "load_time_s": 0.0,
            "encode_time_s": None,
            "sampling_time_s": 0.0,
            "decode_time_s": None,
            "total_latency_s": round(time.perf_counter() - t0, 4),
            "strength_applied": False,
            "strength_skip_reason": "force_mock_no_omnigen2",
            "input_mode": "multi_image_in_context",
            "features": {"multi_image": True, "omnigen2_package": False},
        }
        dbg = {
            k: v
            for k, v in {
                "debug_body": self._save_debug(out_dir, "debug_body.png", body_work),
                "debug_face_crop": self._save_debug(
                    out_dir, "debug_face_crop.png", face_crop
                ),
                "debug_mask": self._save_debug(out_dir, "debug_mask.png", mask),
            }.items()
            if v
        }
        return PipelineResult(
            image=stitched,
            latency_s=time.perf_counter() - t0,
            meta=meta,
            debug_paths=dbg,
        )


def build_prompt(cfg: dict[str, Any], body: Image.Image, face: Image.Image, cache_dir: Path) -> str:
    base = str(cfg.get("prompt", "")).strip()
    hint = describe_hair_length_hint(body, face, cache_dir)
    return (base + hint).strip()
