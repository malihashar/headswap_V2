from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from headswap.comfy.runtime import NodeRuntime, comfy_tensor_to_pil, get_value_at_index, pil_to_comfy_tensor
from headswap.pipelines.base import BasePipeline, PipelineResult, build_prompt
from headswap.preprocess import (
    crop_face_reference,
    crop_with_mask,
    head_hair_mask_from_face,
    lab_histogram_match_face,
    pad_to_ar_blur,
    resize_long_side,
    resize_max_keep_ar,
    soft_composite,
)


def _load_qwen_stack(rt: NodeRuntime, cfg: dict):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    key = (
        f"qwen::{cfg.get('unet_name')}::hs{cfg.get('headswap_lora_strength')}"
        f"::lt{cfg.get('lightning_lora_strength')}::sh{cfg.get('auraflow_shift')}::st{cfg.get('steps')}"
    )
    if key in rt.models:
        return rt.models[key]

    checkpoint = cfg["unet_name"]
    load_meta: dict = {
        "checkpoint": checkpoint,
        "checkpoint_preferred": checkpoint,
        "checkpoint_fallback_used": False,
        "loras_loaded": [],
        "lora_strengths": {},
        "fallbacks": [],
    }

    vae = rt.call("VAELoader", vae_name=cfg["vae_name"])
    clip = rt.call(
        "CLIPLoader",
        clip_name=cfg["clip_name"],
        type=cfg.get("clip_type", "qwen_image"),
        device="default",
    )
    unet = rt.call("UNETLoader", unet_name=checkpoint, weight_dtype="default")
    model = get_value_at_index(unet, 0)

    hs_name = cfg["headswap_lora_name"]
    hs_strength = float(cfg.get("headswap_lora_strength", 1.0))
    model = get_value_at_index(
        rt.call(
            "LoraLoaderModelOnly",
            model=model,
            lora_name=hs_name,
            strength_model=hs_strength,
        ),
        0,
    )
    load_meta["loras_loaded"].append(hs_name)
    load_meta["lora_strengths"][hs_name] = hs_strength
    print(f"[qwen] loaded LoRA {hs_name} strength={hs_strength}")

    lt = float(cfg.get("lightning_lora_strength", 0) or 0)
    lt_name = cfg.get("lightning_lora_name")
    if lt > 0 and lt_name:
        model = get_value_at_index(
            rt.call(
                "LoraLoaderModelOnly",
                model=model,
                lora_name=lt_name,
                strength_model=lt,
            ),
            0,
        )
        load_meta["loras_loaded"].append(lt_name)
        load_meta["lora_strengths"][lt_name] = lt
        print(f"[qwen] loaded LoRA {lt_name} strength={lt}")
    elif lt_name:
        load_meta["fallbacks"].append(f"lightning_lora_skipped_strength_zero:{lt_name}")

    if rt.has("ModelSamplingAuraFlow"):
        model = get_value_at_index(
            rt.call(
                "ModelSamplingAuraFlow",
                model=model,
                shift=float(cfg.get("auraflow_shift", 5)),
            ),
            0,
        )
    else:
        load_meta["fallbacks"].append("model_sampling_auraflow_missing")
    if rt.has("CFGNorm"):
        model = get_value_at_index(
            rt.call(
                "CFGNorm",
                model=model,
                strength=float(cfg.get("cfg_norm_strength", 1.0)),
            ),
            0,
        )
    else:
        load_meta["fallbacks"].append("cfgnorm_missing")

    bundle = {
        "model": model,
        "clip": get_value_at_index(clip, 0),
        "vae": get_value_at_index(vae, 0),
        "load_meta": load_meta,
    }
    rt.models[key] = bundle
    return bundle


def _sample_qwen(rt: NodeRuntime, bundle, body_t, face_t, cfg, prompt: str):
    import torch

    fallbacks: list[str] = []
    flux_kontext_applied = False
    flux_kontext_image_scale_applied = False

    with torch.inference_mode():
        # Official Qwen Image Edit 2511: FluxKontextImageScale on image1 before
        # TextEncodeQwenImageEditPlus and VAEEncode.
        image1 = body_t
        if rt.has("FluxKontextImageScale"):
            scaled = rt.call("FluxKontextImageScale", image=body_t)
            image1 = get_value_at_index(scaled, 0)
            flux_kontext_image_scale_applied = True
        else:
            fallbacks.append("flux_kontext_image_scale_missing")

        body_latent = rt.call("VAEEncode", vae=bundle["vae"], pixels=image1)
        if not rt.has("TextEncodeQwenImageEditPlus"):
            raise KeyError("TextEncodeQwenImageEditPlus node missing — update ComfyUI")
        pos = rt.call(
            "TextEncodeQwenImageEditPlus",
            clip=bundle["clip"],
            prompt=prompt,
            vae=bundle["vae"],
            image1=image1,
            image2=face_t,
        )
        neg = rt.call(
            "TextEncodeQwenImageEditPlus",
            clip=bundle["clip"],
            prompt=str(cfg.get("negative_prompt", "") or ""),
            vae=bundle["vae"],
            image1=image1,
            image2=face_t,
        )
        positive = get_value_at_index(pos, 0)
        negative = get_value_at_index(neg, 0)
        # Official Qwen 2511 blueprint includes Edit Model Reference Method
        # (FluxKontextMultiReferenceLatentMethod) with index_timestep_zero.
        if rt.has("FluxKontextMultiReferenceLatentMethod"):
            positive = get_value_at_index(
                rt.call(
                    "FluxKontextMultiReferenceLatentMethod",
                    conditioning=positive,
                    reference_latents_method="index_timestep_zero",
                ),
                0,
            )
            negative = get_value_at_index(
                rt.call(
                    "FluxKontextMultiReferenceLatentMethod",
                    conditioning=negative,
                    reference_latents_method="index_timestep_zero",
                ),
                0,
            )
            flux_kontext_applied = True

        noise = get_value_at_index(
            rt.call("RandomNoise", noise_seed=int(cfg.get("seed", 46))), 0
        )
        guider = get_value_at_index(
            rt.call(
                "CFGGuider",
                model=bundle["model"],
                positive=positive,
                negative=negative,
                cfg=float(cfg.get("cfg", 1.1)),
            ),
            0,
        )
        sampler = get_value_at_index(
            rt.call("KSamplerSelect", sampler_name=cfg.get("sampler", "euler")), 0
        )
        sigmas = get_value_at_index(
            rt.call(
                "BasicScheduler",
                model=bundle["model"],
                scheduler=cfg.get("scheduler", "simple"),
                steps=int(cfg.get("steps", 6)),
                denoise=float(cfg.get("denoise", 1.0)),
            ),
            0,
        )
        samples = rt.call(
            "SamplerCustomAdvanced",
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=get_value_at_index(body_latent, 0),
        )
        decoded = rt.call(
            "VAEDecode",
            samples=get_value_at_index(samples, 0),
            vae=bundle["vae"],
        )
        image = comfy_tensor_to_pil(get_value_at_index(decoded, 0))
        sample_meta = {
            "reference_latent_used": False,
            "flux_kontext_applied": flux_kontext_applied,
            "flux_kontext_image_scale_applied": flux_kontext_image_scale_applied,
            "fallbacks": fallbacks,
        }
        return image, sample_meta


class QwenBaselinePipeline(BasePipeline):
    """Faithful port of the current Magic Hour Colab Cell 5."""

    name = "qwen_baseline"

    def _ensure_runtime(self) -> NodeRuntime:
        if self.runtime is None:
            self.runtime = NodeRuntime()
        return self.runtime

    def run(self, body: Image.Image, face: Image.Image, out_dir: Path | None = None) -> PipelineResult:
        import torch

        t0 = time.perf_counter()
        rt = self._ensure_runtime()
        bundle = _load_qwen_stack(rt, self.cfg)
        div_by = int(self.cfg.get("div_by", 8))
        max_dim = int(self.cfg.get("max_dim", 576))

        body_pil = resize_max_keep_ar(body.convert("RGB"), max_dim, div_by)
        face_crop = crop_face_reference(
            face,
            self.cache_dir,
            top=float(self.cfg.get("face_top_pad", 0.65)),
            bot=float(self.cfg.get("face_bot_pad", 0.15)),
            side=float(self.cfg.get("face_side_pad", 0.35)),
            include_shoulders=False,
        )
        if self.cfg.get("blur_pad_face", True):
            face_for_model = pad_to_ar_blur(face_crop, body_pil.width / body_pil.height).resize(
                body_pil.size, Image.Resampling.LANCZOS
            )
        else:
            face_for_model = face_crop.resize(body_pil.size, Image.Resampling.LANCZOS)

        prompt = str(self.cfg.get("prompt", "")).strip()
        out, sample_meta = _sample_qwen(
            rt,
            bundle,
            pil_to_comfy_tensor(body_pil, torch),
            pil_to_comfy_tensor(face_for_model, torch),
            self.cfg,
            prompt,
        )
        load_meta = dict(bundle.get("load_meta") or {})
        fallbacks = list(load_meta.get("fallbacks") or []) + list(
            sample_meta.get("fallbacks") or []
        )
        dbg = {
            k: v
            for k, v in {
                "debug_body": self._save_debug(out_dir, "debug_body.png", body_pil),
                "debug_face_crop": self._save_debug(out_dir, "debug_face_crop.png", face_crop),
                "debug_face_for_model": self._save_debug(
                    out_dir, "debug_face_for_model.png", face_for_model
                ),
            }.items()
            if v
        }
        meta = {
            "pipeline": self.name,
            "checkpoint": load_meta.get("checkpoint"),
            "loras_loaded": list(load_meta.get("loras_loaded") or []),
            "lora_strengths": dict(load_meta.get("lora_strengths") or {}),
            "prompt": prompt,
            "crop_size": list(body_pil.size),
            "body_size": list(body_pil.size),
            "reference_latent_used": bool(sample_meta.get("reference_latent_used")),
            "flux_kontext_applied": bool(sample_meta.get("flux_kontext_applied")),
            "flux_kontext_image_scale_applied": bool(
                sample_meta.get("flux_kontext_image_scale_applied")
            ),
            "fallbacks": fallbacks,
        }
        print(
            f"[qwen_baseline] checkpoint={meta['checkpoint']} loras={meta['loras_loaded']} "
            f"strengths={meta['lora_strengths']} crop={meta['crop_size']} "
            f"flux_kontext={meta['flux_kontext_applied']} "
            f"image_scale={meta['flux_kontext_image_scale_applied']} "
            f"fallbacks={fallbacks or 'none'}"
        )
        return PipelineResult(
            image=out,
            latency_s=time.perf_counter() - t0,
            meta=meta,
            debug_paths=dbg,
        )


class QwenImprovedPipeline(BasePipeline):
    """Qwen 2511 + BFS with official prompt, mask crop/stitch, no blur-pad."""

    name = "qwen_improved_mask_crop"

    def _ensure_runtime(self) -> NodeRuntime:
        if self.runtime is None:
            self.runtime = NodeRuntime()
        return self.runtime

    def run(self, body: Image.Image, face: Image.Image, out_dir: Path | None = None) -> PipelineResult:
        import torch

        t0 = time.perf_counter()
        rt = self._ensure_runtime()
        bundle = _load_qwen_stack(rt, self.cfg)
        div_by = int(self.cfg.get("div_by", 8))

        body_full = resize_max_keep_ar(
            body.convert("RGB"), int(self.cfg.get("max_body_dim", 1024)), div_by
        )
        face_crop = crop_face_reference(
            face,
            self.cache_dir,
            top=float(self.cfg.get("face_top_pad", 0.65)),
            bot=float(self.cfg.get("face_bot_pad", 0.25)),
            side=float(self.cfg.get("face_side_pad", 0.35)),
            include_shoulders=bool(self.cfg.get("include_shoulders", True)),
        )
        mask = head_hair_mask_from_face(
            body_full,
            self.cache_dir,
            expand_px=int(self.cfg.get("mask_expand_px", 18)),
            blur_px=int(self.cfg.get("mask_blur_px", 12)),
        )
        crop_img, _, box = crop_with_mask(body_full, mask, pad=12, div_by=div_by)
        crop_work = resize_long_side(crop_img, int(self.cfg.get("crop_long_side", 768)), div_by)
        if self.cfg.get("blur_pad_face", False):
            face_ref = pad_to_ar_blur(face_crop, crop_work.width / crop_work.height).resize(
                crop_work.size, Image.Resampling.LANCZOS
            )
        else:
            face_ref = face_crop.resize(crop_work.size, Image.Resampling.LANCZOS)

        prompt = build_prompt(self.cfg, body_full, face_crop, self.cache_dir)
        edited, sample_meta = _sample_qwen(
            rt,
            bundle,
            pil_to_comfy_tensor(crop_work, torch),
            pil_to_comfy_tensor(face_ref, torch),
            self.cfg,
            prompt,
        )
        stitched = soft_composite(body_full, edited, mask, box)
        stitched = lab_histogram_match_face(stitched, body_full, mask, strength=0.3)
        load_meta = dict(bundle.get("load_meta") or {})
        fallbacks = list(load_meta.get("fallbacks") or []) + list(
            sample_meta.get("fallbacks") or []
        )
        dbg = {
            k: v
            for k, v in {
                "debug_body": self._save_debug(out_dir, "debug_body.png", body_full),
                "debug_face_crop": self._save_debug(out_dir, "debug_face_crop.png", face_crop),
                "debug_mask": self._save_debug(out_dir, "debug_mask.png", mask),
                "debug_crop": self._save_debug(out_dir, "debug_crop.png", crop_work),
                "debug_edited_crop": self._save_debug(out_dir, "debug_edited_crop.png", edited),
            }.items()
            if v
        }
        meta = {
            "pipeline": self.name,
            "checkpoint": load_meta.get("checkpoint"),
            "loras_loaded": list(load_meta.get("loras_loaded") or []),
            "lora_strengths": dict(load_meta.get("lora_strengths") or {}),
            "prompt": prompt,
            "crop_size": list(crop_work.size),
            "body_size": list(body_full.size),
            "face_ref_size": list(face_ref.size),
            "reference_latent_used": bool(sample_meta.get("reference_latent_used")),
            "flux_kontext_applied": bool(sample_meta.get("flux_kontext_applied")),
            "flux_kontext_image_scale_applied": bool(
                sample_meta.get("flux_kontext_image_scale_applied")
            ),
            "fallbacks": fallbacks,
        }
        print(
            f"[qwen_improved] checkpoint={meta['checkpoint']} loras={meta['loras_loaded']} "
            f"strengths={meta['lora_strengths']} crop={meta['crop_size']} "
            f"flux_kontext={meta['flux_kontext_applied']} "
            f"image_scale={meta['flux_kontext_image_scale_applied']} "
            f"fallbacks={fallbacks or 'none'}"
        )
        return PipelineResult(
            image=stitched,
            latency_s=time.perf_counter() - t0,
            meta=meta,
            debug_paths=dbg,
        )
