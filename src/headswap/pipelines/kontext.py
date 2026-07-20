"""FLUX.1 Kontext [dev] first integration — body + face ref → edit → image.

This is intentionally minimal: no Place It / Put It Here LoRA, no InsightFace
alignment, no soft-paste / LAB postprocess. Later stages are marked TODO.
"""
from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from headswap.comfy.full_load import force_sampling_full_load
from headswap.comfy.runtime import (
    NodeRuntime,
    comfy_tensor_to_pil,
    get_value_at_index,
    pil_to_comfy_tensor,
)
from headswap.pipelines.base import BasePipeline, PipelineResult
from headswap.pipelines.errors import PipelineRunError
from headswap.preprocess import crop_face_reference, pad_to_ar_blur, resize_max_keep_ar


# TODO: InsightFace alignment
# TODO: ReferenceLatent refinements (multi-ref method / index_timestep_zero)
# TODO: ConditioningZeroOut
# TODO: Place It / Put It Here LoRA
# TODO: soft paste
# TODO: LAB color match


def _load_kontext_stack(rt: NodeRuntime, cfg: dict) -> dict:
    """Load Kontext UNET + DualCLIP (clip_l + t5) + ae VAE. No LoRAs yet."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    unet_name = cfg["unet_name"]
    clip_name1 = cfg["clip_name1"]
    clip_name2 = cfg["clip_name2"]
    vae_name = cfg["vae_name"]
    key = f"kontext::{unet_name}::{clip_name1}::{clip_name2}::{vae_name}"
    if key in rt.models:
        return rt.models[key]

    load_meta: dict = {
        "checkpoint": unet_name,
        "checkpoint_preferred": unet_name,
        "checkpoint_fallback_used": False,
        "loras_loaded": [],
        "lora_strengths": {},
        "fallbacks": [],
        "clip_name1": clip_name1,
        "clip_name2": clip_name2,
        "vae_name": vae_name,
    }

    unet = rt.call("UNETLoader", unet_name=unet_name, weight_dtype="default")
    model = get_value_at_index(unet, 0)

    # DualCLIPLoader is the native Flux / Kontext text-encoder path.
    if not rt.has("DualCLIPLoader"):
        raise KeyError("DualCLIPLoader node missing — update ComfyUI for FLUX Kontext")
    clip = rt.call(
        "DualCLIPLoader",
        clip_name1=clip_name1,
        clip_name2=clip_name2,
        type=cfg.get("clip_type", "flux"),
    )
    vae = rt.call("VAELoader", vae_name=vae_name)

    # TODO: Place It / Put It Here LoRA via LoraLoaderModelOnly

    bundle = {
        "model": model,
        "clip": get_value_at_index(clip, 0),
        "vae": get_value_at_index(vae, 0),
        "load_meta": load_meta,
    }
    rt.models[key] = bundle
    return bundle


def _sample_kontext(
    rt: NodeRuntime,
    bundle: dict,
    body_t,
    face_t,
    cfg: dict,
    prompt: str,
) -> tuple[Image.Image, dict]:
    """Minimal Kontext edit: encode body (+ face), text prompt, sample, decode."""
    import torch

    fallbacks: list[str] = []
    flux_kontext_image_scale_applied = False
    reference_latent_used = False
    flux_guidance_applied = False

    with torch.no_grad():
        image1 = body_t
        input_h, input_w = int(body_t.shape[1]), int(body_t.shape[2])

        # Prefer Kontext-native scale when available (official workflow).
        if bool(cfg.get("flux_kontext_image_scale", True)) and rt.has("FluxKontextImageScale"):
            scaled = rt.call("FluxKontextImageScale", image=body_t)
            image1 = get_value_at_index(scaled, 0)
            flux_kontext_image_scale_applied = True
        elif bool(cfg.get("flux_kontext_image_scale", True)):
            fallbacks.append("flux_kontext_image_scale_missing")

        encode_h, encode_w = int(image1.shape[1]), int(image1.shape[2])

        body_lat = rt.call("VAEEncode", pixels=image1, vae=bundle["vae"])
        body_latent = get_value_at_index(body_lat, 0)

        # Face ref is encoded so it can be attached once ReferenceLatent is wired.
        # Without alignment / Place It LoRA this is a raw second reference only.
        face_lat = rt.call("VAEEncode", pixels=face_t, vae=bundle["vae"])
        face_latent = get_value_at_index(face_lat, 0)

        pos = rt.call("CLIPTextEncode", text=prompt, clip=bundle["clip"])
        positive = get_value_at_index(pos, 0)
        neg_text = str(cfg.get("negative_prompt", "") or "")
        neg = rt.call("CLIPTextEncode", text=neg_text, clip=bundle["clip"])
        negative = get_value_at_index(neg, 0)

        # Minimal ReferenceLatent so Kontext receives the body (and face) image.
        # This is the smallest working edit path; later quality stages refine it.
        #
        # TODO: InsightFace alignment before encode
        # TODO: ReferenceLatent refinements / FluxKontextMultiReferenceLatentMethod
        # TODO: ConditioningZeroOut on text conditioning
        if rt.has("ReferenceLatent"):
            pos_ref = rt.call(
                "ReferenceLatent",
                conditioning=positive,
                latent=body_latent,
            )
            pos_ref = rt.call(
                "ReferenceLatent",
                conditioning=get_value_at_index(pos_ref, 0),
                latent=face_latent,
            )
            positive = get_value_at_index(pos_ref, 0)

            neg_ref = rt.call(
                "ReferenceLatent",
                conditioning=negative,
                latent=body_latent,
            )
            neg_ref = rt.call(
                "ReferenceLatent",
                conditioning=get_value_at_index(neg_ref, 0),
                latent=face_latent,
            )
            negative = get_value_at_index(neg_ref, 0)
            reference_latent_used = True
        else:
            fallbacks.append("reference_latent_missing")

        # TODO: ConditioningZeroOut

        guidance = float(cfg.get("flux_guidance", cfg.get("cfg", 2.5)))
        if rt.has("FluxGuidance"):
            positive = get_value_at_index(
                rt.call("FluxGuidance", conditioning=positive, guidance=guidance),
                0,
            )
            flux_guidance_applied = True
        else:
            fallbacks.append("flux_guidance_missing")

        # Latent canvas: prefer empty SD3/Flux latent at encode size; else body latent.
        latent_image = body_latent
        if rt.has("EmptySD3LatentImage"):
            empty = rt.call(
                "EmptySD3LatentImage",
                width=encode_w,
                height=encode_h,
                batch_size=1,
            )
            latent_image = get_value_at_index(empty, 0)
        else:
            fallbacks.append("empty_sd3_latent_missing_used_body_latent")

        noise = get_value_at_index(
            rt.call("RandomNoise", noise_seed=int(cfg.get("seed", 46))), 0
        )
        guider = get_value_at_index(
            rt.call(
                "CFGGuider",
                model=bundle["model"],
                positive=positive,
                negative=negative,
                cfg=float(cfg.get("cfg", 1.0)),
            ),
            0,
        )
        sampler = get_value_at_index(
            rt.call("KSamplerSelect", sampler_name=cfg.get("sampler", "euler")),
            0,
        )
        sigmas = get_value_at_index(
            rt.call(
                "BasicScheduler",
                model=bundle["model"],
                scheduler=cfg.get("scheduler", "simple"),
                steps=int(cfg.get("steps", 20)),
                denoise=float(cfg.get("denoise", 1.0)),
            ),
            0,
        )

        with force_sampling_full_load(models=(bundle["model"],)):
            samples = rt.call(
                "SamplerCustomAdvanced",
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent_image,
            )
        del guider

        decoded = rt.call(
            "VAEDecode",
            samples=get_value_at_index(samples, 0),
            vae=bundle["vae"],
        )
        image = comfy_tensor_to_pil(get_value_at_index(decoded, 0))

    sample_meta = {
        "reference_latent_used": reference_latent_used,
        "flux_guidance_applied": flux_guidance_applied,
        "flux_kontext_image_scale_applied": flux_kontext_image_scale_applied,
        "flux_kontext_image_scale_enabled": bool(cfg.get("flux_kontext_image_scale", True)),
        "input_body_size": [input_w, input_h],
        "encode_body_size": [encode_w, encode_h],
        "encode_megapixels": round((encode_w * encode_h) / 1_000_000, 3),
        "fallbacks": fallbacks,
    }
    return image, sample_meta


class FluxKontextPipeline(BasePipeline):
    """First working FLUX Kontext integration (no LoRA / no align / no stitch)."""

    name = "flux_kontext"

    def _ensure_runtime(self) -> NodeRuntime:
        if self.runtime is None:
            self.runtime = NodeRuntime()
        return self.runtime

    def run(
        self, body: Image.Image, face: Image.Image, out_dir: Path | None = None
    ) -> PipelineResult:
        t0 = time.perf_counter()
        run_error: BaseException | None = None
        out: Image.Image | None = None
        sample_meta: dict = {}
        body_pil: Image.Image | None = None
        face_crop: Image.Image | None = None
        face_for_model: Image.Image | None = None
        dbg: dict[str, str] = {}
        load_meta: dict = {}
        fallbacks: list[str] = []
        prompt = str(self.cfg.get("prompt", "")).strip()

        try:
            import torch

            rt = self._ensure_runtime()
            bundle = _load_kontext_stack(rt, self.cfg)
            div_by = int(self.cfg.get("div_by", 8))
            max_dim = int(self.cfg.get("max_dim", 576))

            # Preprocessing identical to qwen_baseline (full-frame, blur-pad face).
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
                face_for_model = pad_to_ar_blur(
                    face_crop, body_pil.width / body_pil.height
                ).resize(body_pil.size, Image.Resampling.LANCZOS)
            else:
                face_for_model = face_crop.resize(
                    body_pil.size, Image.Resampling.LANCZOS
                )

            body_t = pil_to_comfy_tensor(body_pil, torch)
            face_t = pil_to_comfy_tensor(face_for_model, torch)

            out, sample_meta = _sample_kontext(
                rt, bundle, body_t, face_t, self.cfg, prompt
            )

            # TODO: soft paste
            # TODO: LAB color match

            load_meta = dict(bundle.get("load_meta") or {})
            fallbacks = list(load_meta.get("fallbacks") or []) + list(
                sample_meta.get("fallbacks") or []
            )
            dbg = {
                k: v
                for k, v in {
                    "debug_body": self._save_debug(out_dir, "debug_body.png", body_pil),
                    "debug_face_crop": self._save_debug(
                        out_dir, "debug_face_crop.png", face_crop
                    ),
                    "debug_face_for_model": self._save_debug(
                        out_dir, "debug_face_for_model.png", face_for_model
                    ),
                }.items()
                if v
            }
        except BaseException as exc:
            run_error = exc

        latency_s = time.perf_counter() - t0
        crop_size = list(body_pil.size) if body_pil is not None else None
        meta = {
            "pipeline": self.name,
            "checkpoint": load_meta.get("checkpoint"),
            "loras_loaded": list(load_meta.get("loras_loaded") or []),
            "lora_strengths": dict(load_meta.get("lora_strengths") or {}),
            "prompt": prompt,
            "crop_size": crop_size,
            "body_size": crop_size,
            "reference_latent_used": bool(sample_meta.get("reference_latent_used")),
            "flux_guidance_applied": bool(sample_meta.get("flux_guidance_applied")),
            "flux_kontext_image_scale_applied": bool(
                sample_meta.get("flux_kontext_image_scale_applied")
            ),
            "flux_kontext_image_scale_enabled": bool(
                sample_meta.get("flux_kontext_image_scale_enabled")
            ),
            "input_body_size": sample_meta.get("input_body_size"),
            "encode_body_size": sample_meta.get("encode_body_size"),
            "encode_megapixels": sample_meta.get("encode_megapixels"),
            "fallbacks": fallbacks,
            "latency_s": round(latency_s, 4),
        }
        if run_error is not None:
            meta["run_error"] = str(run_error)
            meta["run_error_type"] = type(run_error).__name__

        print(
            f"[flux_kontext] checkpoint={meta.get('checkpoint')} "
            f"loras={meta.get('loras_loaded')} crop={meta.get('crop_size')} "
            f"reference_latent={meta.get('reference_latent_used')} "
            f"image_scale={meta.get('flux_kontext_image_scale_applied')} "
            f"fallbacks={fallbacks or 'none'}"
        )

        if run_error is not None:
            raise PipelineRunError(
                str(run_error),
                meta=meta,
                latency_s=latency_s,
                image=out,
                debug_paths=dbg,
            ) from run_error
        if out is None or body_pil is None:
            raise PipelineRunError(
                "pipeline finished without output image",
                meta=meta,
                latency_s=latency_s,
                debug_paths=dbg,
            )

        return PipelineResult(
            image=out,
            latency_s=latency_s,
            meta=meta,
            debug_paths=dbg,
        )
