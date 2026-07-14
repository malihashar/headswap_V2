from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from headswap.comfy.runtime import (
    NodeRuntime,
    comfy_tensor_to_pil,
    get_value_at_index,
    pil_to_comfy_tensor,
    resolve_model_file,
)
from headswap.pipelines.base import BasePipeline, PipelineResult, build_prompt
from headswap.preprocess import (
    crop_face_reference,
    crop_with_mask,
    head_hair_mask_from_face,
    lab_histogram_match_face,
    resize_long_side,
    resize_max_keep_ar,
    soft_composite,
)


class KleinMaskCropPipeline(BasePipeline):
    """
    FLUX.2 [klein] 4B distilled multi-reference head swap with:
    face prep → head/hair mask → crop → ReferenceLatent(body_crop, face) → stitch.
    """

    name = "klein4b_mask_crop_stitch"

    def _ensure_runtime(self) -> NodeRuntime:
        if self.runtime is None:
            self.runtime = NodeRuntime()
        return self.runtime

    def _load_models(self, rt: NodeRuntime):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        from headswap.comfy.runtime import comfyui_path

        models = Path(comfyui_path()) / "models"
        unet_dir = models / "diffusion_models"
        unet_name = resolve_model_file(
            unet_dir,
            self.cfg.get("unet_name", "flux-2-klein-4b-fp8.safetensors"),
            fallbacks=[self.cfg.get("unet_name_fallback", "flux-2-klein-4b.safetensors")],
        )
        key = f"klein::{unet_name}::{self.cfg.get('bfs_lora_name')}::{self.cfg.get('bfs_lora_strength')}"
        if key in rt.models:
            return rt.models[key]

        unet = rt.call("UNETLoader", unet_name=unet_name, weight_dtype="default")
        model = get_value_at_index(unet, 0)

        strength = float(self.cfg.get("bfs_lora_strength", 0.0) or 0.0)
        lora_name = self.cfg.get("bfs_lora_name")
        if strength > 0 and lora_name:
            loras = models / "loras"
            resolved = resolve_model_file(loras, lora_name, fallbacks=[lora_name])
            model = get_value_at_index(
                rt.call(
                    "LoraLoaderModelOnly",
                    model=model,
                    lora_name=resolved,
                    strength_model=strength,
                ),
                0,
            )

        clip = rt.call(
            "CLIPLoader",
            clip_name=self.cfg.get("clip_name", "qwen_3_4b.safetensors"),
            type=self.cfg.get("clip_type", "flux2"),
            device="default",
        )
        vae = rt.call("VAELoader", vae_name=self.cfg.get("vae_name", "flux2-vae.safetensors"))

        bundle = {"model": model, "clip": get_value_at_index(clip, 0), "vae": get_value_at_index(vae, 0)}
        rt.models[key] = bundle
        return bundle

    def run(self, body: Image.Image, face: Image.Image, out_dir: Path | None = None) -> PipelineResult:
        import torch

        t0 = time.perf_counter()
        rt = self._ensure_runtime()
        bundle = self._load_models(rt)
        div_by = int(self.cfg.get("div_by", 16))

        body_full = resize_max_keep_ar(
            body.convert("RGB"), int(self.cfg.get("max_body_dim", 1280)), div_by=div_by
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
        crop_img, crop_mask, box = crop_with_mask(body_full, mask, pad=12, div_by=div_by)
        crop_work = resize_long_side(crop_img, int(self.cfg.get("crop_long_side", 1024)), div_by=div_by)
        face_ref = resize_long_side(face_crop, int(self.cfg.get("crop_long_side", 1024)), div_by=div_by)

        prompt = build_prompt(self.cfg, body_full, face_crop, self.cache_dir)
        neg = str(self.cfg.get("negative_prompt", "") or "")

        with torch.inference_mode():
            body_t = pil_to_comfy_tensor(crop_work, torch)
            face_t = pil_to_comfy_tensor(face_ref, torch)

            body_lat = rt.call("VAEEncode", pixels=body_t, vae=bundle["vae"])
            face_lat = rt.call("VAEEncode", pixels=face_t, vae=bundle["vae"])

            pos = rt.call("CLIPTextEncode", text=prompt, clip=bundle["clip"])
            conditioning = get_value_at_index(pos, 0)

            # Official Klein multi-ref: chain ReferenceLatent(body) → ReferenceLatent(face)
            if rt.has("ReferenceLatent"):
                ref1 = rt.call(
                    "ReferenceLatent",
                    conditioning=conditioning,
                    latent=get_value_at_index(body_lat, 0),
                )
                ref2 = rt.call(
                    "ReferenceLatent",
                    conditioning=get_value_at_index(ref1, 0),
                    latent=get_value_at_index(face_lat, 0),
                )
                positive = get_value_at_index(ref2, 0)
            else:
                positive = conditioning

            if neg.strip() and rt.has("CLIPTextEncode"):
                neg_c = rt.call("CLIPTextEncode", text=neg, clip=bundle["clip"])
                negative = get_value_at_index(neg_c, 0)
            elif rt.has("ConditioningZeroOut"):
                negative = get_value_at_index(
                    rt.call("ConditioningZeroOut", conditioning=positive), 0
                )
            else:
                negative = positive

            w, h = crop_work.size
            if rt.has("EmptyFlux2LatentImage"):
                empty = rt.call("EmptyFlux2LatentImage", width=w, height=h, batch_size=1)
                latent_image = get_value_at_index(empty, 0)
            else:
                latent_image = get_value_at_index(body_lat, 0)

            steps = int(self.cfg.get("steps", 4))
            if rt.has("Flux2Scheduler"):
                sigmas = get_value_at_index(
                    rt.call("Flux2Scheduler", steps=steps, width=w, height=h), 0
                )
            else:
                sigmas = get_value_at_index(
                    rt.call(
                        "BasicScheduler",
                        model=bundle["model"],
                        scheduler=self.cfg.get("scheduler", "simple"),
                        steps=steps,
                        denoise=float(self.cfg.get("denoise", 1.0)),
                    ),
                    0,
                )

            sampler = get_value_at_index(
                rt.call("KSamplerSelect", sampler_name=self.cfg.get("sampler", "euler")), 0
            )
            noise = get_value_at_index(
                rt.call("RandomNoise", noise_seed=int(self.cfg.get("seed", 46))), 0
            )
            guider = get_value_at_index(
                rt.call(
                    "CFGGuider",
                    model=bundle["model"],
                    positive=positive,
                    negative=negative,
                    cfg=float(self.cfg.get("cfg", 1.0)),
                ),
                0,
            )
            samples = rt.call(
                "SamplerCustomAdvanced",
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent_image,
            )
            decoded = rt.call(
                "VAEDecode",
                samples=get_value_at_index(samples, 0),
                vae=bundle["vae"],
            )
            edited_crop = comfy_tensor_to_pil(get_value_at_index(decoded, 0))

        stitched = soft_composite(body_full, edited_crop, mask, box)
        stitched = lab_histogram_match_face(stitched, body_full, mask, strength=0.3)

        dbg = {
            k: v
            for k, v in {
                "debug_body": self._save_debug(out_dir, "debug_body.png", body_full),
                "debug_face_crop": self._save_debug(out_dir, "debug_face_crop.png", face_crop),
                "debug_mask": self._save_debug(out_dir, "debug_mask.png", mask),
                "debug_crop": self._save_debug(out_dir, "debug_crop.png", crop_work),
                "debug_edited_crop": self._save_debug(out_dir, "debug_edited_crop.png", edited_crop),
            }.items()
            if v
        }
        return PipelineResult(
            image=stitched,
            latency_s=time.perf_counter() - t0,
            meta={
                "pipeline": self.name,
                "prompt": prompt,
                "crop_size": list(crop_work.size),
                "body_size": list(body_full.size),
                "bfs_lora_strength": float(self.cfg.get("bfs_lora_strength", 0) or 0),
            },
            debug_paths=dbg,
        )
