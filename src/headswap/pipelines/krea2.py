"""Krea 2 Identity Edit — two-image head/face swap via community LoRA.

Uses ComfyUI native Krea 2 loaders + comfyui-krea2edit nodes:
  Krea2EditModelPatch + Krea2EditGroundedEncode

Input order (LoRA training):
  image 1 / source = body scene
  image 2 / source_b = identity person (face)
"""
from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from headswap.comfy.full_load import force_sampling_full_load
from headswap.comfy.runtime import (
    NodeRuntime,
    comfy_tensor_to_pil,
    comfyui_path,
    get_value_at_index,
    pil_to_comfy_tensor,
    resolve_model_file,
)
from headswap.pipelines.base import BasePipeline, PipelineResult
from headswap.pipelines.errors import PipelineRunError
from headswap.preprocess import crop_face_reference, resize_max_keep_ar


class Krea2IdentityEditPipeline(BasePipeline):
    name = "krea2_identity_edit"

    def _ensure_runtime(self) -> NodeRuntime:
        if self.runtime is None:
            # Identity Edit nodes live in custom_nodes/comfyui-krea2edit.
            self.runtime = NodeRuntime(init_custom_nodes=True)
        return self.runtime

    def _load_models(self, rt: NodeRuntime) -> dict:
        import torch  # noqa: F401 — ensure CUDA ctx exists before Comfy loaders

        models = Path(comfyui_path()) / "models"
        unet_name = resolve_model_file(
            models / "diffusion_models",
            self.cfg.get("unet_name", "krea2_turbo_fp8_scaled.safetensors"),
        )
        clip_name = resolve_model_file(
            models / "text_encoders",
            self.cfg.get("clip_name", "qwen3vl_4b_fp8_scaled.safetensors"),
        )
        vae_name = resolve_model_file(
            models / "vae",
            self.cfg.get("vae_name", "qwen_image_vae.safetensors"),
        )
        lora_name = self.cfg.get(
            "identity_lora_name", "krea2_identity_edit_v1_2_r64.safetensors"
        )
        lora_strength = float(self.cfg.get("identity_lora_strength", 1.0) or 0.0)
        key = f"krea2::{unet_name}::{clip_name}::{vae_name}::{lora_name}::{lora_strength}"
        if key in rt.models:
            return rt.models[key]

        if not rt.has("Krea2EditModelPatch") or not rt.has("Krea2EditGroundedEncode"):
            raise PipelineRunError(
                "comfyui-krea2edit nodes missing. "
                "Run: bash scripts/setup_krea2_nodes.sh "
                "(or setup_kaggle.sh --krea2) and ensure HEADSWAP_INIT_CUSTOM_NODES=1."
            )

        unet = rt.call("UNETLoader", unet_name=unet_name, weight_dtype="default")
        model = get_value_at_index(unet, 0)

        loras_loaded: list[str] = []
        lora_strengths: dict[str, float] = {}
        if lora_strength > 0 and lora_name:
            resolved = resolve_model_file(
                models / "loras", lora_name, fallbacks=[lora_name]
            )
            if not (models / "loras" / resolved).exists():
                raise FileNotFoundError(
                    f"Krea2 identity LoRA not found at {models / 'loras' / resolved}. "
                    "Run: python scripts/download_krea2.py"
                )
            model = get_value_at_index(
                rt.call(
                    "LoraLoaderModelOnly",
                    model=model,
                    lora_name=resolved,
                    strength_model=lora_strength,
                ),
                0,
            )
            loras_loaded.append(resolved)
            lora_strengths[resolved] = lora_strength
            print(f"[krea2] loaded LoRA {resolved} strength={lora_strength}")

        clip = rt.call(
            "CLIPLoader",
            clip_name=clip_name,
            type=str(self.cfg.get("clip_type", "krea2")),
            device="default",
        )
        vae = rt.call("VAELoader", vae_name=vae_name)

        bundle = {
            "model": model,
            "clip": get_value_at_index(clip, 0),
            "vae": get_value_at_index(vae, 0),
            "load_meta": {
                "checkpoint": unet_name,
                "clip": clip_name,
                "vae": vae_name,
                "loras_loaded": loras_loaded,
                "lora_strengths": lora_strengths,
            },
        }
        rt.models[key] = bundle
        return bundle

    def run(
        self, body: Image.Image, face: Image.Image, out_dir: Path | None = None
    ) -> PipelineResult:
        import torch

        t0 = time.perf_counter()
        rt = self._ensure_runtime()
        bundle = self._load_models(rt)
        div_by = int(self.cfg.get("div_by", 16))
        max_dim = int(self.cfg.get("max_dim", 768))

        # Scene = body (image 1). Person = face crop (image 2).
        scene = resize_max_keep_ar(body.convert("RGB"), max_dim, div_by=div_by)
        face_crop = crop_face_reference(
            face,
            self.cache_dir,
            top=float(self.cfg.get("face_top_pad", 0.55)),
            bot=float(self.cfg.get("face_bot_pad", 0.20)),
            side=float(self.cfg.get("face_side_pad", 0.28)),
            include_shoulders=bool(self.cfg.get("include_shoulders", False)),
        )
        person = resize_max_keep_ar(face_crop.convert("RGB"), max_dim, div_by=div_by)

        w, h = scene.size
        prompt = str(self.cfg.get("prompt", "") or "").strip()
        neg = str(self.cfg.get("negative_prompt", "") or "")
        grounding_px = int(self.cfg.get("grounding_px", 768))
        ref_boost = float(self.cfg.get("ref_boost", 4.0))
        ref_boost_a = float(self.cfg.get("ref_boost_a", 1.0))
        fit_mode = str(self.cfg.get("fit_mode", "fit") or "fit")
        steps = int(self.cfg.get("steps", 10))
        cfg = float(self.cfg.get("cfg", 1.0))
        seed = int(self.cfg.get("seed", 46))
        denoise = float(self.cfg.get("denoise", 1.0))

        with torch.no_grad():
            scene_t = pil_to_comfy_tensor(scene, torch)
            person_t = pil_to_comfy_tensor(person, torch)

            scene_lat = rt.call("VAEEncode", pixels=scene_t, vae=bundle["vae"])
            person_lat = rt.call("VAEEncode", pixels=person_t, vae=bundle["vae"])

            patched = rt.call(
                "Krea2EditModelPatch",
                model=bundle["model"],
                source_latent=get_value_at_index(scene_lat, 0),
                source_latent_b=get_value_at_index(person_lat, 0),
                ref_boost=ref_boost,
                ref_boost_a=ref_boost_a,
                fit_mode=fit_mode,
                vae=bundle["vae"],
                source_image=scene_t,
                source_image_b=person_t,
            )
            model = get_value_at_index(patched, 0)

            pos = rt.call(
                "Krea2EditGroundedEncode",
                clip=bundle["clip"],
                prompt=prompt,
                image=scene_t,
                image_b=person_t,
                grounding_px=grounding_px,
            )
            positive = get_value_at_index(pos, 0)

            # At CFG>1 ground the negative too; at CFG=1 still provide empty grounded neg.
            neg_enc = rt.call(
                "Krea2EditGroundedEncode",
                clip=bundle["clip"],
                prompt=neg,
                image=scene_t,
                image_b=person_t,
                grounding_px=grounding_px,
            )
            negative = get_value_at_index(neg_enc, 0)

            if rt.has("EmptySD3LatentImage"):
                empty = rt.call(
                    "EmptySD3LatentImage", width=w, height=h, batch_size=1
                )
                latent = get_value_at_index(empty, 0)
                empty_node = "EmptySD3LatentImage"
            else:
                latent = get_value_at_index(scene_lat, 0)
                empty_node = "scene_latent_fallback"

            use_full_load = bool(self.cfg.get("force_full_load", False))
            with force_sampling_full_load(models=(model,), enabled=use_full_load):
                samples = rt.call(
                    "KSampler",
                    model=model,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=str(self.cfg.get("sampler", "euler")),
                    scheduler=str(self.cfg.get("scheduler", "simple")),
                    positive=positive,
                    negative=negative,
                    latent_image=latent,
                    denoise=denoise,
                )
            decoded = rt.call(
                "VAEDecode",
                samples=get_value_at_index(samples, 0),
                vae=bundle["vae"],
            )
            out = comfy_tensor_to_pil(get_value_at_index(decoded, 0))

        dbg = {}
        if out_dir is not None:
            dbg = {
                k: v
                for k, v in {
                    "debug_scene": self._save_debug(out_dir, "debug_scene.png", scene),
                    "debug_person": self._save_debug(out_dir, "debug_person.png", person),
                    "debug_face_crop": self._save_debug(
                        out_dir, "debug_face_crop.png", face_crop
                    ),
                }.items()
                if v
            }

        meta = {
            "pipeline": self.name,
            "checkpoint": bundle["load_meta"].get("checkpoint"),
            "loras_loaded": list(bundle["load_meta"].get("loras_loaded") or []),
            "lora_strengths": dict(bundle["load_meta"].get("lora_strengths") or {}),
            "prompt": prompt,
            "scene_size": list(scene.size),
            "person_size": list(person.size),
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "ref_boost": ref_boost,
            "ref_boost_a": ref_boost_a,
            "grounding_px": grounding_px,
            "fit_mode": fit_mode,
            "empty_latent_node": empty_node,
            "force_full_load": use_full_load,
        }
        print(
            f"[krea2] checkpoint={meta['checkpoint']} loras={meta['loras_loaded']} "
            f"scene={meta['scene_size']} person={meta['person_size']} "
            f"steps={steps} cfg={cfg} ref_boost={ref_boost}"
        )
        return PipelineResult(
            image=out,
            latency_s=time.perf_counter() - t0,
            meta=meta,
            debug_paths=dbg,
        )
