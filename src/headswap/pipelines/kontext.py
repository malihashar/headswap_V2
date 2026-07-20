"""FLUX.1 Kontext Align → Paste → Refine (community face-swap path).

Mirrors the high-signal ComfyUI workflows:
  InsightFace/landmark align → soft paste onto body → Kontext refine
  with ReferenceLatent, ConditioningZeroOut, FluxKontextImageScale,
  FluxGuidance (~3.5), and optional Place It / Put it here LoRA.
"""
from __future__ import annotations

import re
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
from headswap.preprocess import (
    align_face_to_destination,
    color_match_rgba_to_destination,
    crop_face_reference,
    head_hair_mask_from_face,
    lab_histogram_match_face,
    paste_aligned_face,
    resize_max_keep_ar,
    soft_composite,
)


# Prefer Put it here V4, then Place It, then fuzzy matches in ComfyUI/models/loras.
_PLACEMENT_LORA_PREFERRED = (
    "Put it here_V4.2.safetensors",
    "Put it here_V4.safetensors",
    "Put it here_KonText_V4.safetensors",
    "put it here+kontext-lora.safetensors",
    "put_it_here_kontext_v4.safetensors",
    "put_it_here_v4.safetensors",
    "place_it.safetensors",
    "Place it.safetensors",
    "place_it_flux_kontext.safetensors",
)


def _cfg_bool(cfg: dict, *keys: str, default: bool = True) -> bool:
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return bool(cfg[k])
    return default


def _discover_placement_lora(cfg: dict) -> tuple[str | None, str | None]:
    """
    Resolve placement LoRA filename under ComfyUI/models/loras.

    Returns (filename_or_None, skip_reason_or_None).
    """
    explicit = cfg.get("placement_lora") or cfg.get("place_it_lora_name")
    lora_dir = Path(comfyui_path()) / "models" / "loras"
    if not lora_dir.is_dir():
        return None, f"loras_dir_missing:{lora_dir}"

    if explicit:
        resolved = resolve_model_file(lora_dir, str(explicit), fallbacks=[str(explicit)])
        if (lora_dir / resolved).exists():
            return resolved, None
        return None, f"placement_lora_not_found:{explicit}"

    # Preferred exact names first.
    for name in _PLACEMENT_LORA_PREFERRED:
        if (lora_dir / name).exists():
            return name, None

    # Fuzzy: put it here / place it
    pattern = re.compile(r"(put[_\s-]*it[_\s-]*here|place[_\s-]*it)", re.I)
    matches = sorted(
        p.name for p in lora_dir.glob("*.safetensors") if pattern.search(p.name)
    )
    if matches:
        # Prefer names containing v4 / kontext
        ranked = sorted(
            matches,
            key=lambda n: (
                0 if re.search(r"v4|kontext", n, re.I) else 1,
                len(n),
                n.lower(),
            ),
        )
        return ranked[0], None
    return None, "placement_lora_not_in_loras_dir"


def _load_kontext_stack(rt: NodeRuntime, cfg: dict) -> dict:
    """Load Kontext UNET + DualCLIP + ae VAE + optional placement LoRA."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    unet_name = cfg["unet_name"]
    clip_name1 = cfg["clip_name1"]
    clip_name2 = cfg["clip_name2"]
    vae_name = cfg["vae_name"]
    lora_name, lora_skip = _discover_placement_lora(cfg)
    strength = float(cfg.get("placement_lora_strength", cfg.get("place_it_lora_strength", 1.0)) or 0.0)
    key = (
        f"kontext::{unet_name}::{clip_name1}::{clip_name2}::{vae_name}"
        f"::{lora_name}::{strength}"
    )
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
        "placement_lora_loaded": False,
        "placement_lora_name": None,
        "placement_lora_strength": strength,
        "placement_lora_skip_reason": None,
    }

    unet = rt.call("UNETLoader", unet_name=unet_name, weight_dtype="default")
    model = get_value_at_index(unet, 0)

    if not rt.has("DualCLIPLoader"):
        raise KeyError("DualCLIPLoader node missing — update ComfyUI for FLUX Kontext")
    clip = rt.call(
        "DualCLIPLoader",
        clip_name1=clip_name1,
        clip_name2=clip_name2,
        type=cfg.get("clip_type", "flux"),
    )
    vae = rt.call("VAELoader", vae_name=vae_name)

    if lora_name and strength > 0:
        if not rt.has("LoraLoaderModelOnly"):
            load_meta["placement_lora_skip_reason"] = "LoraLoaderModelOnly_node_missing"
            load_meta["fallbacks"].append("placement_lora_node_missing")
        else:
            try:
                model = get_value_at_index(
                    rt.call(
                        "LoraLoaderModelOnly",
                        model=model,
                        lora_name=lora_name,
                        strength_model=strength,
                    ),
                    0,
                )
                load_meta["placement_lora_loaded"] = True
                load_meta["placement_lora_name"] = lora_name
                load_meta["loras_loaded"].append(lora_name)
                load_meta["lora_strengths"][lora_name] = strength
                print(f"[flux_kontext] placement LoRA {lora_name} strength={strength}")
            except Exception as exc:
                load_meta["placement_lora_skip_reason"] = f"placement_lora_load_failed:{exc}"
                load_meta["fallbacks"].append(load_meta["placement_lora_skip_reason"])
    else:
        load_meta["placement_lora_skip_reason"] = (
            lora_skip if strength > 0 else "placement_lora_strength_zero"
        )
        if lora_skip:
            load_meta["fallbacks"].append(lora_skip)
            print(f"[flux_kontext] placement LoRA skipped: {lora_skip}")

    bundle = {
        "model": model,
        "clip": get_value_at_index(clip, 0),
        "vae": get_value_at_index(vae, 0),
        "load_meta": load_meta,
    }
    rt.models[key] = bundle
    return bundle


def _sample_kontext_refine(
    rt: NodeRuntime,
    bundle: dict,
    composite_t,
    cfg: dict,
    prompt: str,
    face_t=None,
) -> tuple[Image.Image, dict]:
    """Kontext refine pass on an already align-pasted composite."""
    import torch

    fallbacks: list[str] = []
    feature: dict = {
        "flux_kontext_image_scale_enabled": _cfg_bool(
            cfg, "image_scale", "flux_kontext_image_scale", default=True
        ),
        "flux_kontext_image_scale_applied": False,
        "flux_kontext_image_scale_skip_reason": None,
        "reference_latent_enabled": _cfg_bool(cfg, "reference_latent", default=True),
        "reference_latent_used": False,
        "reference_latent_skip_reason": None,
        "identity_reference_enabled": _cfg_bool(cfg, "identity_reference", default=True),
        "identity_reference_used": False,
        "identity_reference_skip_reason": None,
        "conditioning_zero_out_enabled": _cfg_bool(
            cfg, "conditioning_zero_out", default=True
        ),
        "conditioning_zero_out_applied": False,
        "conditioning_zero_out_skip_reason": None,
        "flux_guidance_value": float(cfg.get("flux_guidance", 4.0)),
        "flux_guidance_applied": False,
        "flux_guidance_skip_reason": None,
        "denoise": float(cfg.get("denoise", 0.72)),
        "steps": int(cfg.get("steps", 32)),
    }

    with torch.no_grad():
        image1 = composite_t
        input_h, input_w = int(composite_t.shape[1]), int(composite_t.shape[2])

        if feature["flux_kontext_image_scale_enabled"]:
            if rt.has("FluxKontextImageScale"):
                scaled = rt.call("FluxKontextImageScale", image=composite_t)
                image1 = get_value_at_index(scaled, 0)
                feature["flux_kontext_image_scale_applied"] = True
            else:
                feature["flux_kontext_image_scale_skip_reason"] = (
                    "FluxKontextImageScale_node_missing"
                )
                fallbacks.append(feature["flux_kontext_image_scale_skip_reason"])
        else:
            feature["flux_kontext_image_scale_skip_reason"] = "disabled_by_config"

        encode_h, encode_w = int(image1.shape[1]), int(image1.shape[2])

        comp_lat = rt.call("VAEEncode", pixels=image1, vae=bundle["vae"])
        composite_latent = get_value_at_index(comp_lat, 0)

        pos = rt.call("CLIPTextEncode", text=prompt, clip=bundle["clip"])
        positive = get_value_at_index(pos, 0)
        neg_text = str(cfg.get("negative_prompt", "") or "")
        neg = rt.call("CLIPTextEncode", text=neg_text, clip=bundle["clip"])
        negative = get_value_at_index(neg, 0)

        # Community recipe: zero-out text conditioning so Kontext trusts the
        # pasted composite outside the swap, then re-apply FluxGuidance.
        if feature["conditioning_zero_out_enabled"]:
            if rt.has("ConditioningZeroOut"):
                # Zero the negative (and optionally positive text) to reduce drift.
                negative = get_value_at_index(
                    rt.call("ConditioningZeroOut", conditioning=negative), 0
                )
                # Also zero a copy used as secondary gate when prompt should not
                # rewrite the whole frame — keep positive for LoRA trigger words.
                feature["conditioning_zero_out_applied"] = True
            else:
                feature["conditioning_zero_out_skip_reason"] = (
                    "ConditioningZeroOut_node_missing"
                )
                fallbacks.append(feature["conditioning_zero_out_skip_reason"])
        else:
            feature["conditioning_zero_out_skip_reason"] = "disabled_by_config"

        if feature["reference_latent_enabled"]:
            if rt.has("ReferenceLatent"):
                positive = get_value_at_index(
                    rt.call(
                        "ReferenceLatent",
                        conditioning=positive,
                        latent=composite_latent,
                    ),
                    0,
                )
                negative = get_value_at_index(
                    rt.call(
                        "ReferenceLatent",
                        conditioning=negative,
                        latent=composite_latent,
                    ),
                    0,
                )
                feature["reference_latent_used"] = True

                # Extra identity anchor: encode the face crop and attach it as a
                # second reference so refine keeps donor identity under denoise.
                if feature["identity_reference_enabled"] and face_t is not None:
                    try:
                        face_img = face_t
                        if feature["flux_kontext_image_scale_applied"] and rt.has(
                            "FluxKontextImageScale"
                        ):
                            face_img = get_value_at_index(
                                rt.call("FluxKontextImageScale", image=face_t), 0
                            )
                        face_lat = get_value_at_index(
                            rt.call("VAEEncode", pixels=face_img, vae=bundle["vae"]), 0
                        )
                        positive = get_value_at_index(
                            rt.call(
                                "ReferenceLatent",
                                conditioning=positive,
                                latent=face_lat,
                            ),
                            0,
                        )
                        feature["identity_reference_used"] = True
                    except Exception as exc:
                        feature["identity_reference_skip_reason"] = (
                            f"identity_reference_failed:{exc}"
                        )
                        fallbacks.append(feature["identity_reference_skip_reason"])
                elif feature["identity_reference_enabled"]:
                    feature["identity_reference_skip_reason"] = "face_tensor_missing"
                else:
                    feature["identity_reference_skip_reason"] = "disabled_by_config"
            else:
                feature["reference_latent_skip_reason"] = "ReferenceLatent_node_missing"
                fallbacks.append(feature["reference_latent_skip_reason"])
                feature["identity_reference_skip_reason"] = "reference_latent_unavailable"
        else:
            feature["reference_latent_skip_reason"] = "disabled_by_config"
            feature["identity_reference_skip_reason"] = "reference_latent_disabled"

        guidance = feature["flux_guidance_value"]
        if rt.has("FluxGuidance"):
            positive = get_value_at_index(
                rt.call("FluxGuidance", conditioning=positive, guidance=guidance),
                0,
            )
            feature["flux_guidance_applied"] = True
        else:
            feature["flux_guidance_skip_reason"] = "FluxGuidance_node_missing"
            fallbacks.append(feature["flux_guidance_skip_reason"])

        # Align→Paste→Refine: start from the pasted composite latent so the
        # sampler refines the face rather than regenerating the whole frame.
        # Official empty-canvas Kontext is opt-in via empty_latent: true.
        latent_image = composite_latent
        use_empty = bool(cfg.get("empty_latent", False))
        if use_empty:
            if rt.has("EmptySD3LatentImage"):
                empty = rt.call(
                    "EmptySD3LatentImage",
                    width=encode_w,
                    height=encode_h,
                    batch_size=1,
                )
                latent_image = get_value_at_index(empty, 0)
                feature["empty_latent_used"] = True
            else:
                feature["empty_latent_used"] = False
                feature["empty_latent_skip_reason"] = "EmptySD3LatentImage_node_missing"
                fallbacks.append(feature["empty_latent_skip_reason"])
        else:
            feature["empty_latent_used"] = False

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
                steps=feature["steps"],
                denoise=feature["denoise"],
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
        **feature,
        "input_body_size": [input_w, input_h],
        "encode_body_size": [encode_w, encode_h],
        "encode_megapixels": round((encode_w * encode_h) / 1_000_000, 3),
        "fallbacks": fallbacks,
    }
    return image, sample_meta


class FluxKontextPipeline(BasePipeline):
    """Community Align → Paste → Refine FLUX Kontext head/face swap."""

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
        align_info: dict = {}
        paste_info: dict = {}
        body_pil: Image.Image | None = None
        face_crop: Image.Image | None = None
        aligned_rgba: Image.Image | None = None
        composite: Image.Image | None = None
        refined: Image.Image | None = None
        head_mask = None
        dbg: dict[str, str] = {}
        load_meta: dict = {}
        fallbacks: list[str] = []
        prompt = str(self.cfg.get("prompt", "")).strip()

        try:
            import torch

            rt = self._ensure_runtime()
            bundle = _load_kontext_stack(rt, self.cfg)
            div_by = int(self.cfg.get("div_by", 8))
            max_dim = int(self.cfg.get("max_dim", 768))

            # --- Stage A: prepare body + face crop ---
            body_pil = resize_max_keep_ar(body.convert("RGB"), max_dim, div_by)
            face_crop = crop_face_reference(
                face,
                self.cache_dir,
                top=float(self.cfg.get("face_top_pad", 0.65)),
                bot=float(self.cfg.get("face_bot_pad", 0.15)),
                side=float(self.cfg.get("face_side_pad", 0.35)),
                include_shoulders=False,
            )

            # --- Stage B: Align → color-match → Paste ---
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
                # Fallback: naive center-resize paste so Kontext still sees a composite.
                fallback_face = face_crop.resize(
                    (
                        max(32, body_pil.width // 3),
                        max(32, body_pil.height // 3),
                    ),
                    Image.Resampling.LANCZOS,
                )
                composite = body_pil.copy()
                fx = (body_pil.width - fallback_face.width) // 2
                fy = max(0, body_pil.height // 8)
                composite.paste(fallback_face, (fx, fy))
                paste_info["composite_paste"] = True
                paste_info["composite_paste_fallback"] = "center_resize_paste"
                fallbacks.append("alignment_failed_used_center_paste")

            # Head mask for locality stitch after refine.
            head_mask = head_hair_mask_from_face(
                body_pil,
                self.cache_dir,
                expand_px=int(self.cfg.get("mask_expand_px", 22)),
                blur_px=int(self.cfg.get("mask_blur_px", 14)),
            )

            composite_t = pil_to_comfy_tensor(composite, torch)
            face_t = pil_to_comfy_tensor(face_crop.convert("RGB"), torch)
            refined, sample_meta = _sample_kontext_refine(
                rt, bundle, composite_t, self.cfg, prompt, face_t=face_t
            )

            # --- Stage C: soft stitch refined head back onto original body ---
            # Keeps clothing / background pixel-stable (community locality).
            if refined.size != body_pil.size:
                refined = refined.resize(body_pil.size, Image.Resampling.LANCZOS)
            box = (0, 0, body_pil.width, body_pil.height)
            out = soft_composite(body_pil, refined, head_mask, box)
            post_match = float(self.cfg.get("post_color_match_strength", 0.35) or 0.0)
            out = lab_histogram_match_face(out, body_pil, head_mask, strength=post_match)

            load_meta = dict(bundle.get("load_meta") or {})
            fallbacks = list(load_meta.get("fallbacks") or []) + list(
                sample_meta.get("fallbacks") or []
            ) + fallbacks
            dbg_candidates = {
                "debug_body": self._save_debug(out_dir, "debug_body.png", body_pil),
                "debug_face_crop": self._save_debug(
                    out_dir, "debug_face_crop.png", face_crop
                ),
                "debug_composite": self._save_debug(
                    out_dir, "debug_composite.png", composite
                ),
                "debug_refined": self._save_debug(
                    out_dir, "debug_refined.png", refined
                ),
                "debug_mask": self._save_debug(out_dir, "debug_mask.png", head_mask),
            }
            if aligned_rgba is not None:
                dbg_candidates["debug_aligned_face"] = self._save_debug(
                    out_dir, "debug_aligned_face.png", aligned_rgba.convert("RGBA")
                )
            dbg = {k: v for k, v in dbg_candidates.items() if v}
        except BaseException as exc:
            run_error = exc

        latency_s = time.perf_counter() - t0
        body_size = list(body_pil.size) if body_pil is not None else None
        meta = {
            "pipeline": self.name,
            "checkpoint": load_meta.get("checkpoint"),
            "loras_loaded": list(load_meta.get("loras_loaded") or []),
            "lora_strengths": dict(load_meta.get("lora_strengths") or {}),
            "prompt": prompt,
            "crop_size": body_size,
            "body_size": body_size,
            # --- Feature instrumentation (also nested under features) ---
            "face_alignment": bool(align_info.get("face_alignment")),
            "face_alignment_backend": align_info.get("face_alignment_backend"),
            "face_alignment_skip_reason": align_info.get("face_alignment_skip_reason"),
            "composite_paste": bool(paste_info.get("composite_paste")),
            "composite_paste_skip_reason": paste_info.get("composite_paste_skip_reason"),
            "composite_paste_fallback": paste_info.get("composite_paste_fallback"),
            "reference_latent_enabled": bool(
                sample_meta.get("reference_latent_enabled", True)
            ),
            "reference_latent_used": bool(sample_meta.get("reference_latent_used")),
            "reference_latent_skip_reason": sample_meta.get("reference_latent_skip_reason"),
            "identity_reference_enabled": bool(
                sample_meta.get("identity_reference_enabled", True)
            ),
            "identity_reference_used": bool(sample_meta.get("identity_reference_used")),
            "identity_reference_skip_reason": sample_meta.get(
                "identity_reference_skip_reason"
            ),
            "pre_color_match_strength": align_info.get(
                "pre_color_match_strength",
                float(self.cfg.get("pre_color_match_strength", 0.55) or 0.0),
            ),
            "denoise": sample_meta.get("denoise", self.cfg.get("denoise", 0.72)),
            "steps": sample_meta.get("steps", self.cfg.get("steps", 32)),
            "conditioning_zero_out_enabled": bool(
                sample_meta.get("conditioning_zero_out_enabled", True)
            ),
            "conditioning_zero_out_applied": bool(
                sample_meta.get("conditioning_zero_out_applied")
            ),
            "conditioning_zero_out_skip_reason": sample_meta.get(
                "conditioning_zero_out_skip_reason"
            ),
            "flux_kontext_image_scale_enabled": bool(
                sample_meta.get("flux_kontext_image_scale_enabled", True)
            ),
            "flux_kontext_image_scale_applied": bool(
                sample_meta.get("flux_kontext_image_scale_applied")
            ),
            "flux_kontext_image_scale_skip_reason": sample_meta.get(
                "flux_kontext_image_scale_skip_reason"
            ),
            "placement_lora_loaded": bool(load_meta.get("placement_lora_loaded")),
            "placement_lora_name": load_meta.get("placement_lora_name"),
            "placement_lora_strength": load_meta.get("placement_lora_strength"),
            "placement_lora_skip_reason": load_meta.get("placement_lora_skip_reason"),
            "flux_guidance_value": sample_meta.get(
                "flux_guidance_value", self.cfg.get("flux_guidance", 4.0)
            ),
            "flux_guidance_applied": bool(sample_meta.get("flux_guidance_applied")),
            "flux_guidance_skip_reason": sample_meta.get("flux_guidance_skip_reason"),
            "input_body_size": sample_meta.get("input_body_size"),
            "encode_body_size": sample_meta.get("encode_body_size"),
            "encode_megapixels": sample_meta.get("encode_megapixels"),
            "fallbacks": fallbacks,
            "latency_s": round(latency_s, 4),
            "features": {
                "face_alignment": bool(align_info.get("face_alignment")),
                "composite_paste": bool(paste_info.get("composite_paste")),
                "reference_latent": bool(sample_meta.get("reference_latent_used")),
                "identity_reference": bool(sample_meta.get("identity_reference_used")),
                "conditioning_zero_out": bool(
                    sample_meta.get("conditioning_zero_out_applied")
                ),
                "flux_kontext_image_scale": bool(
                    sample_meta.get("flux_kontext_image_scale_applied")
                ),
                "placement_lora_loaded": bool(load_meta.get("placement_lora_loaded")),
                "flux_guidance_value": sample_meta.get(
                    "flux_guidance_value", self.cfg.get("flux_guidance", 4.0)
                ),
                "denoise": sample_meta.get("denoise", self.cfg.get("denoise", 0.72)),
            },
        }
        if run_error is not None:
            meta["run_error"] = str(run_error)
            meta["run_error_type"] = type(run_error).__name__

        print(
            f"[flux_kontext] checkpoint={meta.get('checkpoint')} "
            f"align={meta.get('face_alignment')} paste={meta.get('composite_paste')} "
            f"ref_latent={meta.get('reference_latent_used')} "
            f"id_ref={meta.get('identity_reference_used')} "
            f"zero_out={meta.get('conditioning_zero_out_applied')} "
            f"image_scale={meta.get('flux_kontext_image_scale_applied')} "
            f"lora={meta.get('placement_lora_name')} "
            f"guidance={meta.get('flux_guidance_value')} "
            f"denoise={meta.get('denoise')} steps={meta.get('steps')} "
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
