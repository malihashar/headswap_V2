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
    resolve_model_file,
)
from headswap.pipelines.base import BasePipeline, PipelineResult, build_prompt
from headswap.preprocess import (
    align_face_to_destination,
    color_match_rgba_to_destination,
    crop_face_reference,
    crop_with_mask,
    face_on_white_background,
    feathered_soft_composite,
    fit_face_on_square,
    head_hair_mask_from_face,
    lab_histogram_match_face,
    pad_to_square,
    paste_aligned_face,
    resize_long_side,
    resize_max_keep_ar,
)


def _maybe_rmbg_face(
    rt: NodeRuntime, face_t
) -> tuple[object, bool, str | None, str | None]:
    """
    Official BFS Klein runs RMBG on the face reference before VAEEncode.

    Returns (face_tensor, applied, node_name, skip_note).
    Only runs when an RMBG-compatible node is registered; never invents a fallback
    background-removal path.
    """
    # comfyui-rmbg registers "RMBG"; keep a short alias list for common forks.
    for name in ("RMBG", "ImageRemoveBackground+", "easy imageRemBg"):
        if not rt.has(name):
            continue
        try:
            out = rt.call(name, image=face_t)
            return get_value_at_index(out, 0), True, name, None
        except TypeError:
            # comfyui-rmbg typically requires a model widget; match official defaults.
            try:
                out = rt.call(
                    name,
                    image=face_t,
                    model="RMBG-2.0",
                    sensitivity=1.0,
                    process_res=1024,
                    mask_blur=0,
                    mask_offset=0,
                    invert_output=False,
                    refine_foreground=False,
                    background="Alpha",
                    background_color="#222222",
                )
                return get_value_at_index(out, 0), True, name, None
            except Exception as exc:
                return face_t, False, None, f"face_rmbg_call_failed:{name}:{exc}"
        except Exception as exc:
            return face_t, False, None, f"face_rmbg_call_failed:{name}:{exc}"
    return face_t, False, None, "face_rmbg_node_unavailable"


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
        preferred_unet = self.cfg.get("unet_name", "flux-2-klein-4b-fp8.safetensors")
        fallback_unet = self.cfg.get("unet_name_fallback", "flux-2-klein-4b.safetensors")
        unet_name = resolve_model_file(
            unet_dir,
            preferred_unet,
            fallbacks=[fallback_unet],
        )
        strength = float(self.cfg.get("bfs_lora_strength", 1.0) or 0.0)
        lora_name = self.cfg.get("bfs_lora_name")
        key = f"klein::{unet_name}::{lora_name}::{strength}"
        if key in rt.models:
            return rt.models[key]

        load_meta: dict = {
            "checkpoint": unet_name,
            "checkpoint_preferred": preferred_unet,
            "checkpoint_fallback_used": unet_name != preferred_unet,
            "loras_loaded": [],
            "lora_strengths": {},
            "fallbacks": [],
        }
        if unet_name != preferred_unet:
            load_meta["fallbacks"].append(
                f"unet_fallback:{preferred_unet}->{unet_name}"
            )

        unet = rt.call("UNETLoader", unet_name=unet_name, weight_dtype="default")
        model = get_value_at_index(unet, 0)

        if strength > 0 and lora_name:
            loras = models / "loras"
            resolved = resolve_model_file(loras, lora_name, fallbacks=[lora_name])
            if not (loras / resolved).exists():
                load_meta["fallbacks"].append(f"bfs_lora_missing:{resolved}")
                raise FileNotFoundError(
                    f"BFS Klein LoRA not found at {loras / resolved}. "
                    "Run: python scripts/download_models.py --set klein"
                )
            if resolved != lora_name:
                load_meta["fallbacks"].append(f"bfs_lora_resolved:{lora_name}->{resolved}")
            model = get_value_at_index(
                rt.call(
                    "LoraLoaderModelOnly",
                    model=model,
                    lora_name=resolved,
                    strength_model=strength,
                ),
                0,
            )
            load_meta["loras_loaded"].append(resolved)
            load_meta["lora_strengths"][resolved] = strength
            print(f"[klein] loaded LoRA {resolved} strength={strength}")
        elif lora_name:
            load_meta["fallbacks"].append(
                f"bfs_lora_skipped_strength_zero:{lora_name}"
            )
            print(f"[klein] WARNING: BFS LoRA skipped (strength={strength})")
        else:
            load_meta["fallbacks"].append("bfs_lora_name_unset")

        clip = rt.call(
            "CLIPLoader",
            clip_name=self.cfg.get("clip_name", "qwen_3_4b.safetensors"),
            type=self.cfg.get("clip_type", "flux2"),
            device="default",
        )
        vae = rt.call("VAELoader", vae_name=self.cfg.get("vae_name", "flux2-vae.safetensors"))

        bundle = {
            "model": model,
            "clip": get_value_at_index(clip, 0),
            "vae": get_value_at_index(vae, 0),
            "load_meta": load_meta,
        }
        rt.models[key] = bundle
        return bundle

    def run(self, body: Image.Image, face: Image.Image, out_dir: Path | None = None) -> PipelineResult:
        import torch

        t0 = time.perf_counter()
        rt = self._ensure_runtime()
        bundle = self._load_models(rt)
        load_meta = dict(bundle.get("load_meta") or {})
        fallbacks: list[str] = list(load_meta.get("fallbacks") or [])
        div_by = int(self.cfg.get("div_by", 16))
        align_paste_refine = bool(self.cfg.get("align_paste_refine", False))
        denoise = float(self.cfg.get("denoise", 1.0))

        body_full = resize_max_keep_ar(
            body.convert("RGB"), int(self.cfg.get("max_body_dim", 1280)), div_by=div_by
        )
        # Natural face crop for paste / geometry. Never bake white-bg into paste
        # (that produced the oval sticker failure).
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
            top_extend=float(self.cfg.get("mask_top_extend", 1.25)),
            side_extend=float(self.cfg.get("mask_side_extend", 0.60)),
            bot_extend=float(self.cfg.get("mask_bot_extend", 0.40)),
        )
        crop_img, crop_mask, box = crop_with_mask(body_full, mask, pad=12, div_by=div_by)
        crop_work = resize_long_side(crop_img, int(self.cfg.get("crop_long_side", 1024)), div_by=div_by)
        face_nat = resize_long_side(face_crop, int(self.cfg.get("crop_long_side", 1024)), div_by=div_by)
        # BFS sticker face is ONLY for ReferenceLatent identity encode.
        if bool(self.cfg.get("face_white_bg", False)):
            face_ref = face_on_white_background(
                face_nat,
                cache_dir=self.cache_dir,
                force_ellipse=bool(self.cfg.get("face_white_ellipse", True)),
            )
        else:
            face_ref = face_nat

        square_crop = bool(self.cfg.get("square_crop", True))
        crop_content_box: tuple[int, int, int, int] | None = None
        face_ref_fill = float(self.cfg.get("face_ref_fill", 0.0) or 0.0)
        if square_crop:
            crop_work, crop_content_box = pad_to_square(crop_work, fill="edge", div_by=div_by)
            face_nat, _ = pad_to_square(face_nat, fill="edge", div_by=div_by)
            if face_ref_fill > 0:
                face_ref = fit_face_on_square(
                    face_ref,
                    crop_work.size[0],
                    fill_frac=face_ref_fill,
                    bg=(255, 255, 255),
                    div_by=div_by,
                )
            else:
                face_ref, _ = pad_to_square(face_ref, fill=(255, 255, 255), div_by=div_by)
            if face_nat.size != crop_work.size:
                face_nat = face_nat.resize(crop_work.size, Image.Resampling.LANCZOS)
            if face_ref.size != crop_work.size:
                face_ref = face_ref.resize(crop_work.size, Image.Resampling.LANCZOS)

        composite = crop_work
        paste_info: dict = {"composite_paste": False}
        align_info: dict = {}
        # Paste uses natural face (no white canvas). Soft alpha + higher denoise blend.
        if align_paste_refine:
            aligned_rgba, align_info = align_face_to_destination(
                face_nat,
                crop_work,
                self.cache_dir,
                core_min_alpha=float(self.cfg.get("paste_core_min_alpha", 0.78)),
                feather_px=int(self.cfg.get("paste_feather_px", 31)),
            )
            pre_match = float(self.cfg.get("pre_color_match_strength", 0.5) or 0.0)
            if aligned_rgba is not None:
                if pre_match > 0:
                    aligned_rgba = color_match_rgba_to_destination(
                        aligned_rgba, crop_work, strength=pre_match
                    )
                composite, paste_info = paste_aligned_face(crop_work, aligned_rgba)
            else:
                fallbacks.append(
                    align_info.get("face_alignment_skip_reason") or "align_paste_failed"
                )
                paste_info = {
                    "composite_paste": False,
                    "composite_paste_skip_reason": align_info.get(
                        "face_alignment_skip_reason"
                    ),
                }
                composite = crop_work

        prompt = build_prompt(self.cfg, body_full, face_crop, self.cache_dir)
        neg = str(self.cfg.get("negative_prompt", "") or "")

        reference_latent_used = False
        empty_flux2_latent_used = False
        flux2_scheduler_used = False
        face_rmbg_applied = False
        face_rmbg_node = None
        negative_mode = "none"
        paste_refine_latent = False
        steps = int(self.cfg.get("steps", 4))

        with torch.no_grad():
            body_t = pil_to_comfy_tensor(crop_work, torch)
            face_t = pil_to_comfy_tensor(face_ref, torch)
            composite_t = pil_to_comfy_tensor(composite, torch)

            face_t, face_rmbg_applied, face_rmbg_node, rmbg_note = _maybe_rmbg_face(rt, face_t)
            if rmbg_note:
                fallbacks.append(rmbg_note)

            body_lat = rt.call("VAEEncode", pixels=body_t, vae=bundle["vae"])
            face_lat = rt.call("VAEEncode", pixels=face_t, vae=bundle["vae"])
            comp_lat = rt.call("VAEEncode", pixels=composite_t, vae=bundle["vae"])

            pos = rt.call("CLIPTextEncode", text=prompt, clip=bundle["clip"])
            conditioning = get_value_at_index(pos, 0)

            if rt.has("ReferenceLatent"):
                body_latent = get_value_at_index(body_lat, 0)
                face_latent = get_value_at_index(face_lat, 0)
                # Order / repeats bias identity toward Picture 2 without paste-refine.
                # face_body_face = body layout + stronger face ID (recommended for quality).
                ref_order = str(self.cfg.get("reference_order", "body_face") or "body_face")
                face_repeats = max(1, int(self.cfg.get("face_ref_repeats", 1) or 1))

                def _chain_refs(cond, order: str, repeats: int):
                    cur = cond
                    seq = []
                    if order == "face_body_face":
                        seq = ["face"] + ["body"] + (["face"] * max(0, repeats - 1))
                    elif order == "face_body":
                        seq = ["face"] + ["body"]
                    elif order == "face_face_body":
                        seq = (["face"] * repeats) + ["body"]
                    else:
                        # body_face (default) — optional extra face repeats after body
                        seq = ["body"] + (["face"] * repeats)
                    for kind in seq:
                        lat = face_latent if kind == "face" else body_latent
                        cur = get_value_at_index(
                            rt.call("ReferenceLatent", conditioning=cur, latent=lat),
                            0,
                        )
                    return cur, seq

                positive, pos_seq = _chain_refs(conditioning, ref_order, face_repeats)
                # Body + face refs only. Do NOT attach composite as a 3rd
                # ReferenceLatent — that reinjected the paste sticker into
                # conditioning and fought the refine.
                reference_latent_used = True

                neg_c = rt.call("CLIPTextEncode", text=neg, clip=bundle["clip"])
                neg_conditioning = get_value_at_index(neg_c, 0)
                negative, _ = _chain_refs(neg_conditioning, ref_order, face_repeats)
                negative_mode = f"clip_text_reference_latent_{'_'.join(pos_seq)}"
            else:
                positive = conditioning
                fallbacks.append("reference_latent_missing_text_only")
                if neg.strip() and rt.has("CLIPTextEncode"):
                    neg_c = rt.call("CLIPTextEncode", text=neg, clip=bundle["clip"])
                    negative = get_value_at_index(neg_c, 0)
                    negative_mode = "clip_text"
                elif rt.has("ConditioningZeroOut"):
                    negative = get_value_at_index(
                        rt.call("ConditioningZeroOut", conditioning=positive), 0
                    )
                    negative_mode = "conditioning_zero_out"
                    fallbacks.append("negative_conditioning_zero_out")
                else:
                    negative = positive
                    negative_mode = "copied_positive"
                    fallbacks.append("negative_copied_from_positive")

            w, h = crop_work.size
            if align_paste_refine and bool(paste_info.get("composite_paste")):
                latent_image = get_value_at_index(comp_lat, 0)
                paste_refine_latent = True
                empty_flux2_latent_used = False
            elif rt.has("EmptyFlux2LatentImage"):
                empty = rt.call("EmptyFlux2LatentImage", width=w, height=h, batch_size=1)
                latent_image = get_value_at_index(empty, 0)
                empty_flux2_latent_used = True
            else:
                latent_image = get_value_at_index(body_lat, 0)
                fallbacks.append("empty_flux2_latent_missing_used_body_latent")

            use_partial = paste_refine_latent and denoise < 0.999
            if (not use_partial) and rt.has("Flux2Scheduler"):
                sigmas = get_value_at_index(
                    rt.call("Flux2Scheduler", steps=steps, width=w, height=h), 0
                )
                flux2_scheduler_used = True
            else:
                sigmas = get_value_at_index(
                    rt.call(
                        "BasicScheduler",
                        model=bundle["model"],
                        scheduler=self.cfg.get("scheduler", "simple"),
                        steps=steps,
                        denoise=denoise if use_partial else float(self.cfg.get("denoise", 1.0)),
                    ),
                    0,
                )
                if not use_partial:
                    fallbacks.append("flux2_scheduler_missing_used_basic_scheduler")
                flux2_scheduler_used = False

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
            use_full_load = bool(self.cfg.get("force_full_load", True))
            with force_sampling_full_load(
                models=(bundle["model"],), enabled=use_full_load
            ):
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
            edited_crop = comfy_tensor_to_pil(get_value_at_index(decoded, 0))

        if crop_content_box is not None:
            ox, oy, cw, ch = crop_content_box
            edited_crop = edited_crop.crop((ox, oy, ox + cw, oy + ch))

        stitched = feathered_soft_composite(
            body_full,
            edited_crop,
            mask,
            box,
            extra_blur_px=int(self.cfg.get("stitch_feather_px", 10)),
        )
        post_match = float(self.cfg.get("post_color_match_strength", 0.35) or 0.0)
        stitched = lab_histogram_match_face(
            stitched, body_full, mask, strength=post_match
        )

        dbg = {
            k: v
            for k, v in {
                "debug_body": self._save_debug(out_dir, "debug_body.png", body_full),
                "debug_face_crop": self._save_debug(out_dir, "debug_face_crop.png", face_crop),
                "debug_face_ref": self._save_debug(out_dir, "debug_face_ref.png", face_ref),
                "debug_mask": self._save_debug(out_dir, "debug_mask.png", mask),
                "debug_crop": self._save_debug(out_dir, "debug_crop.png", crop_work),
                "debug_composite": self._save_debug(out_dir, "debug_composite.png", composite),
                "debug_edited_crop": self._save_debug(out_dir, "debug_edited_crop.png", edited_crop),
            }.items()
            if v
        }
        meta = {
            "pipeline": self.name,
            "checkpoint": load_meta.get("checkpoint"),
            "checkpoint_preferred": load_meta.get("checkpoint_preferred"),
            "checkpoint_fallback_used": bool(load_meta.get("checkpoint_fallback_used")),
            "loras_loaded": list(load_meta.get("loras_loaded") or []),
            "lora_strengths": dict(load_meta.get("lora_strengths") or {}),
            "bfs_lora_name": self.cfg.get("bfs_lora_name"),
            "bfs_lora_strength": float(self.cfg.get("bfs_lora_strength", 0) or 0),
            "prompt": prompt,
            "crop_size": list(crop_work.size),
            "body_size": list(body_full.size),
            "face_ref_size": list(face_ref.size),
            "square_crop": square_crop,
            "align_paste_refine": align_paste_refine,
            "paste_refine_latent": paste_refine_latent,
            "composite_paste": bool(paste_info.get("composite_paste")),
            "face_alignment": bool(align_info.get("face_alignment")),
            "face_alignment_backend": align_info.get("face_alignment_backend"),
            "reference_order": str(self.cfg.get("reference_order", "body_face") or "body_face"),
            "face_ref_fill": float(self.cfg.get("face_ref_fill", 0) or 0),
            "face_ref_repeats": int(self.cfg.get("face_ref_repeats", 1) or 1),
            "denoise": denoise,
            "steps": steps,
            "reference_latent_used": reference_latent_used,
            "empty_flux2_latent_used": empty_flux2_latent_used,
            "flux2_scheduler_used": flux2_scheduler_used,
            "face_rmbg_applied": face_rmbg_applied,
            "face_rmbg_node": face_rmbg_node,
            "negative_mode": negative_mode,
            "fallbacks": fallbacks,
        }
        print(
            f"[klein] checkpoint={meta['checkpoint']} loras={meta['loras_loaded']} "
            f"strengths={meta['lora_strengths']} crop={meta['crop_size']} "
            f"steps={steps} denoise={denoise} square={square_crop} "
            f"align_paste={align_paste_refine} paste={meta['composite_paste']} "
            f"reference_latent={reference_latent_used} "
            f"face_rmbg={face_rmbg_applied} fallbacks={fallbacks or 'none'}"
        )
        return PipelineResult(
            image=stitched,
            latency_s=time.perf_counter() - t0,
            meta=meta,
            debug_paths=dbg,
        )
