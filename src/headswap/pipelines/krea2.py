"""Krea 2 Identity Edit — two-image head/face swap via community LoRA.

Uses ComfyUI native Krea 2 loaders + comfyui-krea2edit nodes:
  Krea2EditModelPatch + Krea2EditGroundedEncode

Input order (LoRA training):
  image 1 / source = body scene
  image 2 / source_b = identity person (face)

Stage timings (meta["timing_s"]) are always collected so Kaggle runs can
show where wall time goes (bootstrap / load / encode / sample / decode).
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageChops

from headswap.align_paste_swap import run_align_paste_swap
from headswap.comfy.full_load import force_sampling_full_load
from headswap.comfy.runtime import (
    NodeRuntime,
    comfy_tensor_to_pil,
    comfyui_path,
    get_value_at_index,
    pil_mask_to_comfy_tensor,
    pil_to_comfy_tensor,
    resolve_model_file,
)
from headswap.pipelines.base import BasePipeline, PipelineResult
from headswap.pipelines.errors import PipelineRunError
from headswap.profiling.identity_stage_trace import IdentityStageTrace
from headswap.profiling.head_scale_trace import HeadScaleTrace
from headswap.preprocess import (
    FaceBox,
    classify_lighting,
    clamp_crop_away_neighbors,
    clamp_edited_head_scale,
    crop_face_reference,
    crop_with_mask,
    dilate_mask,
    ensure_selected_face_mask_coverage,
    expand_crop_box_for_face_fill,
    feathered_soft_composite,
    hard_freeze_neighbor_faces,
    identity_face_boost_mask,
    lab_histogram_match_face,
    narrow_band_seam_refine,
    pad_to_square,
    pil_to_rgb_np,
    place_face_at_height_frac,
    procrustes_align_edited_crop_to_body_box,
    procrustes_align_generated_to_body,
    resize_contain,
    resize_long_side,
    resize_max_keep_ar,
    resize_to_megapixels,
    select_face_box,
    suppress_neighbor_faces_in_mask,
    expand_crop_box_wide,
)
from headswap.segmentation import build_head_hair_mask


@contextmanager
def _stage(timings: dict[str, float], name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)


def _print_timing_breakdown(
    timings: dict[str, float], total_s: float, *, verbose: bool = True
) -> None:
    if not verbose:
        return
    order = [
        "bootstrap",
        "model_loading",
        "preprocessing",
        "vae_encode",
        "model_patch",
        "grounded_encode_positive",
        "grounded_encode_negative",
        "diffusion_sampling",
        "vae_decode",
        "postprocessing",
        "image_saving",
    ]
    print(f"[krea2 timing] total={total_s:.2f}s")
    accounted = 0.0
    for name in order:
        if name not in timings:
            continue
        sec = float(timings[name])
        accounted += sec
        pct = (100.0 * sec / total_s) if total_s > 0 else 0.0
        print(f"  {name:<28} {sec:7.2f}s  ({pct:5.1f}%)")
    other = total_s - accounted
    if abs(other) >= 0.005:
        pct = (100.0 * other / total_s) if total_s > 0 else 0.0
        print(f"  {'other':<28} {other:7.2f}s  ({pct:5.1f}%)")


def _attention_backend_info() -> dict:
    """Detect which PyTorch SDPA / xFormers path is active."""
    info: dict = {
        "xformers": False,
        "flash_sdp": None,
        "mem_efficient_sdp": None,
        "math_sdp": None,
        "backend": "unknown",
    }
    try:
        import torch

        if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
            info["flash_sdp"] = bool(torch.backends.cuda.flash_sdp_enabled())
            info["mem_efficient_sdp"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
            info["math_sdp"] = bool(torch.backends.cuda.math_sdp_enabled())
        try:
            import xformers  # type: ignore

            info["xformers"] = True
            info["xformers_version"] = getattr(xformers, "__version__", "?")
        except Exception:
            info["xformers"] = False

        if info["xformers"]:
            info["backend"] = "xformers"
        elif info["flash_sdp"]:
            info["backend"] = "flash_sdp"
        elif info["mem_efficient_sdp"]:
            info["backend"] = "mem_efficient_sdp"
        elif info["math_sdp"]:
            info["backend"] = "math_sdp"
        else:
            info["backend"] = "pytorch_default"
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _configure_fast_kernels(*, prefer_flash: bool = True) -> dict:
    """
    Prefer Flash / mem-efficient SDPA over math SDPA. Does not change weights.

    Expected: lower attention latency on Ampere+ (A100) and often on T4 when
    flash/mem-efficient paths are available. Quality unchanged.
    """
    info: dict = {"ok": False, "actions": []}
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
            info["actions"].append("matmul_precision_high")
        except Exception:
            pass

        if prefer_flash and hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            # Keep math as last-resort fallback so exotic shapes still run.
            torch.backends.cuda.enable_math_sdp(True)
            info["actions"].append("sdpa_flash+mem_efficient_preferred")
        info["ok"] = True
        info["attention_after"] = _attention_backend_info()
    except Exception as exc:
        info["error"] = str(exc)
    return info


@contextmanager
def _silence_krea2edit_step_prints() -> Iterator[None]:
    """
    Identity-edit wrapper prints `[krea2edit] STRIDE1-POS...` with flush=True on
    EVERY UNet forward. That forces host sync and can add material overhead.

    Filter builtins.print once for the whole sample (not per-forward wrap).
    Also disable Comfy's progress bar (tqdm can sync each step). Quality unchanged.
    """
    import builtins

    oprint = builtins.print

    def filtered(*a, **k):
        if a and isinstance(a[0], str) and a[0].startswith("[krea2edit]"):
            return None
        return oprint(*a, **k)

    builtins.print = filtered

    # Disable Comfy progress bar during sample (tqdm can sync each step).
    prog_restores: list[Any] = []
    try:
        import comfy.utils as cu

        if hasattr(cu, "set_progress_bar_enabled"):
            cu.set_progress_bar_enabled(False)
            prog_restores.append(("set_progress_bar_enabled", cu))
        elif hasattr(cu, "PROGRESS_BAR_ENABLED"):
            prev = cu.PROGRESS_BAR_ENABLED
            cu.PROGRESS_BAR_ENABLED = False
            prog_restores.append(("PROGRESS_BAR_ENABLED", cu, prev))
    except Exception:
        pass

    try:
        yield
    finally:
        builtins.print = oprint
        for item in prog_restores:
            if item[0] == "set_progress_bar_enabled":
                try:
                    item[1].set_progress_bar_enabled(True)
                except Exception:
                    pass
            elif item[0] == "PROGRESS_BAR_ENABLED":
                try:
                    item[1].PROGRESS_BAR_ENABLED = item[2]
                except Exception:
                    pass


# Process-wide Comfy runtime — survives multiple pipeline instances in one kernel.
_SHARED_RUNTIME: NodeRuntime | None = None


def get_shared_krea2_runtime(*, init_custom_nodes: bool = True) -> NodeRuntime:
    global _SHARED_RUNTIME
    if _SHARED_RUNTIME is None:
        _SHARED_RUNTIME = NodeRuntime(init_custom_nodes=init_custom_nodes)
    return _SHARED_RUNTIME


def reset_shared_krea2_runtime() -> None:
    global _SHARED_RUNTIME
    _SHARED_RUNTIME = None


def _maybe_torch_compile_diffusion(model, *, enabled: bool, mode: str = "reduce-overhead") -> dict:
    """
    Optional torch.compile on the DiT. First call pays compile cost; later steps /
    images benefit. Off by default — enable on A100 multi-image sessions.
    """
    info: dict = {"enabled": False, "compiled": False}
    if not enabled:
        return info
    info["enabled"] = True
    try:
        import torch

        base = getattr(model, "model", None)
        dm = getattr(base, "diffusion_model", None) if base is not None else None
        if dm is None:
            info["error"] = "diffusion_model_missing"
            return info
        if getattr(dm, "_headswap_compiled", False):
            info["compiled"] = True
            info["cache_hit"] = True
            return info
        compiled = torch.compile(dm, mode=mode, fullgraph=False)
        base.diffusion_model = compiled
        # Mark the compiled module if possible
        try:
            compiled._headswap_compiled = True  # type: ignore[attr-defined]
        except Exception:
            pass
        info["compiled"] = True
        info["mode"] = mode
        print(f"[krea2] torch.compile applied mode={mode}")
    except Exception as exc:
        info["error"] = str(exc)
        print(f"[krea2] torch.compile skipped: {exc}")
    return info


def _cuda_util_snapshot() -> dict:
    out: dict = {
        "cuda_available": False,
        "device_name": None,
        "free_mb": None,
        "total_mb": None,
        "allocated_mb": None,
        "reserved_mb": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        out["cuda_available"] = True
        out["device_name"] = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info()
        out["free_mb"] = round(free_b / (1024**2), 1)
        out["total_mb"] = round(total_b / (1024**2), 1)
        out["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
        out["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 1)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _comfy_vram_state() -> dict:
    info: dict = {"vram_state": None, "lowvram": None, "cpu_offload_likely": None}
    try:
        import comfy.model_management as mm

        state = getattr(mm, "vram_state", None)
        info["vram_state"] = str(state)
        name = str(state)
        info["lowvram"] = "LOW" in name or "NO_VRAM" in name
        # NORMAL_VRAM alone does NOT mean streaming when force_full_load is on.
        info["cpu_offload_likely"] = bool(info["lowvram"])
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _print_sampling_diagnostics(
    *,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    force_full_load: bool,
    denoise: float,
    width: int,
    height: int,
    mu_shift: float | None,
    verbose: bool = True,
) -> dict:
    """Collect sampling knobs before KSampler; print only when verbose."""
    attn = _attention_backend_info()
    cuda = _cuda_util_snapshot()
    vram = _comfy_vram_state()
    unet_evals_est = steps if cfg <= 1.0 + 1e-6 else steps * 2
    streaming = (not force_full_load) or bool(vram.get("lowvram"))
    diag = {
        "configured_steps": steps,
        "scheduler": scheduler,
        "sampler": sampler,
        "cfg": cfg,
        "denoise": denoise,
        "force_full_load": force_full_load,
        "layer_streaming_expected": streaming,
        "estimated_unet_evals": unet_evals_est,
        "attention": attn,
        "cuda_before_sample": cuda,
        "comfy_vram": vram,
        "resolution": [width, height],
        "timestep_shift_mu": mu_shift,
        "note_ref_boost_attn_bias": (
            "ref_boost!=1 builds attn logit bias → masked SDPA path "
            "(may skip pure flash); required for identity fidelity"
        ),
        "note_dual_ref_seq": (
            "two-image Identity Edit prepends 2 ref latent token blocks → "
            "~3× image tokens vs T2I; attention ~O(n²) dominates on T4"
        ),
    }
    if not verbose:
        return diag
    print("[krea2 sampling diagnostics]")
    print(f"  configured_steps={steps}")
    print(f"  sampler={sampler}  scheduler={scheduler}")
    print(f"  cfg={cfg}  denoise={denoise}  estimated_unet_evals≈{unet_evals_est}")
    print(
        f"  force_full_load={force_full_load}  "
        f"layer_streaming_expected={streaming}  "
        f"(streaming only if full_load=false or LOW_VRAM)"
    )
    print(
        f"  attention_backend={attn.get('backend')}  xformers={attn.get('xformers')}  "
        f"flash_sdp={attn.get('flash_sdp')}  mem_efficient_sdp={attn.get('mem_efficient_sdp')}"
    )
    print(
        f"  comfy_vram_state={vram.get('vram_state')}  "
        f"(ignore older cpu_offload_likely=NORMAL noise — full_load overrides)"
    )
    print(
        f"  gpu={cuda.get('device_name')}  free_mb={cuda.get('free_mb')}/"
        f"{cuda.get('total_mb')}  alloc_mb={cuda.get('allocated_mb')}  "
        f"reserved_mb={cuda.get('reserved_mb')}"
    )
    if mu_shift is not None:
        print(f"  timestep_shift_mu={mu_shift}  (Turbo author rec: 1.15)")
    print(
        "  cost drivers: dual-ref token sequence + ref_boost attn bias + 12B DiT "
        "(not CPU↔GPU weight streaming when force_full_load=true)"
    )
    try:
        import sys

        sys.stdout.flush()
    except Exception:
        pass
    return diag


def _count_unet_forwards(model, enabled: bool = True):
    """Wrap diffusion forward to count real UNet evaluations during KSampler."""
    counter = {"n": 0, "wrapped": False}
    if not enabled:
        return counter, None

    # ModelPatcher → .model (BaseModel) → .diffusion_model (nn.Module)
    try:
        inner = model
        dm = None
        if hasattr(inner, "model"):
            base = inner.model
            dm = getattr(base, "diffusion_model", None) or getattr(base, "model", None)
        if dm is None or not hasattr(dm, "forward"):
            return counter, None
        orig = dm.forward

        def counted_forward(*args, **kwargs):
            counter["n"] += 1
            return orig(*args, **kwargs)

        dm.forward = counted_forward
        counter["wrapped"] = True

        def restore():
            dm.forward = orig

        return counter, restore
    except Exception:
        return counter, None


class Krea2IdentityEditPipeline(BasePipeline):
    name = "krea2_identity_edit"

    def _ensure_runtime(self, timings: dict[str, float]) -> NodeRuntime:
        if self.runtime is None:
            with _stage(timings, "bootstrap"):
                # Process-wide singleton: second image in the same kernel skips
                # Comfy/custom-node init (~25s saved).
                self.runtime = get_shared_krea2_runtime(init_custom_nodes=True)
        else:
            timings.setdefault("bootstrap", 0.0)
        return self.runtime

    def _load_models(self, rt: NodeRuntime, timings: dict[str, float]) -> dict:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
        # Full-frame A/B only: optional LoRA override (crop_stitch ignores this).
        _ff_lora = self.cfg.get("full_frame_identity_lora_name")
        _ff_mode = str(
            self.cfg.get("multi_person_edit_mode", "crop_stitch") or "crop_stitch"
        ).strip().lower()
        if _ff_lora and _ff_mode in ("full_frame", "fullframe", "full"):
            lora_name = str(_ff_lora)
        lora_strength = float(self.cfg.get("identity_lora_strength", 1.0) or 0.0)
        key = f"krea2::{unet_name}::{clip_name}::{vae_name}::{lora_name}::{lora_strength}"
        if key in rt.models:
            timings.setdefault("model_loading", 0.0)
            timings["_model_cache_hit"] = 1.0
            return rt.models[key]

        if not rt.has("Krea2EditModelPatch") or not rt.has("Krea2EditGroundedEncode"):
            raise PipelineRunError(
                "comfyui-krea2edit nodes missing. "
                "Run: bash scripts/setup_krea2_nodes.sh "
                "(or setup_kaggle.sh --krea2) and ensure HEADSWAP_INIT_CUSTOM_NODES=1."
            )

        with _stage(timings, "model_loading"):
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
        timings["_model_cache_hit"] = 0.0
        return bundle

    def _cheap_negative(self, rt: NodeRuntime, positive, clip, timings: dict[str, float]):
        """
        At CFG≤1 KSampler does not use CFG against the negative, so a full
        Qwen3-VL grounded negative encode is pure waste. Prefer zero-out / empty text.
        """
        with _stage(timings, "grounded_encode_negative"):
            if rt.has("ConditioningZeroOut"):
                neg = rt.call("ConditioningZeroOut", conditioning=positive)
                timings["_negative_mode"] = 0.0  # marker: zero_out
                return get_value_at_index(neg, 0), "conditioning_zero_out"
            if rt.has("CLIPTextEncode"):
                neg = rt.call("CLIPTextEncode", text="", clip=clip)
                timings["_negative_mode"] = 1.0  # empty text
                return get_value_at_index(neg, 0), "clip_text_empty"
            # Last resort: reuse positive (harmless at cfg=1).
            timings["_negative_mode"] = 2.0
            return positive, "copied_positive"


    def _single_person_parity(self) -> bool:
        """SPP-CC: multi uses the same conditioning recipe as single-person."""
        return bool(self.cfg.get("single_person_parity", True))

    def _tight_crop_flags(
        self,
        body_full: Image.Image,
        selected_face: FaceBox | None,
        all_faces: list[FaceBox],
    ) -> dict[str, Any]:
        """Decide crop/mask flags for the selected face.

        With ``single_person_parity`` (Architecture C / SPP-CC), multi-person
        photos use the same mask extents and crop recipe as single-person —
        only neighbor suppress remains for group locality. Legacy isolate /
        tight forks stay available when parity is disabled.
        """
        multi_person = len(all_faces) > 1
        bw, bh = body_full.size
        face_frac = 0.0
        if selected_face is not None and bw * bh > 0:
            face_frac = (selected_face.width * selected_face.height) / float(bw * bh)
        wide_group = (bw / max(1, bh)) >= float(
            self.cfg.get("group_aspect_thresh", 1.25)
        )
        small_face = face_frac > 0 and face_frac < float(
            self.cfg.get("group_face_area_frac", 0.12)
        )
        force_tight = bool(self.cfg.get("force_tight_head_crop", False))
        spp = self._single_person_parity()

        # SPP-CC: never enter the legacy multi tight/isolate mask recipe.
        use_tight = False if spp else bool(
            force_tight
            or (
                bool(self.cfg.get("tight_on_small_face", False))
                and (small_face or (wide_group and small_face))
            )
        )
        # Spatial isolation fork (wider masks / expand_crop) — off under SPP-CC.
        isolate_selected = False if spp else bool(multi_person or small_face)

        top_ext = float(self.cfg.get("mask_top_extend", 1.25))
        side_ext = float(self.cfg.get("mask_side_extend", 0.60))
        bot_ext = float(self.cfg.get("mask_bot_extend", 0.40))
        expand_px = int(self.cfg.get("mask_expand_px", 18))
        crop_pad = 12
        if use_tight:
            top_ext = float(self.cfg.get("multi_mask_top_extend", 1.20))
            side_ext = float(self.cfg.get("multi_mask_side_extend", 0.48))
            bot_ext = float(self.cfg.get("multi_mask_bot_extend", 0.48))
            expand_px = int(self.cfg.get("multi_mask_expand_px", 16))
            crop_pad = int(self.cfg.get("multi_crop_pad", 14))
        elif isolate_selected:
            side_ext = float(
                self.cfg.get("isolate_mask_side_extend", max(side_ext, 0.70))
            )
            top_ext = float(
                self.cfg.get("isolate_mask_top_extend", max(top_ext, 1.70))
            )
            expand_px = int(
                self.cfg.get("isolate_mask_expand_px", max(expand_px, 22))
            )
            crop_pad = int(self.cfg.get("isolate_crop_pad", crop_pad))
        return {
            "multi_person": multi_person,
            "face_frac": face_frac,
            "use_tight": use_tight,
            "isolate_selected": isolate_selected,
            "single_person_parity": spp,
            "small_face": small_face,
            "top_ext": top_ext,
            "side_ext": side_ext,
            "bot_ext": bot_ext,
            "expand_px": expand_px,
            "crop_pad": crop_pad,
        }

    def _build_scene_person(
        self,
        body_full: Image.Image,
        face_crop: Image.Image,
        selected_face: FaceBox | None,
        *,
        div_by: int,
        use_tight: bool,
        top_ext: float,
        side_ext: float,
        bot_ext: float,
        expand_px: int,
        crop_pad: int,
        all_faces: list[FaceBox] | None = None,
        isolate_selected: bool = False,
    ) -> dict[str, Any]:
        """Crop+mask one body face and size the identity reference to match."""
        all_faces = list(all_faces or [])
        multi_person = len(all_faces) > 1
        spp = self._single_person_parity()
        mask, mask_info = build_head_hair_mask(
            body_full,
            self.cache_dir,
            backend=str(self.cfg.get("head_mask_backend", "ellipse") or "ellipse"),
            face_box=selected_face,
            expand_px=expand_px,
            blur_px=int(self.cfg.get("mask_blur_px", 12)),
            top_extend=top_ext,
            side_extend=side_ext,
            bot_extend=bot_ext,
        )
        # Prevent expanded group crops from stitching neighbor faces.
        if multi_person and selected_face is not None:
            mask = suppress_neighbor_faces_in_mask(
                mask, selected_face, all_faces, shrink=0.90
            )

        crop_img, _, box = crop_with_mask(
            body_full, mask, pad=crop_pad, div_by=div_by
        )
        crop_long_before = int(max(crop_img.size))
        expand_info: dict[str, float] = {
            "expanded": 0.0,
            "face_fill_before": 0.0,
            "face_fill_after": 0.0,
            "crop_long_before": float(crop_long_before),
            "crop_long_after": float(crop_long_before),
        }
        neighbor_clamp_info: dict[str, float] = {
            "clamped": 0.0,
            "neighbors_excluded": 0.0,
        }
        # Production multi locality: keep single-person crop recipe, but shrink
        # the window so neighboring face centers never enter the Krea2 scene.
        if multi_person and selected_face is not None and bool(
            self.cfg.get("clamp_crop_away_neighbors", True)
        ):
            box, neighbor_clamp_info = clamp_crop_away_neighbors(
                body_full.size,
                box,
                selected_face,
                all_faces,
                margin_frac=float(self.cfg.get("neighbor_crop_margin_frac", 0.18)),
                # Protect exactly what the stitch mask covers so the clamp can
                # never cut into the selected head/hair region.
                protect_top_frac=top_ext,
                protect_side_frac=side_ext,
                protect_bot_frac=bot_ext,
                protect_pad_px=expand_px + int(self.cfg.get("mask_blur_px", 12)),
            )
            crop_img = body_full.crop(box)
            if neighbor_clamp_info.get("neighbors_excluded", 0) > 0:
                import sys as _sys

                print(
                    "[krea2] clamped crop away from neighbors "
                    f"excluded={int(neighbor_clamp_info['neighbors_excluded'])} "
                    f"box={list(box)}",
                    file=_sys.__stdout__,
                    flush=True,
                )
        # Legacy: expand crop for face-fill (off under SPP-CC / single-person parity).
        if (not spp) and (isolate_selected or use_tight):
            box, expand_info = expand_crop_box_for_face_fill(
                body_full.size,
                box,
                selected_face,
                target_face_area_frac=float(
                    self.cfg.get("multi_target_face_fill", 0.16)
                ),
                min_long_side=int(self.cfg.get("multi_min_crop_long", 448)),
                other_faces=all_faces,
                div_by=div_by,
            )
            crop_img = body_full.crop(box)

        # Match successful single-person inference resolution (default 768),
        # unless crop_margin_mode=wide targets ~1MP with a larger native crop.
        crop_margin_mode = str(
            self.cfg.get("crop_margin_mode", "tight") or "tight"
        ).strip().lower()
        wide_mode = crop_margin_mode in ("wide", "wide_crop", "1mp")
        if wide_mode and selected_face is not None:
            # Recompute a larger crop from the face (shoulders/torso), not an
            # upscale of the tight head crop. Neighbor clamp still applies.
            wide_box = expand_crop_box_wide(
                body_full.size,
                selected_face,
                pad_top_frac=float(self.cfg.get("wide_crop_pad_top_frac", 1.20)),
                pad_side_frac=float(self.cfg.get("wide_crop_pad_side_frac", 1.80)),
                pad_bot_frac=float(self.cfg.get("wide_crop_pad_bot_frac", 2.50)),
                target_mp=float(self.cfg.get("wide_crop_target_mp", 1.0)),
                div_by=div_by,
            )
            if multi_person and bool(self.cfg.get("clamp_crop_away_neighbors", True)):
                wide_box, neighbor_clamp_info = clamp_crop_away_neighbors(
                    body_full.size,
                    wide_box,
                    selected_face,
                    all_faces,
                    margin_frac=float(self.cfg.get("neighbor_crop_margin_frac", 0.18)),
                    protect_top_frac=top_ext,
                    protect_side_frac=side_ext,
                    protect_bot_frac=bot_ext,
                    protect_pad_px=expand_px + int(self.cfg.get("mask_blur_px", 12)),
                )
            box = wide_box
            crop_img = body_full.crop(box)
            # Allow mild upscale so the Krea2 scene lands near ~1MP when the
            # available native crop is smaller (still a cropped region, not FF).
            scene = resize_to_megapixels(
                crop_img,
                target_mp=float(self.cfg.get("wide_crop_target_mp", 1.0)),
                min_mp=float(self.cfg.get("wide_crop_min_mp", 0.85)),
                max_mp=float(self.cfg.get("wide_crop_max_mp", 1.15)),
                max_dim=int(self.cfg.get("wide_crop_max_dim", 1536)),
                div_by=div_by,
                allow_upscale=bool(self.cfg.get("wide_crop_allow_upscale", True)),
            )
        else:
            # Match successful single-person inference resolution (default 768).
            crop_long = int(self.cfg.get("crop_long_side", 768))
            if (not spp) and use_tight and bool(self.cfg.get("multi_boost_crop_long", False)):
                crop_long = max(
                    crop_long, int(self.cfg.get("multi_crop_long_side", 896))
                )
            scene = resize_long_side(crop_img, crop_long, div_by=div_by)
        face_h_frac_scene = 0.55
        hair_boost = float(self.cfg.get("identity_hair_height_boost", 1.30))
        if selected_face is not None and box is not None:
            crop_h = max(1, box[3] - box[1])
            face_h_frac_native = float(selected_face.height) / float(crop_h)
            face_h_frac_scene = min(0.70, face_h_frac_native * hair_boost)

        # Match donor face-to-canvas ratio to measured scene face height.
        # Allowed under SPP (krea2_crop) for oversized-head A/B — gated only by
        # identity_scale_match + multi/isolate, not by single_person_parity.
        scale_match = (
            bool(self.cfg.get("identity_scale_match", True))
            and (multi_person or isolate_selected)
        )
        scale_factor = float(self.cfg.get("identity_scale_factor", 0.90))
        if bool(self.cfg.get("person_match_crop_size", True)):
            if scale_match:
                person = place_face_at_height_frac(
                    face_crop.convert("RGB"),
                    scene.size,
                    height_frac=face_h_frac_scene * scale_factor,
                    fill=(0, 0, 0),
                )
            else:
                person = resize_contain(
                    face_crop.convert("RGB"), scene.size, fill=(0, 0, 0)
                )
        else:
            person = resize_long_side(
                face_crop.convert("RGB"),
                max(scene.size),
                div_by=div_by,
            )
        apply_white_bg = (
            (not spp)
            and use_tight
            and bool(self.cfg.get("face_white_bg", False))
            and bool(self.cfg.get("face_white_bg_on_tight", False))
        )
        if apply_white_bg:
            from headswap.preprocess import face_on_white_background

            person = face_on_white_background(
                person,
                cache_dir=self.cache_dir,
                force_ellipse=True,
            )
        crop_content_box: tuple[int, int, int, int] | None = None
        if bool(self.cfg.get("square_crop", False)):
            scene, crop_content_box = pad_to_square(
                scene, fill="edge", div_by=div_by
            )
            if bool(self.cfg.get("person_match_crop_size", True)):
                if scale_match:
                    person = place_face_at_height_frac(
                        face_crop.convert("RGB"),
                        scene.size,
                        height_frac=face_h_frac_scene * scale_factor,
                        fill=(0, 0, 0),
                    )
                else:
                    person = resize_contain(person, scene.size, fill=(0, 0, 0))

        face_in_crop = 0.0
        if selected_face is not None:
            ca = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            face_in_crop = float(selected_face.width * selected_face.height) / float(ca)

        diag = {
            "faces_detected": len(all_faces),
            "selected_box": (
                None
                if selected_face is None
                else [selected_face.x0, selected_face.y0, selected_face.x1, selected_face.y1]
            ),
            "face_area_frac_body": round(
                float(
                    (selected_face.width * selected_face.height)
                    / float(max(1, body_full.size[0] * body_full.size[1]))
                )
                if selected_face is not None
                else 0.0,
                4,
            ),
            "face_area_frac_crop": round(face_in_crop, 4),
            "face_height_frac_scene": round(float(face_h_frac_scene), 4),
            "identity_scale_match": bool(scale_match),
            "identity_scale_factor": float(scale_factor) if scale_match else 1.0,
            "person_prep": (
                "place_face_at_height_frac" if scale_match else "resize_contain"
            ),
            "single_person_parity": spp,
            "head_mask": mask_info,
            "crop_box": list(box),
            "crop_native_size": [box[2] - box[0], box[3] - box[1]],
            "crop_long_native": int(max(box[2] - box[0], box[3] - box[1])),
            "mask_size": list(mask.size),
            "scene_size": list(scene.size),
            "scene_megapixels": round(
                (scene.size[0] * scene.size[1]) / 1_000_000.0, 4
            ),
            "person_size": list(person.size),
            "body_size": list(body_full.size),
            "use_tight": bool(use_tight),
            "isolate_selected": bool(isolate_selected),
            "multi_person": bool(multi_person),
            "crop_margin_mode": "wide" if wide_mode else "tight",
            "face_white_bg_applied": bool(apply_white_bg),
            "crop_expand": {k: round(float(v), 4) for k, v in expand_info.items()},
            "neighbor_crop_clamp": {
                k: round(float(v), 4) for k, v in neighbor_clamp_info.items()
            },
            "mask_params": {
                "top_ext": top_ext,
                "side_ext": side_ext,
                "bot_ext": bot_ext,
                "expand_px": expand_px,
                "crop_pad": crop_pad,
            },
        }
        if spp and (scale_match or isolate_selected or use_tight):
            import sys

            print(
                "[krea2] WARNING: SPP-CC parity violated "
                f"(scale_match={scale_match} isolate={isolate_selected} "
                f"tight={use_tight})",
                flush=True,
            )
        return {
            "scene": scene,
            "person": person,
            "mask": mask,
            "box": box,
            "crop_content_box": crop_content_box,
            "diag": diag,
        }

    def _log_face_diag(self, diag: dict[str, Any], *, label: str = "prep") -> None:
        import sys

        print(
            f"[krea2 diag:{label}] faces={diag.get('faces_detected')} "
            f"multi={diag.get('multi_person')} tight={diag.get('use_tight')} "
            f"isolate={diag.get('isolate_selected')} "
            f"white_bg={diag.get('face_white_bg_applied')} "
            f"selected={diag.get('selected_box')} "
            f"face_frac_body={diag.get('face_area_frac_body')} "
            f"face_frac_crop={diag.get('face_area_frac_crop')} "
            f"crop_native={diag.get('crop_native_size')} "
            f"crop_long_native={diag.get('crop_long_native')} "
            f"scene={diag.get('scene_size')} person={diag.get('person_size')} "
            f"body={diag.get('body_size')} mask={diag.get('mask_size')} "
            f"expand={diag.get('crop_expand')} mask_params={diag.get('mask_params')}",
            flush=True,
        )

    def _apply_expression_policy(self, prompt: str) -> str:
        """Honor ``preserve_expression`` (default true).

        When false, strip prompt clauses that force the body/scene expression
        onto the result and append donor-expression freedom language. Geometry,
        identity, and hair instructions are left intact.
        """
        text = str(prompt or "").strip()
        if bool(self.cfg.get("preserve_expression", True)):
            return text
        # Known expression-lock clauses (yaml prompt + align_paste refine + add-ons).
        patterns = [
            # Primary yaml CRITICAL block (expression + follow-on Keep mouth…).
            r"CRITICAL:\s*copy the facial expression from the first image exactly"
            r".*?never from the second image\.",
            r"CRITICAL:\s*copy the facial expression from image 1 only[^.]*\.",
            r"Keep mouth shape, smile/no-smile, eye gaze, and\s*"
            r"micro-expressions from the first image only[^.]*\.",
            r"Preserve the facial expression, mouth shape, eye gaze, and head\s*"
            r"pose from the first image exactly\.?",
        ]
        out = text
        for pat in patterns:
            out = re.sub(pat, " ", out, flags=re.IGNORECASE | re.DOTALL)
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r" *\n *", "\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        freedom = (
            "Allow the facial expression from the second image (donor identity) — "
            "do not force the first person's smile, no-smile, or mouth shape onto "
            "the result. Prefer the natural expression of the identity person."
        )
        if "Allow the facial expression from the second image" not in out:
            out = (out + " " + freedom).strip()
        return out

    def _prompt_for_edit(self, *, use_tight: bool, multi_person: bool) -> str:
        prompt = str(self.cfg.get("prompt", "") or "").strip()
        # SPP-CC: identical prompt to single-person (no multi add-ons).
        if self._single_person_parity():
            return self._apply_expression_policy(prompt)
        # Head-scale reminder for group shots (oversized heads are the main fail).
        if multi_person and bool(self.cfg.get("multi_head_scale_prompt", True)):
            prompt = (
                prompt
                + " CRITICAL: match the original head size exactly — the swapped "
                "head must not be larger than the original head or the heads of "
                "nearby people. Keep neck placement and shoulder line unchanged."
            )
        # Hair often survives from the body crop unless we force replacement.
        if bool(self.cfg.get("multi_hair_replace_prompt", True)) and (
            multi_person or use_tight
        ):
            prompt = (
                prompt
                + " Replace the hair completely with the hairstyle from the second "
                "image — hairline, length, color, and parting. Do not leave any of "
                "the first person's original hair. Completely cover and remove the "
                "original face — no ghosting, double face, or leftover ear/cheek "
                "from the first person beside the new head."
            )
        if bool(self.cfg.get("multi_extra_prompt", False)) and (use_tight or multi_person):
            prompt = (
                prompt
                + " The first image crop shows ONE person only — replace only that "
                "person's head. Match the original head size and neck placement "
                "exactly. Do not change other people, arms, or background. "
                "Use ONLY the face identity from the second image — do not copy "
                "clothing, collar, bowtie, or shoulders from the second image. "
                "Match lighting and skin tone to the neck and jaw in the first image."
            )
        return self._apply_expression_policy(prompt)

    def _face_position_label(
        self, selected: FaceBox | None, all_faces: list[FaceBox]
    ) -> str:
        """Relative horizontal position of the selected face among detections."""
        if selected is None or len(all_faces) <= 1:
            return "the primary"
        ordered = sorted(all_faces, key=lambda f: 0.5 * (f.x0 + f.x1))
        idx = next(
            (
                i
                for i, f in enumerate(ordered)
                if f.x0 == selected.x0
                and f.y0 == selected.y0
                and f.x1 == selected.x1
                and f.y1 == selected.y1
            ),
            0,
        )
        n = len(ordered)
        if n == 2:
            return "leftmost" if idx == 0 else "rightmost"
        if idx == 0:
            return "leftmost"
        if idx == n - 1:
            return "rightmost"
        return "center"

    def _prompt_for_full_frame_multi(
        self, selected: FaceBox | None, all_faces: list[FaceBox]
    ) -> str:
        """Prompt for full-frame multi-person edit with hard locality language."""
        base = str(self.cfg.get("prompt", "") or "").strip()
        pos = self._face_position_label(selected, all_faces)
        n = len(all_faces)
        return self._apply_expression_policy(
            base
            + f" There are {n} people in the first image. "
            "LOCALITY (mandatory): edit ONLY the selected person's head, face, and "
            f"hair — the {pos} person. Do not modify, morph, blend, duplicate, or "
            "ghost any other face. Preserve every other face EXACTLY as in image 1. "
            "Preserve all clothing, hands, body pose, necklines, accessories, and the "
            "entire background exactly. Do not invent a second set of eyes/nose/mouth "
            "on anyone. "
            f"Replace ONLY the head, face, and hair of the {pos} person with the "
            "identity from the second image — including that person's hairstyle from "
            "image 2. Completely remove the original hair of the selected person only. "
            "CRITICAL: the new head must be the SAME SIZE as the original head of "
            f"that {pos} person — do not enlarge it relative to the neighbors. "
            "Leave every other person completely unchanged — same faces, hair, "
            "expression, clothing, and hands. Do not add a visible seam or paste line "
            "at the neck. Match lighting and skin tone to the first image. Keep pose, "
            "camera angle, and background identical."
        )

    def _build_full_frame_freeze_mask(
        self,
        body_full: Image.Image,
        selected_face: FaceBox | None,
        all_faces: list[FaceBox],
    ) -> Image.Image:
        """Selected head+hair mask; neighbors carved out aggressively."""
        mask, _info = build_head_hair_mask(
            body_full,
            self.cache_dir,
            backend=str(self.cfg.get("head_mask_backend", "ellipse") or "ellipse"),
            face_box=selected_face,
            expand_px=int(self.cfg.get("full_frame_mask_expand_px", 12)),
            blur_px=int(self.cfg.get("full_frame_mask_blur_px", 12)),
            top_extend=float(
                self.cfg.get(
                    "full_frame_mask_top_extend",
                    self.cfg.get("mask_top_extend", 1.55),
                )
            ),
            side_extend=float(self.cfg.get("full_frame_mask_side_extend", 0.42)),
            bot_extend=float(self.cfg.get("full_frame_mask_bot_extend", 0.40)),
        )
        shrink = float(self.cfg.get("full_frame_neighbor_suppress_shrink", 1.05))
        mask = suppress_neighbor_faces_in_mask(
            mask, selected_face, all_faces, shrink=shrink
        )
        return ensure_selected_face_mask_coverage(
            mask,
            selected_face,
            min_frac=float(self.cfg.get("full_frame_min_selected_mask_frac", 0.35)),
            top_extend=float(
                self.cfg.get(
                    "full_frame_mask_top_extend",
                    self.cfg.get("mask_top_extend", 1.55),
                )
            ),
            side_extend=float(self.cfg.get("full_frame_mask_side_extend", 0.42)),
            bot_extend=float(self.cfg.get("full_frame_mask_bot_extend", 0.40)),
        )

    def _freeze_full_frame_outside_selected(
        self,
        body_full: Image.Image,
        edited: Image.Image,
        freeze_mask: Image.Image,
        selected_face: FaceBox | None,
        all_faces: list[FaceBox],
    ) -> Image.Image:
        """
        Soft-composite full-frame edit onto the original body using the selected
        head mask, then hard-restore neighbor faces from the original.

        Krea2 has no latent region/inpaint mask; this is the strongest available
        freeze for full-frame generation without reverting to crop-and-stitch.
        """
        if edited.size != body_full.size:
            edited = edited.resize(body_full.size, Image.Resampling.LANCZOS)
        if freeze_mask.size != body_full.size:
            freeze_mask = freeze_mask.resize(body_full.size, Image.Resampling.BILINEAR)

        dilate_px = int(self.cfg.get("full_frame_freeze_dilate_px", 6))
        stitch_mask = dilate_mask(freeze_mask, dilate_px) if dilate_px > 0 else freeze_mask
        # Dilate can bleed into neighbors — carve them out again.
        stitch_mask = suppress_neighbor_faces_in_mask(
            stitch_mask,
            selected_face,
            all_faces,
            shrink=float(self.cfg.get("full_frame_neighbor_suppress_shrink", 1.05)),
        )
        # Neighbor carve on tight groups can wipe the selected head — restore it.
        stitch_mask = ensure_selected_face_mask_coverage(
            stitch_mask,
            selected_face,
            min_frac=float(self.cfg.get("full_frame_min_selected_mask_frac", 0.35)),
            top_extend=float(
                self.cfg.get(
                    "full_frame_mask_top_extend",
                    self.cfg.get("mask_top_extend", 1.55),
                )
            ),
            side_extend=float(self.cfg.get("full_frame_mask_side_extend", 0.42)),
            bot_extend=float(self.cfg.get("full_frame_mask_bot_extend", 0.40)),
        )
        w, h = body_full.size
        feather = int(self.cfg.get("full_frame_freeze_feather_px", 12))
        out = feathered_soft_composite(
            body_full,
            edited,
            stitch_mask,
            (0, 0, w, h),
            extra_blur_px=feather,
        )
        if bool(self.cfg.get("full_frame_hard_freeze_neighbors", True)):
            out = hard_freeze_neighbor_faces(
                out,
                body_full,
                selected_face,
                all_faces,
                pad_frac=float(self.cfg.get("full_frame_neighbor_pad_frac", 0.22)),
                expand_top_frac=float(self.cfg.get("full_frame_neighbor_top_frac", 0.35)),
                protect_expand=float(self.cfg.get("full_frame_protect_expand", 1.20)),
            )
        post_match = float(
            self.cfg.get("full_frame_post_color_match_strength", 0.45) or 0.0
        )
        if post_match > 0:
            out = lab_histogram_match_face(
                out, body_full, stitch_mask, strength=post_match
            )
        print(
            f"[krea2 blend] mode=full_frame method=lab_match+feather "
            f"feather_px={feather} lab_strength={post_match}",
            flush=True,
        )
        return out

    def _refine_full_frame_face(
        self,
        out: Image.Image,
        face_crop: Image.Image,
        *,
        rt: NodeRuntime,
        bundle: dict,
        edit_cache_info: dict,
        div_by: int,
        timings: dict[str, float],
        base_seed: int | None,
        prompt: str,
        out_dir: Path | None,
        stitch_debug: dict[str, Image.Image],
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Crop_stitch-quality face refine pass on top of a full_frame result.

        full_frame gives correct body/head proportions but spends the same
        pixel + VLM-grounding budget on the whole scene, so the face itself
        is rendered (and grounded) at a fraction of crop_stitch's effective
        resolution -- e.g. a full-body photo with face_height_frac=0.12 in a
        ~1.25MP scene renders the face at roughly 1/3 the pixel height a
        768px head crop gets. This second pass re-detects the (already
        proportion-correct) head in ``out``, crops+masks it exactly like the
        normal single-person crop_stitch path, re-samples just that region at
        native crop_stitch resolution/grounding, and stitches it back with
        the same feathered composite + LAB match crop_stitch already uses --
        so only pixels inside the head+hair mask can change; body/limb
        geometry from the full_frame pass is untouched by construction.
        """
        diag: dict[str, Any] = {"applied": False}
        if not bool(self.cfg.get("full_frame_face_refine", True)):
            diag["reason"] = "disabled"
            return out, diag

        rgb = pil_to_rgb_np(out)
        selected, faces = select_face_box(
            rgb,
            self.cache_dir,
            index=int(self.cfg.get("body_face_index", 0) or 0),
            policy=str(self.cfg.get("body_face_policy", "largest") or "largest"),
        )
        if selected is None:
            diag["reason"] = "no_face_on_full_frame_output"
            return out, diag

        flags = self._tight_crop_flags(out, selected, faces)
        built = self._build_scene_person(
            out,
            face_crop,
            selected,
            div_by=div_by,
            use_tight=False,
            top_ext=float(flags["top_ext"]),
            side_ext=float(flags["side_ext"]),
            bot_ext=float(flags["bot_ext"]),
            expand_px=int(flags["expand_px"]),
            crop_pad=int(flags["crop_pad"]),
            all_faces=faces,
            isolate_selected=False,
        )
        refine_scene = built["scene"]
        refine_face_h = float(selected.height)
        crop_box = built["box"]
        crop_h = max(1, crop_box[3] - crop_box[1]) if crop_box else 1
        refine_scale = refine_scene.size[1] / float(crop_h)
        diag.update(
            {
                "full_frame_face_height_px": round(refine_face_h, 1),
                "refine_crop_scene_size": list(refine_scene.size),
                "refine_face_height_px_effective": round(
                    refine_face_h * refine_scale, 1
                ),
                "resolution_gain_x": round(max(refine_scale, 1e-6), 3),
            }
        )

        # This pass intentionally reuses the already-loaded bundle (same
        # checkpoint/LoRA as the full_frame first pass): reloading a
        # different LoRA mid-run means a fresh UNETLoader, which is a real
        # VRAM/latency cost. Opt in via full_frame_face_refine_reload_lora
        # to reload with the crop_stitch identity LoRA for max fidelity.
        refine_bundle = bundle
        if bool(self.cfg.get("full_frame_face_refine_reload_lora", False)):
            _mode_before = self.cfg.get("multi_person_edit_mode")
            try:
                self.cfg["multi_person_edit_mode"] = "crop_stitch"
                refine_bundle = self._load_models(rt, timings)
            except Exception as exc:
                diag["reload_lora_failed"] = f"{type(exc).__name__}: {exc}"
                refine_bundle = bundle
            finally:
                self.cfg["multi_person_edit_mode"] = _mode_before

        _mode_before2 = self.cfg.get("multi_person_edit_mode")
        try:
            self.cfg["multi_person_edit_mode"] = "crop_stitch"
            refine_sample = self._sample_edit(
                rt,
                refine_bundle,
                refine_scene,
                built["person"],
                timings,
                prompt=prompt,
                edit_cache_info=edit_cache_info,
                seed=base_seed,
                ref_boost_mask=None,
            )
        except Exception as exc:
            diag["reason"] = f"refine_sample_failed:{type(exc).__name__}: {exc}"
            return out, diag
        finally:
            self.cfg["multi_person_edit_mode"] = _mode_before2

        refined = self._stitch_edited(
            out,
            refine_sample["edited"],
            built["mask"],
            built["box"],
            built["crop_content_box"],
            color_ref=out,
            multi_person=False,
            original_scene=refine_scene,
            debug_stages=stitch_debug,
            skip_head_clamp=False,
        )
        diag.update(
            {
                "applied": True,
                "lora_reloaded": refine_bundle is not bundle,
                "steps": refine_sample.get("steps"),
                "grounding_px": refine_sample.get("grounding_px"),
            }
        )
        print(
            f"[krea2 full_frame_refine] applied=True "
            f"full_frame_face_h={diag['full_frame_face_height_px']}px "
            f"refine_effective_face_h={diag['refine_face_height_px_effective']}px "
            f"gain={diag['resolution_gain_x']}x "
            f"lora_reloaded={diag['lora_reloaded']}",
            flush=True,
        )
        if out_dir is not None:
            self._save_debug(out_dir, "debug_full_frame_refine_crop.png", refine_scene)
            self._save_debug(
                out_dir, "debug_full_frame_refine_edited.png", refine_sample["edited"]
            )
        return refined, diag

    @staticmethod
    def _node_accepts_kwarg(rt: NodeRuntime, node_name: str, key: str) -> bool:
        cls = rt.mappings.get(node_name)
        if cls is None:
            return False
        try:
            types = cls.INPUT_TYPES()  # type: ignore[attr-defined]
        except Exception:
            return False
        opt = types.get("optional") or {}
        req = types.get("required") or {}
        return key in opt or key in req

    def _log_selected_face_delta(
        self,
        label: str,
        body: Image.Image,
        edited: Image.Image,
        selected: FaceBox,
    ) -> dict[str, float]:
        """MSE/PSNR in the selected face box — proves whether the sampler swapped."""
        import sys

        import numpy as np

        if edited.size != body.size:
            edited = edited.resize(body.size, Image.Resampling.LANCZOS)
        a = np.asarray(body.convert("RGB"), dtype=np.float64)
        b = np.asarray(edited.convert("RGB"), dtype=np.float64)
        x0, y0 = max(0, selected.x0), max(0, selected.y0)
        x1, y1 = min(body.size[0], selected.x1), min(body.size[1], selected.y1)
        if x1 <= x0 or y1 <= y0:
            return {"mse": -1.0, "psnr": -1.0}
        patch_a = a[y0:y1, x0:x1]
        patch_b = b[y0:y1, x0:x1]
        mse = float(np.mean((patch_a - patch_b) ** 2))
        psnr = 99.0 if mse <= 1e-12 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
        print(
            f"[krea2 face_delta] {label} selected_box=[{x0},{y0},{x1},{y1}] "
            f"mse={mse:.2f} psnr={psnr:.2f} "
            f"(high mse / low psnr ⇒ sampler changed the face)",
            flush=True,
        )
        return {"mse": round(mse, 3), "psnr": round(psnr, 3)}

    def _pil_content_stats(self, im: Image.Image) -> dict[str, Any]:
        """Compact fingerprint for scene/person tensors entering Krea2."""
        import hashlib

        import numpy as np

        rgb = np.asarray(im.convert("RGB"))
        mean = rgb.astype(np.float64).mean(axis=(0, 1))
        # Non-near-black coverage — distinguishes full-bleed vs tiny stickers.
        nonblack = float((rgb.astype(np.float32).mean(axis=2) > 14.0).mean())
        digest = hashlib.sha1(rgb.tobytes()).hexdigest()[:12]
        return {
            "size": list(im.size),
            "sha1_12": digest,
            "mean_rgb": [round(float(x), 2) for x in mean],
            "nonblack_frac": round(nonblack, 4),
        }

    def _log_krea2_conditioning(
        self,
        *,
        edit_mode: str,
        scene: Image.Image,
        person: Image.Image,
        prompt: str,
        bundle: dict,
        seed: int,
        ref_boost_mask: Image.Image | None,
    ) -> dict[str, Any]:
        """Log every conditioning input that enters shared `_sample_edit`."""
        import hashlib
        import sys

        scene_s = self._pil_content_stats(scene)
        person_s = self._pil_content_stats(person)
        prompt_sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        load = bundle.get("load_meta") or {}
        info = {
            "edit_mode": edit_mode,
            "scene": scene_s,
            "person": person_s,
            "prompt_len": len(prompt),
            "prompt_sha1_12": prompt_sha,
            "prompt_preview": prompt[:120].replace("\n", " "),
            "ref_boost_mask": ref_boost_mask is not None,
            "checkpoint": load.get("checkpoint"),
            "loras_loaded": list(load.get("loras_loaded") or []),
            "lora_strengths": dict(load.get("lora_strengths") or {}),
            "steps": int(self.cfg.get("steps", 8)),
            "cfg": float(self.cfg.get("cfg", 1.0)),
            "seed": int(seed),
            "denoise": float(self.cfg.get("denoise", 1.0)),
            "ref_boost": float(self.cfg.get("ref_boost", 4.0)),
            "ref_boost_a": float(self.cfg.get("ref_boost_a", 1.0)),
            "fit_mode": str(self.cfg.get("fit_mode", "fit") or "fit"),
            "grounding_px": int(self.cfg.get("grounding_px", 768)),
            "sampler_name": str(self.cfg.get("sampler_name", "euler") or "euler"),
            "scheduler": str(self.cfg.get("scheduler", "simple") or "simple"),
            "timestep_shift_mu": float(self.cfg.get("timestep_shift_mu", 1.15) or 1.15),
        }
        print(
            f"[krea2 conditioning] mode={edit_mode} "
            f"scene={scene_s['size']} sha={scene_s['sha1_12']} "
            f"nonblack={scene_s['nonblack_frac']} "
            f"person={person_s['size']} sha={person_s['sha1_12']} "
            f"nonblack={person_s['nonblack_frac']} "
            f"ref_boost_mask={info['ref_boost_mask']} "
            f"prompt_sha={prompt_sha} prompt_len={info['prompt_len']} "
            f"ckpt={info['checkpoint']} loras={info['loras_loaded']} "
            f"strengths={info['lora_strengths']} "
            f"steps={info['steps']} cfg={info['cfg']} seed={info['seed']} "
            f"denoise={info['denoise']} ref_boost={info['ref_boost']}/"
            f"{info['ref_boost_a']} fit={info['fit_mode']} "
            f"grounding_px={info['grounding_px']} "
            f"sampler={info['sampler_name']}/{info['scheduler']} "
            f"shift_mu={info['timestep_shift_mu']}",
            flush=True,
        )
        print(
            f"[krea2 conditioning] prompt_preview={info['prompt_preview']!r}",
            flush=True,
        )
        return info

    def _build_full_frame_inputs(
        self,
        body_full: Image.Image,
        face_crop: Image.Image,
        *,
        div_by: int,
        selected_face: FaceBox | None,
        all_faces: list[FaceBox],
    ) -> dict[str, Any]:
        """Full-body scene + identity ref matching crop_stitch person prep.

        Unlike crop_stitch (head crop @ long-side 768 ≈ 0.59MP), full_frame
        passes the entire photo as Krea2 image-1. Author guidance is ~1–1.5MP
        for two-person context — see ``full_frame_target_mp``.
        """
        max_ff = int(
            self.cfg.get(
                "full_frame_max_dim",
                self.cfg.get(
                    "multi_full_frame_max_dim",
                    self.cfg.get("max_body_dim", 1024),
                ),
            )
        )
        target_mp = self.cfg.get("full_frame_target_mp", 1.25)
        if target_mp is not None:
            scene = resize_to_megapixels(
                body_full.convert("RGB"),
                target_mp=float(target_mp),
                min_mp=float(self.cfg.get("full_frame_min_mp", 1.0)),
                max_mp=float(self.cfg.get("full_frame_max_mp", 1.5)),
                max_dim=max_ff,
                div_by=div_by,
                allow_upscale=bool(self.cfg.get("full_frame_allow_upscale", False)),
            )
        else:
            scene = resize_max_keep_ar(body_full.convert("RGB"), max_ff, div_by=div_by)

        # Identity conditioning parity with successful single-person crop_stitch:
        # that path uses resize_contain (scale_match gated off for single).
        # place_face_at_height_frac stickers were a full_frame-only divergence
        # that starved identity tokens on group shots.
        match_crop_id = bool(self.cfg.get("full_frame_match_crop_identity", True))
        face_h_frac = 0.22
        if selected_face is not None and body_full.size[1] > 0:
            sy = scene.size[1] / float(body_full.size[1])
            face_h_frac = (selected_face.height * sy) / float(max(1, scene.size[1]))
        scale_factor = float(self.cfg.get("identity_scale_factor", 0.90))
        if match_crop_id or not bool(self.cfg.get("identity_scale_match", True)):
            person = resize_contain(face_crop.convert("RGB"), scene.size, fill=(0, 0, 0))
            person_prep = "resize_contain"
        else:
            person = place_face_at_height_frac(
                face_crop.convert("RGB"),
                scene.size,
                height_frac=face_h_frac * scale_factor,
                fill=(0, 0, 0),
                max_height_frac=0.45,
            )
            person_prep = "place_face_at_height_frac"

        freeze_outside = bool(self.cfg.get("full_frame_freeze_outside", True))
        freeze_mask = None
        if freeze_outside and selected_face is not None:
            freeze_mask = self._build_full_frame_freeze_mask(
                body_full, selected_face, all_faces
            )

        # Default ON for full_frame — crop_stitch never passes ref_boost_mask.
        ref_boost_mask = None
        if bool(self.cfg.get("full_frame_ref_boost_mask", True)):
            ref_boost_mask = identity_face_boost_mask(person, self.cache_dir)

        # ── Debug: dump both masks as PNGs so polarity can be inspected visually ──
        # Always runs on full_frame path; download from /tmp/headswap_debug/ in Colab.
        # White=1 (selected head / boosted region), Black=0 (frozen / not boosted).
        try:
            import os as _os
            import time as _time
            _dbg_dir = "/tmp/headswap_debug"
            _os.makedirs(_dbg_dir, exist_ok=True)
            _ts = int(_time.time())
            _sel = selected_face
            if freeze_mask is not None:
                freeze_mask.save(f"{_dbg_dir}/freeze_mask_{_ts}.png")
                print(
                    f"[krea2 debug] freeze_mask saved → {_dbg_dir}/freeze_mask_{_ts}.png  "
                    f"white=selected_head(swap preserved), black=background(restored to original)  "
                    f"size={freeze_mask.size}  "
                    f"selected_box={[_sel.x0,_sel.y0,_sel.x1,_sel.y1] if _sel else None}",
                    flush=True,
                )
            if ref_boost_mask is not None:
                ref_boost_mask.save(f"{_dbg_dir}/ref_boost_mask_person_{_ts}.png")
                print(
                    f"[krea2 debug] ref_boost_mask saved → {_dbg_dir}/ref_boost_mask_person_{_ts}.png  "
                    f"white=boosted_identity_region  size={ref_boost_mask.size}  "
                    f"NOTE: built from person-image coords, NOT scene coords",
                    flush=True,
                )
        except Exception as _e:
            print(f"[krea2 debug] mask dump failed: {_e}", flush=True)

        diag = {
            "faces_detected": len(all_faces),
            "selected_box": (
                None
                if selected_face is None
                else [
                    selected_face.x0,
                    selected_face.y0,
                    selected_face.x1,
                    selected_face.y1,
                ]
            ),
            "edit_mode": "full_frame",
            "scene_size": list(scene.size),
            "scene_megapixels": round((scene.size[0] * scene.size[1]) / 1_000_000.0, 4),
            "person_size": list(person.size),
            "body_size": list(body_full.size),
            "full_frame_target_mp": (
                None
                if self.cfg.get("full_frame_target_mp") is None
                else float(self.cfg.get("full_frame_target_mp"))
            ),
            "face_position": self._face_position_label(selected_face, all_faces),
            "face_height_frac_scene": round(float(face_h_frac), 4),
            "person_prep": person_prep,
            "full_frame_match_crop_identity": match_crop_id,
            "identity_scale_match": bool(self.cfg.get("identity_scale_match", True)),
            "full_frame_freeze_outside": freeze_outside,
            "full_frame_ref_boost_mask": ref_boost_mask is not None,
            "multi_person": True,
            "use_tight": False,
            "isolate_selected": False,
            "face_white_bg_applied": False,
        }
        return {
            "scene": scene,
            "person": person,
            "freeze_mask": freeze_mask,
            "ref_boost_mask": ref_boost_mask,
            "diag": diag,
        }

    def _sample_edit(
        self,
        rt: NodeRuntime,
        bundle: dict,
        scene: Image.Image,
        person: Image.Image,
        timings: dict[str, float],
        *,
        prompt: str,
        edit_cache_info: dict,
        seed: int | None = None,
        ref_boost_mask: Image.Image | None = None,
    ) -> dict[str, Any]:
        """One Krea2 dual-ref sample → decoded PIL. Shared by single + swap-all."""
        import torch
        from headswap.comfy.krea2_edit_fast import clear_krea2_edit_static_cache

        clear_krea2_edit_static_cache()

        w, h = scene.size
        neg = str(self.cfg.get("negative_prompt", "") or "")
        grounding_px = int(self.cfg.get("grounding_px", 768))
        ref_boost = float(self.cfg.get("ref_boost", 4.0))
        ref_boost_a = float(self.cfg.get("ref_boost_a", 1.0))
        # Full-frame A/B only: optional ref_boost override (crop_stitch ignores).
        _ff_mode = str(
            self.cfg.get("multi_person_edit_mode", "crop_stitch") or "crop_stitch"
        ).strip().lower()
        _ff_rb = self.cfg.get("full_frame_ref_boost")
        if _ff_rb is not None and _ff_mode in ("full_frame", "fullframe", "full"):
            ref_boost = float(_ff_rb)
        # Full-frame's scene is the whole photo, not a head crop, so the same
        # grounding_px squeezes a much smaller face into the VLM's grounding
        # encode than crop_stitch's tight crop does. Let full_frame use a
        # higher grounding_px by default to partially compensate.
        _ff_grounding_px = self.cfg.get("full_frame_grounding_px")
        if _ff_grounding_px is not None and _ff_mode in ("full_frame", "fullframe", "full"):
            grounding_px = int(_ff_grounding_px)
        fit_mode = str(self.cfg.get("fit_mode", "fit") or "fit")
        steps = int(self.cfg.get("steps", 8))
        cfg = float(self.cfg.get("cfg", 1.0))
        seed_i = int(self.cfg.get("seed", 46) if seed is None else seed)
        denoise = float(self.cfg.get("denoise", 1.0))
        skip_neg_vlm = bool(self.cfg.get("skip_negative_grounding", True)) and cfg <= 1.0 + 1e-6
        verbose = bool(self.cfg.get("verbose", False))
        negative_mode = "grounded_vlm"
        kernel_info = _configure_fast_kernels(prefer_flash=True)
        count_forwards = bool(self.cfg.get("debug_count_unet_forwards", False))
        torch_compile = bool(self.cfg.get("torch_compile", False))
        compile_mode = str(self.cfg.get("torch_compile_mode", "reduce-overhead"))
        used_ref_boost_mask = False

        lora_name = self.cfg.get("identity_lora_name", "krea2_identity_edit_v1_2_r64.safetensors")
        _ff_lora = self.cfg.get("full_frame_identity_lora_name")
        if _ff_lora and _ff_mode in ("full_frame", "fullframe", "full"):
            lora_name = str(_ff_lora)

        import sys
        print(
            f"[krea2 sampling] mode={_ff_mode} lora={lora_name} "
            f"ref_boost={ref_boost} grounding_px={grounding_px} "
            f"steps={steps} cfg={cfg} seed={seed_i}",
            flush=True,
        )

        with torch.no_grad():
            scene_t = pil_to_comfy_tensor(scene, torch)
            person_t = pil_to_comfy_tensor(person, torch)

            with _stage(timings, "vae_encode"):
                scene_lat = rt.call("VAEEncode", pixels=scene_t, vae=bundle["vae"])
                person_lat = rt.call("VAEEncode", pixels=person_t, vae=bundle["vae"])

            with _stage(timings, "model_patch"):
                patch_kwargs: dict[str, Any] = {
                    "model": bundle["model"],
                    "source_latent": get_value_at_index(scene_lat, 0),
                    "source_latent_b": get_value_at_index(person_lat, 0),
                    "ref_boost": ref_boost,
                    "ref_boost_a": ref_boost_a,
                    "fit_mode": fit_mode,
                    "vae": bundle["vae"],
                    "source_image": scene_t,
                    "source_image_b": person_t,
                }
                # Identity attention boost localized to face — NOT an edit freeze.
                if (
                    ref_boost_mask is not None
                    and self._node_accepts_kwarg(rt, "Krea2EditModelPatch", "ref_boost_mask")
                ):
                    patch_kwargs["ref_boost_mask"] = pil_mask_to_comfy_tensor(
                        ref_boost_mask, torch
                    )
                    used_ref_boost_mask = True
                patched = rt.call("Krea2EditModelPatch", **patch_kwargs)
                model = get_value_at_index(patched, 0)

            compile_info = _maybe_torch_compile_diffusion(
                model, enabled=torch_compile, mode=compile_mode
            )

            with _stage(timings, "grounded_encode_positive"):
                pos = rt.call(
                    "Krea2EditGroundedEncode",
                    clip=bundle["clip"],
                    prompt=prompt,
                    image=scene_t,
                    image_b=person_t,
                    grounding_px=grounding_px,
                )
                positive = get_value_at_index(pos, 0)

            if skip_neg_vlm:
                negative, negative_mode = self._cheap_negative(
                    rt, positive, bundle["clip"], timings
                )
            else:
                with _stage(timings, "grounded_encode_negative"):
                    neg_enc = rt.call(
                        "Krea2EditGroundedEncode",
                        clip=bundle["clip"],
                        prompt=neg,
                        image=scene_t,
                        image_b=person_t,
                        grounding_px=grounding_px,
                    )
                    negative = get_value_at_index(neg_enc, 0)
                negative_mode = "grounded_vlm"

            if rt.has("EmptySD3LatentImage"):
                empty = rt.call(
                    "EmptySD3LatentImage", width=w, height=h, batch_size=1
                )
                latent = get_value_at_index(empty, 0)
                empty_node = "EmptySD3LatentImage"
            else:
                latent = get_value_at_index(scene_lat, 0)
                empty_node = "scene_latent_fallback"

            mu_shift = float(self.cfg.get("timestep_shift_mu", 1.15) or 1.15)
            apply_shift = bool(self.cfg.get("apply_timestep_shift", True))
            if apply_shift and rt.has("ModelSamplingFlux"):
                model = get_value_at_index(
                    rt.call(
                        "ModelSamplingFlux",
                        model=model,
                        max_shift=mu_shift,
                        base_shift=float(self.cfg.get("timestep_base_shift", 0.5) or 0.5),
                        width=w,
                        height=h,
                    ),
                    0,
                )
            elif apply_shift:
                mu_shift = None

            use_full_load = bool(self.cfg.get("force_full_load", True))
            sampler_name = str(self.cfg.get("sampler", "euler"))
            scheduler_name = str(self.cfg.get("scheduler", "simple"))

            sampling_diag = _print_sampling_diagnostics(
                steps=steps,
                cfg=cfg,
                sampler=sampler_name,
                scheduler=scheduler_name,
                force_full_load=use_full_load,
                denoise=denoise,
                width=w,
                height=h,
                mu_shift=mu_shift if apply_shift else None,
                verbose=verbose,
            )
            sampling_diag["kernels"] = kernel_info
            sampling_diag["torch_compile"] = compile_info
            sampling_diag["debug_count_unet_forwards"] = count_forwards
            sampling_diag["edit_forward_cache"] = edit_cache_info
            if verbose:
                print(
                    f"[krea2 kernels] actions={kernel_info.get('actions')} "
                    f"attention={kernel_info.get('attention_after', {}).get('backend')} "
                    f"edit_cache={edit_cache_info.get('installed')}"
                )

            fwd_counter, restore_fwd = _count_unet_forwards(
                model, enabled=count_forwards
            )
            cuda_mid = _cuda_util_snapshot()
            if verbose:
                print(
                    f"[krea2 sampling] pre-sample gpu alloc_mb="
                    f"{cuda_mid.get('allocated_mb')} reserved_mb="
                    f"{cuda_mid.get('reserved_mb')} free_mb={cuda_mid.get('free_mb')}"
                )
            sample_t0 = time.perf_counter()
            with _stage(timings, "diffusion_sampling"):
                with force_sampling_full_load(
                    models=(model,), enabled=use_full_load
                ):
                    with _silence_krea2edit_step_prints():
                        samples = rt.call(
                            "KSampler",
                            model=model,
                            seed=seed_i,
                            steps=steps,
                            cfg=cfg,
                            sampler_name=sampler_name,
                            scheduler=scheduler_name,
                            positive=positive,
                            negative=negative,
                            latent_image=latent,
                            denoise=denoise,
                        )
            sample_s = time.perf_counter() - sample_t0
            if restore_fwd is not None:
                restore_fwd()

            actual_forwards = int(fwd_counter.get("n") or 0)
            sec_per_step = (sample_s / steps) if steps > 0 else None
            sec_per_fwd = (
                (sample_s / actual_forwards) if actual_forwards > 0 else None
            )
            sps = f"{sec_per_step:.2f}" if sec_per_step is not None else "n/a"
            spf = f"{sec_per_fwd:.2f}" if sec_per_fwd is not None else "n/a"
            if verbose:
                print(
                    f"[krea2 sampling result] wall={sample_s:.2f}s  "
                    f"configured_steps={steps}  actual_unet_forwards="
                    f"{actual_forwards if count_forwards else 'skipped'}  "
                    f"sec/step≈{sps}  sec/forward≈{spf}  "
                    f"fwd_wrap_ok={fwd_counter.get('wrapped')}"
                )
            cuda_after = _cuda_util_snapshot()
            sampling_diag["actual_unet_forwards"] = (
                actual_forwards if count_forwards else None
            )
            sampling_diag["sample_wall_s"] = round(sample_s, 4)
            sampling_diag["sec_per_step"] = (
                round(sec_per_step, 4) if sec_per_step is not None else None
            )
            sampling_diag["cuda_after_sample"] = cuda_after

            with _stage(timings, "vae_decode"):
                decoded = rt.call(
                    "VAEDecode",
                    samples=get_value_at_index(samples, 0),
                    vae=bundle["vae"],
                )
                edited = comfy_tensor_to_pil(get_value_at_index(decoded, 0))

        return {
            "edited": edited,
            "sampling_diag": sampling_diag,
            "negative_mode": negative_mode,
            "empty_node": empty_node,
            "use_full_load": use_full_load,
            "skip_neg_vlm": skip_neg_vlm,
            "steps": steps,
            "cfg": cfg,
            "seed": seed_i,
            "ref_boost": ref_boost,
            "ref_boost_a": ref_boost_a,
            "ref_boost_mask_used": used_ref_boost_mask,
            "grounding_px": grounding_px,
            "fit_mode": fit_mode,
            "verbose": verbose,
        }

    def _maybe_procrustes_edited_crop(
        self,
        edited: Image.Image,
        body_full: Image.Image,
        box: tuple[int, int, int, int],
        selected_face: FaceBox | None,
        face_prep_diag: dict[str, Any],
        crop_content_box: tuple[int, int, int, int] | None = None,
    ) -> tuple[Image.Image, bool]:
        """Optional Procrustes on crop_stitch edited crop before stitch.

        Returns ``(image, applied)``. ``applied`` lets the stitch stage skip its
        own head-scale clamp so the head is never rescaled twice.
        """
        if not bool(self.cfg.get("enable_procrustes_correction", False)):
            return edited, False
        aligned, info = procrustes_align_edited_crop_to_body_box(
            edited,
            body_full,
            box,
            self.cache_dir,
            crop_content_box=crop_content_box,
            prefer_body_box=selected_face,
            min_inliers=int(self.cfg.get("procrustes_min_inliers", 3)),
            min_scale=float(self.cfg.get("procrustes_min_scale", 0.40)),
            max_scale=float(self.cfg.get("procrustes_max_scale", 2.50)),
            max_rotation_deg=float(self.cfg.get("procrustes_max_rotation_deg", 12.0)),
            max_translate_frac=float(
                self.cfg.get("procrustes_max_translate_frac", 0.25)
            ),
            max_residual_frac=float(self.cfg.get("procrustes_max_residual_frac", 0.06)),
        )
        face_prep_diag["procrustes_correction"] = info
        import sys

        if info.get("procrustes"):
            print(
                "[krea2] procrustes_correction "
                f"scale={info.get('scale')} rot_deg={info.get('rotation_deg')} "
                f"t={info.get('translation')} inliers={info.get('inliers')} "
                f"residual_px={info.get('residual_px')} "
                f"content_box={info.get('content_box')}",
                flush=True,
            )
            return aligned, True
        print(
            "[krea2] procrustes_correction skipped: "
            f"{info.get('procrustes_reason')}",
            flush=True,
        )
        return edited, False

    def _stitch_edited(
        self,
        canvas: Image.Image,
        edited: Image.Image,
        mask: Image.Image,
        box: tuple[int, int, int, int],
        crop_content_box: tuple[int, int, int, int] | None,
        *,
        color_ref: Image.Image,
        multi_person: bool = False,
        original_scene: Image.Image | None = None,
        debug_stages: dict[str, Image.Image] | None = None,
        skip_head_clamp: bool = False,
    ) -> Image.Image:
        edited_crop = edited
        if crop_content_box is not None:
            ox, oy, cw, ch = crop_content_box
            edited_crop = edited_crop.crop((ox, oy, ox + cw, oy + ch))
        # Clamp / multi stitch boosts only when NOT on SPP-CC parity path.
        stitch_multi = bool(multi_person) and not self._single_person_parity()
        # Clamp oversized heads before composite (group-shot failure mode).
        # Suppressed when Procrustes already resized the head — two independent
        # scale+translate warps stacked produce a doubled / displaced head.
        if stitch_multi and skip_head_clamp:
            import sys

            print(
                "[krea2] head-scale clamp skipped (procrustes already applied)",
                flush=True,
            )
        if stitch_multi and not skip_head_clamp and bool(
            self.cfg.get("clamp_edited_head_scale", True)
        ):
            ref_scene = original_scene
            if ref_scene is None:
                ref_scene = canvas.crop(box).resize(
                    edited_crop.size, Image.Resampling.LANCZOS
                )
            elif ref_scene.size != edited_crop.size:
                ref_scene = ref_scene.resize(
                    edited_crop.size, Image.Resampling.LANCZOS
                )
            edited_crop, clamp_info = clamp_edited_head_scale(
                ref_scene,
                edited_crop,
                self.cache_dir,
                max_height_ratio=float(
                    self.cfg.get("max_edited_head_height_ratio", 1.08)
                ),
                target_ratio=float(self.cfg.get("target_edited_head_height_ratio", 0.98)),
            )
            import sys

            if clamp_info.get("clamped"):
                print(
                    f"[krea2] clamped oversized head "
                    f"ratio {clamp_info['ratio_before']:.2f}→{clamp_info['ratio_after']:.2f} "
                    f"shrink={clamp_info['shrink']:.3f}",
                    flush=True,
                )
        stitch_mask = mask
        # Dilate before feather so the opaque core fully covers the old head
        # (stops original face/ear ghosting next to neighbors).
        if stitch_multi:
            dilate_px = int(self.cfg.get("multi_stitch_mask_dilate_px", 14))
            if dilate_px > 0:
                stitch_mask = dilate_mask(mask, dilate_px)
        # Multi-person: stronger feather + LAB match — jaw/neck seams are the
        # dominant failure mode when the donor face lighting ≠ body flash.
        if stitch_multi:
            feather = int(
                self.cfg.get(
                    "multi_stitch_feather_px",
                    max(14, int(self.cfg.get("stitch_feather_px", 10)) + 4),
                )
            )
            post_match = float(
                self.cfg.get(
                    "multi_post_color_match_strength",
                    max(0.55, float(self.cfg.get("post_color_match_strength", 0.35) or 0.0)),
                )
            )
        else:
            feather = int(self.cfg.get("stitch_feather_px", 10))
            post_match = float(self.cfg.get("post_color_match_strength", 0.35) or 0.0)
        stitched = feathered_soft_composite(
            canvas,
            edited_crop,
            stitch_mask,
            box,
            extra_blur_px=feather,
        )
        if debug_stages is not None:
            debug_stages["stitch_mask"] = stitch_mask.convert("L").copy()
            debug_stages["composite_before_lab"] = stitched.copy()
        stitched = lab_histogram_match_face(
            stitched, color_ref, stitch_mask, strength=post_match
        )
        if debug_stages is not None:
            debug_stages["composite_after_lab"] = stitched.copy()
        # Stage E: optional narrow-band seam refine (face interior locked).
        if bool(self.cfg.get("seam_refine", False)):
            # stitch_mask is crop-local; paste onto full-canvas mask for refine.
            full_mask = Image.new("L", canvas.size, 0)
            x0, y0, x1, y1 = box
            mw, mh = x1 - x0, y1 - y0
            m_r = stitch_mask.convert("L").resize((mw, mh), Image.Resampling.BILINEAR)
            full_mask.paste(m_r, (x0, y0))
            stitched = narrow_band_seam_refine(
                color_ref,
                stitched,
                full_mask,
                erode_px=int(self.cfg.get("seam_refine_erode_px", 8)),
                dilate_px=int(self.cfg.get("seam_refine_dilate_px", 16)),
                strength=float(self.cfg.get("seam_refine_strength", 0.85) or 0.0),
                blur_px=int(self.cfg.get("seam_refine_blur_px", 6)),
            )
        return stitched


    def _run_multi_align_paste(
        self,
        body_full: Image.Image,
        face: Image.Image,
        *,
        selected_face: FaceBox | None,
        all_faces: list[FaceBox],
        rt: NodeRuntime,
        bundle: dict,
        timings: dict[str, float],
        edit_cache_info: dict,
        out_dir: Path | None,
        save_debug: bool,
        base_seed: int,
        t0: float,
    ) -> PipelineResult:
        """Architecture B: align-paste (+ optional masked Krea2 refine) for multi-person."""
        import sys

        refine_fn = None
        do_refine = bool(self.cfg.get("align_paste_krea2_refine", True))
        if do_refine:
            refine_prompt = self._apply_expression_policy(
                str(
                    self.cfg.get(
                        "align_paste_refine_prompt",
                        "Blend the pasted face naturally into the first image. "
                        "CRITICAL: the person must look in exactly the same direction "
                        "as in the first image — same head yaw, eye gaze, and face angle. "
                        "Do not rotate the head toward the camera or front-face the identity. "
                        "Preserve the facial expression, mouth shape, eye gaze, and head "
                        "pose from the first image exactly. Use only facial identity from "
                        "the second image. Do not change clothing, collar, hair silhouette "
                        "outside the face, or neighboring people. Match lighting to the first image.",
                    )
                ).strip()
            )
            def refine_fn(composite_crop, id_matte, face_mask_crop):  # noqa: ANN001
                scene = composite_crop.convert("RGB")
                person = resize_contain(id_matte.convert("RGB"), scene.size, fill=(0, 0, 0))
                # Temporarily lower steps for refine if configured.
                old_steps = self.cfg.get("steps")
                refine_steps = self.cfg.get("align_paste_refine_steps")
                if refine_steps is not None:
                    self.cfg["steps"] = int(refine_steps)
                try:
                    sample_meta = self._sample_edit(
                        rt,
                        bundle,
                        scene,
                        person,
                        timings,
                        prompt=refine_prompt,
                        edit_cache_info=edit_cache_info,
                        seed=base_seed,
                        ref_boost_mask=None,
                    )
                finally:
                    if refine_steps is not None:
                        if old_steps is None:
                            self.cfg.pop("steps", None)
                        else:
                            self.cfg["steps"] = old_steps
                edited = sample_meta["edited"]
                if edited.size != scene.size:
                    edited = edited.resize(scene.size, Image.Resampling.LANCZOS)
                # Keep geometry lock: only blend refined pixels inside face mask.
                # Return (blended, raw) so debug can fork model ID vs soft-lock.
                blended = feathered_soft_composite(
                    scene,
                    edited,
                    face_mask_crop,
                    (0, 0, scene.size[0], scene.size[1]),
                    extra_blur_px=int(self.cfg.get("align_paste_stitch_feather_px", 10)),
                )
                return blended, edited

        with _stage(timings, "align_paste"):
            trace = None
            want_trace = bool(save_debug) or bool(
                self.cfg.get("identity_stage_trace", True)
            )
            if want_trace and out_dir is not None:
                sel_idx = None
                if selected_face is not None and all_faces:
                    for i, f in enumerate(all_faces):
                        if (
                            f.x0 == selected_face.x0
                            and f.y0 == selected_face.y0
                            and f.x1 == selected_face.x1
                            and f.y1 == selected_face.y1
                        ):
                            sel_idx = i
                            break
                trace = IdentityStageTrace(
                    Path(out_dir) / "identity_stages",
                    donor=face,
                    body=body_full,
                    selected=selected_face,
                    selected_index=sel_idx,
                    cache_dir=self.cache_dir,
                    enabled=True,
                )
            ap = run_align_paste_swap(
                body_full,
                face,
                self.cache_dir,
                selected_face=selected_face,
                all_faces=all_faces,
                cfg=self.cfg,
                refine_fn=refine_fn if do_refine else None,
                identity_trace=trace,
            )
        out = ap["image"]
        timings["_multi_person"] = 1.0
        timings["_body_face_count"] = float(len(all_faces))
        timings["_edit_mode_align_paste"] = 1.0
        timings["_align_paste_refine"] = 1.0 if ap["refine_meta"].get("refine_applied") else 0.0

        print(
            f"[krea2] edit_mode=geometry_lock faces={len(all_faces)} "
            f"paste={ap['paste_info'].get('composite_paste')} "
            f"align={ap['align_info'].get('face_alignment')} "
            f"seamless={ap['paste_info'].get('seamless_clone')} "
            f"backend={ap['align_info'].get('face_alignment_backend')} "
            f"refine={ap['refine_meta'].get('refine_applied')} "
            f"pose_relock={ap.get('pose_meta', {}).get('pose_relock')} "
            f"gates={ap.get('gates')}",
            flush=True,
        )

        dbg: dict[str, str] = {}
        if out_dir is not None and save_debug:
            with _stage(timings, "image_saving"):
                debug_imgs = {
                    "debug_body": body_full,
                    "debug_identity_matte": ap["identity_matte"],
                    "debug_align_work": ap["work_crop"],
                    "debug_align_composite": ap["composite_crop"],
                    "debug_align_refined": ap["refined_crop"],
                    "debug_align_mask": ap["face_mask"],
                    "debug_align_mask_crop": ap.get("face_mask_crop"),
                    "debug_pose_before_relock": ap.get("pose_before_relock"),
                    "debug_krea2_raw_edited": ap.get("raw_refined_crop"),
                    "debug_aligned_rgba": ap.get("aligned_rgba"),
                    "debug_final": out,
                }
                debug_imgs = {k: v for k, v in debug_imgs.items() if v is not None}
                dbg = {
                    k: v
                    for k, v in {
                        key: self._save_debug(out_dir, f"{key}.png", img)
                        for key, img in debug_imgs.items()
                    }.items()
                    if v
                }
        else:
            timings.setdefault("image_saving", 0.0)

        total_s = time.perf_counter() - t0
        verbose = bool(self.cfg.get("verbose", False))
        _print_timing_breakdown(timings, total_s, verbose=verbose)
        timing_rounded = {
            k: round(float(v), 4) for k, v in timings.items() if not k.startswith("_")
        }
        meta = {
            "pipeline": self.name,
            "edit_mode": "geometry_lock",
            "multi_person_swap_mode": "align_paste",
            "multi_person_edit_mode": str(
                self.cfg.get("multi_person_edit_mode", "crop_stitch") or "crop_stitch"
            ),
            "multi_person": True,
            "body_face_count": len(all_faces),
            "selected_box": (
                None
                if selected_face is None
                else [
                    selected_face.x0,
                    selected_face.y0,
                    selected_face.x1,
                    selected_face.y1,
                ]
            ),
            "body_size": list(body_full.size),
            "align_paste": {
                "matte_info": ap["matte_info"],
                "align_info": ap["align_info"],
                "paste_info": ap["paste_info"],
                "refine_meta": ap["refine_meta"],
                "pose_meta": ap.get("pose_meta") or {},
                "gates": ap["gates"],
                "box": list(ap["box"]),
                "identity_stage_diagnosis": (ap.get("identity_stage_report") or {}).get(
                    "diagnosis"
                ),
            },
            "checkpoint": bundle["load_meta"].get("checkpoint"),
            "loras_loaded": list(bundle["load_meta"].get("loras_loaded") or []),
            "timing_s": timing_rounded,
            "verbose": verbose,
        }
        return PipelineResult(image=out, latency_s=total_s, meta=meta, debug_paths=dbg)

    def _detect_full_body(
        self,
        body: Image.Image,
        selected_face: FaceBox | None,
        all_faces: list[FaceBox],
    ) -> dict[str, Any]:
        """Classify whether a full/half body (not just a head/portrait) is visible.

        Primary signal (always available, no extra deps): face bbox height as
        a fraction of image height, plus how much frame remains below the
        chin. Portraits/close-ups fill the frame with the face; full-body or
        half-body shots leave a large fraction of the frame for torso+legs
        below the face.

        Optional refinement: when ``rembg`` is installed, reuse the existing
        BiRefNet/rembg person matte (``segmentation._try_birefnet_mask``,
        ungated by passing ``face_box=None``) to measure the actual person
        bbox height fraction, which overrides the geometric heuristic when
        available since it looks at real body pixels instead of a proxy.
        """
        import numpy as np

        w, h = body.size
        if selected_face is None or h <= 0:
            return {
                "method": "none",
                "full_body_detected": False,
                "reason": "no_face",
            }

        face_h_frac = selected_face.height / float(h)
        below_face_frac = max(0.0, h - selected_face.y1) / float(h)
        face_h_frac_max = float(
            self.cfg.get("body_route_face_height_frac_max", 0.30)
        )
        below_frac_min = float(
            self.cfg.get("body_route_below_face_frac_min", 0.38)
        )
        heuristic_full_body = (
            face_h_frac <= face_h_frac_max and below_face_frac >= below_frac_min
        )
        result: dict[str, Any] = {
            "method": "face_geometry",
            "face_height_frac": round(float(face_h_frac), 4),
            "below_face_frac": round(float(below_face_frac), 4),
            "face_height_frac_max": face_h_frac_max,
            "below_face_frac_min": below_frac_min,
            "full_body_detected": bool(heuristic_full_body),
        }

        if bool(self.cfg.get("body_route_use_segmentation", True)):
            try:
                from headswap.segmentation import _try_birefnet_mask

                mask, skip_reason = _try_birefnet_mask(body, None, blur_px=0)
            except Exception as exc:  # pragma: no cover - defensive
                mask, skip_reason = None, f"import_failed:{exc}"
            if mask is not None:
                alpha = np.asarray(mask)
                rows = np.where((alpha > 16).any(axis=1))[0]
                if rows.size > 0:
                    person_h_frac = float(rows.max() - rows.min()) / float(h)
                    person_h_frac_min = float(
                        self.cfg.get("body_route_person_height_frac_min", 0.55)
                    )
                    result.update(
                        {
                            "method": "person_segmentation",
                            "person_height_frac": round(person_h_frac, 4),
                            "person_height_frac_min": person_h_frac_min,
                            "full_body_detected": bool(
                                person_h_frac >= person_h_frac_min
                            ),
                        }
                    )
            else:
                result["segmentation_skip_reason"] = skip_reason

        return result

    def _resolve_body_route(self, body: Image.Image) -> dict[str, Any]:
        """Auto-route to full_frame when a full/half body is detected.

        Full-frame preserves body/head proportions but has historically
        traded off some face quality vs. crop_stitch. When a full body is
        detected, we route to full_frame AND apply the same identity-boost
        overrides the (dark) lighting route uses (full v1.2 LoRA +
        ref_boost=4.0 + ref_boost_mask) so full_frame keeps as much
        crop_stitch-level identity/face quality as the existing recipe
        allows, rather than the plain default full_frame settings.

        When no full body is detected, ``self.cfg`` is left untouched and the
        existing crop_stitch path runs exactly as before — this route only
        ever *adds* a path to full_frame, it never forces crop_stitch off.
        Runs BEFORE model load (same as ``_resolve_lighting_route``) so any
        LoRA/ref_boost overrides are picked up when the models are built.
        """
        enabled = bool(self.cfg.get("enable_body_route", True))
        print(f"[krea2 body_route] enable_body_route={enabled}", flush=True)
        meta: dict[str, Any] = {
            "enable_body_route": enabled,
            "applied": False,
            "faces_detected": None,
            "full_body_detected": False,
            "route": None,
            "reason": "disabled" if not enabled else None,
        }
        if not enabled:
            return meta

        div_by = int(self.cfg.get("div_by", 16))
        probe = resize_max_keep_ar(
            body.convert("RGB"),
            int(self.cfg.get("max_body_dim", 1024)),
            div_by=div_by,
        )
        rgb = pil_to_rgb_np(probe)
        selected, faces = select_face_box(
            rgb,
            self.cache_dir,
            index=int(self.cfg.get("body_face_index", 0) or 0),
            policy=str(self.cfg.get("body_face_policy", "largest") or "largest"),
        )
        n_faces = len(faces)
        meta["faces_detected"] = n_faces
        if selected is None:
            meta["reason"] = "no_face_detected"
            print("[krea2 body_route] no_face_detected -> route unchanged", flush=True)
            return meta

        detection = self._detect_full_body(probe, selected, faces)
        meta.update(detection)
        print(
            f"[krea2 body_route] faces_detected={n_faces} "
            f"method={detection.get('method')} "
            f"face_height_frac={detection.get('face_height_frac')} "
            f"below_face_frac={detection.get('below_face_frac')} "
            f"person_height_frac={detection.get('person_height_frac')} "
            f"full_body_detected={detection.get('full_body_detected')}",
            flush=True,
        )
        if not detection.get("full_body_detected"):
            meta["route"] = "crop_stitch_unchanged"
            meta["reason"] = "no_full_body_detected"
            return meta

        self.cfg["multi_person_edit_mode"] = "full_frame"
        if not self.cfg.get("full_frame_identity_lora_name"):
            self.cfg["full_frame_identity_lora_name"] = (
                "krea2_identity_edit_v1_2.safetensors"
            )
        if self.cfg.get("full_frame_ref_boost") is None:
            self.cfg["full_frame_ref_boost"] = 4.0
        if not bool(self.cfg.get("allow_disable_full_frame_ref_boost_mask", False)):
            self.cfg["full_frame_ref_boost_mask"] = True
        meta["applied"] = True
        meta["route"] = "full_frame"
        meta["reason"] = "full_body_detected"
        meta["multi_person_edit_mode"] = str(self.cfg.get("multi_person_edit_mode"))
        print(
            "[krea2 body_route] resolved_mode=full_frame reason=full_body_detected",
            flush=True,
        )
        return meta

    def _resolve_lighting_route(
        self, body: Image.Image
    ) -> dict[str, Any]:
        """Multi-person lighting route (config-gated, default off).

        When ``enable_lighting_route`` is true and >1 face is detected:
          dark  -> multi_person_edit_mode=full_frame (A/B winner: full v1.2 + rb4)
          else  -> multi_person_edit_mode=crop_stitch (production)

        Single-person: no lighting check; config left unchanged.
        Mutates ``self.cfg['multi_person_edit_mode']`` for this run when multi.
        """
        import sys

        enabled = bool(self.cfg.get("enable_lighting_route", False))
        print(
            f"[krea2 lighting] enable_lighting_route={enabled}",
            flush=True,
        )
        meta: dict[str, Any] = {
            "enable_lighting_route": enabled,
            "applied": False,
            "faces_detected": None,
            "route": None,
            "reason": "disabled" if not enabled else None,
        }
        if not enabled:
            return meta

        div_by = int(self.cfg.get("div_by", 16))
        probe = resize_max_keep_ar(
            body.convert("RGB"),
            int(self.cfg.get("max_body_dim", 1024)),
            div_by=div_by,
        )
        rgb = pil_to_rgb_np(probe)
        selected, faces = select_face_box(
            rgb,
            self.cache_dir,
            index=int(self.cfg.get("body_face_index", 0) or 0),
            policy=str(self.cfg.get("body_face_policy", "largest") or "largest"),
        )
        n_faces = len(faces)
        meta["faces_detected"] = n_faces
        print(
            f"[krea2 lighting] faces_detected={n_faces} multi_person_branch={n_faces > 1}",
            flush=True,
        )
        if n_faces <= 1:
            meta["route"] = "single_person_unchanged"
            meta["reason"] = "single_person_skip_lighting"
            print(
                f"[krea2 lighting] lighting_route=skip faces={n_faces} "
                f"(single-person path unchanged)",
                flush=True,
            )
            return meta

        threshold = float(self.cfg.get("dark_lighting_threshold", 110.0))
        lighting = classify_lighting(rgb, threshold=threshold, box=selected, boxes=faces)
        meta.update(lighting)
        print(
            f"[krea2 lighting] raw_luminance={lighting['lighting_metric']:.2f} "
            f"region={lighting['lighting_region']} "
            f"(face_lum={lighting.get('face_luminance')}, frame_lum={lighting.get('whole_frame_luminance')}) "
            f"threshold={threshold} is_dark={lighting['is_dark']}",
            flush=True,
        )
        if lighting["is_dark"]:
            self.cfg["multi_person_edit_mode"] = "full_frame"
            # Ensure A/B winner (d_ff_full_rb4) overrides are present when routing.
            if not self.cfg.get("full_frame_identity_lora_name"):
                self.cfg["full_frame_identity_lora_name"] = (
                    "krea2_identity_edit_v1_2.safetensors"
                )
            if self.cfg.get("full_frame_ref_boost") is None:
                self.cfg["full_frame_ref_boost"] = 4.0
            # Lighting-routed full_frame always wants identity boost (do not keep
            # a stale notebook False from FULL_FRAME_REF_BOOST_MASK).
            if not bool(
                self.cfg.get("allow_disable_full_frame_ref_boost_mask", False)
            ):
                self.cfg["full_frame_ref_boost_mask"] = True
            route = "full_frame"
            reason = "multi_person_dark"
        else:
            self.cfg["multi_person_edit_mode"] = "crop_stitch"
            route = "crop_stitch"
            reason = "multi_person_normal_lighting"
        meta["applied"] = True
        meta["route"] = route
        meta["reason"] = reason
        meta["multi_person_edit_mode"] = str(self.cfg.get("multi_person_edit_mode"))
        print(
            f"[krea2 lighting] resolved_mode={route} reason={reason}",
            flush=True,
        )
        return meta

    def _maybe_run_magic_hour_face_detection(
        self, body: Image.Image, out_dir: Path | None
    ) -> dict[str, Any] | None:
        """Optional Magic Hour face-detection backend (does not replace geometry).

        LIMITATION: Magic Hour returns cropped face images (path/url) only — no
        bounding boxes or landmarks — so it cannot fully replace the current
        detector's FaceBox geometry for crop_stitch / full_frame without
        additional work. When selected we call the API for audit/count and still
        run the current detector for boxes downstream.
        """
        import os
        import sys
        import tempfile

        backend = str(
            self.cfg.get("face_detection_backend", "current") or "current"
        ).strip().lower()
        if backend not in ("magic_hour", "magichour", "mh"):
            return None

        # Colab may run Magic Hour first to show an interactive face selector
        # and map each returned crop identity to scene geometry. Reuse that
        # completed job instead of uploading/detecting the same image twice.
        precomputed = self.cfg.get("magic_hour_precomputed_result")
        if isinstance(precomputed, dict) and precomputed.get("status") == "complete":
            return dict(precomputed)

        target = self.cfg.get("magic_hour_target_file_path")
        conf = self.cfg.get("magic_hour_confidence_score", 0.5)
        conf_f = None if conf is None else float(conf)
        max_wait = float(self.cfg.get("magic_hour_max_wait_s", 120.0))
        tmp_path: Path | None = None
        try:
            from headswap.magichour.face_detection import detect_faces_magic_hour

            if target:
                result = detect_faces_magic_hour(
                    str(target),
                    confidence_score=conf_f,
                    max_wait_s=max_wait,
                )
            else:
                # Upload local body so MH can see the same scene we edit.
                fd, name = tempfile.mkstemp(prefix="mh_body_", suffix=".png")
                os.close(fd)
                tmp_path = Path(name)
                body.convert("RGB").save(tmp_path, format="PNG")
                result = detect_faces_magic_hour(
                    local_image_path=tmp_path,
                    confidence_score=conf_f,
                    max_wait_s=max_wait,
                )
            mh_meta = {
                "backend": "magic_hour",
                "id": result.id,
                "status": result.status,
                "face_count": result.face_count,
                "faces": [
                    {"path": f.path, "url": f.url} for f in result.faces
                ],
                "credits_charged": result.credits_charged,
                "geometry_from": "current",
                "limitation": (
                    "Magic Hour returns face crop path/url only — no boxes/"
                    "landmarks; current detector still supplies FaceBox geometry."
                ),
            }
            print(
                f"[krea2] face_detection_backend=magic_hour "
                f"id={result.id} status={result.status} "
                f"faces={result.face_count} "
                f"(geometry still from current detector)",
                flush=True,
            )
            if out_dir is not None:
                try:
                    import json

                    (Path(out_dir) / "magic_hour_face_detection.json").write_text(
                        json.dumps(mh_meta, indent=2), encoding="utf-8"
                    )
                except OSError:
                    pass
            return mh_meta
        except Exception as e:
            # Never fail a swap because the audit backend is down/misconfigured.
            print(
                f"[krea2] magic_hour face detection failed "
                f"({type(e).__name__}: {e})",
                flush=True,
            )
            return {
                "backend": "magic_hour",
                "error": f"{type(e).__name__}: {e}",
                "geometry_from": "current",
            }
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def run(
        self, body: Image.Image, face: Image.Image, out_dir: Path | None = None
    ) -> PipelineResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        # Restore multi_person_edit_mode after a lighting-routed run so reused
        # pipeline instances do not leak full_frame into the next image.
        _mode_before = self.cfg.get("multi_person_edit_mode", "crop_stitch")
        _ff_lora_before = self.cfg.get("full_frame_identity_lora_name")
        _ff_rb_before = self.cfg.get("full_frame_ref_boost")
        _ff_rbm_before = self.cfg.get("full_frame_ref_boost_mask")

        try:
            # Optional Magic Hour face-detection audit (geometry still current).
            magic_hour_meta = self._maybe_run_magic_hour_face_detection(body, out_dir)

            # Lighting route must run BEFORE model load so full_frame LoRA/rb
            # overrides are picked up when multi+dark.
            lighting_route_meta = self._resolve_lighting_route(body)

            # Body route runs AFTER lighting so it has the final say: it only
            # ever upgrades crop_stitch -> full_frame when a full body is
            # detected, and otherwise leaves whatever the lighting route (or
            # the default) already decided untouched.
            body_route_meta = self._resolve_body_route(body)

            rt = self._ensure_runtime(timings)
            from headswap.comfy.krea2_edit_fast import (
                clear_krea2_edit_static_cache,
                install_krea2_edit_static_cache,
            )

            edit_cache_info = install_krea2_edit_static_cache()
            clear_krea2_edit_static_cache()  # fresh cache per image
            bundle = self._load_models(rt, timings)
            div_by = int(self.cfg.get("div_by", 16))
            max_dim = int(self.cfg.get("max_dim", 768))
            mask_crop_stitch = bool(self.cfg.get("mask_crop_stitch", True))

            return self._run_after_models(
                body,
                face,
                out_dir,
                t0=t0,
                timings=timings,
                rt=rt,
                bundle=bundle,
                edit_cache_info=edit_cache_info,
                div_by=div_by,
                max_dim=max_dim,
                mask_crop_stitch=mask_crop_stitch,
                lighting_route_meta=lighting_route_meta,
                body_route_meta=body_route_meta,
                magic_hour_meta=magic_hour_meta,
            )
        finally:
            self.cfg["multi_person_edit_mode"] = _mode_before
            self.cfg["full_frame_identity_lora_name"] = _ff_lora_before
            self.cfg["full_frame_ref_boost"] = _ff_rb_before
            self.cfg["full_frame_ref_boost_mask"] = _ff_rbm_before

    def _run_after_models(
        self,
        body: Image.Image,
        face: Image.Image,
        out_dir: Path | None,
        *,
        t0: float,
        timings: dict[str, float],
        rt: NodeRuntime,
        bundle: dict,
        edit_cache_info: dict,
        div_by: int,
        max_dim: int,
        mask_crop_stitch: bool,
        lighting_route_meta: dict[str, Any],
        body_route_meta: dict[str, Any],
        magic_hour_meta: dict[str, Any] | None,
    ) -> PipelineResult:
        import sys

        # face_swap_mode: "single" (default) | "all"
        # "all" only triggers multi-pass when >1 face is detected.
        #
        # FUTURE / DISABLED: multi-face picker + swap-all are implemented below
        # but gated by enable_multi_face_features (default false). Current product
        # always does automatic single-face swap via body_face_policy. To re-enable:
        # set enable_multi_face_features: true and face_swap_mode: all (or use Colab UI).
        multi_face_enabled = bool(self.cfg.get("enable_multi_face_features", False))
        face_swap_mode = str(self.cfg.get("face_swap_mode", "single") or "single").strip().lower()
        if not multi_face_enabled:
            face_swap_mode = "single"
        swap_all = multi_face_enabled and face_swap_mode in ("all", "swap_all", "every")

        body_full: Image.Image | None = None
        mask = None
        box = None
        crop_content_box: tuple[int, int, int, int] | None = None
        freeze_mask: Image.Image | None = None
        ref_boost_mask_img: Image.Image | None = None
        scene: Image.Image
        person: Image.Image
        all_faces: list[FaceBox] = []
        selected_face: FaceBox | None = None
        multi_person = False
        use_tight = False
        face_failures: list[dict[str, Any]] = []
        faces_succeeded: list[int] = []
        face_prep_diag: dict[str, Any] = {}
        stitch_debug: dict[str, Image.Image] = {}
        edit_mode = "crop_stitch"  # crop_stitch | full_frame | legacy_full
        do_stitch = bool(mask_crop_stitch)

        with _stage(timings, "preprocessing"):
            face_crop = crop_face_reference(
                face,
                self.cache_dir,
                top=float(self.cfg.get("face_top_pad", 0.55)),
                bot=float(self.cfg.get("face_bot_pad", 0.02)),
                side=float(self.cfg.get("face_side_pad", 0.28)),
                include_shoulders=bool(self.cfg.get("include_shoulders", False)),
                tight_identity_crop=bool(self.cfg.get("tight_identity_crop", True)),
            )
            if mask_crop_stitch:
                # Locality shell (Klein/Qwen-improved): edit head crop, stitch back.
                # Picture 1 = body crop, Picture 2 = face — never swap order.
                body_full = resize_max_keep_ar(
                    body.convert("RGB"),
                    int(self.cfg.get("max_body_dim", max_dim)),
                    div_by=div_by,
                )
                body_face_policy = str(
                    self.cfg.get("body_face_policy", "largest") or "largest"
                )
                body_face_index = int(self.cfg.get("body_face_index", 0) or 0)
                selected_face, all_faces = select_face_box(
                    pil_to_rgb_np(body_full),
                    self.cache_dir,
                    index=body_face_index,
                    policy=body_face_policy if not swap_all else "largest",
                )
                multi_edit_mode = str(
                    self.cfg.get("multi_person_edit_mode", "crop_stitch") or "crop_stitch"
                ).strip().lower()
                print(
                    f"[krea2 path_selection] final_resolved_mode={multi_edit_mode} "
                    f"faces_detected={len(all_faces)} swap_all={swap_all} "
                    f"full_body_detected={body_route_meta.get('full_body_detected')} "
                    f"body_route_applied={body_route_meta.get('applied')}",
                    flush=True,
                )
                multi_swap_mode = str(
                    self.cfg.get("multi_person_swap_mode", "krea2_crop")
                    or "krea2_crop"
                ).strip().lower()
                # DEPRECATED A/B only: geometry-locked landmark paste.
                # Production default is krea2_crop + single_person_parity.
                if (
                    len(all_faces) > 1
                    and not swap_all
                    and multi_swap_mode
                    in ("align_paste", "align", "paste", "geometry_lock")
                ):
                    multi_person = True
                    edit_mode = "align_paste"
                    do_stitch = False
                    scene = body_full
                    person = face_crop
                    mask = None
                    box = None
                    crop_content_box = None
                    face_prep_diag = {
                        "edit_mode": "align_paste",
                        "faces_detected": len(all_faces),
                        "selected_box": (
                            None
                            if selected_face is None
                            else [
                                selected_face.x0,
                                selected_face.y0,
                                selected_face.x1,
                                selected_face.y1,
                            ]
                        ),
                        "multi_person_swap_mode": multi_swap_mode,
                    }
                    timings["_multi_person"] = 1.0
                    timings["_body_face_count"] = float(len(all_faces))
                    timings["_edit_mode_align_paste"] = 1.0
                    print(
                        f"[krea2] edit_mode=align_paste faces={len(all_faces)} "
                        f"(geometry-locked; Krea2 dual-ref free-gen skipped)",
                        flush=True,
                    )
                # MULTI-FACE: swap every detected face sequentially when requested.
                elif swap_all and len(all_faces) > 1:
                    print(
                        f"[krea2] face_swap_mode=all faces_detected={len(all_faces)} "
                        f"— running {len(all_faces)} inference passes",
                        flush=True,
                    )
                    timings["_multi_person"] = 1.0
                    timings["_body_face_count"] = float(len(all_faces))
                    timings["_face_swap_mode_all"] = 1.0
                    edit_mode = "crop_stitch"
                    do_stitch = True
                elif (
                    multi_edit_mode in ("full_frame", "fullframe", "full")
                    and not swap_all
                    and (
                        len(all_faces) > 1
                        or bool(body_route_meta.get("full_body_detected"))
                    )
                ):
                    # Full-frame Krea2 (no head crop). Author guidance is a full
                    # scene around ~1–1.5MP — do NOT reuse crop_stitch's 1024 body
                    # cap, which starves body/shoulder proportion cues.
                    # Reachable either via >1 detected faces (legacy multi-person
                    # A/B) or via single-person full-body auto-routing
                    # (_resolve_body_route): a full/half body needs the full
                    # scene for correct proportions even with only one face.
                    multi_person = len(all_faces) > 1
                    use_tight = False
                    body_ff = resize_to_megapixels(
                        body.convert("RGB"),
                        target_mp=float(self.cfg.get("full_frame_target_mp", 1.25)),
                        min_mp=float(self.cfg.get("full_frame_min_mp", 1.0)),
                        max_mp=float(self.cfg.get("full_frame_max_mp", 1.5)),
                        max_dim=int(
                            self.cfg.get(
                                "full_frame_max_dim",
                                self.cfg.get("multi_full_frame_max_dim", 2048),
                            )
                        ),
                        div_by=div_by,
                        allow_upscale=bool(
                            self.cfg.get("full_frame_allow_upscale", False)
                        ),
                    )
                    # Re-detect on the full-frame canvas so boxes match scene coords.
                    selected_face, all_faces = select_face_box(
                        pil_to_rgb_np(body_ff),
                        self.cache_dir,
                        index=body_face_index,
                        policy=body_face_policy,
                    )
                    body_full = body_ff
                    # Single source of truth: full_frame always enables identity ref_boost_mask
                    # (overrides stale Colab FULL_FRAME_REF_BOOST_MASK=False / notebook cfg.update).
                    # A/B opt-out: set allow_disable_full_frame_ref_boost_mask=True.
                    if not bool(
                        self.cfg.get("allow_disable_full_frame_ref_boost_mask", False)
                    ):
                        self.cfg["full_frame_ref_boost_mask"] = True
                    built = self._build_full_frame_inputs(
                        body_full,
                        face_crop,
                        div_by=div_by,
                        selected_face=selected_face,
                        all_faces=all_faces,
                    )
                    scene = built["scene"]
                    person = built["person"]
                    freeze_mask = built.get("freeze_mask")
                    ref_boost_mask_img = built.get("ref_boost_mask")
                    # Expose freeze mask in debug as "mask" for overlay inspection.
                    mask = freeze_mask
                    box = (
                        (0, 0, body_full.size[0], body_full.size[1])
                        if freeze_mask is not None
                        else None
                    )
                    crop_content_box = None
                    face_prep_diag = built.get("diag") or {}
                    edit_mode = "full_frame"
                    do_stitch = False
                    self._log_face_diag(face_prep_diag, label="full_frame")
                    print(
                        f"[krea2] edit_mode=full_frame faces={len(all_faces)} "
                        f"policy={body_face_policy} index={body_face_index} "
                        f"position={face_prep_diag.get('face_position')} "
                        f"selected_box="
                        f"{None if selected_face is None else [selected_face.x0, selected_face.y0, selected_face.x1, selected_face.y1]} "
                        f"person_prep={face_prep_diag.get('person_prep')} "
                        f"freeze={freeze_mask is not None} "
                        f"ref_boost_mask={ref_boost_mask_img is not None} "
                        f"scene={list(scene.size)} "
                        f"scene_mp={face_prep_diag.get('scene_megapixels')} "
                        f"person={list(person.size)}",
                        flush=True,
                    )
                    timings["_multi_person"] = 1.0 if multi_person else 0.0
                    timings["_tight_crop"] = 0.0
                    timings["_body_face_count"] = float(len(all_faces))
                    timings["_edit_mode_full_frame"] = 1.0
                    timings["_full_frame_freeze"] = (
                        1.0 if freeze_mask is not None else 0.0
                    )
                    timings["_full_frame_scene_mp"] = float(
                        face_prep_diag.get("scene_megapixels") or 0.0
                    )
                    timings["_full_frame_via_body_route"] = (
                        1.0 if body_route_meta.get("applied") else 0.0
                    )
                    face_prep_diag["routed_via"] = (
                        "body_route" if body_route_meta.get("applied") else "multi_face"
                    )
                    face_prep_diag["single_person_full_body"] = not multi_person
                else:
                    # Default / single-face / multi crop_stitch path.
                    if swap_all and len(all_faces) <= 1:
                        print(
                            "[krea2] face_swap_mode=all but only 1 face detected "
                            "— falling back to single-face path",
                            flush=True,
                        )
                    flags = self._tight_crop_flags(body_full, selected_face, all_faces)
                    multi_person = bool(flags["multi_person"])
                    use_tight = bool(flags["use_tight"])
                    isolate_selected = bool(flags.get("isolate_selected"))
                    face_frac = float(flags["face_frac"])
                    edit_mode = "crop_stitch"
                    do_stitch = True
                    print(
                        f"[krea2] edit_mode=crop_stitch faces_detected={len(all_faces)} "
                        f"policy={body_face_policy} index={body_face_index} "
                        f"face_frac={face_frac:.3f} tight_crop={use_tight} "
                        f"isolate={isolate_selected} "
                        f"selected_box="
                        f"{None if selected_face is None else [selected_face.x0, selected_face.y0, selected_face.x1, selected_face.y1]}",
                        flush=True,
                    )
                    built = self._build_scene_person(
                        body_full,
                        face_crop,
                        selected_face,
                        div_by=div_by,
                        use_tight=use_tight,
                        top_ext=float(flags["top_ext"]),
                        side_ext=float(flags["side_ext"]),
                        bot_ext=float(flags["bot_ext"]),
                        expand_px=int(flags["expand_px"]),
                        crop_pad=int(flags["crop_pad"]),
                        all_faces=all_faces,
                        isolate_selected=isolate_selected,
                    )
                    scene = built["scene"]
                    person = built["person"]
                    mask = built["mask"]
                    box = built["box"]
                    crop_content_box = built["crop_content_box"]
                    face_prep_diag = built.get("diag") or {}
                    face_prep_diag["edit_mode"] = "crop_stitch"
                    self._log_face_diag(face_prep_diag, label="crop_stitch")
                    timings["_multi_person"] = 1.0 if multi_person else 0.0
                    timings["_tight_crop"] = 1.0 if use_tight else 0.0
                    timings["_body_face_count"] = float(len(all_faces))
                    timings["_face_frac"] = float(face_frac)
                    timings["_crop_long_native"] = float(
                        face_prep_diag.get("crop_long_native") or 0
                    )
                    timings["_face_fill_crop"] = float(
                        face_prep_diag.get("face_area_frac_crop") or 0
                    )
            else:
                # Full-frame path (legacy A/B). Scene = body, person = face.
                scene = resize_max_keep_ar(
                    body.convert("RGB"), max_dim, div_by=div_by
                )
                person = resize_contain(
                    face_crop.convert("RGB"), scene.size, fill=(0, 0, 0)
                )
                multi_person = False
                all_faces = []
                selected_face = None
                edit_mode = "legacy_full"
                do_stitch = False

        save_debug = bool(self.cfg.get("save_debug", False))
        verbose = bool(self.cfg.get("verbose", False))
        base_seed = int(self.cfg.get("seed", 46))
        # Geometry RCA only — does not alter conditioning / sampling / stitch math.
        want_head_scale = bool(self.cfg.get("head_scale_trace", False)) or save_debug
        head_scale_trace: HeadScaleTrace | None = None
        if (
            want_head_scale
            and edit_mode == "crop_stitch"
            and body_full is not None
            and box is not None
            and out_dir is not None
        ):
            head_scale_trace = HeadScaleTrace(
                Path(out_dir) / "head_scale_trace",
                cache_dir=self.cache_dir,
                enabled=True,
                body_full=body_full,
                selected=selected_face,
                crop_box=box,
            )
            head_scale_trace.record_prep(
                body_full=body_full,
                selected=selected_face,
                mask=mask,
                crop_box=box,
                scene=scene,
                person=person,
            )

        if edit_mode == "align_paste" and body_full is not None:
            return self._run_multi_align_paste(
                body_full,
                face,
                selected_face=selected_face,
                all_faces=all_faces,
                rt=rt,
                bundle=bundle,
                timings=timings,
                edit_cache_info=edit_cache_info,
                out_dir=out_dir,
                save_debug=save_debug,
                base_seed=base_seed,
                t0=t0,
            )

        # ---------- Swap-all: N× crop → sample → stitch ----------
        # Gated by enable_multi_face_features + face_swap_mode=all (disabled by
        # default). Kept intact for future re-enablement — do not delete.
        if (
            mask_crop_stitch
            and body_full is not None
            and swap_all
            and len(all_faces) > 1
        ):
            canvas = body_full.copy()
            color_ref = body_full  # LAB match vs original body lighting
            last_sample: dict[str, Any] | None = None
            last_scene = body_full
            last_person = face_crop
            for face_i, face_box in enumerate(all_faces):
                try:
                    flags = self._tight_crop_flags(body_full, face_box, all_faces)
                    # Swap-all still isolates each face spatially; keep single-person
                    # identity/prompt/resolution (use_tight only if explicitly forced).
                    use_tight_i = bool(flags["use_tight"])
                    isolate_i = True
                    built = self._build_scene_person(
                        body_full,  # mask/crop from original geometry
                        face_crop,
                        face_box,
                        div_by=div_by,
                        use_tight=use_tight_i,
                        top_ext=float(flags["top_ext"]),
                        side_ext=float(flags["side_ext"]),
                        bot_ext=float(flags["bot_ext"]),
                        expand_px=int(flags["expand_px"]),
                        crop_pad=int(flags["crop_pad"]),
                        all_faces=all_faces,
                        isolate_selected=isolate_i,
                    )
                    self._log_face_diag(
                        built.get("diag") or {}, label=f"swap_all[{face_i}]"
                    )
                    face_prep_diag = built.get("diag") or face_prep_diag
                    prompt_i = self._prompt_for_edit(
                        use_tight=True, multi_person=True
                    )
                    print(
                        f"[krea2] swap-all face {face_i + 1}/{len(all_faces)} "
                        f"box=[{face_box.x0},{face_box.y0},{face_box.x1},{face_box.y1}]",
                        flush=True,
                    )
                    sample = self._sample_edit(
                        rt,
                        bundle,
                        built["scene"],
                        built["person"],
                        timings,
                        prompt=prompt_i,
                        edit_cache_info=edit_cache_info,
                        seed=base_seed + face_i,
                    )
                    edited_i, procrustes_applied_i = self._maybe_procrustes_edited_crop(
                        sample["edited"],
                        body_full,
                        built["box"],
                        face_box,
                        face_prep_diag,
                        crop_content_box=built["crop_content_box"],
                    )
                    with _stage(timings, "postprocessing"):
                        canvas = self._stitch_edited(
                            canvas,
                            edited_i,
                            built["mask"],
                            built["box"],
                            built["crop_content_box"],
                            color_ref=color_ref,
                            multi_person=True,
                            original_scene=built["scene"],
                            debug_stages=stitch_debug,
                            skip_head_clamp=procrustes_applied_i,
                        )
                    faces_succeeded.append(face_i)
                    last_sample = sample
                    mask = built["mask"]
                    box = built["box"]
                    crop_content_box = built["crop_content_box"]
                    scene = built["scene"]
                    person = built["person"]
                except Exception as exc:  # noqa: BLE001 — best-effort per face
                    msg = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[krea2] swap-all face {face_i + 1}/{len(all_faces)} FAILED — {msg}",
                        flush=True,
                    )
                    face_failures.append({"index": face_i, "error": msg})
                    continue

            if not faces_succeeded:
                raise PipelineRunError(
                    "swap-all failed: every detected face failed during inference"
                )
            out = canvas
            if last_sample is None:
                raise PipelineRunError("swap-all produced no successful samples")
            sample_meta = last_sample
            prompt = self._prompt_for_edit(use_tight=True, multi_person=True)
            timings["_tight_crop"] = 1.0
            use_tight = True
            multi_person = True
        else:
            # ---------- Single-pass path (crop_stitch or full_frame) ----------
            if edit_mode == "full_frame" and not bool(
                self.cfg.get("full_frame_use_crop_prompt", True)
            ):
                # Legacy full_frame-only prompt (diverges from crop_stitch).
                prompt = self._prompt_for_full_frame_multi(selected_face, all_faces)
            else:
                # Same prompt builder as crop_stitch. For full_frame parity with
                # the successful single-person path: use_tight=False and do not
                # inject multi_person prompt add-ons (those are crop-path extras).
                prompt_multi = bool(multi_person) and edit_mode != "full_frame"
                prompt = self._prompt_for_edit(
                    use_tight=use_tight if edit_mode != "full_frame" else False,
                    multi_person=prompt_multi,
                )
            cond_info = self._log_krea2_conditioning(
                edit_mode=edit_mode,
                scene=scene,
                person=person,
                prompt=prompt,
                bundle=bundle,
                seed=int(base_seed),
                ref_boost_mask=(
                    ref_boost_mask_img if edit_mode == "full_frame" else None
                ),
            )
            face_prep_diag["krea2_conditioning"] = cond_info
            sample_meta = self._sample_edit(
                rt,
                bundle,
                scene,
                person,
                timings,
                prompt=prompt,
                edit_cache_info=edit_cache_info,
                seed=base_seed,
                ref_boost_mask=ref_boost_mask_img if edit_mode == "full_frame" else None,
            )
            edited = sample_meta["edited"]
            procrustes_applied = False
            if (
                do_stitch
                and body_full is not None
                and box is not None
                and edit_mode != "full_frame"
            ):
                edited, procrustes_applied = self._maybe_procrustes_edited_crop(
                    edited,
                    body_full,
                    box,
                    selected_face,
                    face_prep_diag,
                    crop_content_box=crop_content_box,
                )
                sample_meta["edited"] = edited
            if head_scale_trace is not None:
                head_scale_trace.record_edited(edited)
            with _stage(timings, "postprocessing"):
                out = edited
                if (
                    do_stitch
                    and body_full is not None
                    and mask is not None
                    and box is not None
                ):
                    out = self._stitch_edited(
                        body_full,
                        edited,
                        mask,
                        box,
                        crop_content_box,
                        color_ref=body_full,
                        multi_person=multi_person,
                        original_scene=scene,
                        debug_stages=stitch_debug,
                        skip_head_clamp=procrustes_applied,
                    )
                    if head_scale_trace is not None:
                        head_scale_trace.record_stitched(out)
                        hs_report = head_scale_trace.finalize()
                        face_prep_diag["head_scale_trace"] = {
                            "verdict": hs_report.get("verdict"),
                            "dir": str(Path(out_dir) / "head_scale_trace")
                            if out_dir
                            else None,
                        }
                    # Optional post-stitch neighbor restore. OFF by default —
                    # production locality is crop-window exclusion + mask carve.
                    if (
                        multi_person
                        and selected_face is not None
                        and bool(
                            self.cfg.get("multi_crop_hard_freeze_neighbors", False)
                        )
                    ):
                        before_neighbor_freeze = out.copy()
                        out = hard_freeze_neighbor_faces(
                            out,
                            body_full,
                            selected_face,
                            all_faces,
                            pad_frac=float(
                                self.cfg.get("full_frame_neighbor_pad_frac", 0.22)
                            ),
                            expand_top_frac=float(
                                self.cfg.get("full_frame_neighbor_top_frac", 0.35)
                            ),
                            protect_expand=float(
                                self.cfg.get("full_frame_protect_expand", 1.20)
                            ),
                        )
                        if save_debug:
                            stitch_debug["neighbor_freeze_delta"] = (
                                ImageChops.difference(before_neighbor_freeze, out)
                            )
                            stitch_debug["after_neighbor_freeze"] = out.copy()
                        timings["_multi_crop_hard_freeze"] = 1.0
                        import sys as _sys

                        print(
                            "[krea2] multi crop_stitch hard_freeze_neighbors applied",
                            file=_sys.__stdout__,
                            flush=True,
                        )
                elif edit_mode == "full_frame" and body_full is not None:
                    # Model output is already full-frame; resize to body_full if needed.
                    if out.size != body_full.size:
                        out = out.resize(body_full.size, Image.Resampling.LANCZOS)
                    # Log whether the sampler changed the selected face BEFORE freeze.
                    if selected_face is not None:
                        self._log_selected_face_delta(
                            "full_frame_raw_vs_body",
                            body_full,
                            out,
                            selected_face,
                        )
                    do_clamp = bool(
                        self.cfg.get(
                            "full_frame_clamp_head_scale",
                            False,
                        )
                    ) and bool(self.cfg.get("clamp_edited_head_scale", True))
                    if multi_person and do_clamp:
                        out, clamp_info = clamp_edited_head_scale(
                            body_full,
                            out,
                            self.cache_dir,
                            max_height_ratio=float(
                                self.cfg.get("max_edited_head_height_ratio", 1.08)
                            ),
                            target_ratio=float(
                                self.cfg.get("target_edited_head_height_ratio", 0.98)
                            ),
                        )
                        if clamp_info.get("clamped"):
                            import sys

                            print(
                                f"[krea2] full_frame clamped oversized head "
                                f"ratio {clamp_info['ratio_before']:.2f}→"
                                f"{clamp_info['ratio_after']:.2f}",
                                flush=True,
                            )
                        face_prep_diag["full_frame_clamp"] = clamp_info
                    # Landmark Procrustes (similarity): map generated face → original.
                    # Config-gated; apply before freeze so neighbors stay locked.
                    do_procrustes = bool(
                        self.cfg.get("full_frame_procrustes_align", False)
                    )
                    if do_procrustes:
                        out, proc_info = procrustes_align_generated_to_body(
                            out,
                            body_full,
                            self.cache_dir,
                            prefer_body_box=selected_face,
                            prefer_gen_box=selected_face,
                        )
                        face_prep_diag["full_frame_procrustes"] = proc_info
                        if proc_info.get("procrustes"):
                            print(
                                "[krea2] full_frame procrustes_align "
                                f"scale={proc_info.get('scale')} "
                                f"rot_deg={proc_info.get('rotation_deg')} "
                                f"t={proc_info.get('translation')}",
                                flush=True,
                            )
                        elif verbose:
                            print(
                                "[krea2] full_frame procrustes skipped: "
                                f"{proc_info.get('procrustes_reason')}",
                                flush=True,
                            )
                    # Freeze outside selected face (post-sample locality only).
                    _ff_post_match = float(
                        self.cfg.get("full_frame_post_color_match_strength", 0.45) or 0.0
                    )
                    _ff_feather = int(self.cfg.get("full_frame_freeze_feather_px", 12))
                    if freeze_mask is not None and bool(
                        self.cfg.get("full_frame_freeze_outside", True)
                    ):
                        out = self._freeze_full_frame_outside_selected(
                            body_full,
                            out,
                            freeze_mask,
                            selected_face,
                            all_faces,
                        )
                        print(
                            "[krea2] full_frame freeze_outside applied "
                            f"(hard_neighbors="
                            f"{bool(self.cfg.get('full_frame_hard_freeze_neighbors', True))})",
                            flush=True,
                        )
                        if selected_face is not None:
                            self._log_selected_face_delta(
                                "full_frame_frozen_vs_body",
                                body_full,
                                out,
                                selected_face,
                            )
                    else:
                        # No freeze_mask path: apply whole-frame LAB color-match as minimum
                        # blending treatment (prevents raw resize landing with zero correction).
                        if _ff_post_match > 0:
                            out = lab_histogram_match_face(
                                out, body_full, None, strength=_ff_post_match
                            )
                        print(
                            f"[krea2 blend] mode=full_frame method=lab_match_only "
                            f"lab_strength={_ff_post_match} feather_px=N/A (no freeze_mask)",
                            flush=True,
                        )
            if edit_mode == "full_frame" and body_full is not None:
                out, full_frame_refine_diag = self._refine_full_frame_face(
                    out,
                    face_crop,
                    rt=rt,
                    bundle=bundle,
                    edit_cache_info=edit_cache_info,
                    div_by=div_by,
                    timings=timings,
                    base_seed=base_seed,
                    prompt=prompt,
                    out_dir=out_dir,
                    stitch_debug=stitch_debug,
                )
                face_prep_diag["full_frame_face_refine"] = full_frame_refine_diag
            faces_succeeded = [0] if do_stitch or edit_mode == "full_frame" else []

        dbg = {}
        if out_dir is not None and save_debug:
            with _stage(timings, "image_saving"):
                debug_imgs = {
                    "debug_scene": scene,
                    "debug_person": person,
                    "debug_face_crop": face_crop,
                }
                if body_full is not None:
                    debug_imgs["debug_body"] = body_full
                if mask is not None:
                    debug_imgs["debug_mask"] = mask
                if do_stitch:
                    debug_imgs["debug_crop"] = scene
                    debug_imgs["debug_edited_crop"] = sample_meta["edited"]
                    debug_imgs.update(
                        {
                            f"debug_{stage}": image
                            for stage, image in stitch_debug.items()
                        }
                    )
                if edit_mode == "full_frame":
                    debug_imgs["debug_full_frame_scene"] = scene
                    debug_imgs["debug_full_frame_person"] = person
                    debug_imgs["debug_full_frame_raw"] = sample_meta["edited"]
                    if freeze_mask is not None:
                        debug_imgs["debug_freeze_mask"] = freeze_mask
                    if out is not None:
                        debug_imgs["debug_full_frame_frozen"] = out
                    if ref_boost_mask_img is not None:
                        debug_imgs["debug_ref_boost_mask"] = ref_boost_mask_img
                if swap_all and body_full is not None and out is not None:
                    debug_imgs["debug_swap_all_result"] = out
                if out is not None:
                    debug_imgs["debug_final"] = out
                dbg = {
                    k: v
                    for k, v in {
                        key: self._save_debug(out_dir, f"{key}.png", img)
                        for key, img in debug_imgs.items()
                    }.items()
                    if v
                }
        elif out_dir is not None:
            timings.setdefault("image_saving", 0.0)

        total_s = time.perf_counter() - t0
        _print_timing_breakdown(timings, total_s, verbose=verbose)

        timing_rounded = {
            k: round(float(v), 4) for k, v in timings.items() if not k.startswith("_")
        }
        meta = {
            "pipeline": self.name,
            "checkpoint": bundle["load_meta"].get("checkpoint"),
            "loras_loaded": list(bundle["load_meta"].get("loras_loaded") or []),
            "lora_strengths": dict(bundle["load_meta"].get("lora_strengths") or {}),
            "prompt": prompt,
            "scene_size": list(scene.size),
            "person_size": list(person.size),
            "body_size": list(body_full.size) if body_full is not None else list(scene.size),
            "crop_size": list(scene.size) if mask_crop_stitch else None,
            "mask_crop_stitch": mask_crop_stitch,
            "crop_margin_mode": str(
                self.cfg.get("crop_margin_mode", "tight") or "tight"
            ),
            "enable_procrustes_correction": bool(
                self.cfg.get("enable_procrustes_correction", False)
            ),
            "edit_mode": edit_mode,
            "multi_person_swap_mode": str(
                self.cfg.get("multi_person_swap_mode", "krea2_crop") or "krea2_crop"
            ),
            "multi_crop_hard_freeze_neighbors": bool(
                timings.get("_multi_crop_hard_freeze")
            ),
            "multi_person_edit_mode": str(
                self.cfg.get("multi_person_edit_mode", "crop_stitch") or "crop_stitch"
            ),
            "single_person_parity": self._single_person_parity(),
            "head_mask_backend": str(
                self.cfg.get("head_mask_backend", "ellipse") or "ellipse"
            ),
            "seam_refine": bool(self.cfg.get("seam_refine", False)),
            "full_frame_freeze_outside": bool(
                self.cfg.get("full_frame_freeze_outside", True)
            )
            and freeze_mask is not None,
            "ref_boost_mask_used": bool(sample_meta.get("ref_boost_mask_used")),
            "multi_person": bool(timings.get("_multi_person")),
            "tight_crop": bool(timings.get("_tight_crop")),
            "body_face_count": int(timings.get("_body_face_count") or 0),
            "face_frac": round(float(timings.get("_face_frac") or 0.0), 4),
            "face_prep_diag": face_prep_diag,
            "body_face_policy": str(self.cfg.get("body_face_policy", "largest") or "largest"),
            "body_face_index": int(self.cfg.get("body_face_index", 0) or 0),
            "face_swap_mode": "all" if (swap_all and len(all_faces) > 1) else "single",
            "enable_multi_face_features": bool(multi_face_enabled),
            "faces_succeeded": list(faces_succeeded),
            "face_failures": list(face_failures),
            "lighting_route": lighting_route_meta,
            "body_route": body_route_meta,
            "full_frame_face_refine": face_prep_diag.get("full_frame_face_refine"),
            "face_detection_backend": str(
                self.cfg.get("face_detection_backend", "current") or "current"
            ),
            "magic_hour_face_detection": magic_hour_meta,
            "person_match_crop_size": bool(self.cfg.get("person_match_crop_size", True)),
            "square_crop": bool(self.cfg.get("square_crop", False)),
            "stitch_feather_px": int(self.cfg.get("stitch_feather_px", 10)),
            "post_color_match_strength": float(
                self.cfg.get("post_color_match_strength", 0.35) or 0.0
            ),
            "box": list(box) if box is not None else None,
            "steps": sample_meta["steps"],
            "cfg": sample_meta["cfg"],
            "seed": sample_meta["seed"],
            "ref_boost": sample_meta["ref_boost"],
            "ref_boost_a": sample_meta["ref_boost_a"],
            "grounding_px": sample_meta["grounding_px"],
            "preserve_expression": bool(self.cfg.get("preserve_expression", True)),
            "fit_mode": sample_meta["fit_mode"],
            "empty_latent_node": sample_meta["empty_node"],
            "force_full_load": sample_meta["use_full_load"],
            "skip_negative_grounding": sample_meta["skip_neg_vlm"],
            "negative_mode": sample_meta["negative_mode"],
            "model_cache_hit": bool(timings.get("_model_cache_hit")),
            "verbose": verbose,
            "timing_s": timing_rounded,
            "sampling_diagnostics": sample_meta["sampling_diag"],
        }
        if verbose:
            print(
                f"[krea2] checkpoint={meta['checkpoint']} loras={meta['loras_loaded']} "
                f"scene={meta['scene_size']} person={meta['person_size']} "
                f"mask_crop={mask_crop_stitch} steps={meta['steps']} cfg={meta['cfg']} "
                f"ref_boost={meta['ref_boost']} neg={meta['negative_mode']} "
                f"cache_hit={meta['model_cache_hit']} face_swap_mode={meta['face_swap_mode']}"
            )
        return PipelineResult(
            image=out,
            latency_s=total_s,
            meta=meta,
            debug_paths=dbg,
        )
