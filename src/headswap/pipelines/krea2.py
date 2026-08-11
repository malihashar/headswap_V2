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
    detect_best_face,
    dilate_mask,
    ensure_selected_face_mask_coverage,
    estimate_head_direction_label,
    expand_crop_box_for_face_fill,
    feathered_soft_composite,
    get_face_landmarks5,
    dump_neck_seam_debug,
    hard_freeze_neighbor_faces,
    identity_face_boost_mask,
    build_harmonization_mask,
    harmonize_skin_tone,
    lab_histogram_match_face,
    match_neck_stub_to_head_tone,
    clean_alpha_tails,
    collapse_soft_chin_ghost,
    narrow_band_seam_refine,
    pad_to_square,
    pil_to_rgb_np,
    place_face_at_height_frac,
    procrustes_align_edited_crop_to_body_box,
    procrustes_align_generated_to_body,
    relock_pose_to_destination,
    resize_contain,
    resize_long_side,
    resize_max_keep_ar,
    resize_to_megapixels,
    select_face_box,
    suppress_neighbor_faces_in_mask,
    expand_crop_box_wide,
)
from headswap.headwear_erase import restore_background
from headswap.metrics.head_scale import head_scale_metrics
from headswap.segmentation import build_head_hair_mask, matte_backend_available


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

        # --- Scale-invariant head geometry -------------------------------
        # The mask ellipse is PROPORTIONAL to face size (top/side/bot_extend
        # are multiples of face w/h), but expand_px / blur_px / crop_pad /
        # stitch_feather_px are ABSOLUTE PIXELS. That silently breaks when
        # the face is small in frame: at body_full=1024, 18px expand + 12px
        # blur is 7.3% of a 410px portrait face but 24.4% of a 123px
        # full-body face -- the mask balloons ~3.3x more, relative to the
        # head, for exactly the case that fails.
        #
        # This is NOT only a compositing concern. `crop_with_mask` derives
        # the CROP BOX from `mask_bbox`, so the mask directly determines the
        # image Krea2 receives. Measured with this code (1024px body_full):
        #   portrait   -> head fills 58.2% of the crop the model sees
        #   full-body  -> head fills 28.4% of the crop the model sees
        # while the donor reference is resize_contain'd to fill ~100% of the
        # canvas in BOTH cases. The model is shown a head occupying a third
        # of the frame next to a donor filling all of it, and resolves that
        # mismatch by generating an oversized head/hair -- the exact
        # reported failure. Fixing the mask alone would not have helped;
        # the framing handed to the model is what has to match.
        #
        # Fix: scale the absolute-pixel terms by face height so a
        # small-in-frame face gets the same RELATIVE treatment as a
        # portrait. Capped at 1.0 (scale DOWN only) so the known-good
        # portrait/close-up paths stay bit-identical -- this can only ever
        # change the small-face case that is currently broken.
        geom_scale = 1.0
        if bool(self.cfg.get("head_geometry_scale_invariant", True)) and (
            selected_face is not None
        ):
            ref_face_px = float(
                self.cfg.get("head_geometry_reference_face_px", 410.0) or 410.0
            )
            if ref_face_px > 0:
                geom_scale = min(
                    1.0, float(selected_face.height) / ref_face_px
                )
                geom_scale = max(
                    float(self.cfg.get("head_geometry_min_scale", 0.25)),
                    geom_scale,
                )
        blur_px = int(self.cfg.get("mask_blur_px", 12))
        stitch_feather_px = int(self.cfg.get("stitch_feather_px", 10))
        if geom_scale < 1.0:
            expand_px = max(2, int(round(expand_px * geom_scale)))
            crop_pad = max(2, int(round(crop_pad * geom_scale)))
            blur_px = max(1, int(round(blur_px * geom_scale)))
            stitch_feather_px = max(1, int(round(stitch_feather_px * geom_scale)))
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
            "blur_px": blur_px,
            "stitch_feather_px": stitch_feather_px,
            "geom_scale": round(geom_scale, 4),
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
        blur_px: int | None = None,
    ) -> dict[str, Any]:
        """Crop+mask one body face and size the identity reference to match.

        ``blur_px`` overrides the mask's Gaussian blur radius (default:
        ``mask_blur_px`` from config). Callers that need this crop's mask to
        share an exact shape with a mask built elsewhere (e.g. the full_frame
        refine pass matching the first pass's freeze mask) should pass the
        same blur_px the other mask used, not just the same extend/expand
        values -- blur radius is part of the mask's effective footprint too.
        """
        all_faces = list(all_faces or [])
        multi_person = len(all_faces) > 1
        spp = self._single_person_parity()
        # Edge softness follows what the mask ACTUALLY is. The geometric
        # ellipse only approximates the head, so it needs a generous feather
        # to hide the mismatch. A real silhouette matte tracks the true edge,
        # so the same generous feather instead straddles it -- partially
        # regenerating a rim of background and leaving a thin halo hugging the
        # subject (measured on the hat case: worst background pixel 168 with
        # the wide edge vs 76 with the tight one, and 6x fewer background
        # pixels altered). Tight values are applied ONLY when a matte backend
        # is actually installed, so the ellipse fallback keeps its wide edge.
        tight_edge = bool(self.cfg.get("head_matte_tight_edge", True)) and (
            str(self.cfg.get("head_mask_backend", "ellipse") or "ellipse").strip().lower()
            in ("head_matte", "matte", "silhouette")
        ) and matte_backend_available()
        if tight_edge:
            expand_px = int(self.cfg.get("head_matte_expand_px", 3))
            blur_px = int(self.cfg.get("head_matte_blur_px", 3))
        mask, mask_info = build_head_hair_mask(
            body_full,
            self.cache_dir,
            backend=str(self.cfg.get("head_mask_backend", "ellipse") or "ellipse"),
            hair_margin_frac=float(
                self.cfg.get("head_matte_hair_margin_frac", 0.08)
            ),
            face_box=selected_face,
            expand_px=expand_px,
            blur_px=int(
                blur_px if blur_px is not None else self.cfg.get("mask_blur_px", 12)
            ),
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
        #
        # REVERTED for the single-person full-body route (was briefly gated
        # on via crop_stitch_clamp_head_scale): a real GPU render showed
        # massively oversized/duplicated hair. Root cause (two independent
        # investigations): face_h_frac_native is measured against crop_h,
        # which is the DILATED+BLURRED mask bbox, not the raw face box --
        # already smaller than the true head silhouette -- and then
        # identity_hair_height_boost (1.30) gets applied on top of that
        # against face_crop, which crop_face_reference already pads ~1.55x
        # for hair. The donor reference ends up telling Krea2 "the face+hair
        # is this small" while the mask says "fill this much bigger area" --
        # the model hallucinates extra hair to cover the gap. multi_person/
        # isolate_selected crops don't show this because clamp_crop_away_
        # neighbors independently tightens their crop_h first, masking the
        # same latent formula defect. Not safe to reuse without separately
        # fixing that formula -- out of scope for this pass. Single-person
        # full-body crop_stitch now gets the exact same donor-reference prep
        # as an ordinary portrait (plain resize_contain).
        scale_match = bool(self.cfg.get("identity_scale_match", True)) and (
            multi_person or isolate_selected
        )
        scale_factor = float(self.cfg.get("identity_scale_factor", 0.90))
        # --- Donor/target face-scale normalization (measured, not assumed) ---
        # Krea2Edit is dual-reference attention transfer with NO inpaint mask:
        # it samples the whole crop from an empty latent at denoise=1.0. So the
        # single strongest scale cue it gets is the RELATIVE size of the head in
        # image 1 (scene) vs image 2 (person). `resize_contain` fills the person
        # canvas, putting the donor head at ~55% of it, while the target head
        # occupies whatever fraction the crop geometry happens to yield:
        #   portrait  ~58%  (matches the donor -- and only by accident, because
        #                    the mask ellipse gets clipped at the image top)
        #   full-body ~32%  (nothing to clip against, so the full 1.55x top
        #                    extension applies and the head fills far less)
        # Handing the model a donor head ~2x the target's relative size makes it
        # render the head at donor scale: oversized head/hair, and the original
        # hair silhouette left uncovered around it -- the exact reported
        # "oversized + duplicated hair" failure.
        #
        # Fix: measure BOTH fractions for real and match them. This is not the
        # reverted identity_scale_match, whose formula had two compounding
        # defects: it divided by the dilated+blurred mask bbox and then applied
        # identity_hair_height_boost (1.30) on top of a face_crop that
        # crop_face_reference had ALREADY padded ~1.55x for hair -- double
        # counting. Here the donor's face height is DETECTED inside face_crop
        # (no assumed constant, no hair boost), and the ratio is clamped to 1.0
        # so the known-good portrait path stays a plain resize_contain.
        donor_norm_info: dict[str, Any] = {"applied": False}
        person_prep = "resize_contain"
        person = None
        if (
            bool(self.cfg.get("person_match_crop_size", True))
            and bool(self.cfg.get("donor_scale_normalize", True))
            and not scale_match
            and selected_face is not None
            and box is not None
        ):
            crop_h_raw = max(1, box[3] - box[1])
            target_face_frac = float(selected_face.height) / float(crop_h_raw)
            donor_fb = detect_best_face(pil_to_rgb_np(face_crop), self.cache_dir)
            img_frac: float | None = None
            if donor_fb is not None and donor_fb.height > 4:
                donor_face_frac = float(donor_fb.height) / float(
                    max(1, face_crop.size[1])
                )
                if donor_face_frac > 1e-6:
                    # Height of the donor IMAGE (not just its face) on the
                    # canvas such that its FACE lands at target_face_frac.
                    # donor_scale_factor (<1) further shrinks the person ref so
                    # the model renders a slightly smaller head (full-body
                    # oversized-head A/B). Noop check uses the unfactored
                    # match so the known-good portrait path stays untouched.
                    img_frac = target_face_frac / donor_face_frac
                    matched = min(1.0, max(0.15, img_frac))
                    # Already matched (the portrait case, ~0.99-1.0): fall
                    # through to plain resize_contain so the known-good path
                    # stays bit-identical rather than off-by-a-resample.
                    if matched >= float(
                        self.cfg.get("donor_scale_noop_threshold", 0.97)
                    ):
                        donor_norm_info = {
                            "applied": False,
                            "reason": "already_matched",
                            "target_face_frac": round(target_face_frac, 4),
                            "donor_face_frac": round(donor_face_frac, 4),
                            "donor_img_frac": round(matched, 4),
                        }
                        img_frac = None
                    else:
                        img_frac = matched * float(
                            self.cfg.get("donor_scale_factor", 0.88)
                        )
                        img_frac = min(1.0, max(0.15, img_frac))
                if donor_face_frac > 1e-6 and img_frac is not None:
                    person = place_face_at_height_frac(
                        face_crop.convert("RGB"),
                        scene.size,
                        height_frac=img_frac,
                        fill=(0, 0, 0),
                        max_height_frac=1.0,
                        min_height_frac=0.15,
                    )
                    person_prep = "donor_scale_normalized"
                    donor_norm_info = {
                        "applied": True,
                        "target_face_frac": round(target_face_frac, 4),
                        "donor_face_frac": round(donor_face_frac, 4),
                        "donor_img_frac": round(img_frac, 4),
                    }
                    print(
                        f"[krea2 donor_scale] target_face={target_face_frac:.3f} "
                        f"donor_face={donor_face_frac:.3f} -> donor_img={img_frac:.3f}",
                        flush=True,
                    )
            if person is None and donor_norm_info.get("reason") is None:
                donor_norm_info = {
                    "applied": False,
                    "reason": "no_face_detected_in_donor_crop",
                }
        if person is not None:
            pass
        elif bool(self.cfg.get("person_match_crop_size", True)):
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
                person_prep
                if person_prep != "resize_contain"
                else ("place_face_at_height_frac" if scale_match else "resize_contain")
            ),
            "donor_scale_normalize": donor_norm_info,
            "single_person_parity": spp,
            "tight_edge": bool(tight_edge),
            "stitch_feather_px": (
                int(self.cfg.get("head_matte_stitch_feather_px", 3))
                if tight_edge
                else int(self.cfg.get("stitch_feather_px", 10))
            ),
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

    def _append_headwear_policy(self, prompt: str) -> str:
        """Say what to do with hats/headwear explicitly.

        Diagnosed from a real production failure: the target wore a hat with
        wide side flaps, and the output showed a translucent oval above the
        head plus "wings" at ear level. Those were never invented hair -- they
        were the ORIGINAL HAT surviving the swap. Measured: the hat sat 100%
        inside the stitch mask, so this was not a mask-coverage problem.

        The cause is conditioning. The prompt named glasses ("Remove the
        original face and glasses") but never mentioned headwear, while
        simultaneously demanding the donor's hairstyle and "completely remove
        and replace the original hair". A hat is not hair, so nothing told the
        model what to do with it -- and Krea2 conditions heavily on image 1's
        structure. It resolved the ambiguity by producing a hat/hair hybrid.

        Default is PRESERVE: headwear is functionally clothing, and the prompt
        already keeps clothing from image 1. Preserving it also keeps the
        original silhouette, so no background has to be reconstructed --
        removal would additionally require inpainting whatever the hat
        occluded, which is the far harder problem.
        """
        if not bool(self.cfg.get("headwear_prompt_policy", True)):
            return prompt
        if bool(self.cfg.get("preserve_headwear", True)):
            return (
                prompt
                + " This is a HEAD SWAP, not a face swap: the second person's "
                "hair MUST be transferred, not hidden. CRITICAL: if the person "
                "in the first image is wearing a hat, cap, headscarf, helmet or "
                "any other head covering, it is rigid opaque FABRIC, not hair — "
                "reproduce it exactly as in the first image (same shape, size, "
                "position, colour, matte surface, crisp clean edges, including "
                "any flaps, brim or straps) and NO hair may grow through, over, "
                "on top of, or out of it: no strands, wisps, frizz or fuzz on "
                "its surface or along its edges; its silhouette stays smooth "
                "and hard-edged. But the second person's actual hairstyle, hair "
                "length, texture and volume MUST still be rendered wherever it "
                "would realistically be visible outside that fabric: worn UNDER "
                "the headwear and emerging naturally BELOW its lower edge — at "
                "the sideburns, in front of the ears, and at the nape of the "
                "neck. That visible hair must clearly match the second person's "
                "real hairstyle (its texture, curl/coil pattern and colour), "
                "not the first person's original hair or no hair at all."
            )
        return (
            prompt
            + " CRITICAL: if the person in the first image is wearing a hat, cap, "
            "headscarf or any other head covering, REMOVE it completely and "
            "replace it with the second person's hair. No part of the original "
            "headwear — including flaps, brim, straps or its outline — may remain "
            "or show through. Reconstruct whatever background the headwear was "
            "covering so nothing of its silhouette is left."
        )

    def _apply_expression_policy(self, prompt: str) -> str:
        """Honor ``preserve_expression`` (default false in production yaml).

        When false, strip prompt clauses that force the body/scene expression
        AND eye/head-direction lock onto the result, then append donor
        expression + natural gaze freedom language. Identity and hair
        instructions are left intact.
        """
        text = str(prompt or "").strip()
        if bool(self.cfg.get("preserve_expression", True)):
            return text
        # Known expression-lock + gaze/yaw-lock clauses (yaml prompt + refine).
        patterns = [
            # Primary yaml CRITICAL block (expression + follow-on Keep mouth…).
            r"CRITICAL:\s*copy the facial expression from the first image exactly"
            r".*?never from the second image\.",
            r"CRITICAL:\s*copy the facial expression from image 1 only[^.]*\.",
            r"Keep mouth shape, smile/no-smile, eye gaze, and\s*"
            r"micro-expressions from the first image only[^.]*\.",
            r"Preserve the facial expression, mouth shape, eye gaze, and head\s*"
            r"pose from the first image exactly\.?",
            # Head yaw / eye-direction lock (locks sideways body gaze onto donor).
            r"CRITICAL:\s*keep the head facing the exact same direction as the "
            r"first image\s*[—\-].*?match the first image\.",
            r"The eyes must look in the exact same direction as the first image[^.]*\.",
            r"Head yaw,\s*pitch, and roll must exactly match the first image\.?",
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
            "the result. Prefer the natural expression of the identity person. "
            "Prefer natural donor gaze — eyes looking toward the camera or "
            "following the donor's natural eye direction, not locked to the "
            "body's sideways look from the first image."
        )
        if "Allow the facial expression from the second image" not in out:
            out = (out + " " + freedom).strip()
        return out

    def _prompt_for_edit(
        self,
        *,
        use_tight: bool,
        multi_person: bool,
        direction_hint: str = "",
    ) -> str:
        prompt = str(self.cfg.get("prompt", "") or "").strip()
        # SPP-CC: identical prompt to single-person (no multi add-ons), but the
        # per-image head-direction hint still applies -- it's not a multi-person
        # add-on, it's a concrete replacement for the generic "keep the same
        # direction" boilerplate already in the base prompt (see yaml).
        #
        # single_person_parity defaults true, so _single_person_parity() is
        # true for EVERY render in the default production config -- single or
        # multi-person. That silently made the "Hair often survives..."
        # reinforcement below dead code in production (never appended to any
        # real render), which is the multi_hair_replace_prompt config flag's
        # entire purpose. GPU-confirmed against two known-good reference
        # renders of the same target/donor pair (2026-08-07, predating this
        # session): the one WITHOUT this reinforcement kept the target's own
        # long hair with only the face swapped; the one WITH it correctly
        # replaced the hair with the donor's actual (short, pigtailed)
        # hairstyle -- i.e. a real head swap vs. a face swap, which is this
        # entire project's stated goal. So the hair-force clause is applied
        # here too, unconditionally on multi_hair_replace_prompt (default
        # true), not gated behind multi_person/use_tight like the other
        # multi-only add-ons (head-scale reminder, multi_extra_prompt) that
        # stay skipped for single-person parity as originally intended.
        if self._single_person_parity():
            if bool(self.cfg.get("multi_hair_replace_prompt", True)):
                prompt = (
                    prompt
                    + " Replace the hair completely with the hairstyle from the "
                    "second image — hairline, length, color, and parting. Do not "
                    "leave any of the first person's original hair. Completely "
                    "cover and remove the original face — no ghosting, double "
                    "face, or leftover ear/cheek from the first person beside "
                    "the new head."
                )
            return self._apply_expression_policy(
                self._append_headwear_policy(
                    self._append_direction_hint(prompt, direction_hint)
                )
            )
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
        return self._apply_expression_policy(
            self._append_headwear_policy(
                self._append_direction_hint(prompt, direction_hint)
            )
        )

    def _append_direction_hint(self, prompt: str, direction_hint: str) -> str:
        if not bool(self.cfg.get("head_direction_prompt_hint", True)):
            return prompt
        hint = str(direction_hint or "").strip()
        if not hint:
            return prompt
        return (
            prompt
            + f" CRITICAL — measured head/eye alignment in the first image: {hint}."
        )

    def _measure_head_direction_hint(
        self, body_full: Image.Image, selected_face: FaceBox | None
    ) -> tuple[str, dict[str, Any]]:
        """Compute a concrete, per-image head yaw/roll hint for the prompt.

        This is the prompt-only successor to the pixel-warp head-direction
        fixes (head_direction_relock, procrustes correction) that each
        introduced a new compositing artifact (a whole-image warp produced a
        floating duplicate head; a crop-local warp still streaked warped
        sky into the output even after masking). A wrong or unhelpful text
        hint can't corrupt pixels the way those did, so this is the safer
        default lever to reach for first.
        """
        diag: dict[str, Any] = {"applied": False}
        if not bool(self.cfg.get("head_direction_prompt_hint", True)):
            diag["reason"] = "disabled"
            return "", diag
        if selected_face is None:
            diag["reason"] = "no_selected_face"
            return "", diag
        landmarks, backend, note = get_face_landmarks5(
            pil_to_rgb_np(body_full), self.cache_dir, prefer_box=selected_face
        )
        diag["landmarks_backend"] = backend
        if landmarks is None or backend != "insightface":
            diag["reason"] = note or "landmarks_unavailable"
            return "", diag
        result = estimate_head_direction_label(
            landmarks,
            roll_deg_thresh=float(self.cfg.get("head_direction_roll_deg_thresh", 3.0)),
            yaw_thresh=float(self.cfg.get("head_direction_yaw_thresh", 0.12)),
            yaw_strong_thresh=float(
                self.cfg.get("head_direction_yaw_strong_thresh", 0.30)
            ),
        )
        diag.update(result)
        diag["applied"] = bool(result.get("label"))
        print(
            f"[krea2 head_direction_hint] applied={diag['applied']} "
            f"roll_deg={result.get('roll_deg')} yaw_offset={result.get('yaw_offset')} "
            f"label={result.get('label')!r}",
            flush=True,
        )
        return str(result.get("label") or ""), diag

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
        if bool(self.cfg.get("harmonization_enabled", True)):
            pre_harm = out.copy()
            out = self._apply_harmonization(
                pre_harm,
                body_full,
                freeze_mask,
                box=(0, 0, w, h),
            )
        print(
            f"[krea2 blend] mode=full_frame method=lab_match+feather "
            f"feather_px={feather} lab_strength={post_match} "
            f"harmonize={bool(self.cfg.get('harmonization_enabled', True))}",
            flush=True,
        )
        return out

    def _refine_full_frame_face(
        self,
        out: Image.Image,
        face_crop: Image.Image,
        *,
        body_full: Image.Image,
        rt: NodeRuntime,
        bundle: dict,
        edit_cache_info: dict,
        div_by: int,
        timings: dict[str, float],
        base_seed: int | None,
        prompt: str,
        out_dir: Path | None,
        stitch_debug: dict[str, Image.Image],
        selected_face: FaceBox | None = None,
        all_faces: list[FaceBox] | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Crop_stitch-quality face refine pass on top of a full_frame result.

        full_frame gives correct body/head proportions but spends the same
        pixel + VLM-grounding budget on the whole scene, so the face itself
        is rendered (and grounded) at a fraction of crop_stitch's effective
        resolution -- e.g. a full-body photo with face_height_frac=0.12 in a
        ~1.25MP scene renders the face at roughly 1/3 the pixel height a
        768px head crop gets. This second pass crops+masks the head, re-
        samples just that region at native crop_stitch resolution/grounding,
        and stitches it back so only pixels inside the head+hair mask can
        change; body/limb geometry from the full_frame pass is untouched by
        construction.

        Mask/face-box source (see docs/full_frame_failure_analysis.md §2.1):
        this reuses the SAME ``selected_face``/``all_faces`` the first
        (freeze-composite) pass detected on the pristine ``body_full``, and
        builds its mask with the SAME ``full_frame_mask_*`` extend/expand/
        blur parameters that pass's ``_build_full_frame_freeze_mask`` used
        -- not a fresh detection on the already-generated ``out``, and not
        crop_stitch's own (differently-shaped) ``mask_*`` defaults. Two
        independently-detected, independently-shaped masks landing in
        roughly the same head/hair region was the highest-confidence,
        most-convergent explanation the investigation found for the
        persistent ghost/halo artifact: the ring between the two masks'
        extents only ever got one pass's edit, feathered at a different
        radius, from a possibly different face-box center. Sharing one
        face box and one mask shape means the two composites share one
        boundary instead of two. Falls back to a fresh detection on ``out``
        only if no ``selected_face`` is supplied (defensive/back-compat).

        When ``full_frame_face_refine_procrustes`` is on (off by default,
        see yaml comment -- an earlier related fix, kept available but not
        the primary lever here), the freshly re-sampled head crop is also
        landmark-aligned onto ``body_full`` via
        ``procrustes_align_edited_crop_to_body_box``, the same
        content-local-space, inlier/residual-checked correction crop_stitch
        already uses via ``enable_procrustes_correction``.
        """
        diag: dict[str, Any] = {"applied": False}
        if not bool(self.cfg.get("full_frame_face_refine", True)):
            diag["reason"] = "disabled"
            return out, diag

        selected = selected_face
        faces = list(all_faces or ([selected_face] if selected_face is not None else []))
        diag["reused_pass1_face_box"] = selected is not None
        if selected is None:
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

        # crop_pad has no equivalent in _build_full_frame_freeze_mask (that
        # mask isn't crop-windowed), so keep sourcing it from the normal
        # crop_stitch flag resolution -- only the mask SHAPE itself needs to
        # match pass 1, not the unrelated crop padding.
        flags = self._tight_crop_flags(out, selected, faces)
        # Source the refine crop's "scene" (image 1) from the PRISTINE
        # body_full, not `out` (pass 1's own generated content). Krea2 is an
        # edit model -- it conditions heavily on image 1's structure even at
        # denoise=1.0, so cropping from pass 1's already-soft generated head
        # just re-edits that softness at higher pixel dimensions instead of
        # ever seeing real detail. Cropping from the pristine original gives
        # the refine pass the same sharp, real-photo source crop_stitch uses
        # -- this pass becomes a genuine crop_stitch-quality swap anchored at
        # the target's geometry, not an upscale-and-touch-up of pass 1.
        built = self._build_scene_person(
            body_full,
            face_crop,
            selected,
            div_by=div_by,
            use_tight=False,
            # SAME mask geometry _build_full_frame_freeze_mask used for pass 1
            # (not _tight_crop_flags'/crop_stitch's own mask_* defaults) so
            # the two composites share one boundary. See docstring above.
            top_ext=float(
                self.cfg.get(
                    "full_frame_mask_top_extend", self.cfg.get("mask_top_extend", 1.55)
                )
            ),
            side_ext=float(self.cfg.get("full_frame_mask_side_extend", 0.42)),
            bot_ext=float(self.cfg.get("full_frame_mask_bot_extend", 0.40)),
            expand_px=int(self.cfg.get("full_frame_mask_expand_px", 12)),
            blur_px=int(self.cfg.get("full_frame_mask_blur_px", 24)),
            crop_pad=int(flags["crop_pad"]),
            all_faces=faces,
            isolate_selected=False,
        )
        diag["mask_geometry_source"] = "full_frame_pass1_params"
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

        edited_for_stitch = refine_sample["edited"]
        procrustes_applied = False
        if bool(self.cfg.get("full_frame_face_refine_procrustes", True)):
            aligned, proc_info = procrustes_align_edited_crop_to_body_box(
                edited_for_stitch,
                body_full,
                built["box"],
                self.cache_dir,
                crop_content_box=built["crop_content_box"],
                prefer_body_box=selected,
                min_inliers=int(self.cfg.get("procrustes_min_inliers", 3)),
                min_scale=float(self.cfg.get("procrustes_min_scale", 0.40)),
                max_scale=float(self.cfg.get("procrustes_max_scale", 2.50)),
                max_rotation_deg=float(
                    self.cfg.get("procrustes_max_rotation_deg", 12.0)
                ),
                max_translate_frac=float(
                    self.cfg.get("procrustes_max_translate_frac", 0.25)
                ),
                max_residual_frac=float(
                    self.cfg.get("procrustes_max_residual_frac", 0.06)
                ),
            )
            diag["head_direction_procrustes"] = proc_info
            if proc_info.get("procrustes"):
                edited_for_stitch, blend_info = self._blend_procrustes_face_only(
                    refine_sample["edited"], aligned
                )
                diag["procrustes_face_only_blend"] = blend_info
                procrustes_applied = True
                print(
                    "[krea2 full_frame_refine] head_direction_procrustes "
                    f"scale={proc_info.get('scale')} "
                    f"rot_deg={proc_info.get('rotation_deg')} "
                    f"inliers={proc_info.get('inliers')}",
                    flush=True,
                )
            else:
                print(
                    "[krea2 full_frame_refine] head_direction_procrustes skipped "
                    f"reason={proc_info.get('procrustes_reason')}",
                    flush=True,
                )

        # Reordered architecture (see docs/full_frame_failure_analysis.md §10):
        # this refine pass now runs BEFORE the single freeze composite, on the
        # raw/clamped full_frame sample -- `out` here is NOT yet frozen against
        # body_full. So this blend is generated-vs-generated (this refined crop
        # vs. pass 1's raw full_frame content), not generated-vs-pristine.
        # Color/feather correction toward the true original happens exactly
        # ONCE, in the single freeze composite that runs after this returns --
        # doing a second, independently-parameterized LAB pull here (as before)
        # stacked a second soft-composite boundary with a DIFFERENT feather
        # radius (10px vs the freeze's 30px) and strength (0.35 vs 0.15) on
        # top of the freeze's, which is what produced the residual ghost/halo
        # even after pass-1/pass-2 mask geometry was unified. Skip LAB here
        # entirely (strength=0) and match the freeze's own feather radius so
        # there is exactly one seam, one feather radius, one LAB match, run
        # once, at the end.
        refined = self._stitch_edited(
            out,
            edited_for_stitch,
            built["mask"],
            built["box"],
            built["crop_content_box"],
            color_ref=body_full,
            multi_person=False,
            original_scene=refine_scene,
            debug_stages=stitch_debug,
            skip_head_clamp=procrustes_applied,
            feather_px_override=int(self.cfg.get("full_frame_freeze_feather_px", 30)),
            post_color_match_strength_override=0.0,
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

    def _relock_head_direction(
        self,
        out: Image.Image,
        destination: Image.Image,
        selected_face: FaceBox | None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Re-align the generated face's landmarks onto the original photo's.

        Prompt text alone ("keep head rotation / eye gaze from the first
        image") doesn't reliably stop generative identity edits from
        drifting the head yaw/eye-gaze toward the donor face or a frontal
        default. This applies a similarity transform (rotation + uniform
        scale + translation only -- no shear/stretch, to avoid distorting
        the already-good face quality) mapping the generated face's 5
        landmarks onto the original photo's, then blends the warped result
        back in through a tight face-only mask (hair/body untouched).
        """
        info: dict[str, Any] = {"pose_relock": False, "pose_relock_reason": None}
        if not bool(self.cfg.get("head_direction_relock", True)):
            info["pose_relock_reason"] = "disabled"
            return out, info
        if selected_face is None:
            info["pose_relock_reason"] = "no_selected_face"
            return out, info

        face_mask, _mask_info = build_head_hair_mask(
            out,
            self.cache_dir,
            backend="ellipse",
            face_box=selected_face,
            expand_px=int(self.cfg.get("head_direction_relock_expand_px", 6)),
            blur_px=int(self.cfg.get("head_direction_relock_blur_px", 8)),
            top_extend=float(self.cfg.get("head_direction_relock_top_extend", 0.12)),
            side_extend=float(self.cfg.get("head_direction_relock_side_extend", 0.12)),
            bot_extend=float(self.cfg.get("head_direction_relock_bot_extend", 0.05)),
        )
        relocked, pose_meta = relock_pose_to_destination(
            out,
            destination,
            self.cache_dir,
            face_mask=face_mask,
            use_full_affine=bool(self.cfg.get("head_direction_relock_full_affine", False)),
            core_min_alpha=float(self.cfg.get("head_direction_relock_core_min_alpha", 0.90)),
            feather_px=int(self.cfg.get("head_direction_relock_feather_px", 21)),
            stitch_feather_px=int(self.cfg.get("head_direction_relock_stitch_feather_px", 10)),
            min_scale=float(self.cfg.get("head_direction_relock_min_scale", 0.75)),
            max_scale=float(self.cfg.get("head_direction_relock_max_scale", 1.35)),
            max_rotation_deg=float(
                self.cfg.get("head_direction_relock_max_rotation_deg", 20.0)
            ),
            max_translate_frac=float(
                self.cfg.get("head_direction_relock_max_translate_frac", 0.20)
            ),
        )
        info.update(pose_meta)
        print(
            f"[krea2 head_direction] pose_relock={info.get('pose_relock')} "
            f"reason={info.get('pose_relock_reason')} "
            f"rotation_deg={info.get('rotation_deg')} scale={info.get('scale')}",
            flush=True,
        )
        return relocked, info

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
        edit_mask: Image.Image | None = None,
    ) -> dict[str, Any]:
        """One Krea2 dual-ref sample → decoded PIL. Shared by single + swap-all.

        ``edit_mask`` (EXPERIMENT E1, off unless ``sampling_containment`` is
        set): an L mask in ``scene`` coordinates, white where the model may
        generate. When supplied, the sampler is seeded with the ENCODED scene
        latent instead of an empty one and that mask is attached as
        ``noise_mask``, so ComfyUI's KSamplerX0Inpaint re-pins every latent
        outside the mask to the source at every step
        (``comfy/samplers.py:641``).

        Safe by construction at denoise=1.0: Krea2 is ModelType.FLUX ->
        CONST.noise_scaling = ``sigma*noise + (1-sigma)*latent_image``
        (``comfy/model_sampling.py:97``), and sigma_max is 1.0, so the
        latent_image term is multiplied by ZERO. The starting noise is
        bit-identical whether we pass an empty or an encoded latent -- the
        encoded latent's only effect is to make the mask pin functional.
        """
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

        # Report the LoRA actually LOADED in this bundle, not what the current
        # cfg mode would select: the refine pass flips multi_person_edit_mode
        # to "crop_stitch" for sample-time knobs (ref_boost/grounding_px) but
        # reuses pass 1's already-loaded weights, so a cfg-derived name here
        # misreported which model was really sampling.
        lora_name = (bundle.get("load_meta") or {}).get("loras_loaded")

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
                # The krea2edit node pack requires target_latent to be the SAME
                # latent handed to KSampler.latent_image; otherwise the patch
                # pre-allocates against the wrong tensor. Only relevant when E1
                # containment swaps that latent away from the empty one.
                if bool(
                    self.cfg.get("sampling_containment", False)
                ) and edit_mask is not None and self._node_accepts_kwarg(
                    rt, "Krea2EditModelPatch", "target_latent"
                ):
                    patch_kwargs["target_latent"] = get_value_at_index(scene_lat, 0)
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

            # EXPERIMENT E1: sampling-time containment. Seed the sampler with
            # the encoded scene latent and attach the head mask as noise_mask
            # so pixels outside it are re-pinned to the source every step.
            containment_coverage: float | None = None
            containment_on = bool(
                self.cfg.get("sampling_containment", False)
            ) and edit_mask is not None
            if containment_on:
                latent = dict(get_value_at_index(scene_lat, 0))
                _m = edit_mask.convert("L")
                if _m.size != scene.size:
                    _m = _m.resize(scene.size, Image.Resampling.BILINEAR)
                _mt = pil_mask_to_comfy_tensor(_m, torch)
                latent["noise_mask"] = _mt.reshape(
                    (-1, 1, _mt.shape[-2], _mt.shape[-1])
                )
                empty_node = "scene_latent+noise_mask"
                containment_coverage = round(float(_mt.mean().item()), 4)
                print(
                    "[krea2 containment] scene_latent+noise_mask "
                    f"coverage={containment_coverage:.3f} "
                    f"mask={list(_m.size)} scene={list(scene.size)}",
                    flush=True,
                )
            elif rt.has("EmptySD3LatentImage"):
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
            sampling_diag["sampling_containment"] = bool(containment_on)
            if containment_coverage is not None:
                sampling_diag["noise_mask_coverage"] = containment_coverage
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

    def _maybe_clamp_full_frame_head_scale(
        self, body_full: Image.Image, out: Image.Image
    ) -> tuple[Image.Image, dict[str, Any] | None]:
        """Correct full_frame head scale/position using the target's real face.

        clamp_edited_head_scale detects the face in both the pristine
        ``body_full`` (the geometric anchor) and the raw full_frame sample
        ``out``, and -- only if the generated face is taller than the
        target's real face by more than max_edited_head_height_ratio --
        uniformly shrinks the whole ``out`` image about the generated face's
        own center, then translates it so that (now-shrunk) center lands
        exactly on the target's real face center. That single similarity
        transform fixes scale AND position (both axes) together, anchored to
        face detection the same way the rest of this pipeline is, and
        BORDER_REPLICATE + a shrink-only clamp (0.55-1.0x) keeps it bounded.

        Was gated ``if multi_person and do_clamp`` -- full_frame was
        originally a multi-person-only feature, so that gate made sense
        then. ``_resolve_body_route`` later added a single-person full-body
        route to full_frame, but this gate was never updated, so the clamp
        could never fire for exactly that (now primary) trigger case,
        regardless of the ``full_frame_clamp_head_scale`` config flag.
        ``detect_best_face``'s "could grab the wrong face" risk (the
        original reason this defaulted off) only applies with >1 face in
        frame -- moot for the single-person case this now covers, since
        there's only one face to detect.

        Returns ``(out, None)`` when the clamp is disabled or not needed
        (no clamp_info to report); ``(possibly-clamped out, clamp_info)``
        otherwise.
        """
        do_clamp = bool(
            self.cfg.get("full_frame_clamp_head_scale", False)
        ) and bool(self.cfg.get("clamp_edited_head_scale", True))
        if not do_clamp:
            return out, None
        clamped, clamp_info = clamp_edited_head_scale(
            body_full,
            out,
            self.cache_dir,
            max_height_ratio=float(self.cfg.get("max_edited_head_height_ratio", 1.08)),
            target_ratio=float(self.cfg.get("target_edited_head_height_ratio", 0.98)),
            min_height_ratio=float(self.cfg.get("min_edited_head_height_ratio", 0.92)),
            max_grow=float(self.cfg.get("max_edited_head_grow", 1.45)),
        )
        if clamp_info.get("clamped"):
            print(
                f"[krea2] full_frame clamped oversized head "
                f"ratio {clamp_info['ratio_before']:.2f}→"
                f"{clamp_info['ratio_after']:.2f} shrink={clamp_info['shrink']:.3f}",
                flush=True,
            )
        return clamped, clamp_info

    def _maybe_clamp_crop_stitch_head_scale(
        self, body_full: Image.Image, out: Image.Image
    ) -> tuple[Image.Image, dict[str, Any] | None]:
        """Same safety net as `_maybe_clamp_full_frame_head_scale`, for the
        single-person crop_stitch stitch result on a body-route-detected
        full-body photo.

        crop_stitch's own head-scale clamp (inside `_stitch_edited`, via
        `clamp_edited_head_scale`) is gated by `stitch_multi = multi_person
        and not single_person_parity()`. `single_person_parity` defaults
        true, so for any single-subject photo (which a full-body photo
        almost always is) that clamp never fires -- it's dead code for
        exactly the case `_resolve_body_route` now needs it for. Rather than
        touch the existing, validated SPP-CC gate (used by every crop_stitch
        run today), this is a separate, explicitly-flagged call
        (`crop_stitch_clamp_head_scale`, OFF by default — opt-in only).
        Ordinary portrait/close-up crop_stitch runs are unaffected unless the
        flag is explicitly enabled.

        GPU-observed 2026-08-10: this used to hand the ENTIRE canvas to
        `clamp_edited_head_scale`, which warps whatever image it is given
        about the face center -- fine for a head-sized crop (how the
        stitch_multi path already uses it, on `edited_crop`), but here it
        shrank the WHOLE PHOTO (body, clothes, background together) just to
        fix a head-only size mismatch. That produced two visible defects at
        once: a thin rectangular seam around the entire frame (where the
        shrunk interior meets the hard-pasted, unshrunk original border --
        a subtle geometric misalignment in otherwise-identical background
        content), and a duplicated/terraced hem at the bottom of the dress
        (the same border-fill, now cutting through actual subject content
        instead of empty background/sky). Both disappear by construction if
        the warp only ever touches a head-sized region: crop a generous box
        around the head out of both `out` and `body_full`, clamp just that
        crop (identical geometry to the stitch_multi call site), and
        feather-paste it back -- the body, clothes, and background outside
        that box are never touched, so there is nothing left to seam or
        duplicate.
        """
        do_clamp = bool(
            self.cfg.get("crop_stitch_clamp_head_scale", False)
        ) and bool(self.cfg.get("clamp_edited_head_scale", True))
        if not do_clamp:
            return out, None
        face_o = detect_best_face(pil_to_rgb_np(out), self.cache_dir)
        if face_o is None or face_o.height < 8:
            return out, None
        fh = float(face_o.height)
        cx = 0.5 * (face_o.x0 + face_o.x1)
        cy = 0.5 * (face_o.y0 + face_o.y1)
        # Tight box (close to the actual head+hair mask geometry used
        # elsewhere -- mask_top/side/bot_extend) so the shrink warp's own
        # exposed-border zone (roughly (1-shrink)/2 of the box dimensions)
        # lands close to THIS box's own edge, where the feather below
        # already blends it away. GPU-observed 2026-08-10: a much larger box
        # (2.4/3.2/3.2x) pushed that internal hard-paste boundary deep
        # INSIDE the feathered region instead of at its edge, showing up as
        # its own hard rectangle seam in the middle of the photo.
        # 1.1/1.9/0.9 GPU-confirmed clean of the rectangle seam, but that was
        # WITH the (now-fixed) identity-leaking border_fill -- widening back
        # then reintroduced the rectangle because the leak got worse with
        # more border to patch. That leak is fixed now (border_fill=
        # local_out_crop below), so box size is no longer fighting identity
        # leakage -- but 1.1 on the sides is still too tight for voluminous/
        # wavy donor hair: GPU-observed 2026-08-10, hair reaching past the
        # box's opaque core got blended between its pre-shrink and post-
        # shrink positions in the feather zone, showing as duplicated/
        # doubled hair strands. Widened sides/bottom to actually contain the
        # hair; top was never implicated, left alone.
        half_w = fh * float(self.cfg.get("crop_stitch_head_clamp_side_mult", 1.8))
        top = fh * float(self.cfg.get("crop_stitch_head_clamp_top_mult", 1.9))
        bot = fh * float(self.cfg.get("crop_stitch_head_clamp_bot_mult", 1.3))
        W, H = out.size
        x0, x1 = max(0, int(cx - half_w)), min(W, int(cx + half_w))
        y0, y1 = max(0, int(cy - top)), min(H, int(cy + bot))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return out, None
        box = (x0, y0, x1, y1)
        local_body = body_full
        if local_body.size != out.size:
            local_body = local_body.resize(out.size, Image.Resampling.LANCZOS)
        local_out_crop = out.crop(box)
        local_body_crop = local_body.crop(box)
        clamped_local, clamp_info = clamp_edited_head_scale(
            local_body_crop,
            local_out_crop,
            self.cache_dir,
            max_height_ratio=float(self.cfg.get("max_edited_head_height_ratio", 1.08)),
            target_ratio=float(self.cfg.get("target_edited_head_height_ratio", 0.98)),
            min_height_ratio=float(self.cfg.get("min_edited_head_height_ratio", 0.92)),
            max_grow=float(self.cfg.get("max_edited_head_grow", 1.45)),
            # GPU-observed 2026-08-10: this box is tight around the head, so
            # local_body_crop (the ORIGINAL, un-swapped target) is mostly the
            # target's own head/hair -- its default border-fill role leaked a
            # fragment of the target's real hairline as a disconnected dark
            # island above the new head when the exposed strip overlapped
            # where her original head was. local_out_crop (the already-
            # correct, pre-warp swapped result) can never reintroduce the
            # wrong identity there.
            border_fill=local_out_crop,
        )
        if not clamp_info.get("clamped"):
            return out, clamp_info
        print(
            f"[krea2] crop_stitch (full-body route) clamped head scale "
            f"ratio {clamp_info['ratio_before']:.2f}→"
            f"{clamp_info['ratio_after']:.2f} shrink={clamp_info['shrink']:.3f} "
            f"(local box {box}, whole frame untouched)",
            flush=True,
        )
        import cv2
        import numpy as np

        # Elliptical opaque core, not a rectangle: the box is centered on a
        # round head, and a rectangular inset leaves the mask's straight
        # edges cutting across sky/hair at an angle that doesn't match
        # either -- more visible than an ellipse that roughly tracks the
        # head's own silhouette.
        feather = max(4, int(round(fh * 0.35)))
        bw, bh = clamped_local.size
        m = np.zeros((bh, bw), dtype=np.uint8)
        cv2.ellipse(
            m,
            (bw // 2, bh // 2),
            (max(1, bw // 2 - feather), max(1, bh // 2 - feather)),
            0, 0, 360, 255, -1,
        )
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        alpha = (m.astype(np.float32) / 255.0)[..., None]
        blended = (
            np.asarray(clamped_local.convert("RGB")).astype(np.float32) * alpha
            + np.asarray(local_out_crop.convert("RGB")).astype(np.float32) * (1.0 - alpha)
        )
        blended_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        out_final = out.copy()
        out_final.paste(blended_img, (x0, y0))
        return out_final, clamp_info

    def _maybe_procrustes_edited_crop(
        self,
        edited: Image.Image,
        body_full: Image.Image,
        box: tuple[int, int, int, int],
        selected_face: FaceBox | None,
        face_prep_diag: dict[str, Any],
        crop_content_box: tuple[int, int, int, int] | None = None,
        stitch_mask: Image.Image | None = None,
    ) -> tuple[Image.Image, bool]:
        """Optional Procrustes on crop_stitch edited crop before stitch.

        Returns ``(image, applied)``. ``applied`` lets the stitch stage skip its
        own head-scale clamp so the head is never rescaled twice.
        """
        if not bool(self.cfg.get("enable_procrustes_correction", False)):
            return edited, False
        alignment = str(self.cfg.get("procrustes_alignment", "jaw") or "jaw")
        min_inliers = int(self.cfg.get("procrustes_min_inliers", 3))
        if alignment.lower() == "jaw":
            min_inliers = int(
                self.cfg.get("procrustes_jaw_min_inliers", max(4, min_inliers))
            )
        aligned, info = procrustes_align_edited_crop_to_body_box(
            edited,
            body_full,
            box,
            self.cache_dir,
            crop_content_box=crop_content_box,
            prefer_body_box=selected_face,
            alignment=alignment,
            min_inliers=min_inliers,
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
            tag = "procrustes_jaw" if alignment.lower() == "jaw" else "procrustes"
            print(
                f"[krea2 {tag}] "
                f"alignment={alignment} "
                f"scale={info.get('scale')} rot_deg={info.get('rotation_deg')} "
                f"t={info.get('translation')} inliers={info.get('inliers')} "
                f"residual_px={info.get('residual_px')} "
                f"chin_shift_px={info.get('chin_shift_px')} "
                f"content_box={info.get('content_box')}",
                flush=True,
            )
            if stitch_mask is not None:
                blended, blend_info = self._blend_procrustes_through_mask(
                    edited, aligned, stitch_mask
                )
            else:
                blended, blend_info = self._blend_procrustes_face_only(edited, aligned)
            face_prep_diag["procrustes_blend"] = blend_info
            return blended, True
        print(
            "[krea2] procrustes_correction skipped: "
            f"{info.get('procrustes_reason')}",
            flush=True,
        )
        return edited, False

    def _blend_procrustes_through_mask(
        self,
        edited: Image.Image,
        aligned: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Composite procrustes warp through generation/stitch head mask."""
        info: dict[str, Any] = {"mask_blend": False, "method": "stitch_mask"}
        w, h = edited.size
        m = mask.convert("L")
        if m.size != (w, h):
            m = m.resize((w, h), Image.Resampling.BILINEAR)
        feather = int(self.cfg.get("procrustes_mask_blend_feather_px", 8))
        blended = feathered_soft_composite(
            edited,
            aligned,
            m,
            (0, 0, w, h),
            extra_blur_px=max(2, feather),
        )
        info["mask_blend"] = True
        return blended, info

    def _blend_procrustes_face_only(
        self, edited: Image.Image, aligned: Image.Image
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Only keep the procrustes-warped pixels inside a tight face ellipse.

        ``procrustes_align_edited_crop_to_body_box`` warps the ENTIRE
        rectangular head+hair crop -- face, hair, AND whatever background/sky
        is baked into the crop's top/side padding (that padding assumes hair
        fills it; for short-haired subjects it's mostly open sky). The stitch
        mask that shows this crop into the final image is that same generous
        head+hair mask, so any warp distortion in the sky/background portion
        (cv2.warpAffine reflects source pixels at the border, which streaks
        badly across a smooth gradient sky) becomes visible in the output.
        Blending the warped and unwarped crops through a tight face-only
        mask means only eyes/nose/mouth geometry is corrected; hair and any
        background stay exactly as the (unwarped) generation produced them.
        """
        info: dict[str, Any] = {"face_only_blend": False}
        rgb = pil_to_rgb_np(edited)
        face, _ = select_face_box(rgb, self.cache_dir, index=0, policy="largest")
        if face is None:
            info["face_only_blend_reason"] = "no_face_on_edited_crop"
            return aligned, info
        face_mask, _ = build_head_hair_mask(
            edited,
            self.cache_dir,
            backend="ellipse",
            face_box=face,
            expand_px=int(self.cfg.get("procrustes_face_mask_expand_px", 6)),
            blur_px=int(self.cfg.get("procrustes_face_mask_blur_px", 10)),
            top_extend=float(self.cfg.get("procrustes_face_mask_top_extend", 0.15)),
            side_extend=float(self.cfg.get("procrustes_face_mask_side_extend", 0.15)),
            bot_extend=float(self.cfg.get("procrustes_face_mask_bot_extend", 0.08)),
        )
        w, h = edited.size
        blended = feathered_soft_composite(edited, aligned, face_mask, (0, 0, w, h))
        info["face_only_blend"] = True
        return blended, info

    @staticmethod
    def _generation_mask_on_canvas(
        gen_mask: Image.Image,
        canvas_size: tuple[int, int],
        box: tuple[int, int, int, int] | None,
    ) -> Image.Image:
        """Map crop-local generation mask to full canvas coordinates."""
        if gen_mask.size == canvas_size:
            return gen_mask.convert("L")
        full = Image.new("L", canvas_size, 0)
        if box is None:
            return full
        x0, y0, x1, y1 = box
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        m_r = gen_mask.convert("L").resize((bw, bh), Image.Resampling.BILINEAR)
        full.paste(m_r, (x0, y0))
        return full

    def _apply_harmonization(
        self,
        pre_harm: Image.Image,
        body: Image.Image,
        gen_mask: Image.Image,
        *,
        box: tuple[int, int, int, int] | None = None,
        debug_stages: dict[str, Image.Image] | None = None,
    ) -> Image.Image:
        """Post-stitch color harmonization; gen interior locked to pre_harm."""
        if not bool(self.cfg.get("harmonization_enabled", True)):
            return pre_harm
        gen_full = self._generation_mask_on_canvas(gen_mask, pre_harm.size, box)
        full_harm, harm_ring, harm_info = build_harmonization_mask(
            body,
            gen_full,
            dilate_px=int(self.cfg.get("harmonization_dilate_px", 24)),
            bot_extend_frac=float(self.cfg.get("harmonization_bot_extend", 0.15)),
            skin_thresh=float(self.cfg.get("harmonization_skin_thresh", 0.45)),
            include_arms=bool(self.cfg.get("harmonization_include_arms", True)),
        )
        stitched, apply_info = harmonize_skin_tone(
            pre_harm,
            body,
            gen_full,
            full_harm,
            harm_ring,
            strength=float(self.cfg.get("harmonization_strength", 0.55)),
            feather_px=int(self.cfg.get("harmonization_feather_px", 16)),
            interior_erode_px=int(
                self.cfg.get("neck_stub_tone_match_interior_erode_px", 20)
            ),
        )
        print(
            "[krea2 harmonize] "
            f"gen_bbox={harm_info.get('gen_bbox')} "
            f"full_harm_bbox={harm_info.get('full_harm_bbox')} "
            f"harm_ring_px={harm_info.get('harm_ring_area_px')} "
            f"full_harm_px={harm_info.get('full_harm_area_px')} "
            f"gen_subset={harm_info.get('gen_subset_of_harm')} "
            f"skin_frac={float(harm_info.get('harm_skin_frac', 0.0)):.3f} "
            f"method={apply_info.get('harmonization_method')} "
            f"strength={apply_info.get('harmonization_strength')} "
            f"feather_px={apply_info.get('harmonization_feather_px')} "
            f"gen_interior_max_diff={apply_info.get('gen_interior_max_diff')} "
            f"gen_interior_ok={apply_info.get('gen_interior_ok')} "
            f"applied={apply_info.get('harmonization_applied')} "
            f"reason={apply_info.get('harmonization_reason')}",
            flush=True,
        )
        if debug_stages is not None:
            debug_stages["debug_gen_mask"] = gen_full.copy()
            debug_stages["debug_harmonization_mask"] = full_harm.copy()
            debug_stages["debug_harm_ring_mask"] = harm_ring.copy()
            debug_stages["composite_after_harmonize"] = stitched.copy()
        return stitched

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
        feather_px_override: int | None = None,
        post_color_match_strength_override: float | None = None,
    ) -> Image.Image:
        edited_crop = edited
        if crop_content_box is not None:
            ox, oy, cw, ch = crop_content_box
            edited_crop = edited_crop.crop((ox, oy, ox + cw, oy + ch))
        # Clamp / multi stitch boosts only when NOT on SPP-CC parity path.
        stitch_multi = bool(multi_person) and not self._single_person_parity()
        # Pre-stitch clamp on the edited crop (NOT post-stitch local-box paste).
        # Multi always uses this; single-person opts in via
        # crop_stitch_pre_stitch_clamp. Soft-composites through the head mask
        # below — never call _maybe_clamp_crop_stitch_head_scale here.
        pre_stitch_clamp = stitch_multi or bool(
            self.cfg.get("crop_stitch_pre_stitch_clamp", False)
        )
        # Clamp oversized heads before composite (group-shot failure mode).
        # Suppressed when Procrustes already resized the head — two independent
        # scale+translate warps stacked produce a doubled / displaced head.
        if pre_stitch_clamp and skip_head_clamp:
            import sys

            print(
                "[krea2] head-scale clamp skipped (procrustes already applied)",
                flush=True,
            )
        if pre_stitch_clamp and not skip_head_clamp and bool(
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
            pre_clamp_crop = edited_crop
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
                # Single-person pre-stitch: soft-composite clamped head through
                # the head mask so crop sky/padding is not hard-warped (avoids
                # the sky-rect artifact of post-stitch local-box clamp).
                if not stitch_multi:
                    cw, ch = edited_crop.size
                    edited_crop = feathered_soft_composite(
                        pre_clamp_crop,
                        edited_crop,
                        mask,
                        (0, 0, cw, ch),
                        extra_blur_px=max(
                            2, int(self.cfg.get("head_matte_stitch_feather_px", 3))
                        ),
                    )
        stitch_mask = mask
        # Dilate before feather so the opaque core fully covers the old head
        # (stops original face/ear ghosting next to neighbors). A small dilate
        # applies even on the single-person path now: GPU-observed a faint
        # "double chin" ghost at the jaw even with a correctly-sized mask --
        # the un-dilated core tracked the NEW head's silhouette almost
        # exactly, so a sliver of the OLD jaw sat inside the feather band
        # instead of being fully opaque-covered (2026-08-10). This only grows
        # the opaque core by a few px, not the mask's overall extent (extend
        # the extent -- e.g. mask_bot_extend -- and the crop's own edited
        # clothing/necklace content gets pasted over the target's real
        # clothes instead; verified worse).
        dilate_px = int(
            self.cfg.get("multi_stitch_mask_dilate_px", 14)
            if stitch_multi
            else self.cfg.get("stitch_mask_dilate_px", 4)
        )
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
        if feather_px_override is not None:
            feather = int(feather_px_override)
        if post_color_match_strength_override is not None:
            post_match = float(post_color_match_strength_override)
        stitched = feathered_soft_composite(
            canvas,
            edited_crop,
            stitch_mask,
            box,
            extra_blur_px=feather,
        )
        # Soft mid-alpha under-chin ghost collapse (prefer edited, no dress paint).
        sm = stitch_mask.convert("L")
        if sm.size == canvas.size:
            full_m = sm
        else:
            full_m = Image.new("L", canvas.size, 0)
            x0, y0, x1, y1 = box
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            sm_r = sm.resize((bw, bh), Image.Resampling.BILINEAR)
            full_m.paste(sm_r, (x0, y0))
        soft_m = full_m
        if feather > 0:
            import cv2
            import numpy as np

            k = max(3, int(feather) * 2 + 1)
            soft_m = Image.fromarray(cv2.GaussianBlur(np.asarray(full_m), (k, k), 0))
        soft_m = clean_alpha_tails(soft_m, floor=10, ceil=245)
        if bool(self.cfg.get("collapse_soft_chin_ghost", True)):
            x0, y0, x1, y1 = box
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            edited_full = canvas.copy()
            edited_full.paste(
                edited_crop.convert("RGB").resize((bw, bh), Image.Resampling.LANCZOS),
                (x0, y0),
            )
            stitched = collapse_soft_chin_ghost(
                stitched,
                canvas,
                edited_full,
                soft_m,
                mid_lo=float(self.cfg.get("collapse_soft_chin_ghost_mid_lo", 0.12)),
                mid_hi=float(self.cfg.get("collapse_soft_chin_ghost_mid_hi", 0.88)),
                chin_start_frac=float(
                    self.cfg.get("collapse_soft_chin_ghost_chin_frac", 0.42)
                ),
            )
            if debug_stages is not None:
                debug_stages["composite_after_chin_ghost"] = stitched.copy()
        if debug_stages is not None:
            debug_stages["stitch_mask"] = stitch_mask.convert("L").copy()
            debug_stages["composite_before_lab"] = stitched.copy()
        stitched = lab_histogram_match_face(
            stitched,
            color_ref,
            stitch_mask,
            strength=post_match,
            edge_only_px=int(self.cfg.get("post_color_match_edge_only_px", 24)),
        )
        if debug_stages is not None:
            debug_stages["composite_after_lab"] = stitched.copy()
            gen_full = self._generation_mask_on_canvas(stitch_mask, canvas.size, box)
            overlay, neck_crop, _neck_info = dump_neck_seam_debug(
                stitched,
                gen_full,
                color_ref,
                self.cache_dir,
            )
            debug_stages["debug_gen_mask_boundary_overlay"] = overlay
            debug_stages["debug_neck_seam_crop"] = neck_crop
        if bool(self.cfg.get("harmonization_enabled", True)):
            pre_harm = stitched.copy()
            stitched = self._apply_harmonization(
                pre_harm,
                color_ref,
                mask,
                box=box,
                debug_stages=debug_stages,
            )
        elif bool(self.cfg.get("neck_stub_tone_match", True)):
            stitched = match_neck_stub_to_head_tone(
                stitched,
                stitch_mask,
                ring_px=int(self.cfg.get("neck_stub_tone_match_ring_px", 70)),
                interior_erode_px=int(
                    self.cfg.get("neck_stub_tone_match_interior_erode_px", 20)
                ),
                strength=float(self.cfg.get("neck_stub_tone_match_strength", 0.65)),
                l_strength=float(self.cfg.get("neck_stub_tone_match_l_strength", 0.90)),
                ab_strength=float(self.cfg.get("neck_stub_tone_match_ab_strength", 1.0)),
                passes=int(self.cfg.get("neck_stub_tone_match_passes", 2)),
                ab_gate_scale=float(
                    self.cfg.get("neck_stub_tone_match_ab_gate_scale", 22.0)
                ),
                blur_frac=float(self.cfg.get("neck_stub_tone_match_blur_frac", 0.12)),
            )
            if debug_stages is not None:
                debug_stages["composite_after_neck_tone"] = stitched.copy()
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
        """Route full/half-body photos to the pipeline that actually suits them.

        ARCHITECTURE (revised after repeated ghosting/hand-duplication/scale
        failures traced to full_frame's whole-scene generation + freeze
        composite -- see docs/full_frame_failure_analysis.md):

        full_frame's whole-scene-regeneration + freeze-composite architecture
        was originally built and validated for MULTI-PERSON group photos
        (commit 076f687), where crop_stitch's per-person tight crop has a
        real, distinct problem it solves (neighbor-face isolation/locality).
        Single-person full-body auto-routing to full_frame was added later,
        on the assumption that seeing the whole body was necessary for
        correct proportions. That assumption doesn't hold up: crop_stitch's
        crop window is built directly from the REAL detected face bbox in
        the pristine photo (already-correct geometry, proportional to face
        size regardless of how small the face is in frame -- verified: small
        faces get proportionally MORE context margin from expand_px/crop_pad,
        not less). Every ghost/hand-duplication artifact chased across many
        rounds traced back to full_frame's freeze step letting the model
        touch body/hand pixels and then trying to mask them back -- an
        architecture crop_stitch's single tight-crop-and-stitch doesn't have
        by construction, since body/hands are never inside the edited region
        at all.

        So: single-person full-body -> stays on crop_stitch (its proven,
        ghost-free path). The optional post-stitch head-scale clamp
        (`crop_stitch_clamp_head_scale`) stays OFF by default -- enabling it
        for every full-body run (e42caad+) caused rectangular crop-box seams
        and floating scalp ghosts on real sky plates. Opt in via config after
        A/B. Genuine multi-person full-body photos still route to full_frame,
        unchanged -- that's the case it was actually built and validated for.
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

        if n_faces > 1:
            # Multi-person full-body: full_frame's original, validated use
            # case (commit 076f687) -- unchanged.
            self.cfg["multi_person_edit_mode"] = "full_frame"
            if not self.cfg.get("full_frame_identity_lora_name"):
                self.cfg["full_frame_identity_lora_name"] = (
                    "krea2_identity_edit_v1_2.safetensors"
                )
            if self.cfg.get("full_frame_ref_boost") is None:
                self.cfg["full_frame_ref_boost"] = 4.0
            if not bool(
                self.cfg.get("allow_disable_full_frame_ref_boost_mask", False)
            ):
                self.cfg["full_frame_ref_boost_mask"] = True
            meta["applied"] = True
            meta["route"] = "full_frame"
            meta["reason"] = "multi_person_full_body_detected"
            meta["multi_person_edit_mode"] = str(self.cfg.get("multi_person_edit_mode"))
            print(
                "[krea2 body_route] resolved_mode=full_frame "
                "reason=multi_person_full_body_detected",
                flush=True,
            )
            return meta

        # Single-person full-body: stay on crop_stitch (see docstring).
        #
        # Do NOT force crop_stitch_clamp_head_scale here. e42caad+ turned that
        # clamp on for every full-body single-person run; real desert/sky
        # plates still showed a rectangular head-box paste + floating scalp at
        # the top of frame + pale double-neck V (local-box shrink + elliptical
        # reblend). Production default is clamp OFF -- old soft-stitch look.
        # Opt in via config when validating a safer clamp.
        #
        # REVERTED earlier: raising max_body_dim / scaling mask+feather for
        # this route caused oversized/duplicated hair and a rectangular ghost
        # at the crop-box edge (feather wider than crop_pad). Full-body
        # crop_stitch uses the same crop/mask geometry as a portrait.
        meta["applied"] = True
        meta["route"] = "crop_stitch"
        meta["reason"] = "single_person_full_body_detected"
        meta["multi_person_edit_mode"] = str(
            self.cfg.get("multi_person_edit_mode", "crop_stitch")
        )
        meta["crop_stitch_clamp_head_scale"] = bool(
            self.cfg.get("crop_stitch_clamp_head_scale", False)
        )
        print(
            "[krea2 body_route] resolved_mode=crop_stitch "
            f"(clamp_head_scale={meta['crop_stitch_clamp_head_scale']}) "
            "reason=single_person_full_body_detected",
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
        _cs_clamp_before = self.cfg.get("crop_stitch_clamp_head_scale")
        _max_body_dim_before = self.cfg.get("max_body_dim")
        _mask_expand_before = self.cfg.get("mask_expand_px")
        _mask_blur_before = self.cfg.get("mask_blur_px")
        _stitch_feather_before = self.cfg.get("stitch_feather_px")

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
            self.cfg["crop_stitch_clamp_head_scale"] = _cs_clamp_before
            self.cfg["max_body_dim"] = _max_body_dim_before
            self.cfg["mask_expand_px"] = _mask_expand_before
            self.cfg["mask_blur_px"] = _mask_blur_before
            self.cfg["stitch_feather_px"] = _stitch_feather_before

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
        crop_stitch_feather_px: int | None = None
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
                direction_hint, direction_diag = self._measure_head_direction_hint(
                    body_full, selected_face
                )
                face_prep_diag["head_direction_hint"] = direction_diag
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
                        f"geom_scale={flags.get('geom_scale')} "
                        f"expand_px={flags['expand_px']} blur_px={flags.get('blur_px')} "
                        f"crop_pad={flags['crop_pad']} "
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
                        blur_px=int(flags["blur_px"]),
                    )
                    # Feather the stitch by the same scale-invariant amount the
                    # mask was built with, so the blend zone stays proportional
                    # to the head instead of ballooning on a small-in-frame face.
                    crop_stitch_feather_px = int(flags["stitch_feather_px"])
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
                    direction_hint_i, direction_diag_i = self._measure_head_direction_hint(
                        body_full, face_box
                    )
                    face_prep_diag["head_direction_hint"] = direction_diag_i
                    prompt_i = self._prompt_for_edit(
                        use_tight=True,
                        multi_person=True,
                        direction_hint=direction_hint_i,
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
                        stitch_mask=built["mask"],
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
                    direction_hint=direction_hint,
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
            # EXPERIMENT E1 (crop_stitch only): the stitch mask, cropped to the
            # crop window, is exactly the region the model is allowed to
            # regenerate. Reuses the EXISTING mask -- no new mask geometry.
            edit_mask_scene: Image.Image | None = None
            if (
                edit_mode == "crop_stitch"
                and mask is not None
                and box is not None
                and bool(self.cfg.get("sampling_containment", False))
            ):
                edit_mask_scene = mask.convert("L").crop(box)
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
                edit_mask=edit_mask_scene,
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
                    stitch_mask=mask,
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
                        # match the stitch feather to the mask type (see
                        # _build_scene_person): a real matte earns a tight
                        # edge, the ellipse fallback keeps the wide one.
                        feather_px_override=(face_prep_diag or {}).get("stitch_feather_px"),
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
                    # restore_background BEFORE the clamp, not after (order
                    # matters -- GPU-observed a large translucent double-
                    # exposure "ghost" of the ORIGINAL head hovering above the
                    # new, smaller one, 2026-08-10). restore_background's
                    # matte-union blend assumes ``out`` and ``body_full`` show
                    # the SAME person at the SAME scale/position -- true here,
                    # right after the stitch, since only the head/hair mask
                    # region differs from body_full at this point. Run it
                    # here, it is a clean safety net for any diffusion-crop
                    # background drift the stitch's own feathered_soft_composite
                    # missed. Run it AFTER the clamp instead (as this used to)
                    # and that assumption breaks: the clamp shrinks the WHOLE
                    # frame about the face center, so ``out``'s person is now
                    # at a different scale/position than ``body_full``'s --
                    # body_full's matte then confidently marks the OLD, larger
                    # head's true (original) position as "person" too, and the
                    # blurred matte-union transition there blends in a faint
                    # but large translucent trace of the original head instead
                    # of cleanly showing sky. The clamp's own border-fill (see
                    # clamp_edited_head_scale) already handles the shrink's
                    # exposed edge correctly -- it's a HARD 0/255 paste using
                    # the warp's exact coverage mask, not a soft neural matte,
                    # so it doesn't have this failure mode and can safely run
                    # last.
                    if bool(self.cfg.get("restore_background_enabled", True)):
                        out, rb_info = restore_background(out, body_full)
                        face_prep_diag["restore_background"] = rb_info
                    # Body-route safety net (see _resolve_body_route /
                    # _maybe_clamp_crop_stitch_head_scale docstrings): only
                    # active when a single-person full body was detected --
                    # ordinary portrait crop_stitch runs are unaffected.
                    out, cs_clamp_info = self._maybe_clamp_crop_stitch_head_scale(
                        body_full, out
                    )
                    if cs_clamp_info is not None:
                        face_prep_diag["crop_stitch_body_route_clamp"] = cs_clamp_info
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
                    out, clamp_info = self._maybe_clamp_full_frame_head_scale(
                        body_full, out
                    )
                    if clamp_info is not None:
                        face_prep_diag["full_frame_clamp"] = clamp_info
                    # Landmark Procrustes (similarity): map generated face → original.
                    # Config-gated; apply before refine/freeze so neighbors stay locked.
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
                    # Resolution-boost refine BEFORE the single freeze composite
                    # (see docs/full_frame_failure_analysis.md §10). Previously
                    # this ran AFTER the freeze, on the already-composited `out`,
                    # meaning the freeze and the refine's own stitch each did an
                    # independently-parameterized soft-composite + LAB match in
                    # the same head/hair region -- two seams, two feather radii,
                    # two LAB strengths. Running it here means `out` is still the
                    # raw/clamped (not yet frozen) full_frame sample: refine's own
                    # blend is generated-vs-generated (low risk), and there is
                    # exactly one generated-vs-pristine-original seam left, in the
                    # single freeze composite below.
                    out, full_frame_refine_diag = self._refine_full_frame_face(
                        out,
                        face_crop,
                        body_full=body_full,
                        rt=rt,
                        bundle=bundle,
                        edit_cache_info=edit_cache_info,
                        div_by=div_by,
                        timings=timings,
                        base_seed=base_seed,
                        prompt=prompt,
                        out_dir=out_dir,
                        stitch_debug=stitch_debug,
                        selected_face=selected_face,
                        all_faces=all_faces,
                    )
                    face_prep_diag["full_frame_face_refine"] = full_frame_refine_diag
                    # Re-check scale AFTER refine: the refine pass regenerates
                    # the whole head region from the identity ref, so it can
                    # re-inflate (or shrink) the head that the pre-refine clamp
                    # just corrected. This is the only scale check downstream of
                    # the generation that actually produces the final head.
                    out, clamp_info_post = self._maybe_clamp_full_frame_head_scale(
                        body_full, out
                    )
                    if clamp_info_post is not None:
                        face_prep_diag["full_frame_clamp_post_refine"] = clamp_info_post
                    # Freeze outside selected face (post-sample locality only) --
                    # the ONE composite against the pristine original.
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
            if (do_stitch or edit_mode == "full_frame") and body_full is not None and out is not None:
                out, pose_relock_diag = self._relock_head_direction(
                    out, body_full, selected_face
                )
                face_prep_diag["head_direction_relock"] = pose_relock_diag
            faces_succeeded = [0] if do_stitch or edit_mode == "full_frame" else []

        if (
            body_full is not None
            and out is not None
            and selected_face is not None
        ):
            hs = head_scale_metrics(body_full, out, selected_face, self.cache_dir)
            face_prep_diag["head_scale"] = hs
            ratio = hs.get("head_to_body_scale_ratio")
            lo = float(self.cfg.get("head_scale_ratio_warn_lo", 0.92))
            hi = float(self.cfg.get("head_scale_ratio_warn_hi", 1.08))
            warn = ratio is not None and not (lo <= float(ratio) <= hi)
            print(
                "[krea2 head_scale] "
                f"ratio={ratio} target≈1.0 "
                f"body_face_h={hs.get('body_face_h_px')} "
                f"result_face_h={hs.get('result_face_h_px')} "
                f"warn={warn}",
                flush=True,
            )

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
                # `do_stitch` is hardcoded False for edit_mode=="full_frame" (that
                # path composites via _freeze_full_frame_outside_selected, not
                # _stitch_edited) -- but full_frame's own refine pass DOES call
                # _stitch_edited internally and populates this same stitch_debug
                # dict (stitch_mask, composite_before_lab, composite_after_lab).
                # Without also checking edit_mode here, those refine-pass frames
                # were silently discarded even with save_debug=true -- promote
                # them too so a real run can actually be visually inspected.
                if do_stitch or edit_mode == "full_frame":
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
            "head_direction_relock": face_prep_diag.get("head_direction_relock"),
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
