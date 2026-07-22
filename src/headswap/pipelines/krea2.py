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

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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


@contextmanager
def _stage(timings: dict[str, float], name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)


def _print_timing_breakdown(timings: dict[str, float], total_s: float) -> None:
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
        # NORMAL_VRAM / LOW_VRAM / HIGH_VRAM enum-ish
        name = str(state)
        info["lowvram"] = "LOW" in name or "NO_VRAM" in name
        info["cpu_offload_likely"] = info["lowvram"] or "NORMAL" in name
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
) -> dict:
    """Print the sampling knobs the user asked for before KSampler runs."""
    attn = _attention_backend_info()
    cuda = _cuda_util_snapshot()
    vram = _comfy_vram_state()
    # At cfg<=1 Comfy's optimized path should be ~1 UNet eval/step; cfg>1 ≈ 2×.
    unet_evals_est = steps if cfg <= 1.0 + 1e-6 else steps * 2
    diag = {
        "configured_steps": steps,
        "scheduler": scheduler,
        "sampler": sampler,
        "cfg": cfg,
        "denoise": denoise,
        "force_full_load": force_full_load,
        "cpu_offload_likely_without_full_load": (not force_full_load),
        "estimated_unet_evals": unet_evals_est,
        "attention": attn,
        "cuda_before_sample": cuda,
        "comfy_vram": vram,
        "resolution": [width, height],
        "timestep_shift_mu": mu_shift,
    }
    print("[krea2 sampling diagnostics]")
    print(f"  configured_steps={steps}  (verify progress bar ends at {steps}/{steps})")
    print(f"  sampler={sampler}  scheduler={scheduler}")
    print(f"  cfg={cfg}  denoise={denoise}  estimated_unet_evals/step_budget≈{unet_evals_est}")
    print(
        f"  force_full_load={force_full_load}  "
        f"(False ⇒ Comfy may stream UNet layers CPU↔GPU every step — classic ~10–30s/step)"
    )
    print(f"  attention_backend={attn.get('backend')}  xformers={attn.get('xformers')}  "
          f"flash_sdp={attn.get('flash_sdp')}  mem_efficient_sdp={attn.get('mem_efficient_sdp')}")
    print(f"  comfy_vram_state={vram.get('vram_state')}  cpu_offload_likely={vram.get('cpu_offload_likely')}")
    print(
        f"  gpu={cuda.get('device_name')}  free_mb={cuda.get('free_mb')}/"
        f"{cuda.get('total_mb')}  alloc_mb={cuda.get('allocated_mb')}  "
        f"reserved_mb={cuda.get('reserved_mb')}"
    )
    if mu_shift is not None:
        print(f"  timestep_shift_mu={mu_shift}  (Turbo author rec: 1.15)")
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

    def run(
        self, body: Image.Image, face: Image.Image, out_dir: Path | None = None
    ) -> PipelineResult:
        import torch

        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        rt = self._ensure_runtime(timings)
        bundle = self._load_models(rt, timings)
        div_by = int(self.cfg.get("div_by", 16))
        max_dim = int(self.cfg.get("max_dim", 768))

        with _stage(timings, "preprocessing"):
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
        steps = int(self.cfg.get("steps", 8))
        cfg = float(self.cfg.get("cfg", 1.0))
        seed = int(self.cfg.get("seed", 46))
        denoise = float(self.cfg.get("denoise", 1.0))
        # Default ON: skip expensive VLM negative when CFG won't use it.
        skip_neg_vlm = bool(self.cfg.get("skip_negative_grounding", True)) and cfg <= 1.0 + 1e-6
        save_debug = bool(self.cfg.get("save_debug", True))

        negative_mode = "grounded_vlm"
        kernel_info = _configure_fast_kernels(prefer_flash=True)
        count_forwards = bool(self.cfg.get("debug_count_unet_forwards", False))
        torch_compile = bool(self.cfg.get("torch_compile", False))
        compile_mode = str(self.cfg.get("torch_compile_mode", "reduce-overhead"))

        # Use no_grad — NOT inference_mode. Comfy ModelPatcher.detach / unload
        # mutates tensor version counters; inference tensors raise
        # "Cannot set version_counter for inference tensor", leave the UNet on
        # GPU, then VAEDecode dies with broken LoadedModel.real_model.
        with torch.no_grad():
            scene_t = pil_to_comfy_tensor(scene, torch)
            person_t = pil_to_comfy_tensor(person, torch)

            with _stage(timings, "vae_encode"):
                scene_lat = rt.call("VAEEncode", pixels=scene_t, vae=bundle["vae"])
                person_lat = rt.call("VAEEncode", pixels=person_t, vae=bundle["vae"])

            with _stage(timings, "model_patch"):
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

            # Turbo author rec: mu=1.15. Apply Flux-style shift when available so
            # the 8-step schedule matches distilled training (quality, not speed).
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
                mu_shift = None  # node missing — report in diagnostics

            # CRITICAL: without force_full_load, NORMAL_VRAM partially loads the
            # ~12GB UNet and streams layers CPU↔GPU every denoising step
            # (~10–30s/step on T4 → ~160s for 8 steps). Full residency keeps
            # weights on GPU for the sample only (CLIP already freed).
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
            )
            sampling_diag["kernels"] = kernel_info
            sampling_diag["torch_compile"] = compile_info
            sampling_diag["debug_count_unet_forwards"] = count_forwards
            print(
                f"[krea2 kernels] actions={kernel_info.get('actions')} "
                f"attention={kernel_info.get('attention_after', {}).get('backend')}"
            )

            fwd_counter, restore_fwd = _count_unet_forwards(
                model, enabled=count_forwards
            )
            # Snapshot outside the timed KSampler wall — mem_get_info syncs the device.
            cuda_mid = _cuda_util_snapshot()
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
                    # Silence only the denoising loop — not offload bookkeeping.
                    with _silence_krea2edit_step_prints():
                        samples = rt.call(
                            "KSampler",
                            model=model,
                            seed=seed,
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
            print(
                f"[krea2 sampling result] wall={sample_s:.2f}s  "
                f"configured_steps={steps}  actual_unet_forwards="
                f"{actual_forwards if count_forwards else 'skipped'}  "
                f"sec/step≈{sps}  sec/forward≈{spf}  "
                f"fwd_wrap_ok={fwd_counter.get('wrapped')}"
            )
            if count_forwards and actual_forwards > 0 and steps > 0 and actual_forwards >= steps * 2 - 1:
                print(
                    "[krea2 sampling] WARNING: UNet forwards ≈ 2× steps → CFG is "
                    "likely dual-evaluating each step. Try cfg=1.0 (or 0) to keep "
                    "one forward/step."
                )
            if sec_per_step is not None and sec_per_step > 5.0 and not use_full_load:
                print(
                    "[krea2 sampling] WARNING: >5s/step with force_full_load=false "
                    "strongly suggests CPU↔GPU weight streaming. Set force_full_load: true."
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
            print(
                f"[krea2 sampling] after gpu alloc_mb={cuda_after.get('allocated_mb')} "
                f"free_mb={cuda_after.get('free_mb')}"
            )

            with _stage(timings, "vae_decode"):
                decoded = rt.call(
                    "VAEDecode",
                    samples=get_value_at_index(samples, 0),
                    vae=bundle["vae"],
                )
                out = comfy_tensor_to_pil(get_value_at_index(decoded, 0))

        with _stage(timings, "postprocessing"):
            # No heavy post-process today; stage kept for profile completeness.
            pass

        dbg = {}
        if out_dir is not None and save_debug:
            with _stage(timings, "image_saving"):
                dbg = {
                    k: v
                    for k, v in {
                        "debug_scene": self._save_debug(out_dir, "debug_scene.png", scene),
                        "debug_person": self._save_debug(
                            out_dir, "debug_person.png", person
                        ),
                        "debug_face_crop": self._save_debug(
                            out_dir, "debug_face_crop.png", face_crop
                        ),
                    }.items()
                    if v
                }
        elif out_dir is not None:
            timings.setdefault("image_saving", 0.0)

        total_s = time.perf_counter() - t0
        _print_timing_breakdown(timings, total_s)

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
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "ref_boost": ref_boost,
            "ref_boost_a": ref_boost_a,
            "grounding_px": grounding_px,
            "fit_mode": fit_mode,
            "empty_latent_node": empty_node,
            "force_full_load": use_full_load,
            "skip_negative_grounding": skip_neg_vlm,
            "negative_mode": negative_mode,
            "model_cache_hit": bool(timings.get("_model_cache_hit")),
            "timing_s": timing_rounded,
            "sampling_diagnostics": sampling_diag,
        }
        print(
            f"[krea2] checkpoint={meta['checkpoint']} loras={meta['loras_loaded']} "
            f"scene={meta['scene_size']} person={meta['person_size']} "
            f"steps={steps} cfg={cfg} ref_boost={ref_boost} "
            f"neg={negative_mode} cache_hit={meta['model_cache_hit']}"
        )
        return PipelineResult(
            image=out,
            latency_s=total_s,
            meta=meta,
            debug_paths=dbg,
        )
