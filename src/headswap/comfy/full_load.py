"""
Force full GPU residency for the diffusion UNet during sampling.

Why this exists
---------------
ComfyUI's default NORMAL_VRAM path in load_models_gpu computes::

    lowvram_model_memory = free_vram - minimum_memory_required

``minimum_memory_required`` comes from estimate_memory(), which for Qwen Image
Edit includes dual ~1024² reference latents from TextEncodeQwenImageEditPlus.
On a 16GB T4 that reservation often leaves only a few GB for weights, so
ModelPatcher.partially_load streams layers CPU↔GPU every denoising step
(~minutes/step instead of ~seconds/step).

Colab Cell 5 (~2.8 s/step) runs with the UNet fully resident. Forcing
``force_full_load=True`` on prepare_sampling skips the partial-load branch
(model_load uses the full-weight path) without changing weights, dtype,
sampler, or graph — quality is unchanged; only residency changes.

Lifecycle (matches ComfyUI's intended load/unload rhythm)
---------------------------------------------------------
1. Before sample: free currently-loaded models (CLIP/VAE) so the UNet fits.
2. During sample: force full UNet residency.
3. After sample: unload all GPU models (UNet) via unload_all_models /
   free_memory so VAE decode has VRAM. Without this step, a fully-resident
   UNet can leave only ~tens of MiB free and VAEDecode OOMs on a 16GB card.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterator


def _free_mb(mm: Any, device: Any) -> float | None:
    try:
        return round(mm.get_free_memory(device) / (1024**2), 1)
    except Exception:
        return None


def offload_gpu_models(*, reason: str) -> dict:
    """
    Offload every currently GPU-resident model via ComfyUI model_management.

    Uses ``unload_all_models()`` when available (calls ``free_memory(1e30, …)``
    per device), otherwise ``free_memory(1e30, get_torch_device())``. Then
    ``soft_empty_cache()`` so CUDA returns freed blocks to the allocator.
    """
    info: dict = {
        "ok": False,
        "reason": reason,
        "free_vram_mb_before": None,
        "free_vram_mb_after": None,
        "error": None,
    }
    try:
        import comfy.model_management as mm
    except Exception as exc:
        info["error"] = f"import_failed:{exc}"
        print(f"[full_load] offload skip ({reason}) — ComfyUI not available: {exc}")
        return info

    try:
        device = mm.get_torch_device()
        info["free_vram_mb_before"] = _free_mb(mm, device)
        if hasattr(mm, "unload_all_models"):
            mm.unload_all_models()
            info["api"] = "unload_all_models"
        else:
            mm.free_memory(1e30, device)
            info["api"] = "free_memory"
        if hasattr(mm, "soft_empty_cache"):
            mm.soft_empty_cache()
        info["free_vram_mb_after"] = _free_mb(mm, device)
        info["ok"] = True
        print(
            f"[full_load] offloaded GPU models ({reason}, api={info['api']}) "
            f"free_vram_mb {info['free_vram_mb_before']} → {info['free_vram_mb_after']}"
        )
        try:
            sys.stdout.flush()
        except Exception:
            pass
    except Exception as exc:
        info["error"] = str(exc)
        print(f"[full_load] offload warning ({reason}): {exc}")
    return info


@contextmanager
def force_sampling_full_load() -> Iterator[dict]:
    """
    Context manager: free idle GPU models, force full UNet load for sample,
    then unload the sampling model so VAE decode can load.
    """
    info: dict = {
        "enabled": False,
        "freed_before_sample": False,
        "freed_after_sample": False,
        "force_full_load": False,
        "error": None,
    }
    try:
        import comfy.sampler_helpers as sh
    except Exception as exc:
        info["error"] = f"import_failed:{exc}"
        print(f"[full_load] skip — ComfyUI not available: {exc}")
        yield info
        return

    before = offload_gpu_models(reason="before_sample")
    info["freed_before_sample"] = bool(before.get("ok"))
    info["offload_before"] = before
    if before.get("error") and not info.get("error"):
        info["error"] = f"before_sample:{before['error']}"

    orig = sh.prepare_sampling

    def prepare_sampling_full(
        model,
        noise_shape,
        conds,
        model_options=None,
        force_full_load=False,
        force_offload=False,
    ):
        # Always force full residency for this sampling call — ignore caller flag.
        return orig(
            model,
            noise_shape,
            conds,
            model_options=model_options,
            force_full_load=True,
            force_offload=force_offload,
        )

    sh.prepare_sampling = prepare_sampling_full
    info["enabled"] = True
    info["force_full_load"] = True
    print("[full_load] prepare_sampling patched with force_full_load=True")
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        yield info
    finally:
        sh.prepare_sampling = orig
        print("[full_load] prepare_sampling patch restored")
        try:
            sys.stdout.flush()
        except Exception:
            pass
        # Critical: fully-resident UNet must leave GPU before VAEDecode.
        after = offload_gpu_models(reason="after_sample")
        info["freed_after_sample"] = bool(after.get("ok"))
        info["offload_after"] = after
        if after.get("error") and not info.get("error"):
            info["error"] = f"after_sample:{after['error']}"
