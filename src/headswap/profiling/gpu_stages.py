"""
GPU stage profiler for Qwen baseline vs Colab Cell 5 comparison.

Measurement only — does not change sampling graph or parameters.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

# Colab Cell 5 reference (warm cache, T4-class, 456×576, no Kontext scale).
COLAB_CELL5_REFERENCE = {
    "label": "Colab Cell 5 (warm, ~28s pipeline)",
    "body_size": [456, 576],
    "encode_megapixels": 0.262,
    "flux_kontext_image_scale": False,
    "steps": 6,
    "stages_s": {
        "preprocessing": 0.45,
        "model_loading": 0.0,
        "lora_loading": 0.0,
        "flux_kontext_image_scale": 0.0,
        "vae_encode": None,  # bundled in pipeline block in Colab logs
        "text_encoding": None,
        "scheduler_creation": None,
        "sampling_total": 17.0,  # ~6 × 2.84 s/it from Colab progress bar
        "vae_decode": None,
        "image_saving": 0.0,
        "total_pipeline": 27.9,
    },
    "sampling_s_per_step": 2.84,
}


@dataclass
class StageRecord:
    name: str
    seconds: float
    vram_allocated_mb: float | None = None
    vram_reserved_mb: float | None = None
    vram_peak_mb: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def get_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": False,
        "device_name": None,
        "device_count": 0,
        "device_index": None,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
        if info["cuda_available"]:
            idx = torch.cuda.current_device()
            info["device_index"] = idx
            info["device_name"] = torch.cuda.get_device_name(idx)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def vram_snapshot() -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "allocated_mb": None,
        "reserved_mb": None,
        "peak_mb": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        out["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
        out["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 1)
        out["peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    except Exception:
        pass
    return out


def reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def loaded_models_snapshot() -> dict[str, Any]:
    """Best-effort view of ComfyUI model_management load state."""
    snap: dict[str, Any] = {
        "loaded_model_count": None,
        "loaded_model_types": None,
    }
    try:
        import comfy.model_management as mm

        loaded = getattr(mm, "current_loaded_models", None) or []
        snap["loaded_model_count"] = len(loaded)
        types: list[str] = []
        for entry in loaded:
            model = getattr(entry, "model", entry)
            types.append(type(model).__name__)
        snap["loaded_model_types"] = types
    except Exception:
        pass
    return snap


def describe_latent(latent: Any) -> dict[str, Any]:
    """Extract latent tensor shape for diffusion-size logging."""
    info: dict[str, Any] = {"raw_type": type(latent).__name__}
    try:
        if isinstance(latent, dict):
            if "samples" in latent:
                t = latent["samples"]
                info["shape"] = list(t.shape)
                info["dtype"] = str(getattr(t, "dtype", ""))
                info["device"] = str(getattr(t, "device", ""))
                if len(t.shape) >= 4:
                    info["latent_h"] = int(t.shape[2])
                    info["latent_w"] = int(t.shape[3])
                    info["latent_channels"] = int(t.shape[1])
            return info
        if hasattr(latent, "shape"):
            info["shape"] = list(latent.shape)
            return info
    except Exception as exc:
        info["error"] = str(exc)
    return info


def count_sigmas(sigmas: Any) -> int | None:
    try:
        import torch

        if isinstance(sigmas, torch.Tensor):
            n = int(sigmas.numel())
            return max(0, n - 1) if n > 0 else 0
        if hasattr(sigmas, "__len__"):
            n = len(sigmas)
            return max(0, n - 1) if n > 0 else 0
    except Exception:
        pass
    return None


class GpuStageProfiler:
    def __init__(self) -> None:
        self.gpu = get_gpu_info()
        self.stages: list[StageRecord] = []
        self.sampling_step_times_s: list[float] = []
        self.extras: dict[str, Any] = {}
        self._sampling_hook_installed = False
        self._orig_comfy_sample = None

    @contextmanager
    def stage(self, name: str, **notes: Any) -> Iterator[None]:
        _cuda_sync()
        v0 = vram_snapshot()
        models0 = loaded_models_snapshot()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _cuda_sync()
            dt = time.perf_counter() - t0
            v1 = vram_snapshot()
            models1 = loaded_models_snapshot()
            stage_notes = dict(notes)
            stage_notes["models_before"] = models0
            stage_notes["models_after"] = models1
            if (
                models0.get("loaded_model_count") is not None
                and models1.get("loaded_model_count") is not None
                and models1["loaded_model_count"] < models0["loaded_model_count"]
            ):
                stage_notes["models_unloaded"] = True
            self.stages.append(
                StageRecord(
                    name=name,
                    seconds=dt,
                    vram_allocated_mb=v1.get("allocated_mb"),
                    vram_reserved_mb=v1.get("reserved_mb"),
                    vram_peak_mb=v1.get("peak_mb"),
                    notes=stage_notes,
                )
            )

    def note(self, key: str, value: Any) -> None:
        self.extras[key] = value

    def install_sampling_step_hook(self) -> bool:
        """Patch comfy.samplers.sample once to record per-step wall time."""
        if self._sampling_hook_installed:
            return True
        try:
            import comfy.samplers

            orig = comfy.samplers.sample
            profiler = self

            def wrapped(*args, **kwargs):
                user_cb = kwargs.get("callback")
                step_idx = [-1]
                step_t0 = [None]
                step_times: list[float] = []

                def step_callback(*cb_args, **cb_kwargs):
                    now = time.perf_counter()
                    if step_t0[0] is not None:
                        step_times.append(now - step_t0[0])
                    step_t0[0] = now
                    step_idx[0] += 1
                    if user_cb is not None:
                        user_cb(*cb_args, **cb_kwargs)

                kwargs["callback"] = step_callback
                _cuda_sync()
                t_all = time.perf_counter()
                out = orig(*args, **kwargs)
                _cuda_sync()
                if step_t0[0] is not None:
                    step_times.append(time.perf_counter() - step_t0[0])
                profiler.sampling_step_times_s = step_times
                profiler.extras["sampling_hook_total_s"] = time.perf_counter() - t_all
                return out

            comfy.samplers.sample = wrapped
            profiler._orig_comfy_sample = orig
            profiler._sampling_hook_installed = True
            return True
        except Exception as exc:
            self.extras["sampling_hook_error"] = str(exc)
            return False

    def restore_sampling_hook(self) -> None:
        if not self._sampling_hook_installed or self._orig_comfy_sample is None:
            return
        try:
            import comfy.samplers

            comfy.samplers.sample = self._orig_comfy_sample
        except Exception:
            pass
        self._sampling_hook_installed = False

    def timings_dict(self) -> dict[str, float]:
        out = {s.name: s.seconds for s in self.stages}
        if self.sampling_step_times_s:
            for i, t in enumerate(self.sampling_step_times_s, start=1):
                out[f"sampling_step_{i:02d}"] = t
            out["sampling_steps_n"] = float(len(self.sampling_step_times_s))
            out["sampling_step_mean"] = sum(self.sampling_step_times_s) / len(
                self.sampling_step_times_s
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu,
            "stages": [
                {
                    "name": s.name,
                    "seconds": round(s.seconds, 4),
                    "vram_allocated_mb": s.vram_allocated_mb,
                    "vram_reserved_mb": s.vram_reserved_mb,
                    "vram_peak_mb": s.vram_peak_mb,
                    "notes": s.notes,
                }
                for s in self.stages
            ],
            "sampling_step_times_s": [round(t, 4) for t in self.sampling_step_times_s],
            "extras": self.extras,
            "colab_reference": COLAB_CELL5_REFERENCE,
        }

    def print_report(self, *, total_s: float, label: str = "qwen_baseline") -> None:
        ref = COLAB_CELL5_REFERENCE
        print()
        print("=" * 72)
        print(f"[{label} profile] total={total_s:.2f}s")
        if self.gpu.get("device_name"):
            print(
                f"  GPU: {self.gpu['device_name']} "
                f"(cuda={self.gpu.get('cuda_available')}, "
                f"device={self.gpu.get('device_index')})"
            )
        else:
            print(f"  GPU: unavailable ({self.gpu.get('error', 'no CUDA')})")

        enc = self.extras.get("encode_body_size")
        lat = self.extras.get("latent_shape")
        steps = self.extras.get("sampling_steps")
        print(f"  encode_body_size: {enc}  megapixels={self.extras.get('encode_megapixels')}")
        print(f"  latent_shape: {lat}  sampling_steps={steps}")
        print(f"  model_cache_hit: {self.extras.get('model_cache_hit')}")
        print(f"  flux_kontext_image_scale_enabled: {self.extras.get('flux_kontext_image_scale_enabled')}")
        if self.extras.get("bundle_object_ids"):
            print(f"  bundle_ids (stable=not reloaded): {self.extras.get('bundle_object_ids')}")
        print("-" * 72)
        print(f"  {'stage':<28} {'seconds':>8}  {'%tot':>6}  {'vram_mb':>8}  {'peak_mb':>8}")
        for s in self.stages:
            pct = (100.0 * s.seconds / total_s) if total_s > 0 else 0.0
            vram = s.vram_allocated_mb if s.vram_allocated_mb is not None else float("nan")
            peak = s.vram_peak_mb if s.vram_peak_mb is not None else float("nan")
            note = ""
            if s.notes.get("models_unloaded"):
                note = " [models unloaded]"
            elif s.notes.get("cache_hit"):
                note = " [cache hit]"
            print(
                f"  {s.name:<28} {s.seconds:8.2f}  {pct:5.1f}%  "
                f"{vram:8.1f}  {peak:8.1f}{note}"
            )
        if self.sampling_step_times_s:
            print("-" * 72)
            print("  per-step sampling (s):")
            for i, t in enumerate(self.sampling_step_times_s, start=1):
                ref_step = ref.get("sampling_s_per_step")
                delta = ""
                if ref_step is not None:
                    delta = f"  (Colab ref ~{ref_step:.2f}s, ×{t / ref_step:.2f})"
                print(f"    step {i:02d}: {t:.2f}s{delta}")
            mean = sum(self.sampling_step_times_s) / len(self.sampling_step_times_s)
            print(f"    mean: {mean:.2f}s/step  n={len(self.sampling_step_times_s)}")
        print("-" * 72)
        print(f"  Colab reference total pipeline: {ref['stages_s']['total_pipeline']:.1f}s")
        print(f"  Colab reference sampling (~6 steps): {ref['stages_s']['sampling_total']:.1f}s")
        if total_s > 0 and ref["stages_s"]["total_pipeline"]:
            ratio = total_s / ref["stages_s"]["total_pipeline"]
            print(f"  vs Colab total: ×{ratio:.2f}")
        # Highlight largest stage vs Colab sampling share
        if self.stages:
            top = max(self.stages, key=lambda s: s.seconds)
            print(f"  largest stage: {top.name} ({top.seconds:.2f}s, {100*top.seconds/total_s:.1f}%)")
        print("=" * 72)
