"""OmniGen2 experimental head-swap backend (official multi-image API).

Official sources:
  https://github.com/VectorSpaceLab/OmniGen2
  https://huggingface.co/OmniGen2/OmniGen2

OmniGen2 natively accepts ``input_images=[img1, img2, ...]`` for in-context
edits. For headswap:
  image 1 = body (destination pose / clothing / background)
  image 2 = face crop (source identity)

Requires the VectorSpaceLab OmniGen2 package
(``from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline``).
Does not modify ComfyUI Qwen/Kontext pipelines.
"""
from __future__ import annotations

import inspect
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from headswap.pipelines.base import BasePipeline, PipelineResult
from headswap.pipelines.errors import PipelineRunError
from headswap.preprocess import crop_face_reference, resize_max_keep_ar
from headswap.profiling.gpu_stages import get_gpu_info, reset_vram_peak, vram_snapshot


MODEL_VERSION = "OmniGen2/OmniGen2"
OFFICIAL_REPO = "https://github.com/VectorSpaceLab/OmniGen2"
_DEFAULT_OMNIGEN2_ROOTS = (
    "/tmp/OmniGen2",
    "/kaggle/tmp/OmniGen2",
    "/content/OmniGen2",
)


def _ensure_omnigen2_on_path() -> str | None:
    """
    Upstream OmniGen2 has no setup.py — add the clone root to sys.path.

    Returns the root that was used, or None if already importable / not found.
    """
    import sys

    try:
        import omnigen2  # noqa: F401

        return getattr(omnigen2, "__file__", None) and str(
            Path(omnigen2.__file__).resolve().parents[1]
        )
    except ImportError:
        pass

    candidates: list[Path] = []
    env = os.environ.get("OMNIGEN2_DIR") or os.environ.get("OMNIGEN2_PATH")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(Path(p) for p in _DEFAULT_OMNIGEN2_ROOTS)

    for root in candidates:
        if (root / "omnigen2").is_dir():
            sys.path.insert(0, str(root))
            return str(root)
    return None


def _resolve_model_dir(cfg: dict[str, Any]) -> Path | str:
    local = cfg.get("model_path")
    if local:
        p = Path(str(local)).expanduser()
        if (p / "model_index.json").is_file():
            return p
    return str(cfg.get("model_id", MODEL_VERSION))


def _torch_dtype(name: str):
    import torch

    key = (name or "bf16").lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def _even(n: int, div: int = 16) -> int:
    n = max(div, int(n))
    return n - (n % div)


class OmniGen2PipelineRunner(BasePipeline):
    """In-context head swap via official OmniGen2Pipeline."""

    name = "omnigen2"

    def __init__(self, cfg: dict[str, Any], runtime=None, cache_dir: Path | None = None):
        super().__init__(cfg, runtime=runtime, cache_dir=cache_dir)
        self._pipe = None
        self._load_meta: dict[str, Any] = {}

    def _ensure_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        t0 = time.perf_counter()
        reset_vram_peak()
        v0 = vram_snapshot()
        model_ref = _resolve_model_dir(self.cfg)
        dtype = _torch_dtype(str(self.cfg.get("dtype", "bf16")))
        load_meta: dict[str, Any] = {
            "model_version": MODEL_VERSION,
            "model_ref": str(model_ref),
            "dtype": str(dtype).replace("torch.", ""),
            "official_repo": OFFICIAL_REPO,
            "scheduler_requested": str(self.cfg.get("scheduler", "euler")),
            "enable_model_cpu_offload": bool(
                self.cfg.get("enable_model_cpu_offload", True)
            ),
            "enable_sequential_cpu_offload": bool(
                self.cfg.get("enable_sequential_cpu_offload", False)
            ),
            "gpu": get_gpu_info(),
            "vram_before_load_mb": v0,
        }

        try:
            import torch

            used_root = _ensure_omnigen2_on_path()
            if used_root:
                load_meta["omnigen2_code_root"] = used_root
            from omnigen2.models.transformers.transformer_omnigen2 import (  # type: ignore
                OmniGen2Transformer2DModel,
            )
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import (  # type: ignore
                OmniGen2Pipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "OmniGen2 code not found. Upstream has no setup.py — do NOT pip install -e.\n"
                "  bash scripts/setup_omnigen2.sh\n"
                "  # or: git clone https://github.com/VectorSpaceLab/OmniGen2.git /tmp/OmniGen2\n"
                "  # then: export PYTHONPATH=/tmp/OmniGen2:$PYTHONPATH\n"
                "Do NOT pip install -r OmniGen2/requirements.txt on Kaggle "
                "(it pins torch==2.6.0 and breaks the existing CUDA stack).\n"
                f"Original error: {exc}"
            ) from exc

        # Match official inference.py load order exactly:
        # from_pretrained → replace transformer subclass → then offload/to(device).
        pipe = OmniGen2Pipeline.from_pretrained(
            str(model_ref),
            torch_dtype=dtype,
        )
        pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            str(model_ref),
            subfolder="transformer",
            torch_dtype=dtype,
        )

        sched = str(self.cfg.get("scheduler", "euler")).lower()
        if sched in {"dpmsolver++", "dpmsolver"}:
            from omnigen2.schedulers.scheduling_dpmsolver_multistep import (  # type: ignore
                DPMSolverMultistepScheduler,
            )

            pipe.scheduler = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                solver_order=2,
                prediction_type="flow_prediction",
            )
            load_meta["scheduler"] = "dpmsolver++"
        else:
            load_meta["scheduler"] = "euler"

        if load_meta["enable_sequential_cpu_offload"] and hasattr(
            pipe, "enable_sequential_cpu_offload"
        ):
            pipe.enable_sequential_cpu_offload()
            load_meta["device_placement"] = "sequential_cpu_offload"
        elif load_meta["enable_model_cpu_offload"] and hasattr(
            pipe, "enable_model_cpu_offload"
        ):
            pipe.enable_model_cpu_offload()
            load_meta["device_placement"] = "model_cpu_offload"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)
            load_meta["device_placement"] = device

        load_meta["load_time_s"] = round(time.perf_counter() - t0, 4)
        load_meta["vram_after_load_mb"] = vram_snapshot()
        self._pipe = pipe
        self._load_meta = load_meta
        print(
            f"[omnigen2] loaded {MODEL_VERSION} from {model_ref} "
            f"in {load_meta['load_time_s']}s placement={load_meta['device_placement']}"
        )
        return pipe

    def run(
        self, body: Image.Image, face: Image.Image, out_dir: Path | None = None
    ) -> PipelineResult:
        t0 = time.perf_counter()
        run_error: BaseException | None = None
        out: Image.Image | None = None
        dbg: dict[str, str] = {}
        timing: dict[str, float] = {}
        sample_meta: dict[str, Any] = {}
        body_work: Image.Image | None = None
        face_crop: Image.Image | None = None

        prompt = str(self.cfg.get("prompt", "")).strip()
        negative = str(self.cfg.get("negative_prompt", "") or "")
        steps = int(self.cfg.get("steps", 50))
        text_guidance = float(
            self.cfg.get("guidance", self.cfg.get("text_guidance_scale", 5.0))
        )
        image_guidance = float(
            self.cfg.get("image_guidance", self.cfg.get("image_guidance_scale", 2.5))
        )
        seed = int(self.cfg.get("seed", 0))
        strength = float(self.cfg.get("strength", 1.0))
        crop_size = int(self.cfg.get("crop_size", 1024))
        max_pixels = int(self.cfg.get("max_input_image_pixels", 1024 * 1024))
        cfg_start = float(self.cfg.get("cfg_range_start", 0.0))
        cfg_end = float(self.cfg.get("cfg_range_end", 1.0))

        try:
            import torch

            reset_vram_peak()

            t_pre = time.perf_counter()
            body_work = ImageOps.exif_transpose(body.convert("RGB"))
            body_work = resize_max_keep_ar(body_work, crop_size, div_by=16)
            face_crop = crop_face_reference(
                face,
                self.cache_dir,
                top=float(self.cfg.get("face_top_pad", 0.65)),
                bot=float(self.cfg.get("face_bot_pad", 0.15)),
                side=float(self.cfg.get("face_side_pad", 0.35)),
                include_shoulders=False,
            )
            face_crop = ImageOps.exif_transpose(face_crop.convert("RGB"))
            # Cap face pixels similar to official max_input_image_pixels.
            fw, fh = face_crop.size
            if fw * fh > max_pixels:
                scale = (max_pixels / float(fw * fh)) ** 0.5
                face_crop = face_crop.resize(
                    (
                        max(16, int(fw * scale) // 16 * 16),
                        max(16, int(fh * scale) // 16 * 16),
                    ),
                    Image.Resampling.LANCZOS,
                )
            width = _even(body_work.width)
            height = _even(body_work.height)
            timing["preprocess_s"] = round(time.perf_counter() - t_pre, 4)

            t_load = time.perf_counter()
            cold = self._pipe is None
            pipe = self._ensure_pipe()
            if cold and self._load_meta.get("load_time_s") is not None:
                timing["load_s"] = float(self._load_meta["load_time_s"])
            else:
                timing["load_s"] = round(time.perf_counter() - t_load, 4)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Diffusers + model_cpu_offload: keep generator on CPU (official Accelerate
            # path is fine either way; CUDA generators often break under offload).
            generator = torch.Generator(device="cpu").manual_seed(seed)
            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "input_images": [body_work, face_crop],
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "max_sequence_length": 1024,
                "max_pixels": max_pixels,
                "max_input_image_side_length": int(
                    self.cfg.get("max_input_image_side_length", crop_size)
                ),
                "align_res": False,  # we already chose width/height from the body
                "text_guidance_scale": text_guidance,
                "image_guidance_scale": image_guidance,
                "cfg_range": (cfg_start, cfg_end),
                "negative_prompt": negative,
                "num_images_per_prompt": 1,
                "generator": generator,
                "output_type": "pil",
            }

            sig = inspect.signature(pipe.__call__)
            strength_applied = False
            strength_skip = "OmniGen2Pipeline_has_no_strength_param"
            if "strength" in sig.parameters:
                call_kwargs["strength"] = strength
                strength_applied = True
                strength_skip = None

            # Drop kwargs the installed signature does not accept.
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

            v_before = vram_snapshot()
            t_sample = time.perf_counter()
            try:
                result = pipe(**call_kwargs)
            except torch.cuda.OutOfMemoryError as oom:
                # One automatic retry with sequential offload + smaller canvas.
                if not bool(self.cfg.get("_oom_retried")):
                    print(
                        "[omnigen2] CUDA OOM on sample — retrying with "
                        "sequential_cpu_offload and crop_size=768"
                    )
                    self.cfg["_oom_retried"] = True
                    self.cfg["enable_sequential_cpu_offload"] = True
                    self.cfg["enable_model_cpu_offload"] = False
                    self.cfg["crop_size"] = min(crop_size, 768)
                    self._pipe = None
                    self._load_meta = {}
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA OOM during OmniGen2 sample (will not auto-loop here): {oom}. "
                        "Re-run with crop_size: 768 and enable_sequential_cpu_offload: true "
                        "in configs/omnigen2.yaml"
                    ) from oom
                raise
            timing["sampling_s"] = round(time.perf_counter() - t_sample, 4)
            v_after = vram_snapshot()

            if hasattr(result, "images") and result.images:
                out = result.images[0].convert("RGB")
            else:
                raise RuntimeError("OmniGen2 pipeline returned no images")

            if body_work is not None and out.size != body_work.size:
                out = out.resize(body_work.size, Image.Resampling.LANCZOS)

            sample_meta = {
                "steps": steps,
                "guidance": text_guidance,
                "text_guidance_scale": text_guidance,
                "image_guidance": image_guidance,
                "image_guidance_scale": image_guidance,
                "seed": seed,
                "strength": strength,
                "strength_applied": strength_applied,
                "strength_skip_reason": strength_skip,
                "crop_size": crop_size,
                "output_size": [width, height],
                "cfg_range": [cfg_start, cfg_end],
                "input_mode": "multi_image_in_context",
                "vram_before_sample_mb": v_before,
                "vram_after_sample_mb": v_after,
                "vram_peak_mb": v_after.get("peak_mb"),
            }

            dbg = {
                k: v
                for k, v in {
                    "debug_body": self._save_debug(out_dir, "debug_body.png", body_work),
                    "debug_face_crop": self._save_debug(
                        out_dir, "debug_face_crop.png", face_crop
                    ),
                }.items()
                if v
            }
        except BaseException as exc:
            import traceback

            run_error = exc
            print("[omnigen2] TRACEBACK:\n" + traceback.format_exc(), flush=True)

        latency_s = time.perf_counter() - t0
        timing["total_s"] = round(latency_s, 4)

        meta: dict[str, Any] = {
            "pipeline": self.name,
            "model_version": MODEL_VERSION,
            "official_repo": OFFICIAL_REPO,
            "prompt": prompt,
            "negative_prompt": negative,
            "inference_settings": {
                "steps": steps,
                "guidance": text_guidance,
                "image_guidance": image_guidance,
                "seed": seed,
                "strength": strength,
                "crop_size": crop_size,
                "scheduler": self.cfg.get("scheduler", "euler"),
                "dtype": self.cfg.get("dtype", "bf16"),
                "enable_model_cpu_offload": bool(
                    self.cfg.get("enable_model_cpu_offload", True)
                ),
            },
            "timing_s": {
                k: v
                for k, v in {
                    "load": timing.get("load_s"),
                    "preprocess": timing.get("preprocess_s"),
                    "sampling": timing.get("sampling_s"),
                    "total": timing.get("total_s"),
                }.items()
                if v is not None
            },
            "load_time_s": timing.get("load_s"),
            "encode_time_s": None,
            "sampling_time_s": timing.get("sampling_s"),
            "decode_time_s": None,
            "total_latency_s": timing.get("total_s"),
            "vram": sample_meta.get("vram_after_sample_mb")
            or self._load_meta.get("vram_after_load_mb"),
            "vram_peak_mb": sample_meta.get("vram_peak_mb"),
            "load_meta": dict(self._load_meta or {}),
            **sample_meta,
            "latency_s": round(latency_s, 4),
            "note_encode_decode": (
                "OmniGen2Pipeline __call__ bundles encode/decode inside sampling; "
                "encode_time_s/decode_time_s are null by design."
            ),
        }
        if run_error is not None:
            meta["run_error"] = str(run_error)
            meta["run_error_type"] = type(run_error).__name__

        print(
            f"[omnigen2] model={MODEL_VERSION} steps={steps} "
            f"text_cfg={text_guidance} image_cfg={image_guidance} "
            f"lat={latency_s:.2f}s peak_vram={meta.get('vram_peak_mb')}"
        )

        if run_error is not None:
            raise PipelineRunError(
                str(run_error),
                meta=meta,
                latency_s=latency_s,
                image=out,
                debug_paths=dbg,
            ) from run_error
        if out is None:
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


# Public alias matching registry naming expectations.
OmniGen2HeadSwapPipeline = OmniGen2PipelineRunner
