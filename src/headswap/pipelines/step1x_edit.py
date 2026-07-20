"""Step1X-Edit-v1p2 experimental head-swap backend (Diffusers).

Official sources:
  https://github.com/stepfun-ai/Step1X-Edit
  https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2

Step1X-Edit is a *single-image* instruction editor (Qwen2.5-VL conditioner +
DiT + AutoencoderKL, FlowMatchEulerDiscreteScheduler). It does not natively
accept separate body/face tensors like Qwen Image Edit.

For identity transfer we pack:
  left panel  = body  (image 1 — destination pose / clothing / background)
  right panel = face  (image 2 — source identity)
then crop the edited left panel as the result.

Requires the Peyton-Chen/diffusers ``step1xedit_v1p2`` branch
(``Step1XEditPipelineV1P2``). Does not touch ComfyUI or other pipelines.
"""
from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from headswap.pipelines.base import BasePipeline, PipelineResult
from headswap.pipelines.errors import PipelineRunError
from headswap.preprocess import crop_face_reference, resize_max_keep_ar
from headswap.profiling.gpu_stages import get_gpu_info, reset_vram_peak, vram_snapshot


MODEL_VERSION = "stepfun-ai/Step1X-Edit-v1p2"
DIFFUSERS_BRANCH = "step1xedit_v1p2"


def _resolve_model_dir(cfg: dict[str, Any]) -> Path | str:
    """Prefer local snapshot from download_step1x.py; else HF model id."""
    local = cfg.get("model_path")
    if local:
        p = Path(str(local)).expanduser()
        if (p / "model_index.json").is_file():
            return p
    mid = str(cfg.get("model_id", MODEL_VERSION))
    return mid


def _torch_dtype(name: str):
    import torch

    key = (name or "bfloat16").lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def build_dual_panel(
    body: Image.Image,
    face: Image.Image,
    *,
    size_level: int,
    label: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Pack body (image 1) and face (image 2) into one RGB frame for Step1X.

    Returns (panel, layout_meta) where layout_meta has left crop box for result.
    """
    body = body.convert("RGB")
    face = face.convert("RGB")
    # Match heights; each panel targets ~size_level on the long side of its own crop.
    target_h = max(64, int(size_level))
    def _fit_h(im: Image.Image, h: int) -> Image.Image:
        w = max(1, int(round(im.width * (h / max(1, im.height)))))
        return im.resize((w, h), Image.Resampling.LANCZOS)

    left = _fit_h(body, target_h)
    right = _fit_h(face, target_h)
    gap = 8
    panel_w = left.width + gap + right.width
    panel = Image.new("RGB", (panel_w, target_h), (24, 24, 28))
    panel.paste(left, (0, 0))
    panel.paste(right, (left.width + gap, 0))

    if label:
        draw = ImageDraw.Draw(panel)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.rectangle((0, 0, min(120, left.width), 18), fill=(0, 0, 0))
        draw.text((4, 2), "image 1", fill=(255, 255, 255), font=font)
        rx = left.width + gap
        draw.rectangle((rx, 0, rx + min(120, right.width), 18), fill=(0, 0, 0))
        draw.text((rx + 4, 2), "image 2", fill=(255, 255, 255), font=font)

    meta = {
        "panel_size": [panel.width, panel.height],
        "left_box": [0, 0, left.width, target_h],
        "right_box": [left.width + gap, 0, panel_w, target_h],
        "gap_px": gap,
        "size_level": size_level,
    }
    return panel, meta


def crop_left_panel(edited: Image.Image, layout: dict[str, Any]) -> Image.Image:
    """Recover the body-side result after dual-panel editing."""
    x0, y0, x1, y1 = layout["left_box"]
    # If the pipeline resized the panel, scale the crop box.
    pw, ph = layout["panel_size"]
    sx = edited.width / max(1, pw)
    sy = edited.height / max(1, ph)
    box = (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )
    return edited.crop(box).convert("RGB")


class Step1XEditPipeline(BasePipeline):
    """Instruction-based head swap via Step1X-Edit-v1p2 (Diffusers)."""

    name = "step1x_edit"

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
        dtype = _torch_dtype(str(self.cfg.get("dtype", "bfloat16")))
        load_meta: dict[str, Any] = {
            "model_version": MODEL_VERSION,
            "model_ref": str(model_ref),
            "dtype": str(dtype).replace("torch.", ""),
            "diffusers_branch_required": DIFFUSERS_BRANCH,
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            "text_encoder": "Qwen2_5_VLForConditionalGeneration",
            "vae": "AutoencoderKL",
            "enable_cpu_offload": bool(self.cfg.get("enable_cpu_offload", True)),
            "gpu": get_gpu_info(),
            "vram_before_load_mb": v0,
        }

        try:
            import torch
            from diffusers import Step1XEditPipelineV1P2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Step1X-Edit-v1p2 requires Diffusers with Step1XEditPipelineV1P2. "
                "Install official branch:\n"
                "  pip install 'transformers==4.55.0'\n"
                "  git clone -b step1xedit_v1p2 "
                "https://github.com/Peyton-Chen/diffusers.git && "
                "pip install -e ./diffusers\n"
                f"Original error: {exc}"
            ) from exc

        pipe = Step1XEditPipelineV1P2.from_pretrained(
            str(model_ref),
            torch_dtype=dtype,
        )
        if load_meta["enable_cpu_offload"] and hasattr(pipe, "enable_model_cpu_offload"):
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
            f"[step1x_edit] loaded {MODEL_VERSION} from {model_ref} "
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
        layout: dict[str, Any] = {}
        panel: Image.Image | None = None
        face_crop: Image.Image | None = None
        body_work: Image.Image | None = None

        prompt = str(self.cfg.get("prompt", "")).strip()
        negative = str(self.cfg.get("negative_prompt", "") or "")
        steps = int(self.cfg.get("steps", 50))
        guidance = float(self.cfg.get("guidance", self.cfg.get("true_cfg_scale", 6.0)))
        seed = int(self.cfg.get("seed", 42))
        size_level = int(self.cfg.get("crop_size", self.cfg.get("size_level", 1024)))
        strength = float(self.cfg.get("strength", 1.0))
        think = bool(self.cfg.get("enable_thinking_mode", True))
        reflect = bool(self.cfg.get("enable_reflection_mode", True))
        input_mode = str(self.cfg.get("input_mode", "dual_panel"))

        try:
            import torch

            reset_vram_peak()

            # --- preprocess ---
            t_pre = time.perf_counter()
            body_work = resize_max_keep_ar(
                body.convert("RGB"),
                max(size_level, 512),
                div_by=8,
            )
            face_crop = crop_face_reference(
                face,
                self.cache_dir,
                top=float(self.cfg.get("face_top_pad", 0.65)),
                bot=float(self.cfg.get("face_bot_pad", 0.15)),
                side=float(self.cfg.get("face_side_pad", 0.35)),
                include_shoulders=False,
            )
            if input_mode != "dual_panel":
                raise ValueError(
                    f"Unsupported input_mode={input_mode!r}; "
                    "Step1X-Edit only supports dual_panel for two-image headswap"
                )
            panel, layout = build_dual_panel(
                body_work, face_crop, size_level=size_level, label=True
            )
            timing["preprocess_s"] = round(time.perf_counter() - t_pre, 4)

            # --- load ---
            t_load = time.perf_counter()
            cold = self._pipe is None
            pipe = self._ensure_pipe()
            if cold and self._load_meta.get("load_time_s") is not None:
                timing["load_s"] = float(self._load_meta["load_time_s"])
            else:
                timing["load_s"] = round(time.perf_counter() - t_load, 4)

            # --- sample (encode + denoise + decode happen inside Diffusers __call__) ---
            # Official API has no separate encode/decode hooks; we time the call and
            # expose encode/sampling/decode as best-effort notes when unavailable.
            call_kwargs: dict[str, Any] = {
                "image": panel,
                "prompt": prompt,
                "negative_prompt": negative if negative else None,
                "num_inference_steps": steps,
                "true_cfg_scale": guidance,
                "size_level": size_level,
                "generator": torch.Generator(
                    device="cuda" if torch.cuda.is_available() else "cpu"
                ).manual_seed(seed),
                "enable_thinking_mode": think,
                "enable_reflection_mode": reflect,
            }
            # strength is NOT in the official Step1XEditPipelineV1P2 signature.
            sig = inspect.signature(pipe.__call__)
            strength_applied = False
            strength_skip = "Step1XEditPipelineV1P2_has_no_strength_param"
            if "strength" in sig.parameters:
                call_kwargs["strength"] = strength
                strength_applied = True
                strength_skip = None

            # Drop None kwargs the signature rejects awkwardly.
            if call_kwargs.get("negative_prompt") is None:
                call_kwargs.pop("negative_prompt", None)

            v_before = vram_snapshot()
            t_sample = time.perf_counter()
            result = pipe(**call_kwargs)
            timing["sampling_s"] = round(time.perf_counter() - t_sample, 4)
            v_after = vram_snapshot()

            if hasattr(result, "final_images") and result.final_images:
                edited = result.final_images[0]
            elif hasattr(result, "images") and result.images:
                edited = result.images[0]
            else:
                raise RuntimeError("Step1X pipeline returned no images")

            out = crop_left_panel(edited.convert("RGB"), layout)
            # Restore original body aspect if needed.
            if body_work is not None and out.size != body_work.size:
                out = out.resize(body_work.size, Image.Resampling.LANCZOS)

            sample_meta = {
                "steps": steps,
                "guidance": guidance,
                "true_cfg_scale": guidance,
                "seed": seed,
                "strength": strength,
                "strength_applied": strength_applied,
                "strength_skip_reason": strength_skip,
                "crop_size": size_level,
                "size_level": size_level,
                "enable_thinking_mode": think,
                "enable_reflection_mode": reflect,
                "input_mode": input_mode,
                "layout": layout,
                "vram_before_sample_mb": v_before,
                "vram_after_sample_mb": v_after,
                "vram_peak_mb": v_after.get("peak_mb"),
                "reformat_prompt": getattr(result, "reformat_prompt", None),
            }

            dbg = {
                k: v
                for k, v in {
                    "debug_body": self._save_debug(out_dir, "debug_body.png", body_work),
                    "debug_face_crop": self._save_debug(
                        out_dir, "debug_face_crop.png", face_crop
                    ),
                    "debug_dual_panel": self._save_debug(
                        out_dir, "debug_dual_panel.png", panel
                    ),
                    "debug_edited_panel": self._save_debug(
                        out_dir, "debug_edited_panel.png", edited.convert("RGB")
                    ),
                }.items()
                if v
            }
        except BaseException as exc:
            run_error = exc

        latency_s = time.perf_counter() - t0
        timing["total_s"] = round(latency_s, 4)

        meta: dict[str, Any] = {
            "pipeline": self.name,
            "model_version": MODEL_VERSION,
            "diffusers_branch_required": DIFFUSERS_BRANCH,
            "prompt": prompt,
            "negative_prompt": negative,
            "inference_settings": {
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "strength": strength,
                "crop_size": size_level,
                "enable_thinking_mode": think,
                "enable_reflection_mode": reflect,
                "dtype": self.cfg.get("dtype", "bfloat16"),
                "enable_cpu_offload": bool(self.cfg.get("enable_cpu_offload", True)),
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
            **{k: v for k, v in sample_meta.items() if k != "layout"},
            "dual_panel_layout": layout or None,
            "latency_s": round(latency_s, 4),
            "note_encode_decode": (
                "Step1X Diffusers __call__ bundles VAE encode/decode inside sampling; "
                "encode_time_s/decode_time_s are null by design."
            ),
        }
        if run_error is not None:
            meta["run_error"] = str(run_error)
            meta["run_error_type"] = type(run_error).__name__

        print(
            f"[step1x_edit] model={MODEL_VERSION} steps={steps} guidance={guidance} "
            f"size={size_level} think={think} reflect={reflect} "
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
