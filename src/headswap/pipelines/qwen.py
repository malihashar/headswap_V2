from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from PIL import Image

from headswap.comfy.runtime import NodeRuntime, comfy_tensor_to_pil, get_value_at_index, pil_to_comfy_tensor
from headswap.comfy.full_load import force_sampling_full_load
from headswap.pipelines.base import BasePipeline, PipelineResult, build_prompt
from headswap.pipelines.errors import PipelineRunError
from headswap.preprocess import (
    crop_face_reference,
    crop_with_mask,
    head_hair_mask_from_face,
    lab_histogram_match_face,
    pad_to_ar_blur,
    resize_long_side,
    resize_max_keep_ar,
    soft_composite,
)
from headswap.profiling.gpu_stages import (
    GpuStageProfiler,
    count_sigmas,
    describe_latent,
    reset_vram_peak,
)
from headswap.profiling.reporting import emit_profile_report, profile_timing_meta
from headswap.profiling.vae_bridge_probe import install_vae_decode_probes, print_vae_probe
from headswap.profiling.vram_load_probe import vram_load_probe


@contextmanager
def _stage(timings: dict[str, float] | None, name: str) -> Iterator[None]:
    """Accumulate wall time for a named stage when timings is provided."""
    if timings is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)


@contextmanager
def _track(
    profiler: GpuStageProfiler | None,
    timings: dict[str, float] | None,
    name: str,
    **notes,
) -> Iterator[None]:
    if profiler is not None:
        with profiler.stage(name, **notes):
            yield
    else:
        with _stage(timings, name):
            yield


def _print_timing_breakdown(
    label: str,
    timings: dict[str, float],
    total_s: float,
    *,
    notes: dict[str, str] | None = None,
) -> None:
    notes = notes or {}
    order = [
        "preprocessing",
        "model_loading",
        "lora_loading",
        "flux_kontext_image_scale",
        "vae_encode",
        "text_encoding",
        "diffusion_sampling",
        "vae_decode",
        "postprocessing",
        "image_saving",
    ]
    print(f"[{label} timing] total={total_s:.2f}s")
    accounted = 0.0
    for name in order:
        if name not in timings:
            continue
        sec = float(timings[name])
        accounted += sec
        pct = (100.0 * sec / total_s) if total_s > 0 else 0.0
        note = f"  [{notes[name]}]" if name in notes else ""
        print(f"  {name:<28} {sec:7.2f}s  ({pct:5.1f}%){note}")
    other = total_s - accounted
    if abs(other) >= 0.005:
        pct = (100.0 * other / total_s) if total_s > 0 else 0.0
        print(f"  {'other':<28} {other:7.2f}s  ({pct:5.1f}%)")


def _load_qwen_stack(
    rt: NodeRuntime,
    cfg: dict,
    timings: dict[str, float] | None = None,
    profiler: GpuStageProfiler | None = None,
):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    key = (
        f"qwen::{cfg.get('unet_name')}::hs{cfg.get('headswap_lora_strength')}"
        f"::lt{cfg.get('lightning_lora_strength')}::sh{cfg.get('auraflow_shift')}::st{cfg.get('steps')}"
    )
    if key in rt.models:
        if profiler is not None:
            profiler.note("model_cache_hit", True)
            with profiler.stage("model_loading", cache_hit=True):
                pass
            with profiler.stage("lora_loading", cache_hit=True):
                pass
        elif timings is not None:
            timings["model_loading"] = timings.get("model_loading", 0.0)
            timings["lora_loading"] = timings.get("lora_loading", 0.0)
            timings["_model_cache_hit"] = 1.0
        return rt.models[key]

    if profiler is not None:
        profiler.note("model_cache_hit", False)

    checkpoint = cfg["unet_name"]
    load_meta: dict = {
        "checkpoint": checkpoint,
        "checkpoint_preferred": checkpoint,
        "checkpoint_fallback_used": False,
        "loras_loaded": [],
        "lora_strengths": {},
        "fallbacks": [],
    }

    with _track(profiler, timings, "model_loading"):
        vae = rt.call("VAELoader", vae_name=cfg["vae_name"])
        clip = rt.call(
            "CLIPLoader",
            clip_name=cfg["clip_name"],
            type=cfg.get("clip_type", "qwen_image"),
            device="default",
        )
        unet = rt.call("UNETLoader", unet_name=checkpoint, weight_dtype="default")
        model = get_value_at_index(unet, 0)

    with _track(profiler, timings, "lora_loading"):
        hs_name = cfg["headswap_lora_name"]
        hs_strength = float(cfg.get("headswap_lora_strength", 1.0))
        model = get_value_at_index(
            rt.call(
                "LoraLoaderModelOnly",
                model=model,
                lora_name=hs_name,
                strength_model=hs_strength,
            ),
            0,
        )
        load_meta["loras_loaded"].append(hs_name)
        load_meta["lora_strengths"][hs_name] = hs_strength
        print(f"[qwen] loaded LoRA {hs_name} strength={hs_strength}")

        lt = float(cfg.get("lightning_lora_strength", 0) or 0)
        lt_name = cfg.get("lightning_lora_name")
        if lt > 0 and lt_name:
            model = get_value_at_index(
                rt.call(
                    "LoraLoaderModelOnly",
                    model=model,
                    lora_name=lt_name,
                    strength_model=lt,
                ),
                0,
            )
            load_meta["loras_loaded"].append(lt_name)
            load_meta["lora_strengths"][lt_name] = lt
            print(f"[qwen] loaded LoRA {lt_name} strength={lt}")
        elif lt_name:
            load_meta["fallbacks"].append(f"lightning_lora_skipped_strength_zero:{lt_name}")

        if rt.has("ModelSamplingAuraFlow"):
            model = get_value_at_index(
                rt.call(
                    "ModelSamplingAuraFlow",
                    model=model,
                    shift=float(cfg.get("auraflow_shift", 5)),
                ),
                0,
            )
        else:
            load_meta["fallbacks"].append("model_sampling_auraflow_missing")
        if rt.has("CFGNorm"):
            model = get_value_at_index(
                rt.call(
                    "CFGNorm",
                    model=model,
                    strength=float(cfg.get("cfg_norm_strength", 1.0)),
                ),
                0,
            )
        else:
            load_meta["fallbacks"].append("cfgnorm_missing")

    bundle = {
        "model": model,
        "clip": get_value_at_index(clip, 0),
        "vae": get_value_at_index(vae, 0),
        "load_meta": load_meta,
    }
    rt.models[key] = bundle
    return bundle


def _sample_qwen(
    rt: NodeRuntime,
    bundle,
    body_t,
    face_t,
    cfg,
    prompt: str,
    timings: dict[str, float] | None = None,
    profiler: GpuStageProfiler | None = None,
):
    import torch

    fallbacks: list[str] = []
    flux_kontext_applied = False
    flux_kontext_image_scale_applied = False
    use_kontext_scale = bool(cfg.get("flux_kontext_image_scale", False))
    if profiler is not None:
        profiler.note("flux_kontext_image_scale_enabled", use_kontext_scale)
        profiler.note(
            "bundle_object_ids",
            {
                "model": id(bundle["model"]),
                "clip": id(bundle["clip"]),
                "vae": id(bundle["vae"]),
            },
        )

    # Use no_grad — not inference_mode. ComfyUI may load/unload/partially_unload
    # model weights between nodes (especially VAEDecode after sampling). Weights
    # streamed under inference_mode become inference tensors; Parameter() then
    # raises "Cannot set version_counter for inference tensor". ComfyUI's own
    # set_attr_param only clones those tensors when inference_mode is disabled.
    with torch.no_grad():
        image1 = body_t
        input_h, input_w = int(body_t.shape[1]), int(body_t.shape[2])
        with _track(profiler, timings, "flux_kontext_image_scale"):
            if use_kontext_scale and rt.has("FluxKontextImageScale"):
                scaled = rt.call("FluxKontextImageScale", image=body_t)
                image1 = get_value_at_index(scaled, 0)
                flux_kontext_image_scale_applied = True
            elif use_kontext_scale:
                fallbacks.append("flux_kontext_image_scale_missing")
        encode_h, encode_w = int(image1.shape[1]), int(image1.shape[2])
        if profiler is not None:
            profiler.note("input_body_size", [input_w, input_h])
            profiler.note("encode_body_size", [encode_w, encode_h])
            profiler.note(
                "encode_megapixels", round((encode_w * encode_h) / 1_000_000, 3)
            )
            profiler.note(
                "face_ref_size",
                [int(face_t.shape[2]), int(face_t.shape[1])],
            )

        with _track(profiler, timings, "vae_encode"):
            body_latent = rt.call("VAEEncode", vae=bundle["vae"], pixels=image1)
        body_latent_t = get_value_at_index(body_latent, 0)
        if profiler is not None:
            profiler.note("latent_shape", describe_latent(body_latent_t))

        if not rt.has("TextEncodeQwenImageEditPlus"):
            raise KeyError("TextEncodeQwenImageEditPlus node missing — update ComfyUI")
        with _track(profiler, timings, "text_encoding"):
            pos = rt.call(
                "TextEncodeQwenImageEditPlus",
                clip=bundle["clip"],
                prompt=prompt,
                vae=bundle["vae"],
                image1=image1,
                image2=face_t,
            )
            neg = rt.call(
                "TextEncodeQwenImageEditPlus",
                clip=bundle["clip"],
                prompt=str(cfg.get("negative_prompt", "") or ""),
                vae=bundle["vae"],
                image1=image1,
                image2=face_t,
            )
            positive = get_value_at_index(pos, 0)
            negative = get_value_at_index(neg, 0)
            if rt.has("FluxKontextMultiReferenceLatentMethod"):
                positive = get_value_at_index(
                    rt.call(
                        "FluxKontextMultiReferenceLatentMethod",
                        conditioning=positive,
                        reference_latents_method="index_timestep_zero",
                    ),
                    0,
                )
                negative = get_value_at_index(
                    rt.call(
                        "FluxKontextMultiReferenceLatentMethod",
                        conditioning=negative,
                        reference_latents_method="index_timestep_zero",
                    ),
                    0,
                )
                flux_kontext_applied = True

        full_load_info = None
        if profiler is not None:
            with profiler.stage("scheduler_creation"):
                noise = get_value_at_index(
                    rt.call("RandomNoise", noise_seed=int(cfg.get("seed", 46))), 0
                )
                guider = get_value_at_index(
                    rt.call(
                        "CFGGuider",
                        model=bundle["model"],
                        positive=positive,
                        negative=negative,
                        cfg=float(cfg.get("cfg", 1.1)),
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
                        steps=int(cfg.get("steps", 6)),
                        denoise=float(cfg.get("denoise", 1.0)),
                    ),
                    0,
                )
                n_steps = count_sigmas(sigmas) or int(cfg.get("steps", 6))
                profiler.note("sampling_steps", n_steps)
                profiler.note("sampler_name", cfg.get("sampler", "euler"))
                profiler.note("scheduler_name", cfg.get("scheduler", "simple"))
            hook_ok = profiler.install_sampling_step_hook()
            profiler.note("sampling_step_hook", hook_ok)
            try:
                with force_sampling_full_load(models=(bundle["model"],)) as full_load_info:
                    profiler.note("force_sampling_full_load", full_load_info)
                    with vram_load_probe() as load_probe:
                        with profiler.stage("sampling_total"):
                            samples = rt.call(
                                "SamplerCustomAdvanced",
                                noise=noise,
                                guider=guider,
                                sampler=sampler,
                                sigmas=sigmas,
                                latent_image=body_latent_t,
                            )
                    profiler.note("vram_load_probe", load_probe.report.to_dict())
            finally:
                profiler.restore_sampling_hook()
            print("[vae_probe] MARKER reached_post_sampling_before_del_guider", flush=True)
            # Drop guider so its ModelPatcher / cond refs cannot keep CUDA tensors alive.
            del guider
            print("[vae_probe] MARKER after_del_guider", flush=True)
            print_vae_probe(
                "after_sampling_unload",
                bundle=bundle,
                extra={"has_samples": samples is not None},
            )
        else:
            with _stage(timings, "diffusion_sampling"):
                noise = get_value_at_index(
                    rt.call("RandomNoise", noise_seed=int(cfg.get("seed", 46))), 0
                )
                guider = get_value_at_index(
                    rt.call(
                        "CFGGuider",
                        model=bundle["model"],
                        positive=positive,
                        negative=negative,
                        cfg=float(cfg.get("cfg", 1.1)),
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
                        steps=int(cfg.get("steps", 6)),
                        denoise=float(cfg.get("denoise", 1.0)),
                    ),
                    0,
                )
                with force_sampling_full_load(models=(bundle["model"],)) as full_load_info:
                    samples = rt.call(
                        "SamplerCustomAdvanced",
                        noise=noise,
                        guider=guider,
                        sampler=sampler,
                        sigmas=sigmas,
                        latent_image=body_latent_t,
                    )
                print("[vae_probe] MARKER reached_post_sampling_before_del_guider", flush=True)
                del guider
                print("[vae_probe] MARKER after_del_guider", flush=True)
                print_vae_probe(
                    "after_sampling_unload",
                    bundle=bundle,
                    extra={"has_samples": samples is not None},
                )

        print("[vae_probe] MARKER entered_common_post_sampling_path", flush=True)
        # Free refs that could keep diffusion state alive across the VAE boundary.
        try:
            del noise
        except Exception:
            pass
        try:
            del sampler
        except Exception:
            pass
        try:
            del sigmas
        except Exception:
            pass
        print_vae_probe("after_del_sampler_locals", bundle=bundle)
        import gc

        gc.collect()
        print_vae_probe(
            "after_gc_collect",
            bundle=bundle,
            extra={"note": "if alloc jumps here, a finalizer/refcount is reallocating"},
        )

        latent_for_decode = get_value_at_index(samples, 0)
        latent_meta: dict = {}
        try:
            lat = (
                latent_for_decode["samples"]
                if isinstance(latent_for_decode, dict)
                else latent_for_decode
            )
            latent_meta = {
                "type": type(lat).__name__,
                "shape": list(lat.shape) if hasattr(lat, "shape") else None,
                "device": str(getattr(lat, "device", None)),
                "dtype": str(getattr(lat, "dtype", None)),
            }
        except Exception as exc:
            latent_meta = {"error": str(exc)}
        print_vae_probe(
            "immediately_before_VAEDecode_setup",
            bundle=bundle,
            extra={"latent": latent_meta},
        )

        # Install Comfy monkey-patches BEFORE profiler.stage("vae_decode"), because
        # GpuStageProfiler.stage() calls torch.cuda.synchronize() on entry and that
        # can raise a deferred CUDA OOM before rt.call("VAEDecode") runs.
        decode_probe_events: list = []
        print("[vae_probe] MARKER before_install_vae_decode_probes", flush=True)
        with install_vae_decode_probes(bundle=bundle, runtime=rt) as probe_state:
            print_vae_probe(
                "after_monkeypatch_install_before_profiler_vae_stage",
                bundle=bundle,
            )
            print(
                "[vae_probe] MARKER before_profiler_stage_vae_decode "
                "(stage entry will cuda.synchronize)",
                flush=True,
            )
            with _track(profiler, timings, "vae_decode"):
                print("[vae_probe] MARKER inside_profiler_stage_before_rt.call", flush=True)
                try:
                    decoded = rt.call(
                        "VAEDecode",
                        samples=latent_for_decode,
                        vae=bundle["vae"],
                    )
                except BaseException as decode_exc:
                    print_vae_probe(
                        "rt.call(VAEDecode) RAISED",
                        bundle=bundle,
                        extra={
                            "error": str(decode_exc),
                            "error_type": type(decode_exc).__name__,
                        },
                    )
                    raise
                decode_probe_events = list(probe_state.get("events") or [])
            print_vae_probe(
                "after_VAEDecode_success",
                bundle=bundle,
                extra={"decode_events": len(decode_probe_events)},
            )
        image = comfy_tensor_to_pil(get_value_at_index(decoded, 0))
        sample_meta = {
            "reference_latent_used": False,
            "flux_kontext_applied": flux_kontext_applied,
            "flux_kontext_image_scale_applied": flux_kontext_image_scale_applied,
            "flux_kontext_image_scale_enabled": use_kontext_scale,
            "input_body_size": [input_w, input_h],
            "encode_body_size": [encode_w, encode_h],
            "encode_megapixels": round((encode_w * encode_h) / 1_000_000, 3),
            "fallbacks": fallbacks,
            "force_sampling_full_load": full_load_info,
            "vae_probe_events": decode_probe_events,
        }
        if profiler is not None and "vram_load_probe" in profiler.extras:
            sample_meta["vram_load_probe"] = profiler.extras["vram_load_probe"]
        return image, sample_meta


class QwenBaselinePipeline(BasePipeline):
    """Faithful port of Magic Hour Colab Cell 5 (~28 s warm GPU path)."""

    name = "qwen_baseline"

    def _ensure_runtime(self) -> NodeRuntime:
        if self.runtime is None:
            self.runtime = NodeRuntime()
        return self.runtime

    def run(self, body: Image.Image, face: Image.Image, out_dir: Path | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        profiler = GpuStageProfiler()
        reset_vram_peak()
        run_error: BaseException | None = None
        out: Image.Image | None = None
        sample_meta: dict = {}
        body_pil: Image.Image | None = None
        face_crop: Image.Image | None = None
        face_for_model: Image.Image | None = None
        dbg: dict[str, str] = {}
        load_meta: dict = {}
        fallbacks: list[str] = []
        prompt = str(self.cfg.get("prompt", "")).strip()

        try:
            import torch

            t_bootstrap = time.perf_counter()
            rt = self._ensure_runtime()
            profiler.note("bootstrap_s", round(time.perf_counter() - t_bootstrap, 4))
            bundle = _load_qwen_stack(rt, self.cfg, profiler=profiler)
            div_by = int(self.cfg.get("div_by", 8))
            max_dim = int(self.cfg.get("max_dim", 576))

            with profiler.stage("preprocessing"):
                body_pil = resize_max_keep_ar(body.convert("RGB"), max_dim, div_by)
                face_crop = crop_face_reference(
                    face,
                    self.cache_dir,
                    top=float(self.cfg.get("face_top_pad", 0.65)),
                    bot=float(self.cfg.get("face_bot_pad", 0.15)),
                    side=float(self.cfg.get("face_side_pad", 0.35)),
                    include_shoulders=False,
                )
                if self.cfg.get("blur_pad_face", True):
                    face_for_model = pad_to_ar_blur(
                        face_crop, body_pil.width / body_pil.height
                    ).resize(body_pil.size, Image.Resampling.LANCZOS)
                else:
                    face_for_model = face_crop.resize(body_pil.size, Image.Resampling.LANCZOS)
                body_t = pil_to_comfy_tensor(body_pil, torch)
                face_t = pil_to_comfy_tensor(face_for_model, torch)
                profiler.note("preprocess_body_size", list(body_pil.size))

            out, sample_meta = _sample_qwen(
                rt,
                bundle,
                body_t,
                face_t,
                self.cfg,
                prompt,
                profiler=profiler,
            )
            with profiler.stage("postprocessing"):
                pass

            load_meta = dict(bundle.get("load_meta") or {})
            fallbacks = list(load_meta.get("fallbacks") or []) + list(
                sample_meta.get("fallbacks") or []
            )
            with profiler.stage("image_saving"):
                dbg = {
                    k: v
                    for k, v in {
                        "debug_body": self._save_debug(out_dir, "debug_body.png", body_pil),
                        "debug_face_crop": self._save_debug(out_dir, "debug_face_crop.png", face_crop),
                        "debug_face_for_model": self._save_debug(
                            out_dir, "debug_face_for_model.png", face_for_model
                        ),
                    }.items()
                    if v
                }
        except BaseException as exc:
            run_error = exc
        finally:
            latency_s = time.perf_counter() - t0
            crop_size = list(body_pil.size) if body_pil is not None else None
            meta = {
                "pipeline": self.name,
                "checkpoint": load_meta.get("checkpoint"),
                "loras_loaded": list(load_meta.get("loras_loaded") or []),
                "lora_strengths": dict(load_meta.get("lora_strengths") or {}),
                "prompt": prompt,
                "crop_size": crop_size,
                "body_size": crop_size,
                "reference_latent_used": bool(sample_meta.get("reference_latent_used")),
                "flux_kontext_applied": bool(sample_meta.get("flux_kontext_applied")),
                "flux_kontext_image_scale_applied": bool(
                    sample_meta.get("flux_kontext_image_scale_applied")
                ),
                "flux_kontext_image_scale_enabled": bool(
                    sample_meta.get("flux_kontext_image_scale_enabled")
                ),
                "input_body_size": sample_meta.get("input_body_size"),
                "encode_body_size": sample_meta.get("encode_body_size"),
                "encode_megapixels": sample_meta.get("encode_megapixels"),
                "fallbacks": fallbacks,
                "timing_s": profile_timing_meta(profiler),
                "profile": profiler.to_dict(),
                "latency_s": round(latency_s, 4),
            }
            if sample_meta.get("vram_load_probe") is not None:
                meta["vram_load_probe"] = sample_meta["vram_load_probe"]
            if sample_meta.get("force_sampling_full_load") is not None:
                meta["force_sampling_full_load"] = sample_meta["force_sampling_full_load"]
            if run_error is not None:
                meta["run_error"] = str(run_error)
                meta["run_error_type"] = type(run_error).__name__

            print(
                f"[qwen_baseline] checkpoint={meta.get('checkpoint')} loras={meta.get('loras_loaded')} "
                f"strengths={meta.get('lora_strengths')} crop={meta.get('crop_size')} "
                f"flux_kontext={meta.get('flux_kontext_applied')} "
                f"image_scale={meta.get('flux_kontext_image_scale_applied')} "
                f"scale_enabled={meta.get('flux_kontext_image_scale_enabled')} "
                f"encode_mp={meta.get('encode_megapixels')} "
                f"fallbacks={fallbacks or 'none'}"
            )
            emit_profile_report(
                profiler,
                total_s=latency_s,
                label="qwen_baseline",
                error=str(run_error) if run_error is not None else None,
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


class QwenImprovedPipeline(BasePipeline):
    """Qwen 2511 + BFS with official prompt, mask crop/stitch, no blur-pad."""

    name = "qwen_improved_mask_crop"

    def _ensure_runtime(self) -> NodeRuntime:
        if self.runtime is None:
            self.runtime = NodeRuntime()
        return self.runtime

    def run(self, body: Image.Image, face: Image.Image, out_dir: Path | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        profiler = GpuStageProfiler()
        reset_vram_peak()
        run_error: BaseException | None = None
        out: Image.Image | None = None
        sample_meta: dict = {}
        body_full: Image.Image | None = None
        face_crop: Image.Image | None = None
        face_ref: Image.Image | None = None
        crop_work: Image.Image | None = None
        mask = None
        box = None
        edited: Image.Image | None = None
        dbg: dict[str, str] = {}
        load_meta: dict = {}
        fallbacks: list[str] = []
        prompt = ""

        try:
            import torch

            t_bootstrap = time.perf_counter()
            rt = self._ensure_runtime()
            profiler.note("bootstrap_s", round(time.perf_counter() - t_bootstrap, 4))
            bundle = _load_qwen_stack(rt, self.cfg, profiler=profiler)
            div_by = int(self.cfg.get("div_by", 8))

            with profiler.stage("preprocessing"):
                body_full = resize_max_keep_ar(
                    body.convert("RGB"), int(self.cfg.get("max_body_dim", 1024)), div_by
                )
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
                )
                crop_img, _, box = crop_with_mask(body_full, mask, pad=12, div_by=div_by)
                crop_work = resize_long_side(
                    crop_img, int(self.cfg.get("crop_long_side", 768)), div_by
                )
                if self.cfg.get("blur_pad_face", False):
                    face_ref = pad_to_ar_blur(
                        face_crop, crop_work.width / crop_work.height
                    ).resize(crop_work.size, Image.Resampling.LANCZOS)
                else:
                    face_ref = face_crop.resize(crop_work.size, Image.Resampling.LANCZOS)
                prompt = build_prompt(self.cfg, body_full, face_crop, self.cache_dir)
                crop_t = pil_to_comfy_tensor(crop_work, torch)
                face_t = pil_to_comfy_tensor(face_ref, torch)
                profiler.note("preprocess_body_size", list(body_full.size))
                profiler.note("preprocess_crop_size", list(crop_work.size))

            edited, sample_meta = _sample_qwen(
                rt,
                bundle,
                crop_t,
                face_t,
                self.cfg,
                prompt,
                profiler=profiler,
            )
            with profiler.stage("postprocessing"):
                out = soft_composite(body_full, edited, mask, box)
                out = lab_histogram_match_face(out, body_full, mask, strength=0.3)

            load_meta = dict(bundle.get("load_meta") or {})
            fallbacks = list(load_meta.get("fallbacks") or []) + list(
                sample_meta.get("fallbacks") or []
            )
            with profiler.stage("image_saving"):
                dbg = {
                    k: v
                    for k, v in {
                        "debug_body": self._save_debug(out_dir, "debug_body.png", body_full),
                        "debug_face_crop": self._save_debug(
                            out_dir, "debug_face_crop.png", face_crop
                        ),
                        "debug_mask": self._save_debug(out_dir, "debug_mask.png", mask),
                        "debug_crop": self._save_debug(out_dir, "debug_crop.png", crop_work),
                        "debug_edited_crop": self._save_debug(
                            out_dir, "debug_edited_crop.png", edited
                        ),
                    }.items()
                    if v
                }
        except BaseException as exc:
            run_error = exc
        finally:
            latency_s = time.perf_counter() - t0
            meta = {
                "pipeline": self.name,
                "checkpoint": load_meta.get("checkpoint"),
                "loras_loaded": list(load_meta.get("loras_loaded") or []),
                "lora_strengths": dict(load_meta.get("lora_strengths") or {}),
                "prompt": prompt,
                "crop_size": list(crop_work.size) if crop_work is not None else None,
                "body_size": list(body_full.size) if body_full is not None else None,
                "face_ref_size": list(face_ref.size) if face_ref is not None else None,
                "reference_latent_used": bool(sample_meta.get("reference_latent_used")),
                "flux_kontext_applied": bool(sample_meta.get("flux_kontext_applied")),
                "flux_kontext_image_scale_applied": bool(
                    sample_meta.get("flux_kontext_image_scale_applied")
                ),
                "flux_kontext_image_scale_enabled": bool(
                    sample_meta.get("flux_kontext_image_scale_enabled")
                ),
                "input_body_size": sample_meta.get("input_body_size"),
                "encode_body_size": sample_meta.get("encode_body_size"),
                "encode_megapixels": sample_meta.get("encode_megapixels"),
                "fallbacks": fallbacks,
                "timing_s": profile_timing_meta(profiler),
                "profile": profiler.to_dict(),
                "latency_s": round(latency_s, 4),
            }
            if sample_meta.get("vram_load_probe") is not None:
                meta["vram_load_probe"] = sample_meta["vram_load_probe"]
            if sample_meta.get("force_sampling_full_load") is not None:
                meta["force_sampling_full_load"] = sample_meta["force_sampling_full_load"]
            if run_error is not None:
                meta["run_error"] = str(run_error)
                meta["run_error_type"] = type(run_error).__name__

            print(
                f"[qwen_improved] checkpoint={meta.get('checkpoint')} loras={meta.get('loras_loaded')} "
                f"strengths={meta.get('lora_strengths')} crop={meta.get('crop_size')} "
                f"flux_kontext={meta.get('flux_kontext_applied')} "
                f"image_scale={meta.get('flux_kontext_image_scale_applied')} "
                f"scale_enabled={meta.get('flux_kontext_image_scale_enabled')} "
                f"encode_mp={meta.get('encode_megapixels')} "
                f"fallbacks={fallbacks or 'none'}"
            )
            emit_profile_report(
                profiler,
                total_s=latency_s,
                label="qwen_improved",
                error=str(run_error) if run_error is not None else None,
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
