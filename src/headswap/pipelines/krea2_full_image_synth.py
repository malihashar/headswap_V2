"""Experimental Krea2 full-image identity synthesis (no crop/stitch).

Isolated from the production localized pipeline. Regenerates the entire frame
via dual-ref Krea2 Identity Edit and returns the raw model output as final.

Identity-path fixes (evidence-backed):
  - Do NOT letterbox identity into the scene canvas (was ~23% face pixels →
    scene geometry dominated dual-ref VAE tokens). Pass a face-filled square;
    v1.2 fit_mode handles AR mismatch.
  - Prompt uses trained head/face/hair *replace* vocabulary; never asks to
    preserve the target's hair/beard/jaw.
  - ref_boost_a = 1.0 (do not boost scene); ref_boost elevated for image-2 ID.
  - grounding_px = 768 (docs: lower → stronger edit adherence for changes).
  - Optional ref_boost_mask focuses image-2 attention on the face ellipse.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from headswap.comfy.krea2_edit_fast import install_krea2_edit_static_cache
from headswap.pipelines.base import BasePipeline, PipelineResult
from headswap.pipelines.errors import PipelineRunError
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline, get_shared_krea2_runtime
from headswap.preprocess import (
    identity_face_boost_mask,
    prepare_krea2_identity_person,
    resize_max_keep_ar,
)
from headswap.prompting.scene_describe import (
    build_identity_edit_prompt,
    describe_scene,
)


class Krea2FullImageSynthPipeline(BasePipeline):
    """Full-frame Krea2 synthesis — no mask, crop, stitch, paste, or blend."""

    name = "krea2_full_image_synth"

    def _ensure_runtime(self):
        if self.runtime is None:
            self.runtime = get_shared_krea2_runtime(init_custom_nodes=True)
        return self.runtime

    def _draw_face_overlay(
        self,
        body: Image.Image,
        selected,
        all_faces,
    ) -> Image.Image:
        vis = body.convert("RGB").copy()
        draw = ImageDraw.Draw(vis)
        for i, f in enumerate(all_faces or []):
            color = (0, 255, 80) if (
                selected is not None
                and f.x0 == selected.x0
                and f.y0 == selected.y0
                and f.x1 == selected.x1
                and f.y1 == selected.y1
            ) else (255, 80, 80)
            draw.rectangle([f.x0, f.y0, f.x1, f.y1], outline=color, width=3)
            draw.text((f.x0 + 2, max(0, f.y0 - 12)), f"f{i}", fill=color)
        return vis

    def run(
        self, body: Image.Image, face: Image.Image, out_dir: Path | None = None
    ) -> PipelineResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        dbg: dict[str, str] = {}
        meta: dict[str, Any] = {
            "pipeline": self.name,
            "mode": "full_image_synth_raw",
            "postprocess": "none",
        }

        # Delegate load/sample to production helpers without using its run().
        engine = Krea2IdentityEditPipeline(
            self.cfg, runtime=self.runtime, cache_dir=self.cache_dir
        )

        try:
            t_boot = time.perf_counter()
            rt = self._ensure_runtime()
            engine.runtime = rt
            timings["bootstrap"] = time.perf_counter() - t_boot

            bundle = engine._load_models(rt, timings)
            edit_cache_info = install_krea2_edit_static_cache()

            div_by = int(self.cfg.get("div_by", 16))
            max_dim = int(self.cfg.get("max_dim", 1024))

            t_pre = time.perf_counter()
            body_rgb = body.convert("RGB")
            desc, selected, all_faces = describe_scene(
                body_rgb,
                self.cache_dir,
                face_index=int(self.cfg.get("face_index", 0)),
                face_policy=str(self.cfg.get("face_select_policy", "largest")),
                cfg=self.cfg,
            )
            prompt = build_identity_edit_prompt(
                desc,
                instruction_suffix=self.cfg.get("instruction_suffix") or None,
            )
            if self.cfg.get("prompt_prefix"):
                prompt = f"{str(self.cfg['prompt_prefix']).strip()} {prompt}".strip()

            scene = resize_max_keep_ar(body_rgb, max_dim, div_by=div_by)
            # Face-dominant identity canvas (NOT letterboxed into scene size).
            person, face_crop, id_prep = prepare_krea2_identity_person(
                face,
                self.cache_dir,
                long_side=int(self.cfg.get("identity_long_side", 768)),
                div_by=div_by,
                top=float(self.cfg.get("face_top_pad", 0.70)),
                bot=float(self.cfg.get("face_bot_pad", 0.12)),
                side=float(self.cfg.get("face_side_pad", 0.22)),
                white_bg=bool(self.cfg.get("identity_white_bg", True)),
                square_fill=bool(self.cfg.get("identity_square_fill", True)),
                fill_frac=float(self.cfg.get("identity_fill_frac", 0.92)),
            )
            ref_boost_mask = None
            if bool(self.cfg.get("use_ref_boost_mask", True)):
                ref_boost_mask = identity_face_boost_mask(person, self.cache_dir)
            timings["preprocessing"] = time.perf_counter() - t_pre

            if id_prep.get("identity_starved"):
                print(
                    "[krea2_full_image_synth] WARNING identity canvas still weak: "
                    f"{id_prep}",
                    file=sys.__stdout__,
                    flush=True,
                )

            overlay = self._draw_face_overlay(body_rgb, selected, all_faces)
            print(
                f"[krea2_full_image_synth] faces={len(all_faces)} "
                f"selected={desc.selected_role} scene={scene.size} "
                f"person={person.size} id_frac={id_prep.get('identity_face_area_frac')} "
                f"ref_boost={self.cfg.get('ref_boost')}/"
                f"{self.cfg.get('ref_boost_a')} "
                f"grounding_px={self.cfg.get('grounding_px')} "
                f"steps={self.cfg.get('steps')} "
                f"ref_boost_mask={ref_boost_mask is not None}",
                file=sys.__stdout__,
                flush=True,
            )
            print(
                f"[krea2_full_image_synth] prompt={prompt[:280]}"
                f"{'…' if len(prompt) > 280 else ''}",
                file=sys.__stdout__,
                flush=True,
            )

            sample = engine._sample_edit(
                rt,
                bundle,
                scene,
                person,
                timings,
                prompt=prompt,
                edit_cache_info=edit_cache_info,
                seed=int(self.cfg.get("seed", 46)),
                ref_boost_mask=ref_boost_mask,
            )
            out = sample["edited"]

            # Intentionally no freeze / stitch / color match / head-scale clamp.
            t_save = time.perf_counter()
            if out_dir is not None:
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                dbg = {
                    k: v
                    for k, v in {
                        "debug_body": self._save_debug(out_dir, "debug_body.png", body_rgb),
                        "debug_scene": self._save_debug(out_dir, "debug_scene.png", scene),
                        "debug_person": self._save_debug(out_dir, "debug_person.png", person),
                        "debug_face_crop": self._save_debug(
                            out_dir, "debug_face_crop.png", face_crop
                        ),
                        "debug_faces_overlay": self._save_debug(
                            out_dir, "debug_faces_overlay.png", overlay
                        ),
                        "debug_ref_boost_mask": (
                            self._save_debug(
                                out_dir, "debug_ref_boost_mask.png", ref_boost_mask
                            )
                            if ref_boost_mask is not None
                            else None
                        ),
                        "result_raw": self._save_debug(out_dir, "result_raw.png", out),
                    }.items()
                    if v
                }
            timings["image_saving"] = time.perf_counter() - t_save

            load_meta = dict(bundle.get("load_meta") or {})
            latency_s = time.perf_counter() - t0
            meta.update(
                {
                    "checkpoint": load_meta.get("checkpoint"),
                    "loras_loaded": list(load_meta.get("loras_loaded") or []),
                    "prompt": prompt,
                    "scene_description": desc.to_dict(),
                    "identity_prep": id_prep,
                    "faces_detected": len(all_faces),
                    "selected_face": (
                        None
                        if selected is None
                        else [selected.x0, selected.y0, selected.x1, selected.y1]
                    ),
                    "scene_size": list(scene.size),
                    "person_size": list(person.size),
                    "ref_boost": float(self.cfg.get("ref_boost", 5.0)),
                    "ref_boost_a": float(self.cfg.get("ref_boost_a", 1.0)),
                    "ref_boost_mask_used": bool(sample.get("ref_boost_mask_used")),
                    "grounding_px": int(self.cfg.get("grounding_px", 768)),
                    "steps": int(self.cfg.get("steps", 12)),
                    "cfg": float(self.cfg.get("cfg", 1.0)),
                    "fit_mode": str(self.cfg.get("fit_mode", "fit")),
                    "denoise": float(self.cfg.get("denoise", 1.0)),
                    "max_dim": max_dim,
                    "timing_s": {k: round(v, 4) for k, v in timings.items()},
                    "sample_meta": {
                        k: sample.get(k)
                        for k in (
                            "negative_mode",
                            "empty_node",
                            "sampling_diag",
                        )
                        if k in sample
                    },
                    "latency_s": round(latency_s, 4),
                }
            )
            return PipelineResult(
                image=out,
                latency_s=latency_s,
                meta=meta,
                debug_paths=dbg,
            )
        except BaseException as exc:
            latency_s = time.perf_counter() - t0
            meta["run_error"] = str(exc)
            meta["run_error_type"] = type(exc).__name__
            meta["timing_s"] = {k: round(v, 4) for k, v in timings.items()}
            meta["latency_s"] = round(latency_s, 4)
            raise PipelineRunError(
                str(exc),
                meta=meta,
                latency_s=latency_s,
                debug_paths=dbg,
            ) from exc
