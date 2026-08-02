"""BasePipeline adapter for InsightFace InSwapper (+ optional Krea2 head refine).

InSwapper is always primary. Set ``krea2_refine: true`` for the hybrid path.
"""
from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from headswap.inswap.pipeline import InSwapPipeline
from headswap.pipelines.base import BasePipeline, PipelineResult


class InSwapperPipeline(BasePipeline):
    """Local ArcFace + InSwapper face swap, optional Krea2 head completion."""

    name = "inswapper"

    def __init__(self, cfg: dict, runtime=None, cache_dir: Path | None = None):
        super().__init__(cfg, runtime=runtime, cache_dir=cache_dir)
        device = str(cfg.get("device") or "cuda")
        engine = str(cfg.get("engine") or "inswapper")
        restorer = str(cfg.get("restorer") or "none")
        self._pipe = InSwapPipeline(
            cache_dir=self.cache_dir / "inswap",
            engine=engine,
            restorer=restorer,  # type: ignore[arg-type]
            device=device,
            blend_strength=float(cfg.get("blend_strength", 1.0)),
            color_match_strength=float(cfg.get("color_match_strength", 0.25)),
            restore_fidelity=float(cfg.get("restore_fidelity", 0.5)),
            reblend_after_engine=bool(cfg.get("reblend_after_engine", True)),
            krea2_refine=bool(cfg.get("krea2_refine", False)),
            krea2_force_mock=bool(cfg.get("krea2_force_mock", False)),
            krea2_required=bool(cfg.get("krea2_refine_required", False)),
            krea2_steps=(
                int(cfg["krea2_steps"]) if cfg.get("krea2_steps") is not None else None
            ),
            krea2_seed=(
                int(cfg["krea2_seed"]) if cfg.get("krea2_seed") is not None else None
            ),
            krea2_runtime=runtime,
        )
        self._loaded = False

    def run(
        self, body: Image.Image, face: Image.Image, out_dir: Path | None = None
    ) -> PipelineResult:
        t0 = time.perf_counter()
        if not self._loaded:
            self._pipe.load()
            self._loaded = True
        result = self._pipe.run(
            body,
            face,
            out_dir=out_dir,
            face_policy=str(self.cfg.get("body_face_policy", "largest")),
            face_index=int(self.cfg.get("body_face_index", 0)),
            source_face_policy=str(self.cfg.get("source_face_policy", "largest")),
            source_face_index=int(self.cfg.get("source_face_index", 0)),
            save_intermediates=bool(self.cfg.get("save_debug", True)),
            krea2_refine=bool(self.cfg.get("krea2_refine", False)),
        )
        return PipelineResult(
            image=result.image,
            latency_s=time.perf_counter() - t0,
            meta=result.meta,
            debug_paths=result.debug_paths,
        )
