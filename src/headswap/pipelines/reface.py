"""REFace (WACV 2025) face-swap pipeline wrapper.

Upstream: https://github.com/Sanoojan/REFace
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from headswap.pipelines.base import BasePipeline, PipelineResult
from headswap.pipelines.errors import PipelineRunError


def _find_run_reface_script() -> Path:
    """Locate scripts/run_reface_swap.py regardless of editable-install layout."""
    candidates: list[Path] = []
    env_repo = os.environ.get("HEADSWAP_REPO") or os.environ.get("REPO")
    if env_repo:
        candidates.append(Path(env_repo) / "scripts" / "run_reface_swap.py")
    candidates.extend(
        [
            Path("/content/headswap_V2/scripts/run_reface_swap.py"),
            Path(__file__).resolve().parents[3] / "scripts" / "run_reface_swap.py",
            Path(__file__).resolve().parents[2] / "scripts" / "run_reface_swap.py",
        ]
    )
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        candidates.append(parent / "scripts" / "run_reface_swap.py")

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return cand.resolve()
    raise PipelineRunError(
        "Cannot find scripts/run_reface_swap.py. Tried:\n  - "
        + "\n  - ".join(str(c) for c in list(seen)[:12])
        + "\nRun: cd /content/headswap_V2 && git pull && pip install -e ."
    )


class REFacePipeline(BasePipeline):
    name = "reface"

    def run(
        self, body: Image.Image, face: Image.Image, out_dir: Path | None = None
    ) -> PipelineResult:
        t0 = time.perf_counter()
        reface_root = Path(
            self.cfg.get("reface_root")
            or os.environ.get("REFACE_ROOT")
            or "/content/REFace"
        )
        if not reface_root.is_dir():
            raise PipelineRunError(
                f"REFACE_ROOT missing: {reface_root}. Run scripts/setup_reface_colab.sh first."
            )

        work = Path(out_dir) if out_dir is not None else Path(self.cache_dir) / "reface_run"
        work.mkdir(parents=True, exist_ok=True)
        body_path = work / "body.png"
        face_path = work / "face.png"
        result_path = work / "result.png"
        body.convert("RGB").save(body_path)
        face.convert("RGB").save(face_path)

        script = _find_run_reface_script()
        print(f"[reface] using runner {script}", flush=True)
        cmd = [
            sys.executable,
            str(script),
            "--reface-root",
            str(reface_root),
            "--source",
            str(face_path),
            "--target",
            str(body_path),
            "--save-path",
            str(result_path),
            "--face-policy",
            str(self.cfg.get("body_face_policy", "largest")),
            "--face-index",
            str(int(self.cfg.get("body_face_index", 0))),
            "--output-long-side",
            str(int(self.cfg.get("output_long_side", 1024) or 0)),
            "--ddim-steps",
            str(int(self.cfg.get("ddim_steps", 50))),
            "--scale",
            str(float(self.cfg.get("guidance_scale", 3.5))),
            "--seed",
            str(int(self.cfg.get("seed", 42))),
        ]
        env = os.environ.copy()
        env["REFACE_ROOT"] = str(reface_root)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(reface_root),
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise PipelineRunError(f"REFace launch failed: {exc}") from exc

        if proc.stdout:
            print(proc.stdout, flush=True)
        if proc.returncode != 0:
            raise PipelineRunError(
                "REFace failed:\n"
                + (proc.stderr or proc.stdout or f"exit={proc.returncode}")
            )
        if not result_path.is_file():
            raise PipelineRunError(f"REFace produced no result at {result_path}")

        out = Image.open(result_path).convert("RGB")
        dbg = {}
        if out_dir is not None and bool(self.cfg.get("save_debug", False)):
            dbg["debug_body"] = str(body_path)
            dbg["debug_face"] = str(face_path)
            dbg["debug_final"] = str(result_path)

        return PipelineResult(
            image=out,
            latency_s=time.perf_counter() - t0,
            meta={
                "pipeline": self.name,
                "reface_root": str(reface_root),
                "body_face_policy": self.cfg.get("body_face_policy", "largest"),
                "body_face_index": int(self.cfg.get("body_face_index", 0)),
                "ddim_steps": int(self.cfg.get("ddim_steps", 50)),
                "guidance_scale": float(self.cfg.get("guidance_scale", 3.5)),
                "seed": int(self.cfg.get("seed", 42)),
                "output_long_side": int(self.cfg.get("output_long_side", 1024) or 0),
            },
            debug_paths=dbg,
        )
