"""GHOST 2.0 one-shot head swap pipeline wrapper.

Upstream: https://github.com/ai-forever/ghost-2.0
Runs scripts/run_ghost2_swap.py against a local GHOST2_ROOT checkout.
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


class Ghost2HeadSwapPipeline(BasePipeline):
    name = "ghost2_head_swap"

    def run(
        self, body: Image.Image, face: Image.Image, out_dir: Path | None = None
    ) -> PipelineResult:
        t0 = time.perf_counter()
        ghost_root = Path(
            self.cfg.get("ghost2_root")
            or os.environ.get("GHOST2_ROOT")
            or "/content/ghost-2.0"
        )
        if not ghost_root.is_dir():
            raise PipelineRunError(
                f"GHOST2_ROOT missing: {ghost_root}. Run scripts/setup_ghost2_colab.sh first."
            )

        work = Path(out_dir) if out_dir is not None else Path(self.cache_dir) / "ghost2_run"
        work.mkdir(parents=True, exist_ok=True)
        body_path = work / "body.png"
        face_path = work / "face.png"
        result_path = work / "result.png"
        body.convert("RGB").save(body_path)
        face.convert("RGB").save(face_path)

        # ghost2.py → pipelines → headswap → src → repo root
        repo_root = Path(__file__).resolve().parents[3]
        script = repo_root / "scripts" / "run_ghost2_swap.py"
        if not script.is_file():
            here = Path(__file__).resolve().parent
            script = None
            for parent in [here, *here.parents]:
                candidate = parent / "scripts" / "run_ghost2_swap.py"
                if candidate.is_file():
                    script = candidate
                    repo_root = parent
                    break
            if script is None:
                raise PipelineRunError(
                    "Cannot find scripts/run_ghost2_swap.py relative to "
                    f"{Path(__file__).resolve()}"
                )
        cmd = [
            sys.executable,
            str(script),
            "--ghost-root",
            str(ghost_root),
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
        ]
        if bool(self.cfg.get("use_kandi", False)):
            cmd.append("--use-kandi")

        env = os.environ.copy()
        env["GHOST2_ROOT"] = str(ghost_root)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ghost_root),
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise PipelineRunError(f"GHOST 2.0 launch failed: {exc}") from exc

        if proc.stdout:
            print(proc.stdout, flush=True)
        if proc.returncode != 0:
            raise PipelineRunError(
                "GHOST 2.0 failed:\n"
                + (proc.stderr or proc.stdout or f"exit={proc.returncode}")
            )
        if not result_path.is_file():
            raise PipelineRunError(f"GHOST 2.0 produced no result at {result_path}")

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
                "ghost2_root": str(ghost_root),
                "body_face_policy": self.cfg.get("body_face_policy", "largest"),
                "body_face_index": int(self.cfg.get("body_face_index", 0)),
                "use_kandi": bool(self.cfg.get("use_kandi", False)),
                "output_long_side": int(self.cfg.get("output_long_side", 1024) or 0),
            },
            debug_paths=dbg,
        )
