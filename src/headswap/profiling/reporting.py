"""Reliable stdout profile emission (flush + error-safe fallbacks)."""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from headswap.profiling.gpu_stages import GpuStageProfiler


def flush_stdio() -> None:
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass


def profile_timing_meta(profiler: GpuStageProfiler) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in profiler.timings_dict().items()}


def emit_profile_report(
    profiler: GpuStageProfiler,
    *,
    total_s: float,
    label: str,
    error: str | None = None,
) -> None:
    """Print the full stage profile and always flush stdout/stderr."""
    if error:
        print(f"\n[{label}] pipeline error (partial profile below): {error}", file=sys.stderr)
    try:
        profiler.print_report(total_s=total_s, label=label)
    except Exception as exc:
        print(f"[{label}] profile print failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        try:
            payload = json.dumps(profiler.to_dict(), indent=2, default=str)
            print(payload[:12000])
        except Exception as dump_exc:
            print(f"[{label}] profile JSON fallback failed: {dump_exc}", file=sys.stderr)
    flush_stdio()


def emit_timing_summary(
    *,
    label: str,
    pair_id: str | None,
    latency_s: float,
    meta: dict[str, Any] | None = None,
) -> None:
    """One-line finish banner for pipelines without a full GPU profiler."""
    pair = f" {pair_id}" if pair_id else ""
    print(f"[{label}]{pair} pipeline finished in {latency_s:.2f}s")
    timing = (meta or {}).get("timing_s")
    if isinstance(timing, dict) and timing:
        parts = [f"{k}={float(v):.2f}s" for k, v in timing.items() if not k.startswith("sampling_step_")]
        if parts:
            print(f"  timing: {', '.join(parts[:12])}")
    flush_stdio()


def emit_run_finished(
    *,
    pipeline: str,
    pair_id: str,
    result_meta: dict[str, Any],
    latency_s: float,
    had_error: bool = False,
) -> None:
    """Called by run_eval immediately after pipe.run returns (before score_pair)."""
    if result_meta.get("profile"):
        # Full profile already printed from pipeline finally; confirm completion.
        status = "ERROR (partial)" if had_error else "OK"
        print(f"[{pipeline}] {pair_id} profile emitted ({status}), latency={latency_s:.2f}s")
    else:
        emit_timing_summary(label=pipeline, pair_id=pair_id, latency_s=latency_s, meta=result_meta)
    flush_stdio()
