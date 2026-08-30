"""LivePortrait expression transfer: keep an identity, take an expression.

Why this exists at all. CHECKPOINTs 11-14 spent five sessions and eight
levers trying to control expression through Krea2 -- prompt text (twice),
face_refine, ref_boost (four values), the identity LoRA on and off, and a
donor pre-edit -- and never moved it once. The common factor is that every
one of those routes carries expression through a TEXT channel, and
CHECKPOINT-14 measured that channel as too narrow to represent an expression
at all ("mouth open, head tilted up, mid-exertion" compressed to one of four
canned phrases).

LivePortrait has no text in the loop: it reads the driving image's own dense
keypoints. ``animation_region="lip"`` restricts the edit to the mouth and
leaves the eye region untouched, which is the requirement, expressed as a
parameter rather than as a prompt the model may ignore.

Licensing, verified rather than recalled: LivePortrait's code and weights are
MIT. It uses InsightFace ``buffalo_l`` for face analysis, which is
non-commercial research only -- but this repo ALREADY depends on buffalo_l
(``scripts/download_insightface.py``; it loads on every run), so this module
adds no new restriction. That pre-existing InsightFace constraint is a real
issue for anything shipping commercially and is worth raising separately.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Cached LivePortraitPipeline, keyed by the InferenceConfig/CropConfig fields
# we actually vary. Construction loads five .pth models
# (appearance_feature_extractor, motion_extractor, warping_module,
# spade_generator, stitching_retargeting_module) plus an InsightFace
# FaceAnalysisDIY and a LandmarkRunner -- every call rebuilt all of it, which
# is visible as the full "Load ... done." block in every run log.
#
# Keyed rather than a bare singleton because animation_region and
# flag_relative_motion live in InferenceConfig, not in the per-call args, so
# a plain singleton would silently serve a pipeline configured for the
# PREVIOUS call's region. That is the same class of bug as the stale Colab
# form values that cost this investigation ten runs.
_LP_PIPELINE: dict = {"key": None, "pipeline": None}

# Still-image suffixes. A single still cannot express relative motion (see
# resolve_relative_motion), so this list is load-bearing, not cosmetic.
_STILL_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)


def driving_is_still_image(driving_path: str | Path) -> bool:
    """True when the driving input is a single still rather than a video."""
    return Path(driving_path).suffix.lower() in _STILL_SUFFIXES


def resolve_relative_motion(
    driving_path: str | Path, requested: bool | None
) -> tuple[bool, str]:
    """Decide ``flag_relative_motion``, overriding a wrong request.

    Relative motion transfers the DELTA between the driving frame and the
    driving sequence's own reference frame. With a single driving IMAGE that
    frame is both the reference and the target, so the delta is ~zero and
    essentially nothing transfers.

    That is not a matter of taste, it is arithmetic, and it was measured
    twice on GPU: two full 30s+ runs that loaded every model, reported
    success, and returned a near-copy of the source. So a still driver
    FORCES relative motion off rather than trusting the caller -- a stale
    Colab tab kept re-sending ``True`` and silently burned those runs.

    Returns ``(value, reason)`` so the caller can log why.
    """
    if driving_is_still_image(driving_path):
        if requested:
            return False, (
                "FORCED off: driving input is a single still image, where "
                "relative motion is a mathematical no-op (the delta is "
                "against the same frame). Requested True, overriding."
            )
        return False, "off: single still driving image"
    if requested is None:
        return True, "on: driving input is a video (default)"
    return bool(requested), f"{'on' if requested else 'off'}: caller-specified"


def run_expression_transfer(
    source_path: str | Path,
    driving_path: str | Path,
    out_dir: str | Path,
    *,
    animation_region: str = "lip",
    driving_multiplier: float = 1.0,
    stitching: bool = True,
    relative_motion: bool | None = None,
    live_portrait_dir: str | Path = "/content/LivePortrait",
) -> dict[str, Any]:
    """Run one LivePortrait transfer. Returns paths + the effective config.

    ``source_path`` supplies the identity to KEEP; ``driving_path`` supplies
    the expression to TAKE. Argument names say so deliberately: this repo has
    already lost runs to body/face role confusion.
    """
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import time  # noqa: PLC0415

    lp_dir = Path(live_portrait_dir)
    if not (lp_dir / "inference.py").exists():
        raise FileNotFoundError(
            f"LivePortrait not found at {lp_dir}. Clone it first: "
            "git clone --depth 1 https://github.com/KwaiVGI/LivePortrait"
        )

    source_path, driving_path = Path(source_path), Path(driving_path)
    for f in (source_path, driving_path):
        if not f.exists():
            raise FileNotFoundError(f"missing input image: {f}")

    rel, rel_reason = resolve_relative_motion(driving_path, relative_motion)
    print(
        f"[liveportrait] region={animation_region} relative_motion={rel} "
        f"({rel_reason}) multiplier={driving_multiplier} stitching={stitching}",
        flush=True,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {f for f in out_dir.glob("**/*") if f.is_file()}

    cwd_before = Path.cwd()
    os.chdir(lp_dir)
    if str(lp_dir) not in sys.path:
        sys.path.insert(0, str(lp_dir))
    # LivePortrait's package is literally named `src`, which collides with
    # anything else importing a top-level `src`. Drop stale entries so a
    # second call in the same kernel does not reuse another project's module.
    #
    # Only when we are about to BUILD, though. Purging on a cache hit would
    # re-import these modules underneath a live pipeline, leaving its
    # instances bound to class objects that no longer match the freshly
    # imported ones -- the classic stale-isinstance trap. On a hit the
    # already-imported LivePortrait modules are exactly what we want.
    _will_build = _LP_PIPELINE["pipeline"] is None
    if _will_build:
        for mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
            del sys.modules[mod]
    try:
        from src.config.argument_config import ArgumentConfig  # noqa: PLC0415
        from src.config.crop_config import CropConfig  # noqa: PLC0415
        from src.config.inference_config import InferenceConfig  # noqa: PLC0415
        from src.live_portrait_pipeline import (  # noqa: PLC0415
            LivePortraitPipeline,
        )

        args = ArgumentConfig(
            source=str(source_path),
            driving=str(driving_path),
            output_dir=str(out_dir),
            flag_relative_motion=rel,
            animation_region=str(animation_region),
            flag_stitching=bool(stitching),
            flag_pasteback=True,
            driving_multiplier=float(driving_multiplier),
        )

        def _partial(cls):
            return cls(**{k: v for k, v in args.__dict__.items() if hasattr(cls, k)})

        cache_key = (
            str(lp_dir), str(animation_region), bool(rel), bool(stitching),
            float(driving_multiplier),
        )
        load_s = 0.0
        if _LP_PIPELINE["key"] == cache_key and _LP_PIPELINE["pipeline"] is not None:
            pipeline = _LP_PIPELINE["pipeline"]
            print("[liveportrait] reusing cached pipeline (models resident)",
                  flush=True)
        else:
            _lt0 = time.perf_counter()
            pipeline = LivePortraitPipeline(
                inference_cfg=_partial(InferenceConfig),
                crop_cfg=_partial(CropConfig),
            )
            load_s = round(time.perf_counter() - _lt0, 2)
            _LP_PIPELINE["key"] = cache_key
            _LP_PIPELINE["pipeline"] = pipeline
            print(f"[liveportrait] pipeline built + cached in {load_s}s",
                  flush=True)

        t0 = time.perf_counter()
        pipeline.execute(args)
        latency_s = round(time.perf_counter() - t0, 2)
    finally:
        os.chdir(cwd_before)

    # Glob rather than assume a filename: LivePortrait names outputs from the
    # input stems and emits an extra `_concat` comparison image, and returns
    # a video instead of a jpg for some input shapes.
    after = {f for f in out_dir.glob("**/*") if f.is_file()}
    produced = sorted(after - before, key=lambda f: f.stat().st_mtime)
    primary = [f for f in produced if "concat" not in f.name.lower()]

    return {
        "produced": [str(f) for f in produced],
        "primary": str(primary[-1]) if primary else None,
        "latency_s": latency_s,
        "model_load_s": load_s,
        "animation_region": animation_region,
        "relative_motion": rel,
        "relative_motion_reason": rel_reason,
        "driving_multiplier": driving_multiplier,
        "stitching": stitching,
    }
