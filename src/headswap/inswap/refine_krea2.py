"""Optional Krea2 head-refinement stage after InSwapper.

InSwapper remains the primary identity transfer. When enabled, Krea2 only
refines the selected head region (hair / hairline / beard / ears / forehead /
skin transitions) so the result reads as one consistent person.

Disable for A/B: ``Krea2HeadRefiner(enabled=False)`` or
``InSwapPipeline(krea2_refine=False)``.
"""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from headswap.inswap.engines.base import DetectedFace

# Post-InSwapper head completion — face ID already transferred; finish the head.
HYBRID_REFINE_PROMPT = (
    "The first image already has the correct facial identity from a prior face swap. "
    "Complete ONLY the selected person's head so that hairstyle, hairline, hair length, "
    "hair color, beard, sideburns, ears, forehead, jawline skin, and neck/skin transitions "
    "match the identity person in the second image. "
    "Preserve EXACTLY from the first image: the already-swapped facial identity and features, "
    "pose, head rotation, gaze / eye direction, facial expression, mouth shape, head size, "
    "lighting, clothing, body, and background. "
    "If other people are visible, leave them completely unchanged. "
    "Do not enlarge or shrink the head. Do not regenerate the scene. "
    "Photorealistic, natural skin/hair texture, seamless neck and hairline blend."
)

HYBRID_NEGATIVE_PROMPT = (
    "different face identity, face drift, identity change, enlarged head, bobblehead, "
    "changed pose, changed expression, changed gaze, changed clothing, changed background, "
    "altered other people, blurry hairline, pasted hair, double hair"
)


@dataclass
class HeadCropBundle:
    """Full-head crop prepared for Krea2 refinement."""

    crop: Image.Image
    mask_full: Image.Image
    mask_in_crop: Image.Image
    box: tuple[int, int, int, int]  # x0,y0,x1,y1 in full image
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefineResult:
    image: Image.Image
    head_crop: Image.Image | None = None
    refined_head: Image.Image | None = None
    mask_full: Image.Image | None = None
    box: tuple[int, int, int, int] | None = None
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    debug_paths: dict[str, str] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None


def _to_face_box(face: DetectedFace):
    from headswap.preprocess import FaceBox

    x0, y0, x1, y1 = face.bbox
    return FaceBox(
        int(round(x0)),
        int(round(y0)),
        int(round(x1)),
        int(round(y1)),
        float(face.det_score),
    )


def build_full_head_crop(
    scene: Image.Image,
    selected: DetectedFace,
    cache_dir: Path,
    *,
    top_extend: float = 1.65,
    side_extend: float = 0.65,
    bot_extend: float = 0.45,
    expand_px: int = 20,
    blur_px: int = 12,
    crop_pad: int = 14,
    div_by: int = 16,
) -> HeadCropBundle:
    """Crop a region covering hair, hairline, ears, beard, jaw, neck transition."""
    from headswap.preprocess import crop_with_mask, head_hair_mask_from_face

    face_box = _to_face_box(selected)
    mask = head_hair_mask_from_face(
        scene.convert("RGB"),
        cache_dir,
        expand_px=int(expand_px),
        blur_px=int(blur_px),
        top_extend=float(top_extend),
        side_extend=float(side_extend),
        bot_extend=float(bot_extend),
        face_box=face_box,
    )
    crop, mask_in_crop, box = crop_with_mask(
        scene.convert("RGB"), mask, pad=int(crop_pad), div_by=int(div_by)
    )
    return HeadCropBundle(
        crop=crop,
        mask_full=mask,
        mask_in_crop=mask_in_crop,
        box=box,
        meta={
            "top_extend": top_extend,
            "side_extend": side_extend,
            "bot_extend": bot_extend,
            "expand_px": expand_px,
            "box": list(box),
            "crop_size": list(crop.size),
        },
    )


def default_krea2_refine_cfg(
    *,
    face_policy: str = "largest",
    face_index: int = 0,
    steps: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Config for localized head refine on top of an InSwapper result."""
    from headswap.config import load_config, project_root

    cfg_path = project_root() / "configs" / "krea2_identity_edit.yaml"
    cfg = deepcopy(load_config(str(cfg_path)))
    cfg["pipeline"] = "krea2"
    cfg["name"] = "krea2_inswap_refine"
    cfg["mask_crop_stitch"] = True
    cfg["multi_person_edit_mode"] = "crop_stitch"
    cfg["multi_person_swap_mode"] = "krea2_crop"
    cfg["single_person_parity"] = True
    cfg["body_face_policy"] = face_policy
    cfg["body_face_index"] = int(face_index)
    cfg["prompt"] = HYBRID_REFINE_PROMPT
    cfg["negative_prompt"] = HYBRID_NEGATIVE_PROMPT
    # Generous head extents so hair/beard/ears are inside the edit region.
    cfg["mask_top_extend"] = 1.65
    cfg["mask_side_extend"] = 0.65
    cfg["mask_bot_extend"] = 0.45
    cfg["mask_expand_px"] = 20
    cfg["face_top_pad"] = 0.75
    cfg["face_side_pad"] = 0.28
    cfg["face_bot_pad"] = 0.15
    cfg["save_debug"] = True
    cfg["verbose"] = False
    # Keep hair-replace language on; suppress multi-only add-ons that fight identity lock.
    cfg["multi_hair_replace_prompt"] = True
    cfg["multi_extra_prompt"] = False
    cfg["multi_head_scale_prompt"] = True
    if steps is not None:
        cfg["steps"] = int(steps)
    if seed is not None:
        cfg["seed"] = int(seed)
    return cfg


class Krea2HeadRefiner:
    """Optional second stage: Krea2 head completion after InSwapper."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        cache_dir: Path | str,
        force_mock: bool = False,
        runtime=None,
        cfg: dict[str, Any] | None = None,
        steps: int | None = None,
        seed: int | None = None,
        required: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_mock = bool(force_mock)
        self.runtime = runtime
        self.cfg_overrides = cfg or {}
        self.steps = steps
        self.seed = seed
        self.required = bool(required)
        self._pipe = None
        self._loaded = False

    def load(self) -> None:
        if not self.enabled:
            self._loaded = True
            return
        cfg = default_krea2_refine_cfg(steps=self.steps, seed=self.seed)
        cfg.update(self.cfg_overrides)
        from headswap.pipelines import create_pipeline

        if self.force_mock:
            self._pipe = create_pipeline(cfg, force_mock=True)
        else:
            try:
                from headswap.pipelines.krea2 import get_shared_krea2_runtime

                rt = self.runtime or get_shared_krea2_runtime(init_custom_nodes=True)
                self._pipe = create_pipeline(cfg, runtime=rt)
            except Exception as exc:  # noqa: BLE001
                if self.required:
                    raise RuntimeError(
                        f"Krea2 refine required but unavailable: {exc}"
                    ) from exc
                print(
                    f"[krea2-refine] unavailable ({exc}); stage will be skipped",
                    flush=True,
                )
                self._pipe = None
                self.enabled = False
        self._loaded = True

    def refine(
        self,
        scene_after_inswap: Image.Image,
        identity: Image.Image,
        *,
        selected: DetectedFace,
        face_policy: str = "largest",
        face_index: int = 0,
        out_dir: Path | str | None = None,
    ) -> RefineResult:
        """
        Run Krea2 head refine on the InSwapper canvas.

        ``scene_after_inswap`` is Picture 1 (already face-swapped).
        ``identity`` is Picture 2 (same donor used for InSwapper).
        """
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        out_path = Path(out_dir) if out_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        # Always build / save the head crop for A/B intermediates, even if refine skips.
        crop_bundle = build_full_head_crop(
            scene_after_inswap, selected, self.cache_dir
        )
        dbg: dict[str, str] = {}
        if out_path is not None:
            head_path = out_path / "11_head_crop_for_krea2.png"
            crop_bundle.crop.save(head_path)
            dbg["head_crop"] = str(head_path)
            mask_path = out_path / "11_head_mask.png"
            crop_bundle.mask_full.save(mask_path)
            dbg["head_mask"] = str(mask_path)

        if not self.enabled or self._pipe is None:
            return RefineResult(
                image=scene_after_inswap.convert("RGB"),
                head_crop=crop_bundle.crop,
                refined_head=None,
                mask_full=crop_bundle.mask_full,
                box=crop_bundle.box,
                latency_s=time.perf_counter() - t0,
                meta={"krea2_refine": False, "crop": crop_bundle.meta},
                debug_paths=dbg,
                skipped=True,
                skip_reason="disabled" if not self.enabled else "pipeline_unavailable",
            )

        cfg = getattr(self._pipe, "cfg", {})
        if isinstance(cfg, dict):
            cfg["body_face_policy"] = face_policy
            cfg["body_face_index"] = int(face_index)
            cfg["prompt"] = str(cfg.get("prompt") or HYBRID_REFINE_PROMPT)
            cfg["save_debug"] = True

        krea_out = out_path / "krea2_refine" if out_path is not None else None
        if krea_out is not None:
            krea_out.mkdir(parents=True, exist_ok=True)

        try:
            result = self._pipe.run(
                scene_after_inswap.convert("RGB"),
                identity.convert("RGB"),
                out_dir=krea_out,
            )
        except Exception as exc:  # noqa: BLE001
            if self.required:
                raise
            print(f"[krea2-refine] failed ({exc}); keeping InSwapper result", flush=True)
            return RefineResult(
                image=scene_after_inswap.convert("RGB"),
                head_crop=crop_bundle.crop,
                refined_head=None,
                mask_full=crop_bundle.mask_full,
                box=crop_bundle.box,
                latency_s=time.perf_counter() - t0,
                meta={"krea2_refine": False, "error": str(exc), "crop": crop_bundle.meta},
                debug_paths=dbg,
                skipped=True,
                skip_reason=f"error:{exc}",
            )

        refined_full = result.image.convert("RGB")
        refined_head = None
        # Prefer Krea2's edited crop debug asset when present.
        for key in ("debug_edited_crop", "debug_crop", "debug_edited"):
            p = (result.debug_paths or {}).get(key)
            if p and Path(p).is_file():
                refined_head = Image.open(p).convert("RGB")
                dbg[f"krea2_{key}"] = p
                break
        if refined_head is None:
            # Re-crop from refined full frame using the same box.
            x0, y0, x1, y1 = crop_bundle.box
            refined_head = refined_full.crop((x0, y0, x1, y1))

        if out_path is not None and refined_head is not None:
            rh_path = out_path / "12_krea2_refined_head.png"
            refined_head.save(rh_path)
            dbg["refined_head"] = str(rh_path)
            rf_path = out_path / "13_krea2_refined_full.png"
            refined_full.save(rf_path)
            dbg["refined_full"] = str(rf_path)

        for k, v in (result.debug_paths or {}).items():
            if v:
                dbg[f"krea2_{k}"] = v

        return RefineResult(
            image=refined_full,
            head_crop=crop_bundle.crop,
            refined_head=refined_head,
            mask_full=crop_bundle.mask_full,
            box=crop_bundle.box,
            latency_s=time.perf_counter() - t0,
            meta={
                "krea2_refine": True,
                "krea2_meta": result.meta,
                "crop": crop_bundle.meta,
                "prompt": HYBRID_REFINE_PROMPT[:200],
            },
            debug_paths=dbg,
            skipped=False,
        )


def stitch_refined_head(
    base: Image.Image,
    refined_full: Image.Image,
    mask_full: Image.Image,
    box: tuple[int, int, int, int],
    *,
    feather_px: int = 10,
    color_match_strength: float = 0.35,
) -> Image.Image:
    """Re-stitch a refined head region onto ``base`` (usually the InSwapper canvas).

    Prefer using Krea2's own stitch when available; this helper is for cases where
    only a refined crop is returned, or for A/B re-composite experiments.
    """
    from headswap.preprocess import feathered_soft_composite, lab_histogram_match_face

    if refined_full.size != base.size:
        # If refined_full is a crop, paste via box; if full canvas, resize.
        if refined_full.size == (
            max(1, box[2] - box[0]),
            max(1, box[3] - box[1]),
        ):
            canvas = base.convert("RGB").copy()
            x0, y0, x1, y1 = box
            canvas.paste(refined_full, (x0, y0))
            edited = canvas
        else:
            edited = refined_full.resize(base.size, Image.Resampling.LANCZOS)
    else:
        edited = refined_full

    out = feathered_soft_composite(
        base.convert("RGB"),
        edited,
        mask_full,
        box,
        extra_blur_px=int(feather_px),
    )
    if color_match_strength > 0:
        out = lab_histogram_match_face(
            out, base.convert("RGB"), mask_full, strength=float(color_match_strength)
        )
    return out
