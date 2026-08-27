"""Render every skin/hair mask over the image it was computed from.

Three consecutive fixes for a half-corrected limb and for hair ghosting each
moved ~2% of the pixels they targeted, which means the masks being tuned do
not cover the affected regions at all. A log delta cannot distinguish "the
mask is slightly wrong" from "the mask is not there"; an overlay can.

Lives in src (not scripts) so the pipeline can dump masks mid-run without the
caller having to keep the intermediate images alive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

_COLS = 5
_THUMB_W = 340.0

# Every call used to write straight into the same directory, so running two
# pairs back to back left only the SECOND pair's montage on disk -- the first
# was silently overwritten seconds later, which looked exactly like "the dump
# did not run". Number each call so both survive.
_RUN_N = 0


def _overlay(base: Image.Image, mask: Any, label: str) -> Image.Image:
    """Red = mask on, over a dimmed base, so gaps read at a glance."""
    arr = np.asarray(base.convert("RGB"), dtype=np.float32) * 0.45
    if mask is not None:
        m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[:2] != arr.shape[:2]:
            import cv2  # noqa: PLC0415

            m = cv2.resize(m, (arr.shape[1], arr.shape[0]))
        arr[..., 0] = np.clip(arr[..., 0] + 210.0 * m, 0, 255)
        label = f"{label}  cov={float((m > 0.5).mean()):.2%} px={int((m > 0.5).sum())}"
    else:
        label = f"{label}  -- None (unavailable)"
    out = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, 20], fill=(0, 0, 0))
    d.text((4, 5), label[:78], fill=(255, 255, 0))
    return out


def dump_mask_montage(
    result_pil: Image.Image,
    original_pil: Image.Image,
    out_dir: str | Path,
) -> Path | None:
    """Write <out_dir>/masks_montage.png. Returns the path, or None on failure."""
    global _RUN_N
    try:
        _RUN_N += 1
        out_dir = Path(out_dir) / f"run{_RUN_N:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        res_pil = result_pil.convert("RGB")
        orig_pil = original_pil.convert("RGB").resize(res_pil.size)
        res_np = np.asarray(res_pil, dtype=np.uint8)
        orig_np = np.asarray(orig_pil, dtype=np.uint8)

        from headswap.skin_harmonize import (  # noqa: PLC0415
            _get_person_matte,
            _hsv_skin_mask,
            _semantic_skin_mask,
            person_minus_clothes_mask,
            semantic_clothes_mask,
            semantic_head_mask,
            semantic_person_mask,
            semantic_person_skin_mask,
        )

        def _try(fn, *args):
            try:
                return fn(*args)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[mask_dump] {getattr(fn, '__name__', fn)} FAILED: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                return None

        _matte = _try(_get_person_matte, res_pil)
        _hsv = _try(_hsv_skin_mask, res_np)
        panels = [
            (res_pil, None, "RESULT (reference)"),
            (res_pil, None if _matte is None else np.asarray(_matte, np.float32) / 255.0,
             "rembg person matte (RESULT)"),
            (res_pil, _try(semantic_person_mask, res_np), "semantic person (RESULT)"),
            (res_pil, _try(semantic_person_skin_mask, res_np),
             "semantic BODY SKIN -> drives wash"),
            (res_pil, _try(_semantic_skin_mask, res_np), "_semantic_skin_mask (gate)"),
            (res_pil, _try(person_minus_clothes_mask, res_np, res_pil),
             "rembg person-clothes (the floor)"),
            (res_pil, _try(semantic_clothes_mask, res_np), "semantic clothes (RESULT)"),
            (res_pil, None if _hsv is None else np.asarray(_hsv, np.float32) / 255.0,
             "HSV colour skin (fallback)"),
            (orig_pil, _try(semantic_head_mask, orig_np),
             "semantic HEAD/HAIR (ORIGINAL) -> hair lift"),
            (orig_pil, _try(semantic_clothes_mask, orig_np),
             "semantic clothes (ORIGINAL)"),
        ]

        imgs = []
        for base, mask, label in panels:
            panel = _overlay(base, mask, label)
            imgs.append(panel)
            safe = "".join(c if c.isalnum() else "_" for c in label)[:60]
            panel.save(out_dir / f"{safe}.png")

        rows = (len(imgs) + _COLS - 1) // _COLS
        w, h = imgs[0].size
        scale = min(1.0, _THUMB_W / max(1, w))
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        sheet = Image.new("RGB", (_COLS * tw, rows * th), (18, 18, 18))
        for i, im in enumerate(imgs):
            sheet.paste(im.resize((tw, th)), ((i % _COLS) * tw, (i // _COLS) * th))
        dest = out_dir / "masks_montage.png"
        sheet.save(dest)
        print(
            f"[mask_dump] wrote {dest} ({len(imgs)} panels) -- run #{_RUN_N}; "
            "each render gets its own runNN folder so a later pair cannot "
            "overwrite an earlier one",
            flush=True,
        )
        return dest
    except Exception as exc:  # noqa: BLE001 — diagnostics must never break a run
        print(f"[mask_dump] FAILED: {type(exc).__name__}: {exc}", flush=True)
        return None
