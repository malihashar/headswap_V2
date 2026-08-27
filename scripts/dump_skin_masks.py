"""Dump every mask that decides skin/hair compositing, as a labelled montage.

Three consecutive fixes for a half-corrected leg and for hair ghosting each
moved ~2% of the pixels they targeted, which means the masks they adjust do
not cover the affected regions at all. Guessing which mask is short has cost
more than measuring it. This renders each one over the image so the gap is
visible directly.

Usage (Colab, after a run):
    python scripts/dump_skin_masks.py RESULT.png ORIGINAL_BODY.png -o out/

RESULT is the pipeline's final image; ORIGINAL_BODY is the untouched body
input. Writes <out>/masks_montage.png plus one PNG per mask.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _overlay(base: Image.Image, mask: np.ndarray | None, label: str) -> Image.Image:
    """Red = mask on, over a dimmed base, so gaps are obvious."""
    im = base.convert("RGB").copy()
    arr = np.asarray(im, dtype=np.float32) * 0.45
    if mask is not None:
        m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        if m.shape[:2] != arr.shape[:2]:
            import cv2

            m = cv2.resize(m, (arr.shape[1], arr.shape[0]))
        arr[..., 0] = np.clip(arr[..., 0] + 210.0 * m, 0, 255)
        cov = float((m > 0.5).mean())
        label = f"{label}  cov={cov:.3%}  px={int((m > 0.5).sum())}"
    else:
        label = f"{label}  -- None (unavailable)"
    out = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, 22], fill=(0, 0, 0))
    d.text((5, 6), label, fill=(255, 255, 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("original")
    ap.add_argument("-o", "--out", default="results/_mask_dump")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_pil = Image.open(a.result).convert("RGB")
    orig_pil = Image.open(a.original).convert("RGB").resize(res_pil.size)
    res_np = np.asarray(res_pil, dtype=np.uint8)
    orig_np = np.asarray(orig_pil, dtype=np.uint8)

    from headswap.skin_harmonize import (
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
            print(f"  {fn.__name__} FAILED: {type(exc).__name__}: {exc}", flush=True)
            return None

    matte = _try(_get_person_matte, res_pil)
    panels = [
        (res_pil, None, "RESULT (reference)"),
        (res_pil, None if matte is None else matte / 255.0,
         "rembg person matte (RESULT)"),
        (res_pil, _try(semantic_person_mask, res_np), "semantic person (RESULT)"),
        (res_pil, _try(semantic_person_skin_mask, res_np),
         "semantic BODY SKIN (RESULT)  <- drives the wash"),
        (res_pil, _try(_semantic_skin_mask, res_np), "_semantic_skin_mask (gate)"),
        (res_pil, _try(person_minus_clothes_mask, res_np, res_pil),
         "rembg person - clothes (RESULT)  <- the new floor"),
        (res_pil, _try(semantic_clothes_mask, res_np), "semantic clothes (RESULT)"),
        (res_pil, _try(_hsv_skin_mask, res_np).astype(np.float32) / 255.0
         if _try(_hsv_skin_mask, res_np) is not None else None,
         "HSV colour skin (fallback only)"),
        (orig_pil, _try(semantic_head_mask, orig_np),
         "semantic HEAD/HAIR (ORIGINAL)  <- drives the hair lift"),
        (orig_pil, _try(semantic_clothes_mask, orig_np),
         "semantic clothes (ORIGINAL)"),
    ]

    imgs = []
    for base, mask, label in panels:
        print(f"[dump] {label}", flush=True)
        panel = _overlay(base, mask, label)
        imgs.append(panel)
        safe = "".join(c if c.isalnum() else "_" for c in label)[:60]
        panel.save(out_dir / f"{safe}.png")

    cols = 5
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    scale = min(1.0, 340.0 / w)
    tw, th = int(w * scale), int(h * scale)
    sheet = Image.new("RGB", (cols * tw, rows * th), (18, 18, 18))
    for i, im in enumerate(imgs):
        sheet.paste(im.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    dest = out_dir / "masks_montage.png"
    sheet.save(dest)
    print(f"\n[dump] wrote {dest}  ({len(imgs)} panels)", flush=True)
    print(
        "\nRead it as: any panel where the affected leg or the hair strands are "
        "NOT red is a mask that cannot fix them, no matter how it is tuned.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
