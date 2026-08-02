#!/usr/bin/env python3
"""Run REFace face swap with multi-person face selection.

Expects REFACE_ROOT (clone of Sanoojan/REFace) set up via
scripts/setup_reface_colab.sh.

Source = identity donor, target = body/scene.
For multi-person bodies we crop a padded window around the selected face,
run REFace on that window, then paste it back.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _select_box(faces, policy: str, index: int):
    if not faces:
        raise RuntimeError("No face detected in body image.")
    policy = (policy or "largest").strip().lower()
    if policy == "index":
        i = int(index)
        if i < 0 or i >= len(faces):
            raise RuntimeError(f"BODY_FACE_INDEX={i} out of range (n={len(faces)})")
        return faces[i]
    if policy == "rightmost":
        return max(faces, key=lambda f: (f.x0 + f.x1) / 2.0)
    if policy == "leftmost":
        return min(faces, key=lambda f: (f.x0 + f.x1) / 2.0)
    return max(faces, key=lambda f: f.width * f.height)


def _resize_long_side(im: Image.Image, long_side: int) -> Image.Image:
    if not long_side or long_side <= 0:
        return im
    w, h = im.size
    m = max(w, h)
    if m == long_side:
        return im
    scale = float(long_side) / float(m)
    return im.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)


def _crop_around(im: Image.Image, box, pad_frac: float = 1.35) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = im.size
    fw, fh = max(1, box.width), max(1, box.height)
    pad_x = int(pad_frac * fw)
    pad_y = int(pad_frac * fh)
    x0 = max(0, box.x0 - pad_x)
    y0 = max(0, box.y0 - pad_y)
    x1 = min(w, box.x1 + pad_x)
    y1 = min(h, box.y1 + pad_y)
    return im.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reface-root", type=Path, default=None)
    parser.add_argument("--source", type=Path, required=True, help="Identity face")
    parser.add_argument("--target", type=Path, required=True, help="Body / scene")
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--face-policy", default="largest")
    parser.add_argument("--face-index", type=int, default=0)
    parser.add_argument("--output-long-side", type=int, default=1024)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reface_root = Path(
        args.reface_root
        or os.environ.get("REFACE_ROOT")
        or "/content/REFace"
    ).resolve()
    if not reface_root.is_dir():
        raise SystemExit(f"REFACE_ROOT not found: {reface_root}")

    # Colab Python 3.12 removes ``imp``; patch upstream before calling it.
    try:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from patch_reface_py312 import patch_reface_tree

        touched = patch_reface_tree(reface_root)
        if touched:
            print(f"→ patched REFace for Colab ({len(touched)} file(s))", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ REFace Colab patch skipped: {exc}", flush=True)

    # Import headswap face detector (optional; fall back to full-frame).
    headswap_src = Path(__file__).resolve().parents[1] / "src"
    if str(headswap_src) not in sys.path:
        sys.path.insert(0, str(headswap_src))

    body = _resize_long_side(Image.open(args.target).convert("RGB"), args.output_long_side)
    face = Image.open(args.source).convert("RGB")

    work = args.save_path.parent / "_reface_work"
    if work.exists():
        shutil.rmtree(work)
    src_dir = work / "Source"
    tgt_dir = work / "Target"
    out_dir = work / "results"
    base_dir = work / "Outs"
    src_dir.mkdir(parents=True)
    tgt_dir.mkdir(parents=True)

    paste_box = None
    body_full = body
    try:
        from headswap.preprocess import detect_faces, pil_to_rgb_np

        faces = detect_faces(pil_to_rgb_np(body), work / "cache", allow_prior=False)
        print(f"→ detected {len(faces)} face(s) on body", flush=True)
        if len(faces) >= 1:
            sel = _select_box(faces, args.face_policy, args.face_index)
            if len(faces) >= 2:
                crop, paste_box = _crop_around(body, sel, pad_frac=1.40)
                body = crop
                print(
                    f"→ multi-person: using policy={args.face_policy} "
                    f"box={paste_box} crop={body.size}",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ face select fallback (full frame): {exc}", flush=True)

    face.save(src_dir / "0.png")
    body.save(tgt_dir / "0.png")

    ckpt = reface_root / "models" / "REFace" / "checkpoints" / "saved.ckpt"
    if not ckpt.is_file():
        ckpt = reface_root / "models" / "REFace" / "checkpoints" / "last.ckpt"
    config = reface_root / "models" / "REFace" / "configs" / "project_ffhq.yaml"
    if not ckpt.is_file() or not config.is_file():
        raise SystemExit(f"Missing ckpt/config under {reface_root}/models/REFace")

    cmd = [
        sys.executable,
        str(reface_root / "scripts" / "inference_swap_selected.py"),
        "--outdir",
        str(out_dir),
        "--target_folder",
        str(tgt_dir),
        "--src_folder",
        str(src_dir),
        "--Base_dir",
        str(base_dir),
        "--config",
        str(config),
        "--ckpt",
        str(ckpt),
        "--n_samples",
        "1",
        "--scale",
        str(args.scale),
        "--ddim_steps",
        str(args.ddim_steps),
        "--seed",
        str(args.seed),
    ]
    print("→", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(reface_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        cmd,
        cwd=str(reface_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        raise SystemExit(
            "REFace failed:\n" + (proc.stderr or proc.stdout or f"exit={proc.returncode}")
        )

    # Find result image
    candidates = sorted((out_dir / "results").rglob("*.png")) if (out_dir / "results").exists() else []
    if not candidates:
        candidates = sorted(out_dir.rglob("*.png"))
    # Prefer final composites over masks/crops
    candidates = [p for p in candidates if "mask" not in p.name.lower()]
    if not candidates:
        raise SystemExit(f"No REFace output PNGs under {out_dir}")
    result = Image.open(candidates[-1]).convert("RGB")

    if paste_box is not None:
        canvas = body_full.copy()
        x0, y0, x1, y1 = paste_box
        result = result.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
        canvas.paste(result, (x0, y0))
        result = canvas

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.save_path)
    print(f"✓ saved {args.save_path} size={result.size}", flush=True)


if __name__ == "__main__":
    main()
