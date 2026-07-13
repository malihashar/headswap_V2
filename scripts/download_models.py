#!/usr/bin/env python3
"""Download model weights required by headswap_V2 pipelines."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


MODELS = {
    "klein": [
        (
            "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors",
            "diffusion_models",
            "flux-2-klein-4b.safetensors",
        ),
        (
            "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
            "text_encoders",
            "qwen_3_4b.safetensors",
        ),
        (
            "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/vae/flux2-vae.safetensors",
            "vae",
            "flux2-vae.safetensors",
        ),
        (
            "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1.1_optional_flux-klein_4b.safetensors",
            "loras",
            "bfs_head_v1.1_optional_flux-klein_4b.safetensors",
        ),
    ],
    "qwen": [
        (
            "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
            "diffusion_models",
            "qwen_image_edit_2511_fp8mixed.safetensors",
        ),
        (
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
            "vae",
            "qwen_image_vae.safetensors",
        ),
        (
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "text_encoders",
            "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        ),
        (
            "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "loras",
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        ),
        (
            "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
            "loras",
            "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
        ),
    ],
}


def download(url: str, dest_dir: Path, filename: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"  exists: {filename}")
        return
    print(f"  downloading: {filename}")
    if subprocess.call(["which", "aria2c"], stdout=subprocess.DEVNULL) == 0:
        subprocess.check_call(
            ["aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M", "-d", str(dest_dir), "-o", filename, url]
        )
    else:
        subprocess.check_call(["curl", "-L", url, "-o", str(path)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=os.environ.get("COMFYUI_PATH", "/content/ComfyUI"))
    ap.add_argument("--set", choices=["klein", "qwen", "all"], default="all")
    args = ap.parse_args()
    root = Path(args.comfy) / "models"
    keys = ["klein", "qwen"] if args.set == "all" else [args.set]
    for k in keys:
        print(f"== {k} ==")
        for url, sub, name in MODELS[k]:
            download(url, root / sub, name)
    print("Done.")


if __name__ == "__main__":
    main()
