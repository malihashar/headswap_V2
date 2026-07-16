#!/usr/bin/env bash
# Setup ComfyUI + custom nodes for headswap_V2 (Colab/Kaggle/RunPod/Linux).
set -euo pipefail

BASE="${HEADSWAP_BASE:-/content}"
if [[ -d /workspace ]]; then BASE=/workspace; fi
# Kaggle: keep ComfyUI on the persistent /kaggle/working volume; model weights
# live under /tmp (see scripts/download_models.py / scripts/setup_kaggle.sh).
if [[ -d /kaggle/working ]]; then BASE=/kaggle/working; fi
COMFY="${COMFYUI_PATH:-$BASE/ComfyUI}"

echo "Using COMFYUI_PATH=$COMFY"

if [[ ! -d "$COMFY" ]]; then
  git clone https://github.com/comfyanonymous/ComfyUI "$COMFY"
fi
pip install -q -r "$COMFY/requirements.txt"
pip install -q torchsde einops accelerate spandrel opencv-python pillow pyyaml tqdm

mkdir -p "$COMFY/custom_nodes"
cd "$COMFY/custom_nodes"
[[ -d ComfyUI-KJNodes ]] || git clone https://github.com/kijai/ComfyUI-KJNodes.git
[[ -f ComfyUI-KJNodes/requirements.txt ]] && pip install -q -r ComfyUI-KJNodes/requirements.txt || true

echo "Done. Export COMFYUI_PATH=$COMFY"
