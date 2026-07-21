#!/usr/bin/env bash
# Install ComfyUI custom nodes required for Krea 2 Identity Edit.
# Idempotent. Run after setup_comfyui.sh (or via setup_kaggle.sh --krea2).
set -euo pipefail

COMFY="${COMFYUI_PATH:-}"
if [[ -z "$COMFY" ]]; then
  if [[ -d /kaggle/working/ComfyUI ]]; then
    COMFY=/kaggle/working/ComfyUI
  elif [[ -d /content/ComfyUI ]]; then
    COMFY=/content/ComfyUI
  else
    echo "ERROR: set COMFYUI_PATH or run setup_comfyui.sh first." >&2
    exit 1
  fi
fi

mkdir -p "$COMFY/custom_nodes"
cd "$COMFY/custom_nodes"

# Official Identity Edit node pack (Krea2EditModelPatch + Krea2EditGroundedEncode).
if [[ ! -d comfyui-krea2edit ]]; then
  git clone https://github.com/lbouaraba/comfyui-krea2edit.git
else
  echo "comfyui-krea2edit already present."
fi

# einops is used by the node pack; ensure present even if Comfy base missed it.
python3 -m pip install -q einops

echo "Krea2 edit nodes ready under $COMFY/custom_nodes/comfyui-krea2edit"
echo "Pipeline must boot with HEADSWAP_INIT_CUSTOM_NODES=1 (handled by krea2 pipeline)."
