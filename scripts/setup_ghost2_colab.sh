#!/usr/bin/env bash
# Bootstrap GHOST 2.0 on Google Colab (Drive-cached weights).
# Usage: bash scripts/setup_ghost2_colab.sh
set -euo pipefail

REPO="${HEADSWAP_REPO:-/content/headswap_V2}"
DRIVE_ROOT="${GHOST2_DRIVE_ROOT:-/content/drive/MyDrive/headswap_ghost2}"
GHOST_ROOT="${GHOST2_ROOT:-/content/ghost-2.0}"
WEIGHTS_CACHE="${DRIVE_ROOT}/weights"
RELEASE="https://github.com/ai-forever/ghost-2.0/releases/download/aligner"

echo "→ GHOST 2.0 setup"
echo "  repo=$REPO"
echo "  ghost=$GHOST_ROOT"
echo "  weights_cache=$WEIGHTS_CACHE"

mkdir -p "$DRIVE_ROOT" "$WEIGHTS_CACHE"

# --- clone upstream ---
if [[ -d "$GHOST_ROOT/.git" ]]; then
  git -C "$GHOST_ROOT" pull --ff-only || true
else
  rm -rf "$GHOST_ROOT"
  git clone --depth 1 https://github.com/ai-forever/ghost-2.0.git "$GHOST_ROOT"
fi

mkdir -p "$GHOST_ROOT/repos" \
  "$GHOST_ROOT/aligner_checkpoints" \
  "$GHOST_ROOT/blender_checkpoints" \
  "$GHOST_ROOT/weights" \
  "$GHOST_ROOT/src/losses"

# --- dependency repos ---
clone_dep() {
  local url="$1" dest="$2"
  if [[ -d "$dest/.git" ]]; then
    git -C "$dest" pull --ff-only || true
  else
    rm -rf "$dest"
    git clone --depth 1 "$url" "$dest"
  fi
}

clone_dep https://github.com/chroneus/stylematte.git "$GHOST_ROOT/repos/stylematte"
clone_dep https://github.com/anastasia-yaschenko/EMOCA.git "$GHOST_ROOT/repos/emoca"
clone_dep https://github.com/anastasia-yaschenko/BlazeFace_PyTorch.git "$GHOST_ROOT/repos/BlazeFace_PyTorch"
clone_dep https://github.com/yfeng95/DECA.git "$GHOST_ROOT/repos/DECA"

# --- pip deps (Colab-friendly; skip conda pytorch3d) ---
pip install -q -U pip setuptools wheel
pip install -q face-alignment facenet-pytorch
pip install -q \
  pytorch-lightning lightning omegaconf kornia einops \
  insightface onnx onnxruntime-gpu mediapipe \
  simple-lama-inpainting huggingface-hub==0.25.0 \
  diffusers==0.24.0 transformers==4.48.2 \
  scikit-image lpips imgaug pytorch-msssim h5py \
  torchmetrics opencv-python-headless pillow
# NumPy pin recommended by upstream (best-effort on Colab).
pip install -q "numpy==1.26.4" || pip install -q "numpy<2"

# StyleMatte package path
pip install -q -e "$GHOST_ROOT/repos/stylematte" || true

# --- download / link checkpoints (Drive cache) ---
download() {
  local url="$1" dest="$2"
  if [[ -f "$dest" && -s "$dest" ]]; then
    echo "✓ cached $(basename "$dest")"
    return 0
  fi
  echo "→ downloading $(basename "$dest")…"
  wget -q --show-progress -O "$dest.partial" "$url"
  mv "$dest.partial" "$dest"
}

download "$RELEASE/aligner_1020_gaze_final.ckpt" "$WEIGHTS_CACHE/aligner_1020_gaze_final.ckpt"
download "$RELEASE/blender_lama.ckpt" "$WEIGHTS_CACHE/blender_lama.ckpt"
download "$RELEASE/backbone50_1.pth" "$WEIGHTS_CACHE/backbone50_1.pth"
download "$RELEASE/vgg19-d01eb7cb.pth" "$WEIGHTS_CACHE/vgg19-d01eb7cb.pth"
download "$RELEASE/segformer_B5_ce.onnx" "$WEIGHTS_CACHE/segformer_B5_ce.onnx"
download "$RELEASE/gaze_models.zip" "$WEIGHTS_CACHE/gaze_models.zip"

# StyleMatte synth weights
STYLEMATTE_CKPT="$WEIGHTS_CACHE/stylematte_synth.pth"
download \
  "https://github.com/chroneus/stylematte/releases/download/weights/stylematte_synth.pth" \
  "$STYLEMATTE_CKPT"

# Link into ghost tree
ln -sfn "$WEIGHTS_CACHE/aligner_1020_gaze_final.ckpt" \
  "$GHOST_ROOT/aligner_checkpoints/aligner_1020_gaze_final.ckpt"
ln -sfn "$WEIGHTS_CACHE/blender_lama.ckpt" \
  "$GHOST_ROOT/blender_checkpoints/blender_lama.ckpt"
ln -sfn "$WEIGHTS_CACHE/backbone50_1.pth" "$GHOST_ROOT/weights/backbone50_1.pth"
ln -sfn "$WEIGHTS_CACHE/vgg19-d01eb7cb.pth" "$GHOST_ROOT/weights/vgg19-d01eb7cb.pth"
ln -sfn "$WEIGHTS_CACHE/segformer_B5_ce.onnx" "$GHOST_ROOT/weights/segformer_B5_ce.onnx"

mkdir -p "$GHOST_ROOT/repos/stylematte/stylematte/checkpoints"
ln -sfn "$STYLEMATTE_CKPT" \
  "$GHOST_ROOT/repos/stylematte/stylematte/checkpoints/stylematte_synth.pth"

# Gaze models zip → src/losses/gaze_models
if [[ ! -d "$GHOST_ROOT/src/losses/gaze_models" ]]; then
  echo "→ unpacking gaze_models.zip…"
  mkdir -p "$WEIGHTS_CACHE/gaze_models_unpacked"
  unzip -qo "$WEIGHTS_CACHE/gaze_models.zip" -d "$WEIGHTS_CACHE/gaze_models_unpacked"
  # Zip layout varies; find a gaze_models dir or use unpack root.
  if [[ -d "$WEIGHTS_CACHE/gaze_models_unpacked/gaze_models" ]]; then
    ln -sfn "$WEIGHTS_CACHE/gaze_models_unpacked/gaze_models" \
      "$GHOST_ROOT/src/losses/gaze_models"
  else
    ln -sfn "$WEIGHTS_CACHE/gaze_models_unpacked" \
      "$GHOST_ROOT/src/losses/gaze_models"
  fi
fi

# EMOCA ResNet50 emotion network (required at AlignerModule init)
EMOCA_RESNET_DIR="$GHOST_ROOT/repos/emoca/assets/EmotionRecognition/image_based_networks/ResNet50"
mkdir -p "$(dirname "$EMOCA_RESNET_DIR")" \
  "$GHOST_ROOT/repos/emoca/gdl_apps/EmotionRecognition"
if [[ ! -d "$EMOCA_RESNET_DIR" ]]; then
  echo "→ downloading EMOCA ResNet50 emotion weights…"
  TMP_ZIP="$WEIGHTS_CACHE/emoca_resnet50.zip"
  if [[ ! -f "$TMP_ZIP" ]]; then
    wget -q --show-progress -O "$TMP_ZIP.partial" \
      "https://github.com/anastasia-yaschenko/emoca/releases/download/resnet/ResNet50.zip" \
      || wget -q --show-progress -O "$TMP_ZIP.partial" \
      "https://github.com/anastasia-yaschenko/EMOCA/releases/download/resnet/ResNet50.zip" \
      || true
    if [[ -f "$TMP_ZIP.partial" ]]; then
      mv "$TMP_ZIP.partial" "$TMP_ZIP"
    fi
  fi
  if [[ -f "$TMP_ZIP" ]]; then
    mkdir -p "$WEIGHTS_CACHE/emoca_resnet_unpacked"
    unzip -qo "$TMP_ZIP" -d "$WEIGHTS_CACHE/emoca_resnet_unpacked"
    if [[ -d "$WEIGHTS_CACHE/emoca_resnet_unpacked/ResNet50" ]]; then
      ln -sfn "$WEIGHTS_CACHE/emoca_resnet_unpacked/ResNet50" "$EMOCA_RESNET_DIR"
      mkdir -p "$GHOST_ROOT/repos/emoca/gdl_apps/EmotionRecognition"
      ln -sfn "$WEIGHTS_CACHE/emoca_resnet_unpacked/ResNet50" \
        "$GHOST_ROOT/repos/emoca/gdl_apps/EmotionRecognition/ResNet50" || true
    fi
  else
    echo "⚠ EMOCA ResNet50 zip missing — Aligner init may fail until downloaded."
  fi
fi

# Editable headswap install
if [[ -f "$REPO/pyproject.toml" || -f "$REPO/setup.py" ]]; then
  pip install -q -e "$REPO"
fi

echo "✓ GHOST 2.0 setup complete → $GHOST_ROOT"
echo "  export GHOST2_ROOT=$GHOST_ROOT"
