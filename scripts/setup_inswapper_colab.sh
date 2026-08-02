#!/usr/bin/env bash
# One-click Colab bootstrap for the InsightFace InSwapper experiment (T4-friendly).
# Does NOT touch the Krea2 / ComfyUI stack.
set -euo pipefail

REPO_ROOT="${HEADSWAP_REPO:-/content/headswap_V2}"
CACHE_DIR="${INSWAP_CACHE:-/content/drive/MyDrive/headswap_inswap}"
LOCAL_CACHE="${INSWAP_LOCAL_CACHE:-/content/inswap_cache}"

echo "=== InSwapper Colab setup ==="
echo "  REPO_ROOT=$REPO_ROOT"
echo "  CACHE_DIR=$CACHE_DIR"
echo "  LOCAL_CACHE=$LOCAL_CACHE"

cd "$REPO_ROOT"

python3 -m pip install -q -U pip
# Minimal deps for T4: insightface + CUDA onnxruntime + imaging extras
python3 -m pip install -q \
  "insightface>=0.7.3" \
  "onnxruntime-gpu>=1.16" \
  "opencv-python-headless>=4.8" \
  "numpy>=1.24" \
  "Pillow>=10" \
  "tqdm>=4.66" \
  "matplotlib>=3.7"

# Optional restorers (imported lazily; safe if install fails on some runtimes)
python3 -m pip install -q gfpgan basicsr facexlib || \
  echo "WARN: GFPGAN extras failed to install; set RESTORE=none"

mkdir -p "$CACHE_DIR" "$LOCAL_CACHE"
# Prefer Drive-backed cache with a local symlink tree for fast reads
if [[ -d /content/drive/MyDrive ]]; then
  mkdir -p "$CACHE_DIR/models" "$CACHE_DIR/insightface"
  ln -sfn "$CACHE_DIR/models" "$LOCAL_CACHE/models" || true
  ln -sfn "$CACHE_DIR/insightface" "$LOCAL_CACHE/insightface" || true
  export INSWAP_CACHE_DIR="$LOCAL_CACHE"
else
  export INSWAP_CACHE_DIR="$LOCAL_CACHE"
  mkdir -p "$INSWAP_CACHE_DIR"
fi

export INSIGHTFACE_HOME="${INSWAP_CACHE_DIR}/insightface"
mkdir -p "$INSIGHTFACE_HOME"

echo "→ Downloading models…"
python3 "$REPO_ROOT/scripts/download_inswapper.py" \
  --cache-dir "$INSWAP_CACHE_DIR" \
  --device cuda \
  --with-gfpgan || {
    echo "WARN: CUDA download path failed; retrying CPU providers…"
    python3 "$REPO_ROOT/scripts/download_inswapper.py" \
      --cache-dir "$INSWAP_CACHE_DIR" \
      --device cpu \
      --with-gfpgan
  }

echo "✓ InSwapper setup complete"
echo "  Models: $INSWAP_CACHE_DIR"
