#!/usr/bin/env bash
# Prepare a Google Colab runtime for headswap_V2 (idempotent).
#
# Layout (Colab):
#   /content/ComfyUI                              → ComfyUI (ephemeral disk, fast)
#   /content/drive/MyDrive/headswap_V2/models     → persistent model cache on Drive
#   /content/_hf_dl_staging                       → HF download staging (ephemeral)
#   /content/headswap_V2                          → this repository
#
# Model downloads are opt-in:
#   bash scripts/setup_colab.sh              # ComfyUI + deps only
#   bash scripts/setup_colab.sh --krea2      # + Krea2 Identity Edit nodes & weights
#   bash scripts/setup_colab.sh --klein      # + Klein weights
#   bash scripts/setup_colab.sh --kontext    # + Kontext weights
#   bash scripts/setup_colab.sh --qwen       # + Qwen weights
#
# Run from the repository root after mounting Google Drive (recommended for cache).
set -euo pipefail

REPO_ROOT="$(pwd)"
export COMFYUI_PATH="${COMFYUI_PATH:-/content/ComfyUI}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HEADSWAP_MODEL_STORE="${HEADSWAP_MODEL_STORE:-/content/drive/MyDrive/headswap_V2/models}"
export HEADSWAP_STAGING_DIR="${HEADSWAP_STAGING_DIR:-/content/_hf_dl_staging}"

DOWNLOAD_KONTEXT=0
DOWNLOAD_KLEIN=0
DOWNLOAD_QWEN=0
DOWNLOAD_KREA2=0
REQUIRE_DRIVE=1

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_colab.sh [options]

  (default)     Install ComfyUI + Python deps + aria2. Download NO models.
  --krea2       Install comfyui-krea2edit + download Krea 2 Identity Edit set.
  --kontext     Download FLUX.1 Kontext set.
  --klein       Download FLUX.2 Klein set.
  --qwen        Download Qwen Image Edit 2511 set.
  --no-drive    Allow ephemeral /content model store (not persisted).
  -h, --help    Show this help.

Models default to Google Drive so they survive runtime reconnects.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kontext) DOWNLOAD_KONTEXT=1 ;;
    --klein) DOWNLOAD_KLEIN=1 ;;
    --qwen) DOWNLOAD_QWEN=1 ;;
    --krea2) DOWNLOAD_KREA2=1 ;;
    --no-drive) REQUIRE_DRIVE=0 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

# If Drive is missing and --no-drive was not set, fall back to /content/models
# only when REQUIRE_DRIVE=0; otherwise error with clear instructions.
if [[ "$REQUIRE_DRIVE" -eq 1 && ! -d /content/drive/MyDrive ]]; then
  echo "ERROR: Google Drive is not mounted at /content/drive/MyDrive." >&2
  echo "In a Colab cell, run:" >&2
  echo "  from google.colab import drive" >&2
  echo "  drive.mount('/content/drive')" >&2
  echo "Or re-run with --no-drive to use ephemeral /content/models." >&2
  exit 1
fi

if [[ "$REQUIRE_DRIVE" -eq 0 && ! -d /content/drive/MyDrive ]]; then
  export HEADSWAP_MODEL_STORE="${HEADSWAP_MODEL_STORE:-/content/models}"
fi

echo "=== headswap_V2 Colab setup ==="
echo "Repository:  $REPO_ROOT"
echo "ComfyUI:     $COMFYUI_PATH"
echo "Model store: $HEADSWAP_MODEL_STORE"
echo "Staging:     $HEADSWAP_STAGING_DIR"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "Downloads:   kontext=$DOWNLOAD_KONTEXT klein=$DOWNLOAD_KLEIN qwen=$DOWNLOAD_QWEN krea2=$DOWNLOAD_KREA2"
echo

if [[ ! -f "$REPO_ROOT/scripts/setup_comfyui.sh" ]]; then
  echo "ERROR: Run this script from the repository root (scripts/setup_comfyui.sh not found)." >&2
  exit 1
fi

echo "[1/3] ComfyUI → $COMFYUI_PATH"
bash "$REPO_ROOT/scripts/setup_comfyui.sh"
echo

echo "[2/3] Python dependencies (requirements.txt + editable headswap)"
echo "  Note: do not install hf_xet; setup forces HF_HUB_DISABLE_XET=1."
python3 -m pip install -q -r "$REPO_ROOT/requirements.txt"
python3 -m pip install -q -e "$REPO_ROOT"
# Geometry-lock multi-person path needs InsightFace buffalo_l (auto-download).
python3 "$REPO_ROOT/scripts/download_insightface.py" || echo "WARN: InsightFace download failed; box-paste fallback will be used."
echo "  Done."
echo

echo "[3/3] aria2 (resumable HTTP fallback)"
if command -v aria2c >/dev/null 2>&1; then
  echo "  aria2c already installed."
else
  echo "  Installing aria2 ..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq aria2
fi
echo

mkdir -p "$HEADSWAP_MODEL_STORE" "$HEADSWAP_STAGING_DIR" "$COMFYUI_PATH/models"

DL_COMMON=(
  --comfy "$COMFYUI_PATH"
  --store-dir "$HEADSWAP_MODEL_STORE"
  --staging-dir "$HEADSWAP_STAGING_DIR"
  --backend auto
  --disable-xet
  --manifest "$REPO_ROOT/scripts/models.json"
)

STEP=3
if [[ "$DOWNLOAD_KONTEXT" -eq 1 ]]; then
  STEP=$((STEP + 1))
  echo "[$STEP] Download Kontext models → $HEADSWAP_MODEL_STORE"
  if [[ -f "$REPO_ROOT/scripts/download_kontext.py" ]]; then
    python3 "$REPO_ROOT/scripts/download_kontext.py" "${DL_COMMON[@]}"
  else
    python3 "$REPO_ROOT/scripts/download_models.py" --set kontext "${DL_COMMON[@]}"
  fi
  echo
fi

if [[ "$DOWNLOAD_KLEIN" -eq 1 ]]; then
  STEP=$((STEP + 1))
  echo "[$STEP] Download Klein models → $HEADSWAP_MODEL_STORE"
  python3 "$REPO_ROOT/scripts/download_models.py" --set klein "${DL_COMMON[@]}"
  echo
fi

if [[ "$DOWNLOAD_QWEN" -eq 1 ]]; then
  STEP=$((STEP + 1))
  echo "[$STEP] Download Qwen models → $HEADSWAP_MODEL_STORE"
  python3 "$REPO_ROOT/scripts/download_models.py" --set qwen "${DL_COMMON[@]}"
  echo
fi

if [[ "$DOWNLOAD_KREA2" -eq 1 ]]; then
  STEP=$((STEP + 1))
  echo "[$STEP] Install Krea2 edit custom nodes"
  bash "$REPO_ROOT/scripts/setup_krea2_nodes.sh"
  echo
  STEP=$((STEP + 1))
  echo "[$STEP] Download Krea2 Identity Edit models → $HEADSWAP_MODEL_STORE"
  if [[ -f "$REPO_ROOT/scripts/download_krea2.py" ]]; then
    python3 "$REPO_ROOT/scripts/download_krea2.py" "${DL_COMMON[@]}"
  else
    python3 "$REPO_ROOT/scripts/download_models.py" --set krea2 "${DL_COMMON[@]}"
  fi
  echo
fi

# head_matte mask backend (segmentation.py): intersects the geometric ellipse
# with a real foreground matte so the head mask follows the actual silhouette
# instead of enclosing background. Without rembg this silently degrades to the
# plain ellipse, so install it here rather than leaving it to chance.
# CPU onnxruntime is fine -- one matte per image on a ~1024px frame.
echo "-> Installing rembg (head_matte mask backend)..."
pip install -q rembg 2>&1 | tail -2 || echo "   WARN: rembg install failed; head_matte will fall back to ellipse"
python -c "import rembg" 2>/dev/null \
  && echo "   rembg OK" \
  || echo "   WARN: rembg not importable; head_matte will fall back to ellipse"

# erase_headwear (headwear_erase.py) needs LaMa inpainting. simple-lama-inpainting
# silently downgrades pillow to 9.5.0 and numpy to 1.26.4 as transitive deps, which
# breaks CuPy (built against numpy 2.x) and makes rembg import fail -- so head_matte
# and restore_background() (the halo fix) silently fall back/no-op with NO error,
# and every downstream render looks like a pipeline regression instead of a pin
# problem. Install it first, then force pillow/numpy back to a working pin.
echo "-> Installing simple-lama-inpainting (erase_headwear) + repairing its numpy/pillow downgrade..."
pip install -q simple-lama-inpainting 2>&1 | tail -2 || echo "   WARN: simple-lama-inpainting install failed; erase_headwear will fall back/skip"
pip install -q --force-reinstall --no-deps pillow==11.3.0
# --force-reinstall, matching the pillow line above: a plain version-range
# install can leave numpy's pure-Python files and its compiled C extension
# from two different versions coexisting on disk (pip considers a range
# "satisfied" and skips reinstalling), which surfaces later as an ImportError
# deep in some unrelated import chain (e.g. "cannot import name '_slice'
# from numpy._core.umath") rather than as anything about numpy or this
# install step -- GPU-confirmed. --no-deps keeps this from cascading into
# reinstalling everything else that depends on numpy.
pip install -q --force-reinstall --no-deps numpy==2.3.4
python -c "
import rembg  # noqa: F401
print('   rembg OK after simple-lama pin repair')
" 2>/dev/null || echo "   WARN: rembg broken after simple-lama install; head_matte will fall back to ellipse"

# mediapipe (skin_harmonize.py's limb mask for arms/legs). Without it,
# extend_skin_harmonization silently falls back to a coarse "everything
# below the head" geometric mask -- which can tint clothing, not just skin.
# Not previously installed anywhere in this script; confirmed missing on a
# real Colab run (`[skin_harm] limb_backend=geometric`).
echo "-> Installing mediapipe (skin_harmonize.py limb mask backend)..."
pip install -q mediapipe 2>&1 | tail -2 || echo "   WARN: mediapipe install failed; skin harmonization will fall back to the coarse geometric limb mask"
python -c "import mediapipe" 2>/dev/null \
  && echo "   mediapipe OK" \
  || echo "   WARN: mediapipe not importable; skin harmonization will fall back to the coarse geometric limb mask"

# Skin-vs-clothes segmenter for skin_harmonize.py. Without this model the
# skin mask falls back to colour only, which cannot tell skin from
# skin-coloured fabric -- GPU-observed repainting a cream dress (its "skin"
# region measured L=197, i.e. the garment) by 59 L-points.
echo "-> Downloading selfie multiclass segmenter (skin vs clothes)..."
mkdir -p /content/models
SEG_TFLITE=/content/models/selfie_multiclass_256x256.tflite
# Verify by SIZE, not just existence: a failed curl leaves a short HTML error
# body behind, which -f would accept and the pipeline would then treat as a
# working model. Observed downstream as a run where the semantic restore
# silently fell back and the skin changed by nothing at all.
if [ ! -f "$SEG_TFLITE" ] || [ "$(wc -c < "$SEG_TFLITE")" -lt 100000 ]; then
  rm -f "$SEG_TFLITE"
  curl -fsSL --retry 3 --retry-delay 2 -o "$SEG_TFLITE" \
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite" \
    || echo "   WARN: segmenter download failed"
fi
if [ -f "$SEG_TFLITE" ] && [ "$(wc -c < "$SEG_TFLITE")" -ge 100000 ]; then
  echo "   selfie multiclass segmenter OK ($(wc -c < "$SEG_TFLITE") bytes)"
else
  rm -f "$SEG_TFLITE"
  echo "   WARN: segmenter UNUSABLE - skin/clothes separation will be off,"
  echo "         the original body will be restored, and the pipeline will"
  echo "         fall back to the LAB wash for skin tone."
fi

echo "Setup complete."
echo "COMFYUI_PATH=$COMFYUI_PATH"
echo "HEADSWAP_MODEL_STORE=$HEADSWAP_MODEL_STORE"
echo
if [[ "$DOWNLOAD_KREA2" -eq 1 ]]; then
  echo "Ready to run:"
  echo "  python scripts/run_pipeline.py --config configs/krea2_identity_edit.yaml --pair-id custom_001 --limit 1"
fi
