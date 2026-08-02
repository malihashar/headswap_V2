#!/usr/bin/env bash
# Bootstrap REFace (WACV 2025) on Google Colab with Drive-cached weights.
# Usage: bash scripts/setup_reface_colab.sh
set -euo pipefail

REPO="${HEADSWAP_REPO:-/content/headswap_V2}"
DRIVE_ROOT="${REFACE_DRIVE_ROOT:-/content/drive/MyDrive/headswap_reface}"
REFACE_ROOT="${REFACE_ROOT:-/content/REFace}"
WEIGHTS_CACHE="${DRIVE_ROOT}/weights"

echo "→ REFace setup"
echo "  repo=$REPO"
echo "  reface=$REFACE_ROOT"
echo "  weights_cache=$WEIGHTS_CACHE"

mkdir -p "$DRIVE_ROOT" "$WEIGHTS_CACHE"

# --- clone upstream ---
if [[ -d "$REFACE_ROOT/.git" ]]; then
  git -C "$REFACE_ROOT" pull --ff-only || true
else
  rm -rf "$REFACE_ROOT"
  git clone --depth 1 https://github.com/Sanoojan/REFace.git "$REFACE_ROOT"
fi

# --- pip deps (Colab torch is usually fine; avoid forcing cu117) ---
pip install -q -U pip setuptools wheel
pip install -q huggingface_hub
# Core REFace deps (best-effort pins; Colab may already have newer torch).
pip install -q \
  "pytorch-lightning==1.9.5" "transformers==4.30.2" "omegaconf==2.1.1" \
  "einops==0.4.1" "kornia==0.6.12" "albumentations==1.3.1" \
  "face_alignment==1.4.1" "dlib==19.24.2" "opencv-python-headless" \
  "scikit-image" "lpips" "tqdm" "Pillow" "natsort" "ftfy" "regex" \
  "diffusers==0.30.3" "torchmetrics==0.11.4" || true

pip install -q -e "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers" || true
pip install -q -e "git+https://github.com/openai/CLIP.git@main#egg=clip" || true
pip install -q -e "$REFACE_ROOT" || true

# Editable headswap
if [[ -f "$REPO/pyproject.toml" || -f "$REPO/setup.py" ]]; then
  pip install -q -e "$REPO"
fi

# --- HuggingFace assets (last.ckpt + Other_dependencies) ---
export WEIGHTS_CACHE="$WEIGHTS_CACHE"
export REFACE_ROOT="$REFACE_ROOT"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

cache = Path(os.environ.get("WEIGHTS_CACHE", "/content/drive/MyDrive/headswap_reface/weights"))
cache.mkdir(parents=True, exist_ok=True)
reface = Path(os.environ.get("REFACE_ROOT", "/content/REFace"))

print("→ downloading REFace last.ckpt (≈6GB, Drive-cached)…")
ckpt = hf_hub_download(
    repo_id="Sanoojan/REFace",
    filename="last.ckpt",
    local_dir=str(cache / "hf"),
    local_dir_use_symlinks=False,
)
print("✓", ckpt)

print("→ downloading Other_dependencies…")
deps = snapshot_download(
    repo_id="Sanoojan/REFace",
    allow_patterns=["Other_dependencies/**"],
    local_dir=str(cache / "hf"),
    local_dir_use_symlinks=False,
)
print("✓", deps)

# Wire into REFace tree
ckpt_dir = reface / "models" / "REFace" / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
target_ckpt = ckpt_dir / "last.ckpt"
if not target_ckpt.exists():
    try:
        target_ckpt.symlink_to(ckpt)
    except Exception:
        import shutil
        shutil.copy2(ckpt, target_ckpt)
# Scripts historically look for saved.ckpt
saved = ckpt_dir / "saved.ckpt"
if not saved.exists():
    try:
        saved.symlink_to(target_ckpt.resolve())
    except Exception:
        import shutil
        shutil.copy2(target_ckpt, saved)

# Other_dependencies
src_deps = Path(deps) / "Other_dependencies"
dst_deps = reface / "Other_dependencies"
if src_deps.is_dir():
    if dst_deps.exists() or dst_deps.is_symlink():
        if dst_deps.is_symlink() or dst_deps.is_file():
            dst_deps.unlink()
        else:
            # Keep existing; overlay missing files via symlink of whole tree if empty-ish
            pass
    if not dst_deps.exists():
        try:
            dst_deps.symlink_to(src_deps)
        except Exception:
            import shutil
            shutil.copytree(src_deps, dst_deps)
print("✓ REFace weights linked")
PY

# DLIB landmark fallback if HF folder incomplete
DLIB_DIR="$REFACE_ROOT/Other_dependencies/DLIB_landmark_det"
mkdir -p "$DLIB_DIR"
if [[ ! -f "$DLIB_DIR/shape_predictor_68_face_landmarks.dat" ]]; then
  echo "→ downloading dlib 68-landmark predictor…"
  wget -q --show-progress -O "$DLIB_DIR/shape_predictor_68_face_landmarks.dat" \
    "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat" \
    || true
fi

# Ensure config exists
if [[ ! -f "$REFACE_ROOT/models/REFace/configs/project_ffhq.yaml" ]]; then
  echo "⚠ missing models/REFace/configs/project_ffhq.yaml — check upstream checkout"
fi

echo "✓ REFace setup complete → $REFACE_ROOT"
echo "  export REFACE_ROOT=$REFACE_ROOT"
