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

# Colab is Python 3.12+; upstream still has dead ``import imp`` etc.
if [[ -f "$REPO/scripts/patch_reface_py312.py" ]]; then
  python "$REPO/scripts/patch_reface_py312.py" --reface-root "$REFACE_ROOT"
fi

# --- pip deps (Colab torch is usually fine; avoid forcing cu117) ---
pip install -q -U pip setuptools wheel
pip install -q huggingface_hub gdown

# Critical runtime imports for scripts/inference_swap_selected.py — must succeed.
echo "→ installing REFace Python deps (required)…"
# REFace LDM code expects PL 1.x import paths; force a known-good pin.
pip install -q --force-reinstall "pytorch-lightning==1.9.5"
pip install -q "omegaconf>=2.1,<2.4" "einops" "albumentations" "kornia" \
  "transformers>=4.30,<4.45" "diffusers>=0.24,<0.32" "torchmetrics==0.11.4" \
  "face_alignment" "opencv-python-headless" "scikit-image" "tqdm" "Pillow" \
  "natsort" "ftfy" "regex" "lpips" "timm" "pytorch-fid" "bezier"

# dlib is optional-ish (landmarks); try wheels first.
pip install -q dlib || pip install -q dlib==19.24.2 || echo "⚠ dlib install failed — may still work with face_alignment"

# taming + CLIP: clone + PYTHONPATH (pip -e alone often fails silently on Colab).
# Editable REFace install is optional.
pip install -q -e "$REFACE_ROOT" || true

# Editable headswap
if [[ -f "$REPO/pyproject.toml" || -f "$REPO/setup.py" ]]; then
  pip install -q -e "$REPO"
fi

# Vendor taming/CLIP, patch sources, and smoke-import the real inference chain.
python "$REPO/scripts/ensure_reface_runtime.py" --reface-root "$REFACE_ROOT"

# --- HuggingFace assets (last.ckpt + Other_dependencies) ---
export WEIGHTS_CACHE="$WEIGHTS_CACHE"
export REFACE_ROOT="$REFACE_ROOT"
python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

cache = Path(os.environ.get("WEIGHTS_CACHE", "/content/drive/MyDrive/headswap_reface/weights"))
cache.mkdir(parents=True, exist_ok=True)
reface = Path(os.environ.get("REFACE_ROOT", "/content/REFace"))
hf_dir = cache / "hf"
hf_dir.mkdir(parents=True, exist_ok=True)

print("→ downloading REFace last.ckpt (≈6GB, Drive-cached)…")
ckpt = hf_hub_download(
    repo_id="Sanoojan/REFace",
    filename="last.ckpt",
    local_dir=str(hf_dir),
    local_dir_use_symlinks=False,
)
print("✓", ckpt)

print("→ downloading Other_dependencies (face parsing, arcface, …)…")
try:
    snapshot_download(
        repo_id="Sanoojan/REFace",
        allow_patterns=["Other_dependencies/**"],
        local_dir=str(hf_dir),
        local_dir_use_symlinks=False,
    )
except Exception as exc:
    print(f"⚠ snapshot_download Other_dependencies failed: {exc}")

# Explicit face-parsing weight (required for inference).
face_parse_rel = "Other_dependencies/face_parsing/79999_iter.pth"
try:
    face_parse = hf_hub_download(
        repo_id="Sanoojan/REFace",
        filename=face_parse_rel,
        local_dir=str(hf_dir),
        local_dir_use_symlinks=False,
    )
    print("✓ face parsing", face_parse)
except Exception as exc:
    print(f"⚠ HF face_parsing download failed: {exc}")
    face_parse = None

# ArcFace + DLIB if present on HF
for rel in (
    "Other_dependencies/arcface/model_ir_se50.pth",
    "Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat",
):
    try:
        p = hf_hub_download(
            repo_id="Sanoojan/REFace",
            filename=rel,
            local_dir=str(hf_dir),
            local_dir_use_symlinks=False,
        )
        print("✓", p)
    except Exception as exc:
        print(f"⚠ optional miss {rel}: {exc}")

# Wire checkpoints
ckpt_dir = reface / "models" / "REFace" / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
target_ckpt = ckpt_dir / "last.ckpt"
if not target_ckpt.exists():
    try:
        target_ckpt.symlink_to(ckpt)
    except Exception:
        shutil.copy2(ckpt, target_ckpt)
saved = ckpt_dir / "saved.ckpt"
if not saved.exists():
    try:
        saved.symlink_to(target_ckpt.resolve())
    except Exception:
        shutil.copy2(target_ckpt, saved)

# Merge Other_dependencies into REFace tree (never skip if dest partially exists).
src_deps = hf_dir / "Other_dependencies"
dst_deps = reface / "Other_dependencies"
dst_deps.mkdir(parents=True, exist_ok=True)

def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

if src_deps.is_dir():
    for path in src_deps.rglob("*"):
        if path.is_file():
            rel = path.relative_to(src_deps)
            _link_or_copy(path, dst_deps / rel)

# Ensure face parsing file specifically
parse_dst = dst_deps / "face_parsing" / "79999_iter.pth"
if not parse_dst.is_file():
    if face_parse and Path(face_parse).is_file():
        _link_or_copy(Path(face_parse), parse_dst)
    else:
        # Google Drive fallback (official README link).
        print("→ gdown face parsing 79999_iter.pth …")
        try:
            import gdown  # type: ignore
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
            import gdown  # type: ignore
        parse_dst.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(
            id="154JgKpzCPW82qINcVieuPH3fZ2e0P812",
            output=str(parse_dst),
            quiet=False,
        )

if parse_dst.is_file() and parse_dst.stat().st_size > 1_000_000:
    print(f"✓ face parsing ready ({parse_dst.stat().st_size} bytes)")
else:
    raise SystemExit(
        f"Face parsing weight still missing at {parse_dst}. "
        "Re-run setup or download manually."
    )
print("✓ REFace weights linked")
PY

pip install -q gdown || true

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
