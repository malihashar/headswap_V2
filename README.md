# headswap_V2 — Magic Hour head swap prototype

Higher-quality open-source head swap for Magic Hour, focused on **determinism** and **local edits**.

## Winner (recommended promote)

**Primary:** `klein4b_mask_crop_stitch` — FLUX.2 [klein] 4B distilled + head/hair mask + crop → multi-ref edit → soft stitch  
**Fallback:** `qwen_improved_mask_crop` — same locality pattern on current Qwen 2511 + BFS V5  

Model/URL validation (official sources only): see [`docs/VALIDATION.md`](docs/VALIDATION.md).

The current production full-frame Qwen+Lightning path (`qwen_baseline`) stays as the control / legacy arm.

## What’s in this repo

| Path | Purpose |
| --- | --- |
| `configs/klein4b.yaml` | Klein 4B masked crop-stitch (main prototype) |
| `configs/qwen_baseline.yaml` | Faithful port of current Colab Cell 5 |
| `configs/qwen_improved.yaml` | Qwen + official BFS prompt + mask crop-stitch |
| `src/headswap/` | Pipelines, preprocess, metrics, eval harness |
| `workflows/` | Upstream ComfyUI Klein + BFS Klein graphs (reference) |
| `scripts/` | Setup (Colab/Kaggle/ComfyUI), model download, eval, compare |
| `notebooks/colab_compare.ipynb` | GPU Colab runner |
| `results/COMPARISON.md` | Auto-written after `run_compare.py` |

## Architecture (Klein)

1. Detect face → expand to approximate **head+hair** mask  
2. Crop editable region; keep full body for stitch  
3. Face reference: tight head crop **with shoulders** (no blur-pad)  
4. FLUX.2 Klein 4B multi-ref edit (`ReferenceLatent` body_crop + face), 4 steps, CFG 1.0  
5. Soft-alpha stitch + mild LAB skin match  
6. Metrics: face detect, optional ArcFace identity, body PSNR outside mask, smoothness heuristic  

BFS Flux Klein LoRA is **on by default** (`bfs_lora_strength: 1.0`) and is a required download.

## Quick start (CPU mock / CI)

```bash
cd ~/repos/headswap_V2
python3 -m pip install -e ".[dev]"
python3 scripts/prepare_eval_set.py 24
python3 scripts/run_compare.py          # mock all pipelines + write results/COMPARISON.md
pytest -q
```

### Custom real photos (1 pair)

Place your images at:

```text
data/custom/body.png   # destination / body
data/custom/face.png   # source face / head
```

Then benchmark **one** pipeline (image + timings + `metrics.json`, then exit):

```bash
python3 scripts/prepare_eval_set.py --custom
python3 scripts/run_pipeline.py --config configs/qwen_baseline.yaml --limit 1
# equivalents:
#   headswap-run --config configs/qwen_baseline.yaml --limit 1
#   python3 -m headswap.cli --config configs/qwen_baseline.yaml --limit 1
```

For a full three-way eval (unchanged):

```bash
python3 scripts/run_compare.py --gpu --limit 1
```

Outputs land under `results/<pipeline>/images/custom_001/` (`result.png`, debug crops/masks, plus `metrics.json`).

## Google Colab

Large HF/Xet downloads often stall on Colab. This repo downloads over **classic HTTP** (`HF_HUB_DISABLE_XET=1`), writes **only to local staging**, verifies against [`scripts/models.json`](scripts/models.json), then promotes complete files to **Google Drive** and symlinks into ComfyUI. Partials never land on Drive. If Hub HTTP stalls (&lt;1 MiB / 5 min), the downloader kills that attempt and falls back to resumable `aria2c`.

Minimal setup on a fresh Colab GPU runtime:

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!git clone https://github.com/malihashar/headswap_V2.git || true
%cd /content/headswap_V2
!git pull --ff-only

# Recommended for rate limits / gated assets
from huggingface_hub import login
login()  # or: import os; os.environ["HF_TOKEN"] = "hf_..."

!bash scripts/setup_colab.sh
!python scripts/run_compare.py --gpu --limit 12
```

`scripts/setup_colab.sh` is idempotent:

1. Requires Drive mounted at `/content/drive/MyDrive`
2. Installs ComfyUI + `requirements.txt` (do **not** install `hf_xet`)
3. Installs `aria2` for the HTTP resume fallback
4. Downloads missing Klein/Qwen weights into `/content/_hf_dl_staging`, verifies size, promotes to `/content/drive/MyDrive/headswap_V2/models/…`, symlinks into `/content/ComfyUI/models/…`

On a later runtime, complete Drive files are skipped and only re-linked.

## Experimental: Step1X-Edit (`step1x_edit`)

Prototype Diffusers backend using **Step1X-Edit-v1p2** (ReasonEdit-S) — the highest-quality public Step1X image-editing checkpoint on GEdit-Bench / KRIS-Bench when thinking + reflection are enabled.

| Item | Value |
| --- | --- |
| Official repo | https://github.com/stepfun-ai/Step1X-Edit |
| Weights | https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2 |
| Diffusers branch | `Peyton-Chen/diffusers` @ `step1xedit_v1p2` (`Step1XEditPipelineV1P2`) |
| Scheduler | `FlowMatchEulerDiscreteScheduler` |
| Text encoder | `Qwen2_5_VLForConditionalGeneration` (+ `Qwen2_5_VLProcessor`) |
| VAE | `AutoencoderKL` |
| Recommended settings | `steps=50`, `true_cfg_scale=6` (`guidance`), thinking+reflection on, `size_level`/`crop_size=1024`, `bfloat16` |
| Disk | ~42 GiB Diffusers snapshot |
| VRAM | Full bf16 load needs a large GPU (~40–80 GiB class); default config uses `enable_cpu_offload: true` |

**Variants (why v1p2):** v1.0 / v1.1 are older single-pass editors; v1p2-preview is weaker on KRIS overall; **v1p2 + thinking + reflection** is the top published open checkpoint.

**Head-swap adaptation:** Step1X accepts **one** image + text. This pipeline builds a dual panel — left = body (image 1 / guy), right = face (image 2 / Ronaldo) — runs the instruction edit, then crops the left panel.

**Limitation:** Official `__call__` has **no `strength`/denoise** parameter (config key kept for parity; recorded unused in metrics). Encode/decode timings are null because Diffusers bundles them inside sampling.

### Kaggle commands

```python
%cd /kaggle/working/headswap_V2
!bash scripts/setup_kaggle.sh   # ComfyUI optional for this pipeline

# Diffusers branch (required — stock PyPI diffusers lacks Step1XEditPipelineV1P2)
!pip install 'transformers==4.55.0'
!git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git /tmp/diffusers-step1x
!pip install -e /tmp/diffusers-step1x

# Weights → /tmp/models/Step1X-Edit-v1p2 (~42 GiB, resumable)
!python scripts/download_step1x.py

# custom_001 = body guy + Ronaldo face
!python scripts/run_pipeline.py --config configs/step1x_edit.yaml --pair-id custom_001 --limit 1
```

Outputs: `results/step1x_edit/images/…` and `results/step1x_edit/metrics.json`.

Does **not** change `qwen_baseline` or `flux_kontext`.

## Experimental: OmniGen2 (`omnigen2`)

Prototype using **OmniGen2** in-context multi-image editing — native `input_images=[body, face]` (no dual-panel hack).

| Item | Value |
| --- | --- |
| Official repo | https://github.com/VectorSpaceLab/OmniGen2 |
| Weights | https://huggingface.co/OmniGen2/OmniGen2 |
| Pipeline class | `omnigen2…OmniGen2Pipeline` (`trust_remote_code=True`) |
| Scheduler | Flow-Match Euler (default) or `dpmsolver++` |
| Conditioner | Qwen2.5-VL (`mllm`) · VAE `AutoencoderKL` |
| Recommended | `steps=50`, `text_guidance_scale=5.0` (`guidance`), `image_guidance_scale=2.5` (`image_guidance`, in-context tip 2.5–3.0) |
| VRAM | ~17 GB native (RTX 3090-class); `enable_model_cpu_offload` ≈ −50% VRAM |

Official in-context prompt style (README tip #5): *“Edit the first image: replace … from the second image …”*.

### Kaggle commands

```python
%cd /kaggle/working/headswap_V2
!git pull --ff-only   # need omnigen2 files on the remote first

# Clone OmniGen2 code ONLY — do NOT pip install its requirements.txt
# (that file pins torch==2.6.0 and will break Kaggle's torch/CUDA stack).
# Upstream also has no setup.py, so `pip install -e` fails.
!bash scripts/setup_omnigen2.sh

# Weights → /tmp/models/OmniGen2 (resumable)
!python scripts/download_omnigen2.py

# custom_001 = body guy + Ronaldo face
!python scripts/run_pipeline.py --config configs/omnigen2.yaml --pair-id custom_001 --limit 1
```

Outputs: `results/omnigen2/`.

**Limitation:** No `strength`/denoise in the official API (config key recorded unused). Encode/decode timings null (bundled in `__call__`). Needs the OmniGen2 git tree on `PYTHONPATH` (handled by `setup_omnigen2.sh` / auto-path), not Comfy-only weights.

## Kaggle

Kaggle splits storage into two filesystems:

| Path | Role | Capacity |
| --- | --- | --- |
| `/kaggle/working` | 20GB loop device | ComfyUI + notebook outputs only |
| `/tmp` (overlay root) | ~1T free | **model store + HF staging** |

**Default model path is FLUX.1 Kontext** (`configs/flux_kontext.yaml`). Klein / Qwen downloads are opt-in.

### Fresh session (recommended)

```python
%cd /kaggle/working
!git clone https://github.com/malihashar/headswap_V2.git || true
%cd /kaggle/working/headswap_V2
!git pull --ff-only

from huggingface_hub import login
login()  # optional for Kontext (Comfy-Org mirrors are public); useful for rate limits

# ComfyUI + Python deps only (NO model download)
!bash scripts/setup_kaggle.sh

# Kontext weights → /tmp/models (never /kaggle/working)
!python scripts/download_kontext.py

# or equivalently in one step:
# !bash scripts/setup_kaggle.sh --kontext

!python scripts/run_pipeline.py --config configs/flux_kontext.yaml --limit 1
```

### `setup_kaggle.sh` flags

| Command | Effect |
| --- | --- |
| `bash scripts/setup_kaggle.sh` | ComfyUI + deps + aria2 only; **no models** |
| `bash scripts/setup_kaggle.sh --kontext` | + FLUX.1 Kontext set → `/tmp/models` |
| `bash scripts/setup_kaggle.sh --klein` | + FLUX.2 Klein (+ BFS) set → `/tmp/models` |
| `bash scripts/setup_kaggle.sh --qwen` | + Qwen Image Edit 2511 set → `/tmp/models` |

### One-command bootstrap

In a notebook with GPU + Internet enabled:

```python
%cd /kaggle/working/headswap_V2
!bash scripts/bootstrap_kaggle.sh            # default: Kontext models
# !bash scripts/bootstrap_kaggle.sh --klein  # Klein instead
# !bash scripts/bootstrap_kaggle.sh --no-models
```

On a completely fresh runtime (repo not cloned yet):

```python
!curl -fsSL https://raw.githubusercontent.com/malihashar/headswap_V2/main/scripts/bootstrap_kaggle.sh | bash
```

[`scripts/bootstrap_kaggle.sh`](scripts/bootstrap_kaggle.sh) is idempotent: pull/clone → ComfyUI + deps → `/tmp` Kontext models (symlink-only if already complete) → if `data/custom/body.png` + `face.png` exist, prepare the 1-pair set and `run_compare.py --gpu --limit 1`.

Defaults:

- `COMFYUI_PATH=/kaggle/working/ComfyUI`
- `HEADSWAP_MODEL_STORE=/tmp/models`
- `HEADSWAP_STAGING_DIR=/tmp/_hf_dl_staging`

On start the downloader prints `store_dir`, `staging_dir`, and `df -h /tmp` / `df -h /kaggle/working` so you can confirm downloads are not hitting the 20GB loop.

## GPU run (Colab / RunPod)

```bash
export COMFYUI_PATH=/content/ComfyUI   # or /workspace/ComfyUI
bash scripts/setup_comfyui.sh
pip install -U huggingface_hub
python3 scripts/download_models.py --set klein --verify-only   # confirm URLs first
python3 scripts/download_models.py --set all                   # required only (hf_hub_download)
# optional Klein extras (BFS LoRA, bf16 UNET):
# python3 scripts/download_models.py --set klein --include-optional
python3 scripts/prepare_eval_set.py 24
# Optional: replace data/eval/bodies + faces with real consented photos
python3 scripts/run_compare.py --gpu --limit 12
```

Single pipeline:

```bash
python3 -m headswap.cli --help
# or
PYTHONPATH=src python3 -c "from headswap.cli import main_run; main_run()" --config configs/klein4b.yaml
```

Use the CLI entry points after `pip install -e .`:

```bash
headswap-prepare-eval
headswap-run --config configs/klein4b.yaml --mock
headswap-compare
```

## Eval criteria

- **Success:** face detected + identity above threshold (when InsightFace available) + not pathologically smooth  
- **Body preserve PSNR** outside mask (masked pipelines should score high)  
- **Latency p95** vs ~30s budget (match current ~28s Colab)  

Composite score for promote decision:

`0.6 * success + 0.2 * body_psnr_norm + 0.2 * identity - 0.15 if over latency budget`

## Notes

- Synthetic eval images are for harness validation. Swap in real photos for product decisions.  
- Prefer **Klein 4B** (Apache 2.0). Do **not** ship Klein 9B commercially (non-commercial license).  
- OpenCV SSD mask is a portable fallback; swap in SAM3/BiRefNet for production masks when available.  
