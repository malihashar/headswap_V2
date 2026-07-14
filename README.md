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
| `scripts/` | Setup, model download, eval, compare |
| `notebooks/colab_compare.ipynb` | GPU Colab runner |
| `results/COMPARISON.md` | Auto-written after `run_compare.py` |

## Architecture (Klein)

1. Detect face → expand to approximate **head+hair** mask  
2. Crop editable region; keep full body for stitch  
3. Face reference: tight head crop **with shoulders** (no blur-pad)  
4. FLUX.2 Klein 4B multi-ref edit (`ReferenceLatent` body_crop + face), 4 steps, CFG 1.0  
5. Soft-alpha stitch + mild LAB skin match  
6. Metrics: face detect, optional ArcFace identity, body PSNR outside mask, smoothness heuristic  

BFS Flux Klein LoRA is **off by default** (`bfs_lora_strength: 0.0`); A/B with `0.8–1.0`.

## Quick start (CPU mock / CI)

```bash
cd ~/repos/headswap_V2
python3 -m pip install -e ".[dev]"
python3 scripts/prepare_eval_set.py 24
python3 scripts/run_compare.py          # mock all pipelines + write results/COMPARISON.md
pytest -q
```

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
