# Running headswap_V2 on Google Colab

This guide is for the **Krea 2 Identity Edit** demo notebook intended for sharing with stakeholders.

## Open the notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. **File → Upload notebook** and choose `notebooks/krea2_identity_edit.ipynb` from this repo  
   — or clone the repo in Colab and open that path.
3. **Runtime → Change runtime type → GPU**. Prefer **A100** (40 GB). T4 works but is slower.

## What the notebook does

| Step | Action |
| --- | --- |
| GPU check | Confirms CUDA before large downloads |
| Drive mount | Caches ~18 GB weights under `MyDrive/headswap_V2/models` |
| Clone | `https://github.com/malihashar/headswap_V2.git` → `/content/headswap_V2` |
| Setup | `scripts/setup_colab.sh` + `scripts/setup_krea2_nodes.sh` |
| HF login | Token with **read** access |
| Download | `scripts/download_krea2.py` (skips files already complete on Drive) |
| Upload | Body (scene) + face (identity) |
| Run | `run_eval` → `configs/krea2_identity_edit.yaml` |
| Results | Side-by-side preview, timings, download |

The notebook **orchestrates** existing scripts and pipelines. It does not embed a second copy of the diffusion graph.

## Paths (Colab)

| Role | Path |
| --- | --- |
| Repository | `/content/headswap_V2` |
| ComfyUI | `/content/ComfyUI` |
| Model cache (persistent) | `/content/drive/MyDrive/headswap_V2/models` |
| HF staging | `/content/_hf_dl_staging` |
| Demo outputs | `/content/headswap_outputs/` |

## First run vs reconnect

- **First run:** Drive mount + full model download (~18 GB, once).
- **Later sessions:** Drive still has weights → download step verifies quickly; ComfyUI may need a short re-install on a fresh runtime.

## CLI equivalent (advanced)

From the repo root on Colab after Drive is mounted:

```bash
export COMFYUI_PATH=/content/ComfyUI
export HEADSWAP_MODEL_STORE=/content/drive/MyDrive/headswap_V2/models
export HEADSWAP_STAGING_DIR=/content/_hf_dl_staging
export HF_HUB_DISABLE_XET=1

bash scripts/setup_colab.sh
bash scripts/setup_krea2_nodes.sh
python scripts/download_krea2.py
python scripts/prepare_eval_set.py --custom data/custom
python scripts/run_pipeline.py \
  --config configs/krea2_identity_edit.yaml \
  --pair-id custom_001 \
  --limit 1
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No GPU detected` | Runtime → Change runtime type → GPU |
| Drive not mounted | Re-run the mount cell; or `bash scripts/setup_colab.sh --no-drive` (ephemeral) |
| HF rate limit / 401 | `login()` with a read token |
| Slow (~3–4 min) on T4 | Expected; switch to A100 for the intended demo latency |
| OOM on decode | Restart runtime; ensure only one pipeline is loaded |

## Sharing with others

Send `notebooks/krea2_identity_edit.ipynb` (or a Colab link after **File → Save a copy in Drive**).  
Recipients only need a Colab GPU runtime and a Hugging Face token — no local setup.
