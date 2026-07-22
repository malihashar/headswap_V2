# Running headswap_V2 on Google Colab

Internal **production demo** for Krea 2 Identity Edit (Magic Hour engineers).

## Open & run

1. Upload `notebooks/krea2_identity_edit.ipynb` to Colab.
2. **Runtime → Change runtime type → GPU** (prefer **A100**).
3. Edit §1 knobs if needed → **Runtime → Run all**.

### Expected runtime (warm)

| GPU | Typical |
| --- | --- |
| A100 40 GB | ~1–2 min / image |
| T4 16 GB | ~3–4 min / image |

## User knobs (§1)

| Knob | Default | Meaning |
| --- | --- | --- |
| `SEED` | `46` | Deterministic sampling seed |
| `PROMPT` | yaml default | Optional prompt override |
| `STEPS` | `8` | Denoising steps |
| `CFG` | `1.0` | Guidance scale |
| `OUTPUT_LONG_SIDE` | `1024` | Final body canvas long side |
| `STITCH` | `True` | Mask → crop → edit → soft stitch |
| `DEBUG` | `False` | Verbose logs + debug assets |
| `IDENTITY_THRESH` | `0.35` | Post-run identity warning / fail gate |
| `BODY_PSNR_THRESH` | `28.0` | Post-run body-preserve warning |
| `PINNED_COMMIT` | `None` | Optional exact `git` commit to checkout |

## Output package

Each successful §5 run writes:

`/content/headswap_outputs/run_YYYYMMDD_HHMMSS/`

| File | Contents |
| --- | --- |
| `result.png` | Final image |
| `run_config.json` | Knobs, versions, model sizes, face boxes, reproduce hints |
| `metrics.json` | Latency + automatic quality scores |
| `timing.json` | Stage timing breakdown |
| `debug/` | Intermediates **only if** `DEBUG=True` |

Also refreshes `/content/headswap_outputs/HEADSWAP_RESULT.png`.

## Reproduce a prior run

1. Open that run’s `run_config.json`.
2. Copy `reproduce.knobs_from_config` into §1 (and set `PINNED_COMMIT` if you need the exact repo revision).
3. Re-upload the same body/face → Run all.

## Paths

| Role | Path |
| --- | --- |
| Repo | `/content/headswap_V2` |
| ComfyUI | `/content/ComfyUI` |
| Model cache | `/content/drive/MyDrive/headswap_V2/models` |
| Outputs | `/content/headswap_outputs/` |

## First custom-node install

On a brand-new runtime, after the first `setup_krea2_nodes.sh`, **restart once**, then Run all from §1.

## HF login

Not required — Krea2 assets used here are public.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| No GPU | Runtime → GPU (A100 preferred) |
| Missing / bad model size | `python scripts/download_krea2.py` |
| No face / multi-face | Clear single-subject photo; largest face is used |
| Need traceback | `DEBUG=True` in §1 |
| Identity score `None` | Install insightface (optional); body metrics still run |
