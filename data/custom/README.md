# Custom real-photo eval pair

Place exactly two images here (Colab upload cell writes these for you):

| File | Role |
| --- | --- |
| `body.png` | Destination / body image (pose, clothes, background kept) |
| `face.png` | Source face / head identity to swap onto the body |

These PNGs are **gitignored**. Do not commit them. A `git reset --hard`
must never restore someone else's demo pair over your uploads.

Then prepare the 1-pair eval set:

```bash
python scripts/prepare_eval_set.py --custom
```

This writes `data/eval/pairs.json` with a single pair and copies the images into
`data/eval/bodies/` + `data/eval/faces/` so `run_compare.py` / `headswap-run`
work unchanged.
