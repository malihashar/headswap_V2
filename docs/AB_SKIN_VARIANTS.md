# Skin-strategy A/B — how to run it

Five strategies, all of which keep the head swap identical. The only thing
they disagree about is how the BODY's skin tone is made to match the new head.

Single-pair eyeballing picked the wrong winner repeatedly: a change that
helped one pair regressed another, and a change that moved 2% of the pixels
looked the same by eye as one that moved none. Six pairs x five arms with two
measured numbers settles it in one run.

## Cell A — upload 6–7 pairs

Paste as one Colab cell. For each pair it asks for the BODY, then the DONOR
face. Enter a blank name when finished.

```python
#@title Upload A/B pairs (body + face per pair)
from io import BytesIO
from pathlib import Path
from PIL import Image
from IPython.display import display, Markdown
from google.colab import files

DEST = Path("/content/headswap_V2/data/custom/ab_pairs")
DEST.mkdir(parents=True, exist_ok=True)

def pick(prompt):
    display(Markdown(f"**{prompt}** — select exactly 1 file"))
    up = files.upload()
    if len(up) != 1:
        raise ValueError(f"select exactly 1 file, got {len(up)}")
    name, data = next(iter(up.items()))
    return Image.open(BytesIO(data)).convert("RGB")

n = 0
while True:
    label = input("\nPair name (blank to finish): ").strip()
    if not label:
        break
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    d = DEST / safe
    d.mkdir(parents=True, exist_ok=True)
    pick(f"[{safe}] BODY (scene to keep)").save(d / "body.png")
    pick(f"[{safe}] DONOR face (head to put on)").save(d / "face.png")
    n += 1
    print(f"  saved {d}")

print(f"\n{n} pairs ready in {DEST}")
for d in sorted(DEST.iterdir()):
    if d.is_dir():
        print(" ", d.name)
```

## Cell B — run all five arms

Roughly `pairs x 5 x ~2 min`, so ~1 hour for 6 pairs. Variant B is ~2x slower
than the others (cfg>1 means two UNet evaluations per step). Progress prints
per render, and `results.json` is rewritten after every one, so a crash or a
disconnect never loses completed work.

```python
!cd /content/headswap_V2 && python scripts/ab_skin_variants.py \
    --pairs data/custom/ab_pairs \
    --config configs/krea2_identity_edit.yaml \
    -o results/_ab_skin_variants
```

## Cell C — read the result

```python
from IPython.display import Markdown, Image as IPImage, display
from pathlib import Path
R = Path("/content/headswap_V2/results/_ab_skin_variants")
display(Markdown((R / "REPORT.md").read_text()))
for m in sorted(R.glob("montage_*.png")):
    print(m.name); display(IPImage(str(m)))
```

## The two numbers

- **`tone_gap`** = `|face L − body-skin L|` measured inside the result.
  **Lower is better.** This is the artifact being chased: a body that does not
  match its own head. Measured on the output alone, so every arm is scored the
  same way regardless of how it got there.
- **`identity`** = ArcFace cosine, donor vs result. **Higher is better.**
  Present specifically to catch a variant that "wins" on tone by damaging the
  face — the exact trade that kept recurring.

Neither number can see a composite seam, a mask edge or a recoloured garment,
so **look at the montages before concluding.** The report says this too.

## The five arms

| key | what it changes | hypothesis |
|---|---|---|
| `A_prompt_only` | nothing (current default) | baseline |
| `B_cfg_guidance` | `cfg=1.8` | at cfg=1.0 there is no classifier-free guidance, so the recolour clause carries no guidance weight at all |
| `C_named_tone` | measures the donor's cheek L, names it in words | turns "match the other image" from an inference into a literal instruction |
| `D_skin_repaint` | 2nd pass, skin mask as `noise_mask` | model *renders* skin instead of recolouring it, head physically pinned |
| `E_lab_wash` | old composited restore + LAB wash | control; the only arm that can produce a composite boundary |

If `E` wins on tone the trade is real and worth re-opening. If it loses on
both tone and artifacts, raw-model is settled and the wash can be deleted.
