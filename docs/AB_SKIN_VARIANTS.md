# Skin-strategy A/B — how to run it

Five strategies, all of which keep the head swap identical. The only thing
they disagree about is how the BODY's skin tone is made to match the new head.

Single-pair eyeballing picked the wrong winner repeatedly: a change that
helped one pair regressed another, and a change that moved 2% of the pixels
looked the same by eye as one that moved none. Six pairs x five arms with two
measured numbers settles it in one run.

## Cell A — upload every image at once

One upload, one click. Select **all** your files in the picker together --
bodies and faces, any order. They are paired by the number in the filename,
so name them `body1.png` + `face1.png`, `body2.png` + `face2.png`, and so on.

The old version called `files.upload()` once per image inside a loop. That
widget silently does nothing if the cell loses focus between calls, which is
how thirteen of fourteen pair directories ended up empty. One call cannot fail
that way.

```python
#@title Upload all pairs at once (body1/face1, body2/face2, ...)
import re, shutil
from io import BytesIO
from pathlib import Path
from PIL import Image
from google.colab import files

DEST = Path("/content/headswap_V2/data/custom/ab_pairs")
shutil.rmtree(DEST, ignore_errors=True)
DEST.mkdir(parents=True)

print("Select ALL images at once (body1.png, face1.png, body2.png, ...)")
up = files.upload()

imgs = {}
for name, data in up.items():
    nums = re.findall(r"\d+", name)
    if not nums:
        print(f"  ! {name}: no number in filename, skipped")
        continue
    kind = "body" if name.lower().startswith("body") else (
        "face" if name.lower().startswith("face") else None)
    if kind is None:
        print(f"  ! {name}: must start with 'body' or 'face', skipped")
        continue
    imgs.setdefault(nums[0], {})[kind] = Image.open(BytesIO(data)).convert("RGB")

n = 0
for idx in sorted(imgs, key=lambda x: int(x)):
    pair = imgs[idx]
    if "body" not in pair or "face" not in pair:
        print(f"  ! pair{idx}: missing {'body' if 'body' not in pair else 'face'}, skipped")
        continue
    d = DEST / f"pair{idx}"
    d.mkdir(parents=True, exist_ok=True)
    pair["body"].save(d / "body.png")
    pair["face"].save(d / "face.png")
    n += 1
    print(f"  pair{idx}: ok")

print(f"\n{n} complete pairs in {DEST}")
```

### If the upload widget still misbehaves

Drag the files straight into `/content/` using the folder icon in the left
sidebar, then run this instead -- no widget involved at all:

```python
import re, shutil
from pathlib import Path
from PIL import Image

SRC, DEST = Path("/content"), Path("/content/headswap_V2/data/custom/ab_pairs")
shutil.rmtree(DEST, ignore_errors=True); DEST.mkdir(parents=True)
n = 0
for b in sorted(SRC.glob("body*.*")):
    nums = re.findall(r"\d+", b.name)
    if not nums:
        continue
    f = next((p for p in SRC.glob(f"face{nums[0]}.*")), None)
    if f is None:
        print(f"  ! no face{nums[0]} for {b.name}"); continue
    d = DEST / f"pair{nums[0]}"; d.mkdir(parents=True, exist_ok=True)
    Image.open(b).convert("RGB").save(d / "body.png")
    Image.open(f).convert("RGB").save(d / "face.png")
    n += 1; print(f"  pair{nums[0]}: {b.name} + {f.name}")
print(f"\n{n} complete pairs")
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
