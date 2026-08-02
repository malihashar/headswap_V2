# Head scale geometry RCA — findings (instrumentation only)

## Question

Where does the swapped head first become larger than the original target head?

## Production path note

Affine / paste / seamless-clone are **N/A** on production `krea2_crop` + SPP.
Ground truth coords = `body_full`.

## Run analyzed

`results/_krea2_full_vs_localized/night_group_001` (same 3-person night scene
family as the bobblehead failure). Offline analysis of `localized/debug_*`
dumps + body.

Artifacts: `results/_head_scale_trace/night_group_offline/head_scale_trace/`

## FIRST_ENLARGEMENT_STAGE = S3_edited

| Stage | face_h (body) | ratio vs S0 | Notes |
|---|---|---|---|
| S0_original | 54 | 1.00 | target face on body |
| S1_crop | 54 | 1.00 | crop window preserves face size |
| S2_scene | 58 | 1.07 | within tolerance (detector noise) |
| **S3_edited** | **72** | **1.33** | **first exceedance** — local S3/S2 = **1.24** |
| S4_stitched | 98 | 1.81 | further growth after stitch/reveal |

User failure image (scaled to body): rightmost face ~1.62× original height;
rightmost/middle ~2.41 vs original ~1.20 — confirms final bobblehead.

## Answers to the investigation questions

1. **Final head size** = face painted inside the Krea2 (or mock) edited crop,
   then resized into the body crop box by stitch.
2. **Ground truth** = `body_full` pixels.
3. **Affine** = N/A on this path.
4. **Donor vs target scale** = identity `resize_contain` biases conditioning but
   does not set output size; output size appears at S3.
5. **Mask bounds** = stitch can *reveal* more of an already-large S3 head (S4
   ratio rises further) but is not the first stage.
6. **Seamless clone** = N/A on crop_stitch SPP.
7. **Krea2/edit enlargement** = **YES — first failure is S3_edited** (face
   taller inside the same crop canvas than the original scene face).

## Do not fix yet (per plan)

No scale clamps, identity changes, prompt tweaks, or mask edits until a
follow-up explicitly targets **S3_edited** (lock generated face height to
S2 scene face height — e.g. geometry clamp on edited crop before stitch,
or crop/conditioning that prevents free head expansion). Prefer fixing S3
before compensating at S4.
