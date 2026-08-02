# Decision gate: Krea2 multi via single-person conditioning

## Question

Can Krea2 produce single-person quality on a selected face from a multi-person
image if the conditioning after face selection is effectively identical to the
single-person pipeline?

## Phase 1 result — Conditioning parity

**PASS**

- Recipe (A_solo vs B): exact match on `person_prep`, mask extents, prompt SHA,
  sampler knobs, `use_tight=false`, `isolate_selected=false`, SPP on.
- Geometry (A_ctrl vs B on same multi body): face fill, head-to-canvas, scene
  size, person nonblack frac identical (neighbor clamp did not alter the crop
  when neighbors were already outside the selected-head window; when clamp
  fires, protect extents = mask recipe so face fill stays within tolerance).
- Align_paste control (C): diverges by construction (`identity_face_only_matte`,
  face-only refine mask, different scene size).

Artifacts: `results/_conditioning_parity/REPORT.json`

## Phase 2 result — Quality

Local run used `--force-mock` (no NVIDIA GPU). Mock identity/gaze numbers are
**not decisive** for Krea2 capability.

Re-run on Colab/GPU:

```bash
PYTHONPATH=src python scripts/compare_single_vs_multi.py \
  --solo-body <solo.png> --multi-body <group.png> --face <id.png> \
  --out results/_conditioning_parity_gpu --run-krea2
```

## Architectural decision

**YES_ARCHITECTURE** (production simplified now)

```text
detect faces → select face → single-person pipeline → stitch
```

- Default `multi_person_swap_mode: krea2_crop` + `single_person_parity: true`
- `align_paste` demoted to deprecated A/B control only
- Allowed multi-only logic: face select + neighbor exclusion from crop

If a real GPU quality gate later shows identity/gaze still far worse than solo
**under this locked parity**, flip to **NO**: stop Krea2 multi architecture
investment and evaluate alternative identity-edit models — do not add more
preprocessing heuristics.
