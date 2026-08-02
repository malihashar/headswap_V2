# Conditioning parity decision gate

Can Krea2 produce single-person quality on a selected face from a multi-person image if conditioning after face selection is effectively identical to the single-person pipeline?

## Detection
- A (solo): 1 face(s), selected=[286, 118, 499, 436]
- B (multi): 3 faces, selected=[368, 142, 444, 331]

## Phase 1 — Conditioning parity
- parity_ok: **True** (recipe=True, geometry=True)
- failed: []
- neighbor_clamp_B: {'clamped': 0.0, 'neighbors_excluded': 0.0, 'neighbors_unexcludable': 0.0, 'crop_w_before': 224.0, 'crop_h_before': 432.0, 'crop_w_after': 224.0, 'crop_h_after': 432.0}
- note: Recipe: A_solo vs B exact keys. Geometry: A_ctrl (same multi body, selected-only faces) vs B (neighbors) — isolates neighbor exclusion.

### Recipe (A_solo vs B)
| metric | A_solo | B | ok |
|---|---|---|---|
| person_prep | 'resize_contain' | 'resize_contain' | True |
| crop_long_side | 768 | 768 | True |
| mask_top_ext | 1.55 | 1.55 | True |
| mask_side_ext | 0.6 | 0.6 | True |
| mask_bot_ext | 0.4 | 0.4 | True |
| mask_expand_px | 18 | 18 | True |
| prompt_sha1_12 | 'c046c89dfcc6' | 'c046c89dfcc6' | True |
| steps | 8 | 8 | True |
| cfg | 1.0 | 1.0 | True |
| ref_boost | 3.5 | 3.5 | True |
| ref_boost_a | 1.6 | 1.6 | True |
| grounding_px | 768 | 768 | True |
| fit_mode | 'fit' | 'fit' | True |
| use_tight | False | False | True |
| isolate_selected | False | False | True |
| single_person_parity | True | True | True |

### Geometry (A_ctrl vs B — same multi body)
| metric | A_ctrl | B | ok |
|---|---|---|---|
| person_prep | 'resize_contain' | 'resize_contain' | True |
| crop_long_side | 768 | 768 | True |
| mask_top_ext | 1.55 | 1.55 | True |
| mask_side_ext | 0.6 | 0.6 | True |
| mask_bot_ext | 0.4 | 0.4 | True |
| mask_expand_px | 18 | 18 | True |
| prompt_sha1_12 | 'c046c89dfcc6' | 'c046c89dfcc6' | True |
| steps | 8 | 8 | True |
| cfg | 1.0 | 1.0 | True |
| ref_boost | 3.5 | 3.5 | True |
| ref_boost_a | 1.6 | 1.6 | True |
| grounding_px | 768 | 768 | True |
| fit_mode | 'fit' | 'fit' | True |
| use_tight | False | False | True |
| isolate_selected | False | False | True |
| single_person_parity | True | True | True |
| face_area_frac_crop | 0.1484 | 0.1484 | True |
| face_height_frac_scene | 0.5687 | 0.5687 | True |
| person_nonblack_frac | 0.4165 | 0.4165 | True |
| head_to_canvas | 0.5687 | 0.5687 | True |
| scene_size | [384, 768] | [384, 768] | True |

## Align_paste control (A_ctrl vs C) — expected divergence
- person_prep: {'A_ctrl': 'resize_contain', 'C': 'identity_face_only_matte'}
- face_area_frac_crop: {'A_ctrl': 0.1484, 'C': 0.11}
- scene_size: {'A_ctrl': [384, 768], 'C': [240, 544]}

## Decision
- **YES_ARCHITECTURE**
- Conditioning parity holds. Quality metrics are mock-only and not decisive. Production multi must use select→single-person crop path so a real GPU gate can answer YES/NO on Krea2 capability. align_paste demoted to A/B control only.
- action: `simplify_to_krea2_crop_spp`

## Quality (Phase 2)
{
  "ran": true,
  "mock_only": true,
  "A": {
    "identity_cosine": 0.8588,
    "gaze_eye_line_delta_deg": 1.729,
    "body_eye_line_deg": -2.119,
    "result_eye_line_deg": -0.39,
    "head_height_ratio_body": 0.3105,
    "neighbor_outside_selected_psnr": 15.283,
    "edit_mode": null,
    "scene_size": null,
    "mock": true
  },
  "B": {
    "identity_cosine": 0.9415,
    "gaze_eye_line_delta_deg": 1.251,
    "body_eye_line_deg": 1.791,
    "result_eye_line_deg": 0.54,
    "head_height_ratio_body": 0.2953,
    "neighbor_outside_selected_psnr": 18.962,
    "edit_mode": null,
    "scene_size": null,
    "mock": true
  }
}
