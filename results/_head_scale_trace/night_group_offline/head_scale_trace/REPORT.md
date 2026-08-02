# Head scale geometry RCA

Path: `krea2_crop_spp`
Affine: N/A on krea2_crop production path (no landmark affine)

## FIRST_ENLARGEMENT_STAGE = `S3_edited`

Tolerance: face_h ratio > 1.12
S0 face_h_body: 54.0
Ratios vs S0: {
  "S1_crop": 1.0,
  "S2_scene": 1.0741,
  "S3_vs_S2_local": 1.2414,
  "S3_edited": 1.3333,
  "S4_stitched": 1.8148
}
S3 vs S2 (local crop): 1.2414

## Stages
- **S0_original**: face_h_body=54.0 face_h_local=54.0 face_h_frac=0.1776 head_h_body=None notes={'coord': 'body_full'}
- **S1_crop**: face_h_body=54.0 face_h_local=54.0 face_h_frac=0.3068 head_h_body=None notes={'coord': 'crop_native', 'crop_box_body': [221, 12, 381, 188], 'crop_native_size': [160, 176]}
- **S2_scene**: face_h_body=58.0 face_h_local=58.0 face_h_frac=0.3295 head_h_body=None notes={'coord': 'scene_inference', 'scene_size': [160, 176], 'scale_x': 1.0, 'scale_y': 1.0}
- **S3_edited**: face_h_body=72.0 face_h_local=72.0 face_h_frac=0.4091 head_h_body=None notes={'coord': 'edited_crop_after_krea2', 'edited_face_h_over_scene_face_h': 1.2414}
- **S4_stitched**: face_h_body=98.0 face_h_local=98.0 face_h_frac=0.3224 head_h_body=None notes={'coord': 'body_full_after_stitch', 'matched_to_selected': True, 'stitched_face_h_over_original': 1.8148}

## Interpretation rule
- If first is S1/S2: crop window / scene framing already wrong.
- If first is S3_edited: Krea2 (or mock) enlarged the head inside the crop.
- If first is S4_stitched: stitch mapping / mask reveal enlarged the head.
- Do not add correction factors elsewhere until this stage is fixed.
