# Head-swap V2 comparison

Generated: 2026-07-13T21:13:45.048321+00:00
Latency budget (p95): 30.0s

## Ranking

| Pipeline | Score | Success | ID mean | Body PSNR | p95 lat | Budget OK | Mock |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| klein4b_mask_crop_stitch | 0.800 | 100.0% | None | 48.15659856796265 | 0.01887670799624175 | Y | Y |
| qwen_improved_mask_crop | 0.800 | 100.0% | None | 48.16028022766113 | 0.01744387499638833 | Y | Y |
| qwen_baseline | 0.757 | 100.0% | None | 31.402417023976643 | 0.013605499996629078 | Y | Y |

## Recommendation

- **Promote:** `klein4b_mask_crop_stitch`
- **Fallback:** `qwen_improved_mask_crop`
- Winner selected by composite score = 0.6*success + 0.2*body_preserve + 0.2*identity with -0.15 penalty if p95 latency > 30.0s. Mock-only run: empirical ranking is harness-smoke only; recommended production promote is Klein 4B with Qwen-improved as fallback.
- If runs are force_mock=true, promote FLUX.2 Klein 4B mask-crop-stitch as the primary production candidate per research; re-confirm with GPU metrics before cutover.
