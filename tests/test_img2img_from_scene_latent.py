"""Sampling must start from the SOURCE latent, not pure noise.

Every render used EmptySD3LatentImage, so the model regenerated the entire
frame from noise with nothing anchoring the original photo. Measured across an
8-pair sweep: a black robe came back as a bare torso, a lace dress as a white
top plus a patterned skirt, a silver chain appeared on an athlete wearing
none, and the crown of the head was cropped off. No prompt wording can prevent
that -- "keep the clothing exactly as it is" cannot constrain a sampler that
never saw the clothing as a starting point, which is why several rounds of
prompt rewriting changed nothing.

Seeding from the scene latent with denoise < 1 is neither masking nor
compositing: the whole frame is preserved to the same degree, so there is no
boundary anywhere and no seam can appear.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_img2img_branch_exists_before_the_empty_latent_branch():
    i_img = KREA2.find("elif denoise < 0.999 and scene_lat is not None:")
    i_empty = KREA2.find('elif rt.has("EmptySD3LatentImage"):')
    assert i_img > 0, "no img2img branch; sampling always starts from noise"
    assert i_img < i_empty, (
        "the empty-noise branch is reached first, so denoise < 1 would "
        "under-denoise noise instead of preserving the source"
    )


def test_img2img_uses_no_noise_mask():
    """A noise_mask would reintroduce a boundary, which is the seam problem."""
    seg = KREA2[KREA2.find("elif denoise < 0.999"):KREA2.find('elif rt.has("EmptySD3LatentImage")')]
    assert "noise_mask" not in seg, (
        "img2img must preserve the frame globally; a mask would create an "
        "edge and reintroduce the composite seam it exists to avoid"
    )


def test_production_denoise_is_below_one():
    from headswap.config import load_config

    cfg = load_config(ROOT / "configs" / "krea2_identity_edit.yaml")
    assert 0.0 < cfg["denoise"] < 1.0, (
        "denoise 1.0 makes the img2img branch unreachable and restores "
        "full-frame regeneration"
    )


def test_ref_boost_is_not_the_rejected_value():
    """CHECKPOINT-06: 7.0 rejected (overcooks face); 4.0 defensible."""
    from headswap.config import load_config

    cfg = load_config(ROOT / "configs" / "krea2_identity_edit.yaml")
    # 5.5 is the adopted T4 value; 7.0 is the one CHECKPOINT-06 rejected.
    assert cfg["ref_boost"] <= 6.0, (
        f"ref_boost {cfg['ref_boost']} is at or near the value "
        "docs/PIPELINE_STATE.md CHECKPOINT-06 rejected for synthetic skin"
    )
