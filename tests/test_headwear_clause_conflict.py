"""Removing headwear must not leave "keep the clothing" arguing against it.

Measured: with the removal clause appended (chars=943, confirmed in the run
log) a Nike cap still survived the swap. The base T4 sentence says "Keep the
clothing, pose, body shape, background and lighting exactly as they are" --
and a cap IS clothing. _append_headwear_policy's own default preserves
headwear on exactly that reasoning. At denoise=0.85 the sampler starts from
a source latent that already contains the hat, so a preserve instruction is
all the excuse it needs.

When removal is off, T4's prompt must stay byte-identical: CHECKPOINT-10
records that the 564-char text is part of the approved recipe and that
prompt length alone moves framing on this route.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_preservation_is_scoped_below_the_neck_when_removing():
    assert '"Keep the clothing below the neck, pose, body shape, "' in KREA2


def test_original_wording_is_kept_for_the_default_path():
    """T4's approved 564-char prompt must be unchanged when removal is off."""
    assert '"Keep the clothing, pose, body shape, background and "' in KREA2


def test_both_branches_hang_off_the_removal_flag():
    i = KREA2.find('"Keep the clothing below the neck')
    assert i > 0
    window = KREA2[max(0, i - 400):i + 600]
    assert 'simple_full_body_remove_headwear' in window
