"""Carrier 3 -- Krea2EditGroundedEncode -- and the skin log that lied.

The scene reaches the sampler three ways. ref_boost_a scales one and was
swept to 0.3 (default 1.6) with the cap unmoved and the polo wrecked;
denoise governs the second and cannot be lowered without losing clothing;
CHECKPOINT-17 measured a noise_mask working and reverted it anyway because
the pin boundary seams. That leaves grounding, which decides what the model
believes it is looking at before the prompt is read, and which nothing tried
so far scales.

These tests verify the WIRING and the honesty of the instruments. They
cannot verify model behaviour -- that needs the GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()
CHAIN = (ROOT / "src" / "headswap" / "chain.py").read_text()


def test_effective_grounding_px_is_surfaced_on_the_route_meta():
    """A sweep must read the value the SAMPLER used, not the one it asked
    for. An earlier arm returned byte-identical images while the log showed
    the setting changing, because the setting never reached the node."""
    assert '"grounding_px": sample_meta.get("grounding_px"),' in KREA2


def test_sweep_reads_the_effective_value_not_its_own_request():
    assert 'eff = (res.meta or {}).get("grounding_px")' in CHAIN
    assert '"effective_px": eff' in CHAIN


def test_sweep_sets_both_grounding_keys():
    """full_frame_grounding_px OVERRIDES grounding_px when the route resolves
    to full_frame (krea2.py:2588-2590). Setting only one would sweep nothing
    on that route while still printing a moving number."""
    assert 'keys = ("grounding_px", "full_frame_grounding_px")' in CHAIN


def test_sweep_restores_every_key_it_touched():
    src = CHAIN[CHAIN.index("def sweep_grounding("):CHAIN.index("def _strip(")]
    assert "finally:" in src
    assert 'prev = {k: pipe.cfg.get(k) for k in (*keys, "seed")}' in src


def test_sweep_does_not_reintroduce_a_mask():
    """CHECKPOINT-17 closed masks on this route on a measurement, not a
    preference: noise_mask worked (inside=35.40, outside=0.91) and still
    seamed at the pin boundary."""
    src = CHAIN[CHAIN.index("def sweep_grounding("):CHAIN.index("def _strip(")]
    for banned in ("edit_mask", "sampling_containment", "noise_mask",
                   "build_head_hair_mask", "erase_headwear"):
        assert banned not in src, banned


def test_raw_model_does_not_print_the_composited_skin_log():
    """Under raw_model the restore block is gated off, so _head_restored is
    False for a reason that is not a failure. The else-branch was printing
    "the ORIGINAL body was restored and the model's skin was discarded" --
    on a route where no restore ran, nothing was discarded and the wash is
    gated off -- directly under the raw_model line saying the opposite."""
    i = KREA2.index("_wash_default = not _head_restored")
    j = KREA2.index("simple_full_body_skin_harmonize", i)
    block = KREA2[i:j]
    assert "if _raw_model:" in block, "raw_model must be handled first"
    assert "elif _head_restored and _wash_default:" in block, (
        "the composited branches must be reachable only when NOT raw_model"
    )
    # The false sentence must not be reachable on the raw route.
    assert block.index("if _raw_model:") < block.index(
        "person-skin restore did not run"
    )
