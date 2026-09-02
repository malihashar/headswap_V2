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


def _body(name: str) -> str:
    """A function's source with its DOCSTRING removed.

    Necessary, not fussy: these functions document the mechanisms they must
    not use, so a plain substring search for "noise_mask" matches the
    paragraph explaining why noise_mask was abandoned. That is the same
    false-positive that once made a gate-counting test report three gates for
    two call sites.
    """
    import ast

    tree = ast.parse(CHAIN)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == name)
    stmts = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
    lines = CHAIN.splitlines()
    return "\n".join(lines[stmts[0].lineno - 1:fn.end_lineno])


BANNED = ("edit_mask", "sampling_containment", "noise_mask",
          "build_head_hair_mask", "erase_headwear")


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
    src = _body("sweep_grounding")
    assert "finally:" in src
    assert 'prev = {k: pipe.cfg.get(k) for k in (*keys, "seed")}' in src


def test_sweep_does_not_reintroduce_a_mask():
    """CHECKPOINT-17 closed masks on this route on a measurement, not a
    preference: noise_mask worked (inside=35.40, outside=0.91) and still
    seamed at the pin boundary."""
    src = _body("sweep_grounding")
    for banned in BANNED:
        assert banned not in src, banned


def test_raw_model_does_not_print_the_composited_skin_log():
    """Under raw_model the restore block is gated off, so _head_restored is
    False for a reason that is not a failure. The else-branch was printing
    "the ORIGINAL body was restored and the model's skin was discarded" --
    on a route where no restore ran, nothing was discarded and the wash is
    gated off -- directly under the raw_model line saying the opposite."""
    i = KREA2.index("_wash_default = not _head_restored")
    j = KREA2.index("_wash_will_run = not _raw_model and bool(", i)
    block = KREA2[i:j]
    assert "if _raw_model:" in block, "raw_model must be handled first"

    # Past this point the branches must check the REAL outcome
    # (_wash_will_run), not just the computed intent (_wash_default) --
    # printing intent alone lies whenever a caller explicitly overrides
    # the skin_harmonize cfg key. GPU-confirmed: this exact log claimed
    # "LAB wash ON" on a render where the wash had been explicitly
    # disabled and never ran.
    j2 = KREA2.index("simple_full_body_restore_stripped_garment", j)
    branches = KREA2[j:j2]
    assert "if _head_restored and _wash_will_run:" in branches
    assert "elif _wash_will_run:" in branches
    # The false sentence must not be reachable on the raw route.
    full = KREA2[i:j2]
    assert full.index("if _raw_model:") < full.index(
        "person-skin restore did not run"
    )


# --- After carrier 3: guidance ---------------------------------------------
#
# GPU result, hat pair, effective_px confirmed moving 768/512/384/256: the cap
# is unchanged on all four arms. That eliminates the last scene-conditioning
# path, so nothing structurally pins the cap and the model is simply not
# following the sentence. That is guidance, which has never been swept here.


def _guidance_src() -> str:
    return _body("sweep_guidance")


def test_guidance_sweep_reads_effective_values():
    """Every prior sweep that mattered was read off the requested value at
    least once. cfg and ref_boost must come back from the sampler."""
    assert '"cfg": sample_meta.get("cfg"),' in KREA2
    assert '"effective_cfg": meta.get("cfg")' in _guidance_src()
    assert '"effective_ref_boost": meta.get("ref_boost")' in _guidance_src()


def test_guidance_sweep_does_not_touch_lora_strength():
    """identity_lora_strength is part of the model cache key (krea2.py:550)
    and rt.models never evicts, so each swept value would add a full ~28GB
    bundle -- the OOM that already happened once this project."""
    src = _guidance_src()
    assert "identity_lora_strength" not in src
    assert 'prev = {k: pipe.cfg.get(k) for k in ("cfg", "ref_boost", "seed")}' in src


def test_guidance_sweep_restores_and_uses_no_mask():
    src = _guidance_src()
    assert "finally:" in src
    for banned in BANNED:
        assert banned not in src, banned


def test_negative_prompt_is_logged_every_render():
    """run_simple_full_body writes the headwear negative into self.cfg
    permanently, and its own log is gated on the value still being empty --
    so it prints on the first render of a kernel and never again while
    staying active. Four grounding arms were read without it being visible
    that a headwear negative was in play throughout."""
    assert '[krea2 negative]' in KREA2
    i = KREA2.index('neg = str(self.cfg.get("negative_prompt", "") or "")')
    assert '[krea2 negative]' in KREA2[i:i + 900], (
        "must log at sample time, where the value is actually consumed"
    )


def test_negative_prompt_log_says_when_cfg_disables_it():
    """At cfg<=1.0 classifier-free guidance is off and the negative text does
    nothing at all -- a non-empty negative in the log would otherwise read as
    an active mechanism."""
    assert "classifier-free guidance is OFF" in KREA2


def test_effective_negative_is_on_the_route_meta():
    assert '"negative_prompt": neg,' in KREA2
    assert '"negative_prompt": sample_meta.get("negative_prompt"),' in KREA2
