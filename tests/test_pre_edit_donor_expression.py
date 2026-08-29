"""Pre-edit the donor's expression to match the target's, before T4 runs.

CHECKPOINT-11/12/13 (docs/PIPELINE_STATE.md) measured that the donor's
expression survives T4's main pass regardless of prompt text, face_refine, or
ref_boost -- all three attempts edited the TARGET side or the conditioning
strength, and none of them moved it. This is the first attempt at the OTHER
side: editing the DONOR image itself (pure Krea2 self-edit -- scene=person=
the donor photo, no mask) before it ever reaches T4's existing two-step pass
(main pass + face_refine). Off by default, so T4's own render cannot change
unless explicitly enabled.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


class _Pipe(Krea2IdentityEditPipeline):
    def __init__(self, cfg):
        self.cfg = dict(cfg or {})
        self.cache_dir = ROOT / "results" / "_cache"


def test_disabled_by_default():
    assert 'self.cfg.get("pre_edit_donor_expression", False)' in KREA2


def test_disabled_returns_face_unchanged_and_touches_nothing():
    """With the flag off, rt/bundle/timings/edit_cache_info must never be
    used -- passing None for all of them must not raise, and the face object
    must come back untouched (same object), so the call site in
    run_simple_full_body is safe to make unconditionally."""
    pipe = _Pipe({})
    face = object()  # sentinel: must not be opened/copied when disabled
    out, diag = pipe._edit_donor_expression(None, None, face, None, {}, {}, 16)
    assert out is face
    assert diag == {"applied": False, "reason": "disabled"}


def test_measurement_is_forced_regardless_of_the_prompt_hint_flag():
    """_edit_donor_expression must not silently no-op just because the
    unrelated expression_prompt_hint flag (a different feature: appending the
    same measurement to T4's OWN prompt, rejected in CHECKPOINT-11) happens
    to be off. It must call _measure_expression_hint with force=True so the
    two features cannot couple."""
    assert "self._measure_expression_hint(body_full, force=True)" in KREA2


def test_measure_expression_hint_gate_is_bypassable_with_force():
    pipe = _Pipe({"expression_prompt_hint": False})
    _, diag_off = pipe._measure_expression_hint(None, None)
    assert diag_off["reason"] == "disabled"
    _, diag_forced = pipe._measure_expression_hint(None, None, force=True)
    assert diag_forced.get("reason") != "disabled"


def test_call_site_runs_before_crop_face_reference():
    """The edited donor image must flow into the SAME crop/prompt/sampling
    chain T4 already uses -- inserted before crop_face_reference, not as a
    parallel path, so run_simple_full_body's own logic needs no other
    change."""
    i_edit = KREA2.find("face, pre_edit_expr_diag = self._edit_donor_expression(")
    i_crop = KREA2.find("face_crop = crop_face_reference(")
    assert i_edit > 0 and i_crop > i_edit


def test_own_sampling_knobs_are_independent_of_t4s():
    """Must not silently inherit T4's ref_boost=5.5/denoise=0.85/cfg=1.8 --
    those are tuned for a scene != person head transplant, not a same-image
    expression nudge, and must be restorable independently of the main pass.

    Asserts the INTENT (each knob reads its own config key, with a default
    that differs from T4's) rather than pinning exact numbers. Pinning them
    is what broke this test on the first legitimate retune: these values are
    expected to move every GPU round, while "independent of T4" is the
    property that must actually hold.
    """
    t4_value = {
        "pre_edit_donor_expression_denoise": 0.85,
        "pre_edit_donor_expression_ref_boost": 5.5,
        "pre_edit_donor_expression_cfg": 1.8,
    }
    for key in (
        "pre_edit_donor_expression_denoise",
        "pre_edit_donor_expression_ref_boost",
        "pre_edit_donor_expression_cfg",
        "pre_edit_donor_expression_steps",
    ):
        m = re.search(rf'self\.cfg\.get\("{key}", ([0-9.]+)\)', KREA2)
        assert m, f"{key} must be read from cfg with its own default"
        if key in t4_value:
            assert float(m.group(1)) != t4_value[key], (
                f"{key} defaults to T4's own value ({t4_value[key]}); this "
                "pass must not silently inherit the head-transplant tuning"
            )


def test_identity_lora_is_disabled_for_this_pass_by_default():
    """The identity LoRA is trained to make image A's head look like image
    B's. With scene == person == the donor photo that collapses to
    "reproduce this exact head", which pins the output to the input and
    overpowers the prompt -- measured twice on GPU (no change at
    ref_boost=2.0; identity drift but still no expression change at 0.5).

    It is baked into the loaded weights, so it has to be dropped at LOAD
    time, not tuned away per call.
    """
    assert 'self.cfg.get("pre_edit_donor_expression_disable_lora", True)' in KREA2


def test_lora_free_bundle_does_not_disturb_t4s_own_bundle():
    """Dropping the LoRA must be scoped to this pass: identity_lora_strength
    is restored in a finally, so T4's main pass and face_refine still load
    (and cache-hit) their own LoRA-loaded bundle."""
    i_method = KREA2.find("def _edit_donor_expression(")
    i_next = KREA2.find("\n    def run_simple_full_body(")
    body = KREA2[i_method:i_next]
    i_zero = body.find('self.cfg["identity_lora_strength"] = 0.0')
    assert i_zero > 0, "the LoRA must be dropped by zeroing its strength"
    assert "_ls_before" in body[i_zero:], "the previous strength must be restored"
    assert 'self.cfg["identity_lora_strength"] = _ls_before' in body


def test_original_cfg_values_are_restored_after_the_self_edit():
    """The self-edit temporarily overwrites self.cfg['denoise'/'ref_boost'/
    'cfg'/'steps'] to sample with its own knobs; it must restore the previous
    values afterward (same pattern _two_pass_refine already uses), or T4's
    own main pass immediately after would run with the wrong settings."""
    i_method = KREA2.find("def _edit_donor_expression(")
    i_next = KREA2.find("\n    def run_simple_full_body(")
    body = KREA2[i_method:i_next]
    assert "old = {k: self.cfg.get(k) for k in" in body
    assert "finally:" in body
    assert "self.cfg[key] = val" in body


def test_reports_into_pipeline_meta():
    assert '"pre_edit_donor_expression": pre_edit_expr_diag,' in KREA2


def test_no_hint_available_returns_face_unchanged():
    """If the target's expression can't be measured (no face detected, no
    landmarks), the donor image must pass through unedited rather than
    guessing -- the same discipline _measure_expression_hint already applies
    to itself."""
    pipe = _Pipe({"pre_edit_donor_expression": True})
    face = object()
    out, diag = pipe._edit_donor_expression(None, None, face, None, {}, {}, 16)
    assert out is face
    assert diag["applied"] is False
