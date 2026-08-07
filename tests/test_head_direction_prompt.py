"""Regression guard for the head-yaw/gaze-direction prompt clause.

Diffusion identity edits sometimes turn the generated head/eyes away from
the original photo's direction (no post-hoc geometric pose correction runs
in the production crop_stitch/full_frame paths -- see
Krea2IdentityEditPipeline._apply_expression_policy). The main lever against
that is prompt strength, so this locks in the explicit head-yaw/gaze clause
in the production config and confirms it survives preserve_expression=False
(it's a physical-geometry instruction, not an expression-lock clause, so it
must not be stripped by _apply_expression_policy).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

CFG_PATH = ROOT / "configs" / "krea2_identity_edit.yaml"


def test_production_prompt_locks_head_direction():
    cfg = yaml.safe_load(CFG_PATH.read_text())
    prompt = cfg["prompt"]
    assert "keep the head facing the exact same direction as the first image" in prompt
    assert "eyes must look in the exact same direction as the first image" in prompt


def test_head_direction_clause_survives_preserve_expression_false():
    pipe = Krea2IdentityEditPipeline.__new__(Krea2IdentityEditPipeline)
    cfg = yaml.safe_load(CFG_PATH.read_text())
    pipe.cfg = {
        "prompt": cfg["prompt"],
        "preserve_expression": False,
        "single_person_parity": True,
    }
    out = pipe._prompt_for_edit(use_tight=False, multi_person=False)
    # Expression-lock language is stripped...
    assert "copy the facial expression from the first image exactly" not in out.lower()
    # ...but the head-direction/geometry clause is a physical-pose instruction,
    # not an expression-lock clause, and must remain regardless.
    assert "keep the head facing the exact same direction as the first image" in out
