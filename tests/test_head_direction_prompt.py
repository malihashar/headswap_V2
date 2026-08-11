"""Regression guard for head-yaw/gaze prompt policy.

Yaml still contains an explicit head-yaw/gaze lock clause (useful when
``preserve_expression`` is true). When preserve_expression is false
(production default), ``_apply_expression_policy`` must strip that lock and
prefer natural donor gaze so sideways body look is not forced onto the donor.
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
    assert cfg.get("preserve_expression") is False


def test_head_direction_clause_stripped_when_preserve_expression_false():
    pipe = Krea2IdentityEditPipeline.__new__(Krea2IdentityEditPipeline)
    cfg = yaml.safe_load(CFG_PATH.read_text())
    pipe.cfg = {
        "prompt": cfg["prompt"],
        "preserve_expression": False,
        "single_person_parity": True,
    }
    out = pipe._prompt_for_edit(use_tight=False, multi_person=False)
    assert "copy the facial expression from the first image exactly" not in out.lower()
    assert "keep the head facing the exact same direction as the first image" not in out
    assert "eyes must look in the exact same direction as the first image" not in out
    assert "Allow the facial expression from the second image" in out
    assert "natural donor gaze" in out.lower() or "toward the camera" in out.lower()


def test_head_direction_clause_kept_when_preserve_expression_true():
    pipe = Krea2IdentityEditPipeline.__new__(Krea2IdentityEditPipeline)
    cfg = yaml.safe_load(CFG_PATH.read_text())
    pipe.cfg = {
        "prompt": cfg["prompt"],
        "preserve_expression": True,
        "single_person_parity": True,
    }
    out = pipe._prompt_for_edit(use_tight=False, multi_person=False)
    assert "keep the head facing the exact same direction as the first image" in out
