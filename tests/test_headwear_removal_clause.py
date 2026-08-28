"""Explicit headwear-removal clause for run_simple_full_body. Off by default.

_append_headwear_policy already documents the underlying bug: a hat on the
ORIGINAL person survived a swap as a translucent oval with "wings" at ear
level, because "replace the head... with none of the first person's head
remaining" never named headwear specifically -- a hat is not obviously part
of "the head" to the model. That helper is only wired into the
_prompt_for_edit/crop_stitch route; run_simple_full_body builds its own
prompt and never calls it, same dead-code pattern already found this session
for the hair-force and expression clauses.

This wires the REMOVAL half (Ali asked that an original hat be replaced by
the donor's hair, not preserved) directly into T4's prompt, gated off by
default per this file's own rule: every prompt addition must be A/B'd on
face_fraction before landing.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_off_by_default():
    assert 'self.cfg.get("simple_full_body_remove_headwear", False)' in KREA2


def test_disabled_when_prompt_is_overridden():
    """A caller-supplied prompt must win outright, same rule as the inline
    expression hint -- this must not silently append onto an override."""
    i = KREA2.find('if not _prompt_override and bool(')
    j = KREA2.find('self.cfg.get("simple_full_body_remove_headwear"')
    assert i > 0 and 0 < j - i < 80


def test_clause_targets_removal_not_preservation():
    # Scoped to just the appended clause block, not the surrounding comment
    # that legitimately discusses the dead helper's PRESERVE branch by name.
    i = KREA2.find('CRITICAL: if the person in the first image is wearing a')
    j = KREA2.find('[krea2 headwear] remove-and-replace', i)
    block = KREA2[i:j]
    assert i > 0
    assert "REMOVE it" in block
    assert "completely and replace it with the second person's hair" in block
    # The dead helper's PRESERVE branch text is a different clause entirely
    # ("hair MUST be transferred, not hidden") and must not be what landed.
    assert "not hidden" not in block


def test_applied_after_the_base_t4_sentence_not_inside_it():
    """Appended, not spliced -- keeps the base T4 sentence byte-identical
    when this flag is off, same discipline as the inline expression hint."""
    i_base_end = KREA2.find("Do not turn any clothed area into skin.")
    i_flag = KREA2.find('simple_full_body_remove_headwear')
    assert 0 < i_base_end < i_flag
