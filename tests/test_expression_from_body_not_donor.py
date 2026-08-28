"""Expression comes from the BODY photo; only identity comes from the donor.

The donor image is a portrait, so its expression, gaze and head angle ride
along with the identity unless the prompt says otherwise. Observed: a
sprinting athlete inherited the donor's studio smile, and subjects looked at
the camera instead of where the original subject was looking.

preserve_expression / _apply_expression_policy already exist for this, but
they only rewrite prompts built by _prompt_for_edit. run_simple_full_body --
the production route -- builds its own prompt and never calls them, so that
machinery was dead here. The clauses therefore live in the prompt itself.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()

# The whole module: these clauses appear only in the run_simple_full_body
# default prompt, so a file-wide search is unambiguous and cannot drift out of
# a fixed-size window when the prompt is edited.
PROMPT = KREA2


def test_identity_only_from_donor():
    assert "ONLY the identity from the second image" in PROMPT, (
        "the donor is not restricted to identity, so its portrait expression "
        "and gaze come along with the face"
    )


def test_expression_pinned_to_the_body_image():
    for clause in ("same mouth", "where they look", "same head angle"):
        assert clause in PROMPT, f"expression clause missing: {clause}"


def test_donor_expression_explicitly_refused():
    assert "Do not copy the expression, gaze or head angle of the second" in PROMPT


def test_clothing_removal_is_forbidden_outright():
    """A robed subject came back with a bare torso -- the garment was removed."""
    assert "Never remove, open" in PROMPT
    # Source-literal: the clause wraps across two lines in krea2.py, so match
    # only the part that stays on one line.
    assert "never expose a chest, torso or" in PROMPT
