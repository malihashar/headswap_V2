"""Expression comes from the BODY photo; only identity comes from the donor.

The donor image is a portrait, so its expression, gaze and head angle ride
along with the identity unless the prompt says otherwise. Observed twice: a
sprinting athlete inherited the donor's open-mouth studio grin, and subjects
looked at the camera instead of where the original subject was looking.

The first attempt added "the expression stays exactly as it is in the first
image" as a LATER clause while the prompt still opened with "replace the head
from the first image with the head from the second image ... exactly as they
appear". Copying a portrait's head pixels IS copying its smile, so the two
clauses contradicted each other and the earlier, stronger one won -- the grin
came back. The fix was to remove the transplant framing entirely: the prompt
now asks to change WHO the person looks like, not to paste a head.

preserve_expression / _apply_expression_policy already exist for this, but
they only rewrite prompts built by _prompt_for_edit. run_simple_full_body --
the production route -- builds its own prompt and never calls them, so that
machinery is dead here. The clauses therefore live in the prompt itself.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()

# The whole module MINUS its comments: these clauses appear only in the
# run_simple_full_body default prompt, so a file-wide search is unambiguous and
# cannot drift out of a fixed-size window when the prompt is edited. Comments
# are stripped because the code comments quote the old, banned wording verbatim
# in order to explain why it was removed -- a plain file-wide search would then
# find the banned string in the very comment warning against it.
PROMPT = "\n".join(
    line for line in KREA2.splitlines() if not line.lstrip().startswith("#")
)


def test_framed_as_identity_change_not_head_transplant():
    assert "Change identity only." in PROMPT
    assert "Change who the person in the first image " in PROMPT


def test_no_instruction_to_copy_the_donor_head_verbatim():
    """The regression that reintroduced the donor's smile."""
    for banned in (
        "with the head from the second image",
        "exactly as they appear in the ",
    ):
        assert banned not in PROMPT, (
            f"{banned!r} tells the model to copy the donor's rendered head, "
            "which copies its expression, gaze and head angle too"
        )


def test_identity_only_from_donor():
    assert "take only identity" in PROMPT, (
        "the donor is not restricted to identity, so its portrait expression "
        "and gaze come along with the face"
    )


def test_performance_enumerated_and_pinned_to_the_body_image():
    """Named individually: each of these failed separately in practice."""
    for clause in (
        "absolute priority",
        "looking direction",
        "pupil position",
        "eyelid openness",
        "eyebrow position",
        "mouth shape and lip pose",
        "head pitch, yaw and roll",
    ):
        assert clause in PROMPT, f"performance clause missing: {clause}"


def test_donor_expression_explicitly_refused():
    assert "Do not copy the second image's expression, gaze, pupil position" in PROMPT


def test_skin_tone_change_survived_the_rewrite():
    assert "Change the skin colour of the body" in PROMPT


def test_clothing_removal_is_forbidden_outright():
    """A robed subject came back with a bare torso -- the garment was removed."""
    assert "Never remove, open" in PROMPT
    # Source-literal: the clause wraps across two lines in krea2.py, so match
    # only the part that stays on one line.
    assert "never expose a chest, torso or" in PROMPT
