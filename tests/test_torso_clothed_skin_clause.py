"""Second signal for the skip-skin-clause decision: torso garment
classification, not whole-silhouette skin colour.

_measure_visible_skin measures "what fraction of ALL person-pixels below the
face are skin-coloured" -- and praying hands held at chest height inflate
that fraction past the covered threshold even when the torso underneath is
fully robed. Measured on the real failing case: 13.7% (> 6% cutoff) on a
subject wearing a full desert robe with only hands and face bare, so the
skin-recolour clause was kept and the render came back shirtless.

_measure_torso_clothed asks a narrower question over a shoulder-width band
-- wide enough that a hand-sized blob cannot dominate it -- using the
mediapipe CLOTHES class rather than colour. It is wired in as an additive
OR: it can only turn "keep" into "drop", never the reverse, so it cannot
regress any case the original measurement already handles correctly.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_torso_measurement_is_gated_the_same_as_the_skin_measurement():
    i = KREA2.find('if bool(self.cfg.get("skip_skin_clause_when_covered", False)):')
    assert i > 0
    block = KREA2[i:i + 1200]
    assert "_measure_torso_clothed" in block, (
        "the torso measurement must run inside the same "
        "skip_skin_clause_when_covered gate as the skin measurement"
    )


def test_decision_is_an_additive_or_not_a_replacement():
    i = KREA2.find("_skin_says_covered or _torso_says_covered")
    assert i > 0, (
        "the drop decision must OR the two signals -- a replacement would "
        "risk regressing cases the original measurement already gets right"
    )


def test_torso_fails_closed_to_not_covered():
    """Any error/unmeasurable case must not force-drop the clause."""
    i = KREA2.find("def _measure_torso_clothed")
    body = KREA2[i:i + 4500]
    # The only True return must be the real measurement's comparison.
    assert body.count("return True, diag") == 0
    assert "return clothes_frac >= thresh, diag" in body
    # Every early return (face not found, degenerate box, no segmenter, too
    # little person, exception) must return None, which the caller treats
    # as bool(None) == False -- never covered by default.
    early_returns = [
        line for line in body.split("\n")
        if "return None, diag" in line or "return False, diag" in line
    ]
    assert len(early_returns) >= 4, (
        "expected multiple fail-safe early returns before the real "
        "measurement; found fewer than expected -- verify no path was "
        "changed to fail open"
    )


def test_torso_box_is_shoulder_width_not_face_width():
    """A face-width box would put the hands at the box's center of mass
    too, reproducing the exact confound this exists to avoid."""
    i = KREA2.find("def _measure_torso_clothed")
    body = KREA2[i:i + 3000]
    assert "0.6 * fw" in body, (
        "the torso box side margin looks narrower than intended -- a "
        "face-width box does not dilute a hand-sized skin region enough"
    )


def test_uses_the_real_clothes_classifier_not_colour():
    i = KREA2.find("def _measure_torso_clothed")
    body = KREA2[i:i + 3000]
    assert "semantic_clothes_mask" in body
    assert "semantic_person_mask" in body


def test_log_line_names_which_signal_fired():
    """DROPPED must say WHY -- skin fraction, torso fraction, or both --
    so a future investigation can tell the two mechanisms apart from the
    log alone, the same discipline every other measurement in this file
    already follows."""
    i = KREA2.find('print(f"[krea2 skin_clause] DROPPED')
    assert i > 0
    seg = KREA2[max(0, i - 400):i]
    assert "_why" in seg
