"""Relative motion must be forced OFF for a single still driving image.

Two full GPU runs (30s+ each) loaded every LivePortrait model, reported
success, and returned a near-copy of the source, because
flag_relative_motion was True with a single driving IMAGE. Relative motion
transfers the delta between the driving frame and the driving sequence's own
reference frame; with one image that is the same frame, so the delta is
~zero and nothing transfers.

Both runs were caused by a stale Colab tab re-sending the old value, which
is exactly why this is enforced in repo code (synced by the setup cell)
rather than left to a notebook form field (cached in the browser tab).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.expression_transfer import (
    driving_is_still_image,
    resolve_relative_motion,
)


def test_still_image_is_detected():
    for name in ("body.png", "a.JPG", "x.jpeg", "y.webp"):
        assert driving_is_still_image(name), name


def test_video_is_not_a_still():
    for name in ("d0.mp4", "clip.MOV", "x.avi"):
        assert not driving_is_still_image(name), name


def test_relative_motion_forced_off_for_a_still_even_when_requested():
    """The caller asking for True must NOT win -- this is arithmetic, not a
    preference, and a stale caller kept asking for it."""
    value, reason = resolve_relative_motion("body.png", True)
    assert value is False
    assert "FORCED" in reason


def test_relative_motion_off_for_a_still_when_not_requested():
    value, _ = resolve_relative_motion("body.png", False)
    assert value is False
    value, _ = resolve_relative_motion("body.png", None)
    assert value is False


def test_video_defaults_to_relative_motion_on():
    value, _ = resolve_relative_motion("drive.mp4", None)
    assert value is True


def test_video_respects_an_explicit_caller_choice():
    """Only the still-image case is an override; a video caller keeps
    control, because relative motion is meaningful there."""
    assert resolve_relative_motion("drive.mp4", False)[0] is False
    assert resolve_relative_motion("drive.mp4", True)[0] is True


def test_reason_is_returned_for_logging():
    """A silent override would be its own trap -- the run log must say the
    value was changed and why."""
    for driving, requested in (("body.png", True), ("drive.mp4", None)):
        _, reason = resolve_relative_motion(driving, requested)
        assert isinstance(reason, str) and reason.strip()


def test_pipeline_is_cached_and_keyed_not_a_bare_singleton():
    """Rebuilding LivePortraitPipeline per call reloads five .pth models plus
    InsightFace and a landmark runner -- visible as the full "Load ... done."
    block in every run log.

    Keyed, because animation_region and flag_relative_motion live in
    InferenceConfig rather than the per-call args: a bare singleton would
    serve a pipeline configured for the PREVIOUS call's region, silently.
    """
    src = (ROOT / "src" / "headswap" / "expression_transfer.py").read_text()
    assert "_LP_PIPELINE" in src
    assert 'cache_key = (' in src
    assert 'str(animation_region)' in src, "region must be part of the key"
    assert '_LP_PIPELINE["key"] == cache_key' in src


def test_sys_modules_purge_only_on_cache_miss():
    """Purging LivePortrait's `src` modules on a cache HIT would re-import
    them underneath a live pipeline, leaving instances bound to stale class
    objects."""
    src = (ROOT / "src" / "headswap" / "expression_transfer.py").read_text()
    assert "_will_build = _LP_PIPELINE[\"pipeline\"] is None" in src
    assert "if _will_build:" in src


def test_output_detection_uses_mtime_not_a_set_difference():
    """LivePortrait overwrites the same filenames on every call.

    Detecting output as `after - before` therefore found nothing on the
    SECOND run: the paths already existed, so the diff was empty, primary
    came back None, and the caller silently fell back to its unmodified
    input -- reporting lp_applied=False while the log showed "Animated
    image: ...". A good result was discarded and the failure looked like
    LivePortrait not working.
    """
    src = (ROOT / "src" / "headswap" / "expression_transfer.py").read_text()
    assert "_t_start" in src
    assert "st_mtime >= _t_start" in src
    # Match the CODE, not prose: "after - before" also appears in the
    # comment explaining why it was removed, which is exactly the kind of
    # false positive that made an earlier gate-counting test report three
    # gates for two call sites.
    import re
    assert not re.search(r"^\s*produced\s*=\s*sorted\(after\s*-\s*before",
                         src, re.M), (
        "set-difference detection cannot see overwritten files"
    )


def test_missing_output_is_warned_not_silent():
    """The fallback is legitimate, but it must be visible: a silent fallback
    is what made a working LivePortrait pass look like a broken one."""
    src = (ROOT / "src" / "headswap" / "expression_transfer.py").read_text()
    assert "no output written under" in src
