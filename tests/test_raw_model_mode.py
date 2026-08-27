"""Raw-model mode: ship the model's render, with no compositing at all.

Every post-stage creates a boundary between model-rendered pixels and original
pixels, and that boundary is visible exactly when the two sides differ -- which
is when the stage was thought necessary. Stacking them moved seams instead of
removing them, ending in a visible hard-edged patch across an athlete's arm.
The stages had never actually been switched off together, so their value was
assumed rather than measured. This mode measures it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_raw_model_flag_exists_and_defaults_off():
    assert 'self.cfg.get("simple_full_body_raw_model", False)' in KREA2


def test_raw_model_is_reachable_from_the_environment():
    """The caller may have no convenient place to set a cfg key."""
    assert 'os.environ.get("HEADSWAP_RAW_MODEL")' in KREA2


def test_raw_model_disables_every_compositing_stage():
    """Any stage left enabled reintroduces a boundary and defeats the mode."""
    for gate in (
        "if not _raw_model and selected_face is not None and bool(",
        "if not _raw_model and selected_face is not None and not _head_restored",
        'if not _raw_model and bool(self.cfg.get("simple_full_body_skin_harmonize"',
        "if _repaint_on and not _raw_model and selected_face is not None:",
    ):
        assert gate in KREA2, f"stage not gated by _raw_model: {gate}"


def test_raw_model_is_announced_in_the_log():
    assert "[krea2 raw_model]" in KREA2, (
        "a run with all post-processing off must say so, or its output is "
        "indistinguishable from a run where the stages silently no-op'd"
    )
