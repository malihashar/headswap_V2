"""Generation-time garment containment: prevent clothing from being stripped
in the first place, instead of detecting and patching it afterward.

History (see test_restore_stripped_garment.py for the post-render patch it
supersedes for this specific defect): a post-render pixel-diff patch fixed
colour and coverage on GPU, but a residual sleeve/cuff GEOMETRIC misalignment
survived -- the render drew the arm at a different shape than the source
photo, and no pixel patch can correct geometry the sampler already produced.
Ali's direction: the fix has to be at generation/sampling time.

An earlier sampling-containment attempt on this exact route (CHECKPOINT-17,
docs/PIPELINE_STATE.md) was reverted for a seam. The postmortem measured the
cause as TEMPORAL, not spatial: ComfyUI's KSamplerX0Inpaint blends a
feathered mask's weight identically at every denoising step, so two regions
render under fixed, different conditions for the whole schedule and meet
"with nothing reconciling them" -- not a mistuned mask. DifferentialDiffusion
(vendored in this repo's ComfyUI, previously unused) makes the mask's
effective threshold move with the step's sigma instead, which is the
documented fix for exactly that mechanism.

Grep-on-source-text pattern, same as the other structural tests on this
branch: no isolated-exec, no GPU. These pin STRUCTURE, not pixel output --
that needs the GPU validation matrix in the call site's own comment.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()
SKIN_HARM = (ROOT / "src" / "headswap" / "skin_harmonize.py").read_text()


def _function_body(text: str, def_line: str) -> str:
    i = text.find(def_line)
    assert i > 0, f"could not find {def_line!r}"
    j = text.find("\ndef ", i + 10)
    return text[i: j if j > 0 else len(text)]


def test_flag_defaults_off():
    i = KREA2.find('"simple_full_body_garment_containment"')
    assert i > 0
    window = KREA2[i:i + 60]
    assert "False" in window


def test_mask_unions_head_and_skin_vetoes_clothes():
    body = _function_body(SKIN_HARM, "def build_garment_containment_mask(")
    assert "semantic_person_skin_mask" in body
    assert "semantic_clothes_mask" in body
    # Clothes must act as a veto (multiplied against 1 - clothes), not just
    # be unioned in -- a pixel the clothes classifier fires on must never
    # end up in the free region regardless of what the skin/head mask says.
    assert "1.0 - np.clip(clothes" in body


def test_mask_fails_closed():
    body = _function_body(SKIN_HARM, "def build_garment_containment_mask(")
    assert body.count("return None, info") >= 2


def test_not_the_head_only_ellipse_from_the_reverted_attempt():
    """The reverted CHECKPOINT-17 attempt protected everything except the
    head, which would also pin genuinely-bare skin (hands, forearms) to the
    ORIGINAL body's tone -- this function must NOT reduce to that shape by
    using only a head mask with no skin term."""
    body = _function_body(SKIN_HARM, "def build_garment_containment_mask(")
    assert "semantic_head_mask" not in body, (
        "expected semantic_person_skin_mask (head+skin union) rather than "
        "semantic_head_mask alone -- a head-only free region reintroduces "
        "the mismatched-body-tone problem the earlier attempt would have had"
    )


def test_differential_diffusion_wired_into_containment_branch():
    i = KREA2.find("if containment_on and rt.has(\"DifferentialDiffusion\")")
    assert i > 0
    # Must be positioned after the ModelSamplingFlux patch and before the
    # KSampler call, so it wraps whatever model object sampling actually
    # uses.
    i_flux = KREA2.find('"ModelSamplingFlux"')
    i_ksampler = KREA2.find('"KSampler"')
    assert 0 < i_flux < i < i_ksampler


def test_differential_diffusion_has_a_fallback_log_not_a_crash():
    i = KREA2.find("if containment_on and rt.has(\"DifferentialDiffusion\")")
    assert i > 0
    window = KREA2[i:i + 1200]
    assert "elif containment_on:" in window


def test_denoise_forced_to_one_and_restored():
    i = KREA2.find("_prev_denoise = self.cfg.get(\"denoise\")")
    assert i > 0
    window = KREA2[i:i + 2200]
    assert 'self.cfg["denoise"] = 1.0' in window
    assert 'self.cfg["denoise"] = _prev_denoise' in window
    assert "finally:" in window


def test_sampling_containment_flag_forced_and_restored():
    i = KREA2.find("_prev_sampling_containment = self.cfg.get(\"sampling_containment\")")
    assert i > 0
    window = KREA2[i:i + 2200]
    assert 'self.cfg["sampling_containment"] = True' in window
    assert 'self.cfg["sampling_containment"] = _prev_sampling_containment' in window


def test_edit_mask_actually_passed_at_the_call_site():
    """Unlike the always-None edit_mask on this route before this change --
    _sample_edit's containment_on check requires edit_mask is not None, and
    nothing on T4's call site ever supplied one."""
    assert "edit_mask=_containment_mask" in KREA2


def test_falls_through_unmasked_when_flag_off_or_mask_build_fails():
    i = KREA2.find("_containment_mask = None")
    assert i > 0
    window = KREA2[i:i + 2500]
    assert 'if _containment_mask is None:' in window or "mask build failed" in window
