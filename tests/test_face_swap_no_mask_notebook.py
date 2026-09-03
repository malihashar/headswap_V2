"""The face-swap notebook must take the genuinely mask-free route.

Ali's requirement was explicit and repeated: no masks, just Krea generation.
Two earlier attempts failed it, both silently:

  1. `crop_stitch` -- regenerates a small face crop and pastes it back
     through a soft mask. The stitch boundary cutting across headwear is
     what produced the ghosted/haloed hat.
  2. Widening that stitch mask -- the cfg overrides were written into the
     notebook but the pipeline IGNORED them. GPU log showed
     `mask_params={'top_ext': 1.55, 'side_ext': 0.6, 'expand_px': 6}`
     against requested 2.6/1.2/36. Tuning a mask that never applied looked
     like "the fix didn't help" rather than "the fix never ran".

`full_frame` is not mask-free either -- it builds a freeze mask and does
LAB-match + feather compositing.

The only route in this codebase with zero post-generation compositing is
`run_simple_full_body()` with `raw_model=True` AND `face_refine` off. The
`raw_model` flag alone is not sufficient: it gates body_restore, the LAB
skin wash and skin_repaint, but the face_refine pass is gated separately
and calls `feathered_soft_composite` -- a mask.

These tests read the notebook JSON and the pipeline source; no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "face_swap_no_mask.ipynb"
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def _run_cell_source() -> str:
    nb = json.loads(NB_PATH.read_text())
    code_cells = [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    ]
    run = [s for s in code_cells if "Krea2IdentityEditPipeline(" in s]
    assert len(run) == 1, "expected exactly one cell that builds the pipeline"
    return run[0]


def test_notebook_exists_and_parses():
    nb = json.loads(NB_PATH.read_text())
    assert nb["cells"], "notebook has no cells"


def test_forces_the_mask_free_route():
    """simple_full_body is only reachable with the body route ENABLED. An
    earlier notebook set enable_body_route=False, which forced the masking
    crop_stitch path -- the exact opposite of the requirement."""
    src = _run_cell_source()
    assert '"enable_body_route": True' in src
    assert '"simple_full_body_prompt": FACE_SWAP_PROMPT' in src


def test_bust_shots_also_take_the_simple_route():
    """Without lowering this floor, a portrait/bust crop falls through to
    crop_stitch (which masks) instead of the single-pass route."""
    src = _run_cell_source()
    assert '"simple_path_below_face_frac_min"' in src
    assert 'self.cfg.get("simple_path_below_face_frac_min", 0.38)' in KREA2, (
        "the pipeline key this notebook overrides has been renamed or "
        "removed -- the override would now be silently ignored, which is "
        "exactly the failure mode this file exists to prevent"
    )


def test_body_restore_constrains_what_may_change():
    """raw_model must be OFF so body_restore runs.

    With raw_model=True the render ships raw, and a full-frame pass at
    denoise=0.85 is free to redraw every pixel. Measured on GPU: a black
    robe came back tan and the hat was reshaped, with the prompt asking
    for neither. A prompt biases that; it cannot guarantee it.

    body_restore is the control. Per its own implementation note it keeps
    the generated HEAD and the generated SKIN -- so face identity and donor
    skin tone both survive -- while clothes and background "come back from
    the original verbatim". That is exactly "change only what we want to
    change", and it is why the route stays full-frame rather than becoming
    a face-only crop: bare skin has to take the donor's tone, which a face
    crop can never do because the model never sees the arms.
    """
    src = _run_cell_source()
    assert '"simple_full_body_raw_model": False' in src, (
        "raw_model must be OFF -- with it on, clothing/background drift is "
        "unconstrained and was measured changing colour"
    )
    assert '"simple_full_body_restore_body": True' in src


def test_face_refine_left_enabled():
    """An earlier version set simple_full_body_face_refine=False, reading
    "no masks" as maximally as possible. Head-swap production does NOT
    disable it -- chain.py's skip_refine=True sets refine_max_face_frac
    =0.25, i.e. "refine when the face is under 25% of frame". On a
    full-body shot the face is ~8% of frame (~84px at 1024 output), so
    production refines, and that pass is what carries identity. Turning it
    off left identity 84px to survive in, and it did not."""
    src = _run_cell_source()
    assert '"simple_full_body_face_refine": False' not in src
    assert '"simple_full_body_refine_max_face_frac": 0.25' in src


def test_prompt_carries_the_skin_tone_clause():
    """A face-only instruction leaves a donor-toned face on target-toned
    arms and hands. T4 carries a skin clause for this reason; a face swap
    needs the same."""
    p = _face_swap_prompt()
    assert "skin colour" in p
    assert "already bare" in p


def test_sampling_recipe_is_not_casually_overridden():
    """denoise / cfg / seed / output resolution are a tuned recipe shared
    with head-swap and must not be forked without measured reason.

    ref_boost IS forked here (5.5 -> 7.0), deliberately: a head swap also
    transplants the donor's hair, which carries much of perceived identity,
    while this route keeps the target's own hair -- so the face is the only
    channel the donor arrives through, and needs weighting up. 7.0 is the
    config's own documented identity-fidelity value, not invented.
    """
    src = _run_cell_source()
    for key in ('"denoise"', '"cfg"', '"body_min_long_side"'):
        assert key not in src, (
            f"{key} is overridden here -- it should stay on the shared "
            "head-swap recipe unless there is measured reason to fork it"
        )


def test_no_skin_painting_stages():
    """body_restore already keeps the model's OWN rendered skin, so the LAB
    wash has nothing to fix -- but it defaults to auto-on, and when
    mediapipe pose fails it falls back to a geometric limb mask (lines
    drawn between joints). GPU-measured: that painted a hard-edged tan blob
    across praying hands, not following the hand silhouette at all.
    skin_repaint is pinned too so enabling body_restore cannot admit it."""
    src = _run_cell_source()
    assert '"simple_full_body_skin_harmonize": False' in src
    assert '"simple_full_body_skin_repaint": False' in src


def test_the_two_flags_this_notebook_relies_on_still_gate_what_it_thinks():
    """Pins the pipeline side of the contract. If either gate is renamed or
    its default flips, the notebook's override becomes a no-op and the
    'mask-free' claim quietly stops being true."""
    assert 'self.cfg.get("simple_full_body_raw_model", True)' in KREA2
    assert 'self.cfg.get("simple_full_body_face_refine", True)' in KREA2


def test_face_refine_is_the_only_composite_in_the_simple_route():
    """If a NEW compositing stage is ever added to run_simple_full_body,
    this test fails and whoever added it has to decide whether the
    notebook needs another flag turned off."""
    i = KREA2.find("def run_simple_full_body")
    assert i > 0
    j = KREA2.find("\n    def ", i + 10)
    body = KREA2[i:j if j > 0 else len(KREA2)]
    composite_calls = [
        line.strip()
        for line in body.splitlines()
        if "feathered_soft_composite(" in line
        or "soft_composite(" in line
        or "alpha_composite(" in line
    ]
    # Exactly one call site, and it lives in the face_refine block.
    assert len(composite_calls) == 1, (
        f"expected exactly 1 compositing call in run_simple_full_body, "
        f"found {len(composite_calls)}: {composite_calls}"
    )


def _face_swap_prompt() -> str:
    src = _run_cell_source()
    ns: dict = {}
    i = src.find("FACE_SWAP_PROMPT = (")
    j = src.find("\n)\n", i)
    assert 0 < i < j
    exec(src[i:j + 2], {}, ns)  # noqa: S102
    return ns["FACE_SWAP_PROMPT"]


def test_prompt_leads_with_replacement_not_preservation():
    """CHECKPOINT-10 measured that a clause buried after prohibitions is
    simply not acted on. A 1574-char version of this prompt that was almost
    entirely preservation instructions, with "take only facial identity
    from the second image" sitting dead last, produced output where
    identity did not transfer AT ALL -- the model preserved everything,
    including the original face.

    The replacement instruction must come first, before any preservation
    clause, exactly as T4's working prompt does.
    """
    p = _face_swap_prompt()
    i_replace = p.find("replace the head")
    i_keep = p.find("Keep the clothing")
    assert i_replace >= 0 and i_keep > 0, p
    assert i_replace < i_keep, (
        "the replacement instruction must precede the preservation clause"
    )


def test_prompt_asks_for_head_replacement_not_face_only():
    """Identity is bought at generation time with a strong HEAD-replacement
    instruction -- measured, that is what transferred identity, while
    face-only wordings transferred none (keeping the target's hair leaves
    the face as the only donor channel, and the model just preserves what
    is already in the source latent).

    The target's own hair and headwear are put back afterwards by
    body_restore's keep_original_hair control, so the generated hairstyle
    is discarded and only the generated face and skin survive.
    """
    p = _face_swap_prompt()
    assert "replace the head" in p
    src = _run_cell_source()
    assert '"simple_full_body_restore_keep_original_hair": True' in src, (
        "a head-replacement prompt without this control would also replace "
        "the target's hair and headwear"
    )


def test_keep_original_hair_runs_after_the_original_head_union():
    """GPU-measured regression: an earlier version subtracted hair+
    accessories from the keep-mask BEFORE the "unioned ORIGINAL head
    footprint" step ran. That step exists specifically to make sure the
    previous person's hair is REPLACED -- it re-adds the original head
    region to "generated wins" because segmenter recall on the GENERATED
    frame alone misses hair strands. Running the subtraction first meant
    the union silently undid it a few lines later: the keep mask shrank
    71608->64278px, then grew right back to include hair/headwear. The
    hat was replaced anyway despite the flag being on.

    The subtraction must be the LAST mutation of the keep-mask before it
    is used, i.e. positioned after this union in source order.
    """
    i_union = KREA2.index(
        "unioned ORIGINAL head footprint"
    )
    i_keep = KREA2.index(
        "simple_full_body_restore_keep_original_hair"
    )
    assert i_union < i_keep, (
        "keep_original_hair must run AFTER the original-head-footprint "
        "union, or that union re-adds hair/headwear to the keep-mask and "
        "silently undoes the subtraction"
    )


def test_prompt_carries_the_forcing_phrase():
    """T4 relies on "with none of the first person's head remaining" to
    overcome img2img at denoise=0.85 keeping what is already in the source
    latent. The face variant needs the equivalent."""
    p = _face_swap_prompt()
    assert "none of the first person's face remaining" in p


def test_prompt_stays_near_t4_length():
    """CHECKPOINT-11 measured that prompt LENGTH alone moves how much of
    the face the model rebuilds. T4's approved text is 564 chars; the
    1574-char version transferred no identity."""
    p = _face_swap_prompt()
    assert len(p) < 700, (
        f"prompt is {len(p)} chars -- drifting back toward the bloated "
        "version whose instructions the model stopped acting on"
    )


def test_headwear_is_kept_not_replaced():
    """The head-swap route REPLACES headwear with the donor's hair. A face
    swap must keep it."""
    src = _run_cell_source()
    assert '"simple_full_body_remove_headwear": False' in src


def test_experimental_garment_work_is_off():
    """Branched off the head-swap line, which carries experimental garment
    machinery. All of it must stay off here."""
    src = _run_cell_source()
    for key in (
        '"simple_full_body_protect_garments": False',
        '"skip_skin_clause_when_covered": False',
        '"simple_full_body_garment_containment": False',
        '"simple_full_body_restore_stripped_garment": False',
    ):
        assert key in src, f"missing: {key}"


def test_run_cell_reads_uploads_from_disk_not_kernel_state():
    """The run cell must not depend on cell 2's widget objects still being
    alive in the kernel.

    That dependency broke in practice: `NameError: name 'body_uploaders' is
    not defined`, from a stale cached copy of the cell being executed. Any
    of a runtime restart, a re-run of cell 2, or Colab restoring an older
    saved copy could trigger it. Cell 2 now writes each file to disk the
    moment it is picked, and the run cell globs that directory, so the two
    cells share no in-memory state at all.
    """
    src = _run_cell_source()
    assert "body_uploaders" not in src and "face_uploaders" not in src, (
        "run cell still references cell 2's widget lists -- it must read "
        "the upload directory from disk instead"
    )
    assert 'UPLOAD_DIR / f"body_{i:02d}.png"' in src
    assert 'UPLOAD_DIR / f"face_{i:02d}.png"' in src


def _upload_cell_source() -> str:
    nb = json.loads(NB_PATH.read_text())
    code_cells = [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    ]
    upload = [s for s in code_cells if "files.upload()" in s]
    assert len(upload) == 1, "expected exactly one cell that uploads images"
    return upload[0]


def test_upload_cell_uses_colabs_native_uploader():
    """`ipywidgets.FileUpload` renders in Colab but its contents frequently
    never sync back to the kernel -- an earlier version of this cell built
    26 such buttons and silently saved nothing, so the run cell found an
    empty directory. `google.colab.files.upload()` is the native picker and
    returns the bytes directly.
    """
    src = _upload_cell_source()
    assert "from google.colab import files" in src
    assert "FileUpload(" not in src, (
        "ipywidgets.FileUpload is unreliable in Colab -- its value often "
        "never reaches the kernel"
    )
    assert "enable_custom_widget_manager" not in src, (
        "this switches Colab to a widget manager that can break core "
        "ipywidgets; it was added speculatively and is not needed"
    )


def test_upload_cell_persists_files_and_clears_stale_ones():
    """Files must land on disk in the run cell's expected naming, and a
    re-run must not leave older pairs behind to be picked up as real."""
    src = _upload_cell_source()
    assert 'UPLOAD_DIR / f"body_{i:02d}.png"' in src
    assert 'UPLOAD_DIR / f"face_{i:02d}.png"' in src
    assert "CLEAR_PREVIOUS" in src


def test_upload_cell_refuses_mismatched_counts():
    """Pairing is positional, so an unequal number of bodies and faces
    would silently mis-pair every image after the mismatch."""
    src = _upload_cell_source()
    assert "!=" in src and "must match 1:1" in src


def _setup_cell_source() -> str:
    nb = json.loads(NB_PATH.read_text())
    code_cells = [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    ]
    setup = [s for s in code_cells if "setup_colab.sh" in s]
    assert len(setup) == 1, "expected exactly one setup cell"
    return setup[0]


def test_setup_cell_does_not_import_numpy_before_reinstalling_it():
    """`import torch` pulls numpy into the kernel. setup_colab.sh then
    force-reinstalls numpy on disk (repairing simple-lama's downgrade), so a
    kernel that imported numpy first is left holding the OLD compiled
    extension against NEW .py files. The first fresh submodule import during
    a render then dies with "cannot import name '_slice' from
    numpy._core.umath" -- 30+ seconds in, from an unrelated import chain.
    GPU-confirmed twice.

    The GPU check must therefore not import torch; nvidia-smi answers the
    same question without loading numpy.
    """
    src = _setup_cell_source()
    # Comments in this cell legitimately discuss `import torch` as the thing
    # being avoided -- only executable lines are the contract.
    code_lines = [
        ln for ln in src.splitlines() if not ln.strip().startswith("#")
    ]
    assert not any("import torch" in ln for ln in code_lines), (
        "importing torch here loads numpy before setup reinstalls it, which "
        "leaves this kernel in a mixed numpy state"
    )
    assert "nvidia-smi" in src


def test_setup_cell_verifies_numpy_in_a_subprocess():
    """The kernel deliberately has not imported numpy, so the health check
    has to happen in a fresh interpreter to mean anything."""
    src = _setup_cell_source()
    assert "numpy._core.strings" in src
    assert "sys.executable" in src


def test_run_cell_guards_against_a_mixed_numpy_state():
    """If it happens anyway (e.g. a stale session), fail immediately with an
    actionable message instead of deep inside a render."""
    src = _run_cell_source()
    assert "numpy._core.strings" in src
    assert "Restart session" in src


def test_run_reports_the_actual_route_taken():
    """The mask-tuning failure went unnoticed because nothing surfaced which
    path actually ran. The notebook must read the route back from meta and
    say so, using the key the pipeline really sets."""
    src = _run_cell_source()
    assert '.get("edit_mode"' in src, (
        "must read edit_mode -- run_simple_full_body sets 'mode'/'edit_mode', "
        "not 'route', so a 'route' lookup silently returns nothing"
    )
    assert '"mode": "simple_full_body"' in KREA2
    assert '"edit_mode": "simple_full_body"' in KREA2
    assert "WARNING" in src and "simple_full_body" in src


def test_refine_mask_top_extend_defaults_to_existing_behaviour():
    """face_refine is a SEPARATE generation composited back through
    build_head_hair_mask. Its top extent was mask_top_extend (1.30), i.e.
    1.3 face-heights above the face box -- squarely into the hair. So the
    refine re-generated hair by construction, undoing any main-pass hair
    preservation, and generated hair does not look real.

    refine_mask_top_extend lets the face-swap route pull that boundary
    down to the face. It MUST default to top_ext so every existing caller
    (head swap included, where regenerating the donor's hair is the whole
    point) is byte-for-byte unchanged.

    This narrows an existing composite's reach; it does not add a mask.
    """
    i = KREA2.index('self.cfg.get("refine_mask_top_extend"')
    window = KREA2[i:i + 80]
    assert "top_ext" in window, (
        "must default to top_ext, or enabling this key silently changes "
        "head-swap behaviour too"
    )
    # And it must actually be the value handed to the mask builder.
    j = KREA2.index("refine_top_ext = float(")
    k = KREA2.index("build_head_hair_mask(", j)
    seg = KREA2[k:k + 400]
    assert "top_extend=refine_top_ext" in seg, (
        "the new key is computed but not used -- the exact class of bug "
        "that made a mask-widening 'fix' silently never apply earlier"
    )


SKIN_HARM = (ROOT / "src" / "headswap" / "skin_harmonize.py").read_text()


def _fn_body(text: str, def_line: str) -> str:
    i = text.index(def_line)
    j = text.find("\ndef ", i + 10)
    return text[i:j if j > 0 else len(text)]


def test_region_restore_uses_the_skin_only_mask_not_the_hair_one():
    """THE fix for the visible ring around the head.

    Every seam this project produced came from a keep-mask that included
    HAIR: semantic_person_skin_mask is hair u accessories u face-skin u
    body-skin, so the "generated wins" region ended on fine hair strands --
    the hardest edge in the image to blend.

    _semantic_skin_mask is face-skin u body-skin ONLY; hair, accessories and
    clothes are excluded by label. The boundary then falls on the hairline,
    collar and sleeve -- real edges present in BOTH images at the SAME pixel
    coordinates, since the render is img2img from the original's own latent
    and is pixel-aligned with it.
    """
    body = _fn_body(SKIN_HARM, "def restore_all_but_face_and_skin(")
    assert "_semantic_skin_mask(" in body
    # Match CALLS, not prose -- the docstring names the rejected mask on
    # purpose, to record why it is rejected.
    assert "semantic_person_skin_mask(" not in body, (
        "that mask includes hair -- using it puts the boundary back through "
        "hair strands, which is the ring around the head"
    )
    assert "semantic_hair_accessories_mask(" not in body


def test_region_restore_blurs_a_continuous_field_not_a_binary_mask():
    """Skin-to-skin must have no boundary at all. Same reasoning as
    extend_skin_harmonization: 'a binary mask ALWAYS has an edge'."""
    body = _fn_body(SKIN_HARM, "def restore_all_but_face_and_skin(")
    assert "GaussianBlur" in body
    # The blurred field is `w`. Thresholding IT would reinstate the hard
    # edge the blur exists to remove. (`skin > 0.5` is fine -- that is a
    # pixel-count diagnostic, not the blend weight.)
    assert "(w > " not in body, (
        "must not threshold the blurred field -- that reinstates the hard "
        "edge the blur exists to remove"
    )
    assert "* w + " in body, "the blend must use the continuous field"


def test_region_restore_fails_closed():
    body = _fn_body(SKIN_HARM, "def restore_all_but_face_and_skin(")
    assert body.count("return rendered, info") >= 2, (
        "segmenter-unavailable and too-little-skin must both ship the "
        "render untouched rather than blend against a bad mask"
    )


def test_region_restore_is_off_by_default_and_runs_after_refine():
    """Head swap must be untouched, and the face it keeps must be the
    high-resolution one the refine pass produced."""
    i = KREA2.index('"simple_full_body_keep_original_except_face_skin"')
    assert "False" in KREA2[i:i + 80]
    assert (KREA2.index("restore_all_but_face_and_skin(")
            > KREA2.index("[krea2 face_refine] mask bot_extend=")), (
        "must run after face_refine, or it keeps the low-resolution face"
    )


def test_refine_identity_overrides_default_to_inheriting():
    """The refine pass can now sample harder than the main pass, with its
    own prompt -- but all three knobs must default to None = inherit, so
    head swap is byte-for-byte unchanged.

    Sampling harder in the refine is safe ONLY because the region stage
    runs after it: everything the refine renders outside face+skin is
    replaced by original pixels, so it cannot damage hair, clothing or
    background however far it travels.
    """
    for key in ("simple_full_body_refine_denoise",
                "simple_full_body_refine_ref_boost"):
        i = KREA2.index(f'self.cfg.get("{key}")')
        # no default argument => None => inherit
        assert KREA2[i:i + len(key) + 22].rstrip().endswith('")'), (
            f"{key} must default to None (inherit), not a literal value"
        )


def test_refine_restores_main_pass_conditioning_afterwards():
    """The overrides are written into self.cfg, which is shared. Without a
    finally-restore they would leak into every later render in the same
    kernel -- exactly the sticky-config class of bug already fixed once in
    chain.py this project."""
    i = KREA2.index('_rf_prev = {')
    seg = KREA2[i:i + 2600]
    assert "finally:" in seg
    assert "self.cfg[_k] = _v" in seg


SETUP_SH = (ROOT / "scripts" / "setup_colab.sh").read_text()


def test_segmenter_is_fetched_before_the_big_weight_downloads():
    """setup_colab.sh runs under `set -euo pipefail`. With the segmenter
    fetch sitting LAST, any earlier failure -- a killed or timed-out
    multi-GB weight download -- aborted the run before it, while those
    weights, already on Drive, survived. Setup then looked complete with no
    tflite, and the region restore silently skipped."""
    i_seg = SETUP_SH.index("selfie_multiclass_256x256.tflite")
    i_dl = SETUP_SH.index("DL_COMMON=(")
    assert i_seg < i_dl, (
        "the segmenter must be fetched before the large model downloads"
    )


def test_segmenter_is_cached_on_drive():
    """/content is ephemeral and wiped on every Colab reconnect, so a
    /content-only copy vanishes while the Drive-backed weights persist --
    which is how it went missing in the first place."""
    i = SETUP_SH.index("SEG_CACHE=")
    assert "HEADSWAP_MODEL_STORE" in SETUP_SH[i:i + 120]


def test_both_segmenter_checks_use_the_same_size_gate():
    """_semantic_skin_mask used exists() only while the loader also required
    >100_000 bytes, so a truncated download passed one gate and failed the
    other with a different message about the same broken file."""
    assert SKIN_HARM.count("getsize(") >= 3
    i = SKIN_HARM.index("def _semantic_skin_mask(")
    body = SKIN_HARM[i:i + 1800]
    assert "100_000" in body


def test_preflight_helper_exists_for_callers_to_refuse_to_run():
    """A caller must be able to fail loudly rather than ship a render with
    no region protection -- the silent skip cost a full debugging cycle."""
    assert "def semantic_segmenter_available(" in SKIN_HARM


def test_preflight_checks_the_mediapipe_package_not_just_the_weights():
    """GPU-observed: the .tflite was present (16MB) so the preflight
    returned True, the run proceeded, and the region restore then skipped
    anyway with ModuleNotFoundError: No module named 'mediapipe'. Checking
    the weights alone recreates the exact silent degradation this helper
    exists to prevent."""
    i = SKIN_HARM.index("def semantic_segmenter_available(")
    body = SKIN_HARM[i:i + 2000]
    assert "import mediapipe" in body
    assert "pip install mediapipe" in body


def test_erode_runs_before_the_blur():
    """Blurring a mask does NOT move its 0.5 crossing -- it softens it in
    place. At feather_px=9 the kernel is 5px, so the boundary sat exactly
    ON the hairline, merely 5px soft: the hardest place in the image to
    hide a seam, and where the top-of-head artifact appeared.

    Eroding FIRST relocates the crossing into forehead/cheek skin, where
    both sides are flat tone and misregistration is invisible. Verified
    numerically: erode_px=0 -> boundary at the mask edge; erode_px=6 ->
    6px inside it.
    """
    body = _fn_body(SKIN_HARM, "def restore_all_but_face_and_skin(")
    i_erode = body.index("cv2.erode(")
    i_blur = body.index("cv2.GaussianBlur(")
    assert i_erode < i_blur, (
        "erode must precede the blur -- eroding after would soften an "
        "already-placed boundary instead of moving it"
    )


def test_erode_defaults_to_zero():
    """Default must be a no-op so head swap and the previous face-swap
    behaviour are byte-for-byte unchanged."""
    body = _fn_body(SKIN_HARM, "def restore_all_but_face_and_skin(")
    assert "erode_px: int = 0" in body
    assert 'self.cfg.get("face_skin_restore_erode_px", 0)' in KREA2
