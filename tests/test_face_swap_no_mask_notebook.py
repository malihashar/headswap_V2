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


def test_raw_model_on_and_face_refine_off():
    """raw_model alone is NOT enough -- face_refine is gated separately and
    composites via feathered_soft_composite."""
    src = _run_cell_source()
    assert '"simple_full_body_raw_model": True' in src
    assert '"simple_full_body_face_refine": False' in src


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
