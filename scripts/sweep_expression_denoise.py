#!/usr/bin/env python3
"""Sweep the single-image expression edit over denoise, then run one T4 swap.

Exists as a SCRIPT rather than a notebook cell on purpose. Colab caches
notebook `#@param` values in the browser tab, and a notebook saved to Drive
stops re-fetching from GitHub altogether -- between them, ten consecutive GPU
runs of this investigation executed stale code while the logs showed a fresh
commit. A script is pulled by `git pull` like everything else, so the file
that runs is the file in the repo.

Usage (Colab), as a cell that never needs to change again:

    !cd /content/headswap_V2 && git pull -q && python scripts/sweep_expression_denoise.py
    from IPython.display import Image, display
    display(Image('/content/headswap_V2/results/single_edit_test/sweep_montage.png'))

What it measures. Step 1 edits the donor's expression with upstream's
single-image graph; denoise governs how far it may travel from the donor.
At 1.0 the donor is regenerated outright and the likeness goes with it; lower
values let the source latent anchor the face but eventually stop the
expression moving. The useful value is the HIGHEST denoise whose expression
still changes while ArcFace identity stays high, and that trade is only
visible with the arms side by side.

Upstream's own caveat, so the ceiling is not mistaken for a bug:
"structure-preserving i2i -- can't guarantee 1:1 content preservation,
confirmed not solved by us or the wider community."
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    """Put ComfyUI + headswap on the path the same way the notebook does."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "colab_env", REPO / "scripts" / "colab_env.py"
    )
    colab_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(colab_env)
    paths = colab_env.apply_env(colab_env.default_paths(use_drive=False))
    colab_env.ensure_import_path(REPO)
    comfy = str(paths.get("comfyui", "/content/ComfyUI"))
    if comfy not in sys.path:
        sys.path.insert(0, comfy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--denoise",
        default="1.0,0.8,0.65,0.5",
        help="comma-separated denoise values for the expression edit",
    )
    ap.add_argument(
        "--instruction",
        default=(
            "change this person's facial expression so they are not smiling, "
            "with a closed relaxed mouth and a neutral, serious expression. "
            "Keep their eyes, identity, hair and pose exactly the same."
        ),
    )
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument(
        "--swap-with",
        default="best",
        help="'best' (highest identity), 'none', or a denoise value like 0.65",
    )
    ap.add_argument("--pair-dir", default=str(REPO / "data" / "custom" / "my_pair"))
    ap.add_argument("--out-dir", default=str(REPO / "results" / "single_edit_test"))
    args = ap.parse_args()

    _bootstrap()
    os.chdir(REPO)

    from PIL import Image

    from headswap.config import load_config
    from headswap.metrics.scoring import identity_cosine
    from headswap.pipelines import create_pipeline
    from headswap.pipelines.krea2 import get_shared_krea2_runtime

    # Absolute before the runtime is built -- ComfyUI's init chdirs into its
    # own tree, so a relative path silently resolves elsewhere afterwards.
    pair = Path(args.pair_dir)
    if not pair.is_absolute():
        pair = (REPO / pair).resolve()
    body_path, face_path = pair / "body.png", pair / "face.png"
    for f in (body_path, face_path):
        if not f.exists():
            print(f"ERROR: missing {f} -- upload a pair first.", file=sys.stderr)
            return 2
    body_im = Image.open(body_path).convert("RGB")
    face_im = Image.open(face_path).convert("RGB")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = get_shared_krea2_runtime(init_custom_nodes=True)
    cfg = load_config(REPO / "configs" / "krea2_identity_edit.yaml")
    cfg.update(
        {"seed": int(args.seed), "verbose": False, "pre_edit_donor_expression": False}
    )
    pipe = create_pipeline(cfg, runtime=runtime)

    denoises = [float(x) for x in str(args.denoise).split(",") if x.strip()]
    print(f"\n=== sweeping denoise {denoises} ===", flush=True)

    arms: list[dict] = []
    for dn in denoises:
        pipe.cfg["single_edit_denoise"] = dn
        res = pipe.edit_single_image(face_im, args.instruction)
        idc = identity_cosine(face_im, res["image"])
        path = out_dir / f"edit_denoise_{dn}.png"
        res["image"].save(path)
        arms.append(
            {"denoise": dn, "image": res["image"], "identity": idc, "path": path}
        )
        print(
            f"  denoise={dn:<5} identity_vs_donor={idc}  -> {path.name}",
            flush=True,
        )

    print("\n  denoise | identity vs donor")
    print("  --------|------------------")
    for a in arms:
        print(f"  {a['denoise']:<7} | {a['identity']}")
    print(
        "\n  Pick the HIGHEST denoise whose expression actually changed while\n"
        "  identity stays high. Too low: the smile comes back. Too high: the\n"
        "  donor stops being the donor.",
        flush=True,
    )

    # Montage: donor first, then each arm, so the trade is visible at a glance
    # without opening four files.
    _montage([("donor", face_im)] + [(f"dn={a['denoise']} id={a['identity']}", a["image"]) for a in arms],
             out_dir / "sweep_montage.png")
    print(f"\nmontage -> {out_dir / 'sweep_montage.png'}", flush=True)

    if str(args.swap_with).lower() == "none":
        return 0

    if str(args.swap_with).lower() == "best":
        scored = [a for a in arms if a["identity"] is not None]
        chosen = max(scored, key=lambda a: a["identity"]) if scored else arms[-1]
    else:
        target = float(args.swap_with)
        chosen = min(arms, key=lambda a: abs(a["denoise"] - target))

    print(
        f"\n=== T4 swap using denoise={chosen['denoise']} "
        f"(identity={chosen['identity']}) ===",
        flush=True,
    )
    t0 = time.perf_counter()
    result = pipe.run(body_im, chosen["image"], out_dir=out_dir)
    print(f"swap done in {time.perf_counter() - t0:.0f}s", flush=True)

    final = out_dir / "final_result.png"
    result.image.save(final)
    print(f"id_donor_vs_final  = {identity_cosine(face_im, result.image)}")
    print(f"id_edited_vs_final = {identity_cosine(chosen['image'], result.image)}")
    print(f"final -> {final}", flush=True)
    return 0


def _montage(items, out_path: Path, pad: int = 8) -> None:
    """Label-free side-by-side strip, heights normalised."""
    from PIL import Image, ImageDraw

    h = 320
    tiles = []
    for label, im in items:
        w = max(1, int(im.width * h / max(1, im.height)))
        tiles.append((label, im.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)))
    total_w = sum(t.width for _, t in tiles) + pad * (len(tiles) + 1)
    canvas = Image.new("RGB", (total_w, h + 28 + pad * 2), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = pad
    for label, t in tiles:
        canvas.paste(t, (x, 28 + pad))
        draw.text((x + 2, 6), str(label), fill=(235, 235, 235))
        x += t.width + pad
    canvas.save(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
