"""A/B five skin-tone strategies across many pairs, head held constant.

Every strategy in this file keeps the head swap identical -- the disagreement
is only about how the BODY's skin tone is made to match the new head. That is
the one question left after raw-model shipping removed the composite seams,
and single-pair eyeballing has repeatedly picked the wrong winner: a change
that helped one pair regressed another, and a change that moved 2% of pixels
was indistinguishable by eye from one that moved none.

Run: python scripts/ab_skin_variants.py --pairs data/custom/ab_pairs \
        --config configs/krea2_identity_edit.yaml -o results/_ab_skin_variants

Writes per-variant PNGs, a per-pair montage, and REPORT.md ranking variants by
measured tone match and head preservation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------------
# The five strategies.
# --------------------------------------------------------------------------
# Each is a cfg overlay on the production config. `raw_model` (ship the
# model's own frame, no compositing) stays ON everywhere except E, because
# turning it off is what reintroduces the composite boundary -- E exists
# precisely to re-measure that claim rather than assume it.
VARIANTS: list[dict[str, Any]] = [
    {
        "key": "A_prompt_only",
        "label": "A: raw model, prompt only (current default)",
        "why": "Baseline. Model does the recolour unaided at cfg=1.0.",
        "cfg": {"simple_full_body_raw_model": True},
    },
    {
        "key": "B_cfg_guidance",
        "label": "B: raw model + cfg=1.8 (prompt guidance ON)",
        "why": (
            "At cfg=1.0 there is no classifier-free guidance, so the recolour "
            "clause carries no guidance weight at all. If adherence is the "
            "bottleneck this is the single highest-leverage knob. Costs ~2x "
            "sampling time (two UNet evals per step)."
        ),
        "cfg": {"simple_full_body_raw_model": True, "cfg": 1.8},
    },
    {
        "key": "C_named_tone",
        "label": "C: raw model + measured tone named in the prompt",
        "why": (
            "'Match the second image' asks the model to infer a target it may "
            "not attend to. Measuring the donor's cheek L and naming it in "
            "words ('very fair', 'deep brown') turns an inference into a "
            "literal instruction."
        ),
        "cfg": {"simple_full_body_raw_model": True},
        "name_tone": True,
    },
    {
        "key": "D_skin_repaint",
        "label": "D: second pass, model renders skin, head pinned by noise_mask",
        "why": (
            "Keeps the head physically unchangeable (noise_mask re-pins every "
            "latent outside the skin mask each step) while the model RENDERS "
            "the skin rather than recolouring it. Costs a third sampling pass."
        ),
        "cfg": {
            "simple_full_body_raw_model": True,
            "simple_full_body_skin_repaint": True,
        },
    },
    {
        "key": "E_lab_wash",
        "label": "E: composited restore + LAB wash (the old path)",
        "why": (
            "The pre-raw-model behaviour, kept as the control. It is the only "
            "arm that can produce a composite boundary, so if it wins on tone "
            "the trade is real and worth re-opening; if it loses on both tone "
            "and artifacts, raw-model is settled."
        ),
        "cfg": {
            "simple_full_body_raw_model": False,
            "simple_full_body_restore_body": True,
            "simple_full_body_skin_harmonize": True,
        },
    },
]


def _tone_words(l_value: float) -> str:
    """Map a measured LAB L to plain words the text encoder can act on."""
    for thresh, words in (
        (60.0, "very deep, dark brown"),
        (95.0, "deep brown"),
        (130.0, "medium brown"),
        (165.0, "light tan"),
        (200.0, "fair, pale"),
    ):
        if l_value < thresh:
            return words
    return "very fair, very pale"


def _donor_tone_words(face: Image.Image) -> tuple[str, float | None]:
    try:
        from headswap.preprocess import detect_best_face
        from headswap.skin_harmonize import _cheek_lab_stats

        fnp = np.asarray(face.convert("RGB"), dtype=np.uint8)
        box = detect_best_face(face)
        if box is None:
            return "", None
        lab, _ = _cheek_lab_stats(
            fnp, int(box.x0), int(box.y0), int(box.x1), int(box.y1)
        )
        return _tone_words(float(lab[0])), float(lab[0])
    except Exception:  # noqa: BLE001
        return "", None


# --------------------------------------------------------------------------
# Metrics. Both are computed on the RESULT alone plus the donor face, so no
# hand-labelling is needed and every arm is scored identically.
# --------------------------------------------------------------------------
def _tone_gap(result: Image.Image) -> float | None:
    """|face L - visible body-skin L| in the result. Lower is better.

    This is the number the whole exercise is about: a body that does not match
    its own head. Measured inside the result so it is independent of which
    strategy produced it.
    """
    try:
        from headswap.preprocess import detect_best_face
        from headswap.skin_harmonize import (
            _cheek_lab_stats,
            _robust_lab_stats,
            semantic_person_skin_mask,
        )
        import cv2

        rnp = np.asarray(result.convert("RGB"), dtype=np.uint8)
        box = detect_best_face(result)
        if box is None:
            return None
        face_lab, _ = _cheek_lab_stats(
            rnp, int(box.x0), int(box.y0), int(box.x1), int(box.y1)
        )
        skin = semantic_person_skin_mask(rnp)
        if skin is None:
            return None
        skin = skin.copy()
        skin[: max(0, min(rnp.shape[0], int(box.y1)))] = 0.0
        px = rnp[skin > 0.5]
        # A bust shot has little skin below the chin -- a neck and the tops of
        # two shoulders. The old 500px floor rejected exactly those frames, so
        # tone_gap came back None on every arm and the comparison could not
        # answer the question it exists to answer. Scale the floor to the
        # frame and keep it small: 200px of neck is a perfectly good sample of
        # body tone, and the robust (median-based) statistic does not need a
        # large region to be stable.
        _floor = max(200, int(0.0005 * rnp.shape[0] * rnp.shape[1]))
        if px.shape[0] < _floor:
            # Last resort: rembg person minus clothes, below the chin. Catches
            # the case where the class segmenter labels a neck as "face skin"
            # (so it was zeroed with the head) and finds nothing below it.
            try:
                from headswap.skin_harmonize import person_minus_clothes_mask

                alt = person_minus_clothes_mask(rnp, result)
                if alt is not None:
                    alt = alt.copy()
                    alt[: max(0, min(rnp.shape[0], int(box.y1)))] = 0.0
                    px = rnp[alt > 0.5]
            except Exception:  # noqa: BLE001
                pass
        if px.shape[0] < 200:
            return None
        body_lab, _ = _robust_lab_stats(
            cv2.cvtColor(px.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)
            .astype(np.float32)
            .reshape(-1, 3)
        )
        return abs(float(face_lab[0]) - float(body_lab[0]))
    except Exception:  # noqa: BLE001
        return None


def _head_identity(face: Image.Image, result: Image.Image) -> float | None:
    """ArcFace cosine donor-vs-result. Higher is better; guards the head.

    Without this a variant could 'win' on tone by mangling the face, which is
    exactly the trade this whole session kept falling into.
    """
    try:
        from headswap.metrics.scoring import identity_cosine

        return identity_cosine(face, result)
    except Exception:  # noqa: BLE001
        return None


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, 22], fill=(0, 0, 0))
    d.text((5, 6), text[:90], fill=(255, 255, 0))
    return out


def _montage(panels: list[Image.Image], dest: Path, cols: int = 6) -> None:
    if not panels:
        return
    thumb_w = 320
    scaled = []
    for p in panels:
        s = thumb_w / max(1, p.width)
        scaled.append(p.resize((thumb_w, max(1, int(p.height * s)))))
    h = max(p.height for p in scaled)
    rows = (len(scaled) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * h), (18, 18, 18))
    for i, p in enumerate(scaled):
        sheet.paste(p, ((i % cols) * thumb_w, (i // cols) * h))
    sheet.save(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="dir of <name>/body.png + <name>/face.png")
    ap.add_argument("--config", default="configs/krea2_identity_edit.yaml")
    ap.add_argument("-o", "--out", default="results/_ab_skin_variants")
    ap.add_argument("--only", default="", help="comma-separated variant keys")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip renders already on disk (default on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="re-render everything")
    a = ap.parse_args()

    from headswap.config import load_config
    from headswap.pipelines import create_pipeline

    pairs_dir = Path(a.pairs)
    pairs = sorted(
        d for d in pairs_dir.iterdir()
        if d.is_dir() and (d / "body.png").exists() and (d / "face.png").exists()
    )
    if not pairs:
        print(f"No pairs in {pairs_dir} (need <name>/body.png + <name>/face.png)")
        return 2

    variants = VARIANTS
    if a.only:
        want = {k.strip() for k in a.only.split(",") if k.strip()}
        variants = [v for v in VARIANTS if v["key"] in want]

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(a.config)

    print(f"[ab] {len(pairs)} pairs x {len(variants)} variants "
          f"= {len(pairs) * len(variants)} renders", flush=True)

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        # Re-check on use, not just at listing time. An interrupted upload
        # leaves a directory holding face.png and no body.png, and crashing on
        # it threw away an hour of completed renders for the pairs before it.
        if not (pair / "body.png").exists() or not (pair / "face.png").exists():
            print(
                f"[ab] SKIP {pair.name}: incomplete pair "
                f"(body={ (pair/'body.png').exists() } "
                f"face={ (pair/'face.png').exists() }) -- re-upload it",
                flush=True,
            )
            continue
        try:
            body = Image.open(pair / "body.png").convert("RGB")
            face = Image.open(pair / "face.png").convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"[ab] SKIP {pair.name}: unreadable ({exc})", flush=True)
            continue
        tone_words, donor_l = _donor_tone_words(face)
        panels = [_label(body, f"{pair.name}: BODY"),
                  _label(face, f"{pair.name}: DONOR")]

        for v in variants:
            cfg = dict(base_cfg)
            cfg.update(v["cfg"])
            if v.get("name_tone") and tone_words:
                cfg["simple_full_body_tone_words"] = tone_words
            tag = f"{pair.name}/{v['key']}"
            # Resume: a render already on disk is not repeated. A crash or a
            # Colab disconnect part-way through an hour-long sweep should cost
            # the remaining renders, not the finished ones.
            _done = out_dir / v["key"] / f"{pair.name}.png"
            if a.resume and _done.exists():
                print(f"[ab] SKIP {tag}: already rendered", flush=True)
                continue
            print(f"\n===== {tag} =====", flush=True)
            t0 = time.perf_counter()
            rec: dict[str, Any] = {
                "pair": pair.name, "variant": v["key"], "label": v["label"],
                "donor_face_L": round(donor_l, 1) if donor_l else None,
            }
            try:
                pipe = create_pipeline(cfg)
                res = pipe.run(body, face, None)
                img = res.image if hasattr(res, "image") else res["image"]
                vdir = out_dir / v["key"]
                vdir.mkdir(parents=True, exist_ok=True)
                img.save(vdir / f"{pair.name}.png")
                rec["seconds"] = round(time.perf_counter() - t0, 1)
                rec["tone_gap"] = _tone_gap(img)
                rec["identity"] = _head_identity(face, img)
                panels.append(_label(
                    img,
                    f"{v['key']}  gap={rec['tone_gap']}  id={rec['identity']}",
                ))
                print(f"[ab] {tag}: tone_gap={rec['tone_gap']} "
                      f"identity={rec['identity']} {rec['seconds']}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[ab] {tag} FAILED: {rec['error']}", flush=True)
                traceback.print_exc()
            rows.append(rec)
            (out_dir / "results.json").write_text(json.dumps(rows, indent=2))

        _montage(panels, out_dir / f"montage_{pair.name}.png")

    # ---- report -----------------------------------------------------------
    lines = ["# Skin-tone strategy A/B", "",
             f"{len(pairs)} pairs x {len(variants)} variants.", "",
             "`tone_gap` = |face L - body-skin L| in the RESULT (lower is "
             "better; this is the artifact being chased).  ",
             "`identity` = ArcFace cosine donor vs result (higher is better; "
             "guards against winning on tone by mangling the head).", ""]

    lines += ["## Per-variant means", "",
              "| variant | mean tone_gap | mean identity | mean s | fails |",
              "|---|---|---|---|---|"]
    summary = []
    for v in variants:
        vr = [r for r in rows if r["variant"] == v["key"]]
        gaps = [r["tone_gap"] for r in vr if r.get("tone_gap") is not None]
        ids = [r["identity"] for r in vr if r.get("identity") is not None]
        secs = [r["seconds"] for r in vr if r.get("seconds") is not None]
        fails = sum(1 for r in vr if r.get("error"))
        mg = round(float(np.mean(gaps)), 1) if gaps else None
        mi = round(float(np.mean(ids)), 3) if ids else None
        ms = round(float(np.mean(secs)), 1) if secs else None
        summary.append((v, mg, mi))
        lines.append(f"| {v['key']} | {mg} | {mi} | {ms} | {fails} |")

    ranked = [s for s in summary if s[1] is not None]
    ranked.sort(key=lambda s: s[1])
    if ranked:
        best = ranked[0]
        lines += ["", "## Reading this", "",
                  f"Lowest mean tone_gap: **{best[0]['key']}** ({best[1]}).", "",
                  "Do not take that alone: check `identity` for the same row. A "
                  "variant that wins on tone while dropping identity has "
                  "recreated the trade this A/B exists to settle. Open "
                  "`montage_<pair>.png` before concluding -- tone_gap cannot "
                  "see a composite seam, a mask edge, or a recoloured garment.",
                  ""]

    lines += ["## Variants", ""]
    for v in variants:
        lines += [f"### {v['key']} — {v['label']}", "", v["why"], ""]

    (out_dir / "REPORT.md").write_text("\n".join(lines))
    print(f"\n[ab] wrote {out_dir/'REPORT.md'}", flush=True)
    print("\n".join(lines[:24]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
