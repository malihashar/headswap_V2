"""Head-scale geometry tracer for Krea2 crop_stitch RCA.

Measures face/head size at each pipeline stage in body_full coordinates so we
can name the FIRST stage where the selected head becomes larger than the
original target. Geometry only — no identity metrics.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from headswap.preprocess import (
    FaceBox,
    detect_best_face,
    get_face_landmarks5,
    mask_bbox,
    pil_to_rgb_np,
)


@dataclass
class StageGeom:
    name: str
    image_size: list[int]
    face_box: list[int] | None = None
    head_box: list[int] | None = None  # mask bbox when available
    face_w: float | None = None
    face_h: float | None = None
    head_w: float | None = None
    head_h: float | None = None
    face_w_frac: float | None = None
    face_h_frac: float | None = None
    head_w_frac: float | None = None
    head_h_frac: float | None = None
    face_area_frac: float | None = None
    # Mapped into body_full coordinates (ground truth frame).
    face_box_body: list[int] | None = None
    head_box_body: list[int] | None = None
    face_w_body: float | None = None
    face_h_body: float | None = None
    head_w_body: float | None = None
    head_h_body: float | None = None
    iod: float | None = None
    eye_line_deg: float | None = None
    landmarks5: list[list[float]] | None = None
    landmarks_backend: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)
    overlay_path: str | None = None


def _box_list(b: FaceBox | tuple[int, int, int, int] | list[int] | None) -> list[int] | None:
    if b is None:
        return None
    if isinstance(b, FaceBox):
        return [int(b.x0), int(b.y0), int(b.x1), int(b.y1)]
    return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]


def _wh(box: list[int] | None) -> tuple[float | None, float | None]:
    if box is None:
        return None, None
    return float(max(0, box[2] - box[0])), float(max(0, box[3] - box[1]))


def _map_box_to_body(
    box: list[int] | None,
    *,
    local_size: tuple[int, int],
    crop_box_body: tuple[int, int, int, int] | list[int] | None,
) -> list[int] | None:
    """Map a box in crop/scene pixels into body_full via the native crop rectangle."""
    if box is None or crop_box_body is None:
        return None
    lx0, ly0, lx1, ly1 = box
    bx0, by0, bx1, by1 = [int(v) for v in crop_box_body]
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)
    lw, lh = max(1, local_size[0]), max(1, local_size[1])
    sx = bw / float(lw)
    sy = bh / float(lh)
    return [
        int(round(bx0 + lx0 * sx)),
        int(round(by0 + ly0 * sy)),
        int(round(bx0 + lx1 * sx)),
        int(round(by0 + ly1 * sy)),
    ]


def _eye_stats(lm: np.ndarray | None) -> tuple[float | None, float | None]:
    if lm is None or len(lm) < 2:
        return None, None
    le, re = lm[0], lm[1]
    dx, dy = float(re[0] - le[0]), float(re[1] - le[1])
    return round(math.hypot(dx, dy), 3), round(math.degrees(math.atan2(dy, dx)), 3)


def measure_stage(
    name: str,
    image: Image.Image,
    cache_dir,
    *,
    face: FaceBox | list[int] | None = None,
    head_mask: Image.Image | None = None,
    crop_box_body: tuple[int, int, int, int] | list[int] | None = None,
    map_to_body: bool = False,
    prefer_detect: bool = False,
    notes: dict[str, Any] | None = None,
) -> StageGeom:
    """Measure face/head geometry on ``image`` (optionally map into body coords)."""
    rgb = pil_to_rgb_np(image.convert("RGB"))
    iw, ih = image.size
    face_box = _box_list(face)
    if prefer_detect or face_box is None:
        det = detect_best_face(rgb, cache_dir)
        if det is not None:
            face_box = _box_list(det)

    head_box = None
    if head_mask is not None:
        hm = head_mask
        if hm.size != image.size:
            hm = hm.resize(image.size, Image.Resampling.NEAREST)
        head_box = list(mask_bbox(hm, pad=0))

    fw, fh = _wh(face_box)
    hw, hh = _wh(head_box)
    img_area = float(max(1, iw * ih))

    lm, lm_backend, _ = get_face_landmarks5(
        rgb,
        cache_dir,
        prefer_box=None
        if face_box is None
        else FaceBox(face_box[0], face_box[1], face_box[2], face_box[3], 1.0),
    )
    iod, eye_deg = _eye_stats(lm)

    face_body = (
        _map_box_to_body(face_box, local_size=(iw, ih), crop_box_body=crop_box_body)
        if map_to_body
        else (face_box if crop_box_body is None else None)
    )
    # When image IS body_full, face_box is already body coords.
    if not map_to_body and crop_box_body is None:
        face_body = face_box
        head_body = head_box
    else:
        head_body = (
            _map_box_to_body(head_box, local_size=(iw, ih), crop_box_body=crop_box_body)
            if map_to_body
            else None
        )

    fwb, fhb = _wh(face_body)
    hwb, hhb = _wh(head_body)

    return StageGeom(
        name=name,
        image_size=[iw, ih],
        face_box=face_box,
        head_box=head_box,
        face_w=None if fw is None else round(fw, 2),
        face_h=None if fh is None else round(fh, 2),
        head_w=None if hw is None else round(hw, 2),
        head_h=None if hh is None else round(hh, 2),
        face_w_frac=None if fw is None else round(fw / float(iw), 4),
        face_h_frac=None if fh is None else round(fh / float(ih), 4),
        head_w_frac=None if hw is None else round(hw / float(iw), 4),
        head_h_frac=None if hh is None else round(hh / float(ih), 4),
        face_area_frac=None
        if fw is None or fh is None
        else round((fw * fh) / img_area, 4),
        face_box_body=face_body,
        head_box_body=head_body,
        face_w_body=None if fwb is None else round(fwb, 2),
        face_h_body=None if fhb is None else round(fhb, 2),
        head_w_body=None if hwb is None else round(hwb, 2),
        head_h_body=None if hhb is None else round(hhb, 2),
        iod=iod,
        eye_line_deg=eye_deg,
        landmarks5=None if lm is None else lm.tolist(),
        landmarks_backend=lm_backend,
        notes=dict(notes or {}),
    )


def overlay_stage(
    image: Image.Image,
    stage: StageGeom,
    *,
    crop_box: list[int] | None = None,
    title: str | None = None,
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    if crop_box is not None:
        draw.rectangle(crop_box, outline=(255, 60, 60), width=3)
    if stage.head_box is not None:
        draw.rectangle(stage.head_box, outline=(0, 140, 255), width=2)
    if stage.face_box is not None:
        draw.rectangle(stage.face_box, outline=(0, 255, 80), width=2)
    if stage.landmarks5:
        for i, (x, y) in enumerate(stage.landmarks5[:5]):
            r = 3
            color = (0, 255, 255) if i < 2 else (255, 200, 0)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        if len(stage.landmarks5) >= 2:
            a, b = stage.landmarks5[0], stage.landmarks5[1]
            draw.line([a[0], a[1], b[0], b[1]], fill=(255, 255, 0), width=2)
    label = title or stage.name
    fh = stage.face_h_body if stage.face_h_body is not None else stage.face_h
    draw.text((6, 6), f"{label} face_h={fh}", fill=(255, 255, 255))
    return out


def first_enlargement_stage(
    stages: dict[str, StageGeom],
    *,
    tol: float = 1.12,
) -> dict[str, Any]:
    """
    Find the earliest stage where face height (body coords preferred) exceeds
    the S0 baseline by ``tol``.
    """
    order = [
        "S0_original",
        "S1_crop",
        "S2_scene",
        "S3_edited",
        "S4_stitched",
    ]
    s0 = stages.get("S0_original")
    if s0 is None or s0.face_h_body is None and s0.face_h is None:
        return {
            "FIRST_ENLARGEMENT_STAGE": "unknown",
            "reason": "missing S0 face measurement",
            "tol": tol,
        }
    base_h = float(s0.face_h_body if s0.face_h_body is not None else s0.face_h or 0)
    if base_h < 1:
        return {
            "FIRST_ENLARGEMENT_STAGE": "unknown",
            "reason": "S0 face_h < 1",
            "tol": tol,
        }

    ratios: dict[str, float | None] = {}
    first = "none"
    # S1/S2: compare face_h in their local frame to expected mapped size —
    # for crop/scene use face_h_body mapped back (should ≈ S0 if geometry preserved).
    for name in order[1:]:
        st = stages.get(name)
        if st is None:
            ratios[name] = None
            continue
        # Prefer body-mapped face height for cross-stage compare.
        h = st.face_h_body
        if h is None and name in ("S2_scene", "S3_edited"):
            # Fall back to local face_h ratio vs S2 baseline for S3.
            h = None
        if name == "S3_edited":
            s2 = stages.get("S2_scene")
            if (
                s2 is not None
                and s2.face_h is not None
                and st.face_h is not None
                and float(s2.face_h) > 0
            ):
                r_local = float(st.face_h) / float(s2.face_h)
                ratios["S3_vs_S2_local"] = round(r_local, 4)
                if first == "none" and r_local > tol:
                    first = "S3_edited"
                    ratios[name] = (
                        round(float(st.face_h_body) / base_h, 4)
                        if st.face_h_body is not None
                        else round(r_local, 4)
                    )
                    continue
        if h is None:
            h = st.face_h_body if st.face_h_body is not None else st.face_h
        if h is None:
            ratios[name] = None
            continue
        r = float(h) / base_h
        ratios[name] = round(r, 4)
        if first == "none" and r > tol:
            first = name

    return {
        "FIRST_ENLARGEMENT_STAGE": first,
        "tol": tol,
        "s0_face_h_body": base_h,
        "ratios_vs_s0": ratios,
        "s3_vs_s2_local": ratios.get("S3_vs_S2_local"),
    }


class HeadScaleTrace:
    """Collect S0–S4 geometry dumps for one crop_stitch swap."""

    def __init__(
        self,
        out_dir: Path | None,
        *,
        cache_dir,
        enabled: bool = True,
        body_full: Image.Image | None = None,
        selected: FaceBox | None = None,
        crop_box: tuple[int, int, int, int] | list[int] | None = None,
    ) -> None:
        self.enabled = bool(enabled) and out_dir is not None
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.cache_dir = cache_dir
        self.body_full = body_full.convert("RGB") if body_full is not None else None
        self.selected = selected
        self.crop_box = list(crop_box) if crop_box is not None else None
        self.stages: dict[str, StageGeom] = {}
        self.meta: dict[str, Any] = {
            "affine_matrix": None,
            "affine_note": "N/A on krea2_crop production path (no landmark affine)",
            "path": "krea2_crop_spp",
        }
        if self.enabled and self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save_overlay(self, name: str, image: Image.Image, stage: StageGeom) -> None:
        if not self.enabled or self.out_dir is None:
            return
        crop = self.crop_box if name in ("S0_original", "S1_crop", "S4_stitched") else None
        # For body-space overlays use body-mapped boxes when drawing on body.
        vis_stage = stage
        if name in ("S0_original", "S1_crop", "S4_stitched") and stage.face_box_body:
            vis_stage = StageGeom(**{**asdict(stage), "face_box": stage.face_box_body,
                                     "head_box": stage.head_box_body or stage.head_box})
        ov = overlay_stage(image, vis_stage, crop_box=crop, title=name)
        path = self.out_dir / f"{name}_overlay.png"
        ov.save(path)
        stage.overlay_path = str(path)

    def record_prep(
        self,
        *,
        body_full: Image.Image,
        selected: FaceBox | None,
        mask: Image.Image | None,
        crop_box: tuple[int, int, int, int] | list[int],
        scene: Image.Image,
        person: Image.Image | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.body_full = body_full.convert("RGB")
        self.selected = selected
        self.crop_box = list(crop_box)

        s0 = measure_stage(
            "S0_original",
            self.body_full,
            self.cache_dir,
            face=selected,
            head_mask=mask,
            notes={"coord": "body_full"},
        )
        self.stages["S0_original"] = s0
        self._save_overlay("S0_original", self.body_full, s0)
        if mask is not None:
            mask.convert("L").save(self.out_dir / "S0_head_mask.png")

        # S1: crop rectangle on body (geometry of window, not yet resized).
        crop_native = self.body_full.crop(tuple(self.crop_box))
        face_in_crop = None
        if selected is not None:
            face_in_crop = FaceBox(
                selected.x0 - self.crop_box[0],
                selected.y0 - self.crop_box[1],
                selected.x1 - self.crop_box[0],
                selected.y1 - self.crop_box[1],
                selected.conf,
            )
        mask_in_crop = None
        if mask is not None:
            mask_in_crop = mask.crop(tuple(self.crop_box))
        s1 = measure_stage(
            "S1_crop",
            crop_native,
            self.cache_dir,
            face=face_in_crop,
            head_mask=mask_in_crop,
            crop_box_body=self.crop_box,
            map_to_body=True,
            notes={
                "coord": "crop_native",
                "crop_box_body": self.crop_box,
                "crop_native_size": list(crop_native.size),
            },
        )
        self.stages["S1_crop"] = s1
        # Overlay on body with crop rect.
        body_ov = overlay_stage(
            self.body_full,
            StageGeom(
                **{
                    **asdict(s0),
                    "face_box": s0.face_box,
                    "head_box": s0.head_box,
                    "name": "S1_crop",
                }
            ),
            crop_box=self.crop_box,
            title="S1_crop",
        )
        body_ov.save(self.out_dir / "S1_crop_overlay.png")
        s1.overlay_path = str(self.out_dir / "S1_crop_overlay.png")
        crop_native.save(self.out_dir / "S1_crop_native.png")

        s2 = measure_stage(
            "S2_scene",
            scene,
            self.cache_dir,
            face=None,
            prefer_detect=True,
            crop_box_body=self.crop_box,
            map_to_body=True,
            notes={
                "coord": "scene_inference",
                "scene_size": list(scene.size),
                "scale_x": round(
                    (self.crop_box[2] - self.crop_box[0]) / float(max(1, scene.size[0])),
                    4,
                ),
                "scale_y": round(
                    (self.crop_box[3] - self.crop_box[1]) / float(max(1, scene.size[1])),
                    4,
                ),
            },
        )
        self.stages["S2_scene"] = s2
        self._save_overlay("S2_scene", scene, s2)
        scene.save(self.out_dir / "S2_scene.png")
        if person is not None:
            person.save(self.out_dir / "S2_person.png")

    def record_edited(self, edited: Image.Image) -> None:
        if not self.enabled:
            return
        s3 = measure_stage(
            "S3_edited",
            edited,
            self.cache_dir,
            face=None,
            prefer_detect=True,
            crop_box_body=self.crop_box,
            map_to_body=True,
            notes={"coord": "edited_crop_after_krea2"},
        )
        s2 = self.stages.get("S2_scene")
        if s2 is not None and s2.face_h and s3.face_h:
            s3.notes["edited_face_h_over_scene_face_h"] = round(
                float(s3.face_h) / float(s2.face_h), 4
            )
        self.stages["S3_edited"] = s3
        self._save_overlay("S3_edited", edited, s3)
        edited.save(self.out_dir / "S3_edited.png")

    def record_stitched(self, result: Image.Image) -> None:
        if not self.enabled:
            return
        # Detect face near original selected location.
        prefer = self.selected
        s4 = measure_stage(
            "S4_stitched",
            result,
            self.cache_dir,
            face=prefer,
            prefer_detect=True,
            notes={"coord": "body_full_after_stitch"},
        )
        # If detection drifted, pick face closest to original selected center.
        if self.selected is not None and s4.face_box is not None:
            from headswap.preprocess import detect_faces

            faces = detect_faces(pil_to_rgb_np(result), self.cache_dir)
            if faces:
                scx = 0.5 * (self.selected.x0 + self.selected.x1)
                scy = 0.5 * (self.selected.y0 + self.selected.y1)

                def dist(f: FaceBox) -> float:
                    return (0.5 * (f.x0 + f.x1) - scx) ** 2 + (
                        0.5 * (f.y0 + f.y1) - scy
                    ) ** 2

                best = min(faces, key=dist)
                s4 = measure_stage(
                    "S4_stitched",
                    result,
                    self.cache_dir,
                    face=best,
                    notes={"coord": "body_full_after_stitch", "matched_to_selected": True},
                )
        s0 = self.stages.get("S0_original")
        if s0 is not None and s0.face_h_body and s4.face_h_body:
            s4.notes["stitched_face_h_over_original"] = round(
                float(s4.face_h_body) / float(s0.face_h_body), 4
            )
        self.stages["S4_stitched"] = s4
        self._save_overlay("S4_stitched", result, s4)
        result.save(self.out_dir / "S4_stitched.png")

    def _scale_drift_strip(self) -> Image.Image | None:
        if self.body_full is None:
            return None
        panels: list[Image.Image] = []
        colors = {
            "S0_original": (0, 255, 80),
            "S1_crop": (255, 60, 60),
            "S2_scene": (255, 200, 0),
            "S3_edited": (255, 0, 255),
            "S4_stitched": (0, 220, 255),
        }
        for name in (
            "S0_original",
            "S1_crop",
            "S2_scene",
            "S3_edited",
            "S4_stitched",
        ):
            st = self.stages.get(name)
            path = self.out_dir / f"{name}_overlay.png" if self.out_dir else None
            if path is not None and path.is_file():
                im = Image.open(path).convert("RGB")
            elif name == "S1_crop" and self.out_dir and (self.out_dir / "S1_crop_overlay.png").is_file():
                im = Image.open(self.out_dir / "S1_crop_overlay.png").convert("RGB")
            else:
                continue
            # Fit height
            h = 280
            scale = h / max(1, im.size[1])
            im = im.resize((max(1, int(im.size[0] * scale)), h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (im.size[0], h + 24), (20, 20, 20))
            canvas.paste(im, (0, 24))
            draw = ImageDraw.Draw(canvas)
            fh = None if st is None else (st.face_h_body or st.face_h)
            draw.text((4, 4), f"{name} h={fh}", fill=colors.get(name, (255, 255, 255)))
            panels.append(canvas)
        if not panels:
            return None
        gap = 8
        w = sum(p.size[0] for p in panels) + gap * (len(panels) - 1)
        strip = Image.new("RGB", (w, panels[0].size[1]), (10, 10, 10))
        x = 0
        for p in panels:
            strip.paste(p, (x, 0))
            x += p.size[0] + gap
        return strip

    def finalize(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        verdict = first_enlargement_stage(self.stages)
        report = {
            "enabled": True,
            "meta": self.meta,
            "crop_box_body": self.crop_box,
            "selected_box": _box_list(self.selected),
            "stages": {k: asdict(v) for k, v in self.stages.items()},
            "verdict": verdict,
        }
        if self.out_dir is not None:
            (self.out_dir / "HEAD_SCALE_TRACE.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            strip = self._scale_drift_strip()
            if strip is not None:
                strip.save(self.out_dir / "SCALE_DRIFT.png")
            lines = [
                "# Head scale geometry RCA",
                "",
                f"Path: `{self.meta.get('path')}`",
                f"Affine: {self.meta.get('affine_note')}",
                "",
                f"## FIRST_ENLARGEMENT_STAGE = `{verdict.get('FIRST_ENLARGEMENT_STAGE')}`",
                "",
                f"Tolerance: face_h ratio > {verdict.get('tol')}",
                f"S0 face_h_body: {verdict.get('s0_face_h_body')}",
                f"Ratios vs S0: {json.dumps(verdict.get('ratios_vs_s0'), indent=2)}",
                f"S3 vs S2 (local crop): {verdict.get('s3_vs_s2_local')}",
                "",
                "## Stages",
            ]
            for name in (
                "S0_original",
                "S1_crop",
                "S2_scene",
                "S3_edited",
                "S4_stitched",
            ):
                st = self.stages.get(name)
                if st is None:
                    lines.append(f"- {name}: missing")
                    continue
                lines.append(
                    f"- **{name}**: face_h_body={st.face_h_body} face_h_local={st.face_h} "
                    f"face_h_frac={st.face_h_frac} head_h_body={st.head_h_body} "
                    f"notes={st.notes}"
                )
            lines += [
                "",
                "## Interpretation rule",
                "- If first is S1/S2: crop window / scene framing already wrong.",
                "- If first is S3_edited: Krea2 (or mock) enlarged the head inside the crop.",
                "- If first is S4_stitched: stitch mapping / mask reveal enlarged the head.",
                "- Do not add correction factors elsewhere until this stage is fixed.",
            ]
            (self.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (self.out_dir / "REPORT.json").write_text(
                json.dumps({"verdict": verdict, "stages_summary": {
                    k: {
                        "face_h_body": v.face_h_body,
                        "face_h": v.face_h,
                        "face_h_frac": v.face_h_frac,
                        "notes": v.notes,
                    }
                    for k, v in self.stages.items()
                }}, indent=2),
                encoding="utf-8",
            )
        return report


def analyze_debug_dir(
    *,
    body: Image.Image,
    scene_or_crop: Image.Image,
    edited: Image.Image,
    result: Image.Image,
    mask: Image.Image | None,
    selected: FaceBox,
    crop_box: tuple[int, int, int, int] | list[int],
    out_dir: Path,
    cache_dir,
) -> dict[str, Any]:
    """Offline RCA from saved debug_scene / debug_edited_crop / result."""
    trace = HeadScaleTrace(
        out_dir,
        cache_dir=cache_dir,
        enabled=True,
        body_full=body,
        selected=selected,
        crop_box=crop_box,
    )
    trace.record_prep(
        body_full=body,
        selected=selected,
        mask=mask,
        crop_box=crop_box,
        scene=scene_or_crop,
    )
    trace.record_edited(edited)
    trace.record_stitched(result)
    return trace.finalize()
