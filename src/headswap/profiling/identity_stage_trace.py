"""Stage-by-stage identity loss tracer for geometry-lock / Krea2 pipelines.

Saves numbered intermediate images and logs ArcFace cosine vs the donor
identity and vs the original body face so we can prove which stage kills ID.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from headswap.metrics.scoring import identity_cosine, psnr
from headswap.preprocess import FaceBox, get_face_landmarks5, pil_to_rgb_np


@dataclass
class StageRecord:
    index: int
    name: str
    present: bool
    path: str | None = None
    size: list[int] | None = None
    face_bbox: list[int] | None = None
    face_area_pct: float | None = None
    landmarks5: list[list[float]] | None = None
    landmarks_backend: str | None = None
    mask_area_pct: float | None = None
    identity_cosine_vs_donor: float | None = None
    identity_cosine_vs_body_face: float | None = None
    mse_vs_body_face: float | None = None
    psnr_vs_body_face: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)


class IdentityStageTrace:
    """Collect ordered stage images + identity metrics into a debug folder."""

    def __init__(
        self,
        out_dir: Path | None,
        *,
        donor: Image.Image,
        body: Image.Image,
        selected: FaceBox | None,
        selected_index: int | None = None,
        cache_dir=None,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled) and out_dir is not None
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.donor = donor.convert("RGB")
        self.body = body.convert("RGB")
        self.selected = selected
        self.selected_index = selected_index
        self.cache_dir = cache_dir
        self.stages: list[StageRecord] = []
        self._body_face_crop = self._crop_selected(self.body)
        if self.enabled and self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _crop_selected(self, image: Image.Image) -> Image.Image | None:
        if self.selected is None:
            return None
        w, h = image.size
        x0 = max(0, self.selected.x0)
        y0 = max(0, self.selected.y0)
        x1 = min(w, self.selected.x1)
        y1 = min(h, self.selected.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        # Slight pad for ArcFace.
        pad = int(0.15 * max(x1 - x0, y1 - y0))
        return image.crop(
            (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
        )

    def _face_stats(self, image: Image.Image) -> dict[str, Any]:
        rgb = pil_to_rgb_np(image)
        h, w = rgb.shape[:2]
        area = float(h * w) if h and w else 1.0
        bbox = None
        area_pct = None
        if self.selected is not None and image.size == self.body.size:
            bbox = [
                self.selected.x0,
                self.selected.y0,
                self.selected.x1,
                self.selected.y1,
            ]
            area_pct = round(
                100.0
                * (self.selected.width * self.selected.height)
                / max(1.0, float(self.body.size[0] * self.body.size[1])),
                3,
            )
        lm, backend, _ = get_face_landmarks5(
            rgb, self.cache_dir, prefer_box=self.selected if image.size == self.body.size else None
        )
        return {
            "size": [w, h],
            "face_bbox": bbox,
            "face_area_pct": area_pct,
            "landmarks5": lm.tolist() if lm is not None else None,
            "landmarks_backend": backend,
            "pixel_area": area,
        }

    def add(
        self,
        name: str,
        image: Image.Image | None,
        *,
        mask: Image.Image | None = None,
        region: Image.Image | None = None,
        notes: dict[str, Any] | None = None,
        force_full_frame_metrics: bool = False,
    ) -> StageRecord:
        idx = len(self.stages) + 1
        if not self.enabled:
            rec = StageRecord(index=idx, name=name, present=image is not None)
            self.stages.append(rec)
            return rec
        if image is None:
            rec = StageRecord(
                index=idx,
                name=name,
                present=False,
                notes=dict(notes or {}),
            )
            self.stages.append(rec)
            self._print(rec)
            return rec

        im = image.convert("RGB") if image.mode != "RGBA" else image
        # Save RGBA stages as PNG with alpha when present.
        path = self.out_dir / f"{idx:02d}_{name}.png"
        im.save(path)

        stats = self._face_stats(im.convert("RGB"))
        # Identity metrics on the selected face crop when possible.
        probe = region
        if probe is None:
            if force_full_frame_metrics or self.selected is None:
                probe = im.convert("RGB")
            else:
                # Map selected box if full-frame; else use whole image (crop space).
                if im.size == self.body.size:
                    probe = self._crop_selected(im.convert("RGB")) or im.convert("RGB")
                else:
                    probe = im.convert("RGB")

        id_donor = identity_cosine(self.donor, probe)
        id_body = (
            identity_cosine(self._body_face_crop, probe)
            if self._body_face_crop is not None
            else None
        )
        mse = psnr_v = None
        if self._body_face_crop is not None and probe is not None:
            a = np.asarray(
                self._body_face_crop.resize(probe.size, Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            b = np.asarray(probe.convert("RGB"), dtype=np.float32)
            if a.shape == b.shape:
                mse = float(np.mean((a - b) ** 2))
                psnr_v = float(psnr(a, b))

        mask_pct = None
        if mask is not None:
            m = np.asarray(mask.convert("L"))
            mask_pct = round(100.0 * float((m > 127).mean()), 3)

        rec = StageRecord(
            index=idx,
            name=name,
            present=True,
            path=str(path),
            size=stats["size"],
            face_bbox=stats["face_bbox"],
            face_area_pct=stats["face_area_pct"],
            landmarks5=stats["landmarks5"],
            landmarks_backend=stats["landmarks_backend"],
            mask_area_pct=mask_pct,
            identity_cosine_vs_donor=round(id_donor, 4) if id_donor is not None else None,
            identity_cosine_vs_body_face=round(id_body, 4) if id_body is not None else None,
            mse_vs_body_face=round(mse, 3) if mse is not None else None,
            psnr_vs_body_face=round(psnr_v, 3) if psnr_v is not None else None,
            notes=dict(notes or {}),
        )
        self.stages.append(rec)
        self._print(rec)
        return rec

    def add_missing(self, name: str, reason: str) -> StageRecord:
        return self.add(name, None, notes={"missing_reason": reason})

    def _print(self, rec: StageRecord) -> None:
        bits = [
            f"[id_trace {rec.index:02d}] {rec.name}",
            f"present={rec.present}",
        ]
        if rec.size:
            bits.append(f"size={rec.size[0]}x{rec.size[1]}")
        if rec.face_area_pct is not None:
            bits.append(f"face_area%={rec.face_area_pct}")
        if rec.mask_area_pct is not None:
            bits.append(f"mask_area%={rec.mask_area_pct}")
        if rec.identity_cosine_vs_donor is not None:
            bits.append(f"id_vs_donor={rec.identity_cosine_vs_donor}")
        if rec.identity_cosine_vs_body_face is not None:
            bits.append(f"id_vs_body={rec.identity_cosine_vs_body_face}")
        if rec.mse_vs_body_face is not None:
            bits.append(f"mse_vs_body={rec.mse_vs_body_face}")
        if rec.notes:
            bits.append(f"notes={rec.notes}")
        print(" ".join(bits), file=sys.__stdout__, flush=True)

    def diagnose(self) -> dict[str, Any]:
        """Prove Case A (never strong ID) vs Case B (postprocess kills ID)."""
        by_name = {s.name: s for s in self.stages if s.present}
        missing_krea = not by_name.get("06_raw_krea2_output", StageRecord(0, "", False)).present

        # Prefer donor cosine trajectory.
        donor_traj = [
            (s.name, s.identity_cosine_vs_donor)
            for s in self.stages
            if s.present and s.identity_cosine_vs_donor is not None
        ]
        body_traj = [
            (s.name, s.identity_cosine_vs_body_face)
            for s in self.stages
            if s.present and s.identity_cosine_vs_body_face is not None
        ]

        worst_drop = None
        for i in range(1, len(donor_traj)):
            prev_n, prev_v = donor_traj[i - 1]
            cur_n, cur_v = donor_traj[i]
            drop = float(prev_v) - float(cur_v)
            if worst_drop is None or drop > worst_drop["drop"]:
                worst_drop = {
                    "from": prev_n,
                    "to": cur_n,
                    "drop": round(drop, 4),
                    "from_id": prev_v,
                    "to_id": cur_v,
                }

        raw = by_name.get("06_raw_krea2_output")
        final = by_name.get("10_final") or by_name.get("09_after_blend_stitch")
        paste = by_name.get("07b_after_paste_seamless") or by_name.get(
            "07a_after_paste_alpha"
        )
        pre_cm = by_name.get("07_aligned_before_color_match")
        post_cm = by_name.get("07_aligned_after_color_match")

        if missing_krea:
            # Geometry-lock path: identity must come from paste stages.
            paste_id = paste.identity_cosine_vs_donor if paste else None
            final_id = final.identity_cosine_vs_donor if final else None
            body_final = final.identity_cosine_vs_body_face if final else None
            classification = "geometry_lock_no_krea2"
            if (
                pre_cm
                and post_cm
                and pre_cm.identity_cosine_vs_donor is not None
                and post_cm.identity_cosine_vs_donor is not None
                and (pre_cm.identity_cosine_vs_donor - post_cm.identity_cosine_vs_donor)
                > 0.08
            ):
                classification = "case_b_color_match_suppresses_identity"
            elif (
                paste_id is not None
                and final_id is not None
                and paste_id - final_id > 0.08
            ):
                classification = "case_b_postprocess_after_paste"
            elif paste_id is not None and paste_id < 0.35:
                classification = "case_a_weak_paste_identity"
            elif paste_id is not None and paste_id < 0.55:
                classification = "case_a_krea2_bypassed_paste_ceiling"
            elif body_final is not None and body_final > 0.55 and (
                final_id is None or final_id < 0.35
            ):
                classification = "case_a_output_still_original_identity"
            return {
                "classification": classification,
                "krea2_ran": False,
                "worst_donor_drop": worst_drop,
                "donor_trajectory": donor_traj,
                "body_trajectory": body_traj,
                "selected_index": self.selected_index,
                "evidence": (
                    "Krea2 stage 06 absent (align_paste_krea2_refine=false). "
                    "Identity can only come from landmark paste + postprocess. "
                    f"Paste/final donor cosine≈{paste_id}/{final_id}."
                ),
            }

        raw_id = raw.identity_cosine_vs_donor if raw else None
        final_id = final.identity_cosine_vs_donor if final else None
        if raw_id is not None and raw_id < 0.35:
            classification = "case_a_raw_krea2_weak_identity"
        elif (
            raw_id is not None
            and final_id is not None
            and raw_id - final_id > 0.08
        ):
            classification = "case_b_postprocess_identity_loss"
        else:
            classification = "geometry_or_mask_review"
        return {
            "classification": classification,
            "krea2_ran": True,
            "worst_donor_drop": worst_drop,
            "donor_trajectory": donor_traj,
            "body_trajectory": body_traj,
            "selected_index": self.selected_index,
            "evidence": f"raw_id={raw_id} final_id={final_id}",
        }

    def write_report(self) -> dict[str, Any]:
        report = {
            "selected_index": self.selected_index,
            "selected_box": (
                None
                if self.selected is None
                else [
                    self.selected.x0,
                    self.selected.y0,
                    self.selected.x1,
                    self.selected.y1,
                ]
            ),
            "donor_size": list(self.donor.size),
            "body_size": list(self.body.size),
            "stages": [asdict(s) for s in self.stages],
            "diagnosis": self.diagnose(),
        }
        if self.enabled and self.out_dir is not None:
            path = self.out_dir / "IDENTITY_STAGE_REPORT.json"
            path.write_text(json.dumps(report, indent=2, default=str))
            # Human-readable markdown summary
            md = self.out_dir / "IDENTITY_STAGE_REPORT.md"
            lines = [
                "# Identity stage report",
                "",
                f"- selected_index: `{self.selected_index}`",
                f"- classification: **{report['diagnosis'].get('classification')}**",
                f"- krea2_ran: `{report['diagnosis'].get('krea2_ran')}`",
                f"- evidence: {report['diagnosis'].get('evidence')}",
                "",
                "| # | stage | donor_id | body_id | mse_vs_body | size |",
                "|---|-------|----------|---------|-------------|------|",
            ]
            for s in self.stages:
                lines.append(
                    f"| {s.index} | {s.name} | {s.identity_cosine_vs_donor} | "
                    f"{s.identity_cosine_vs_body_face} | {s.mse_vs_body_face} | "
                    f"{s.size} |"
                )
            md.write_text("\n".join(lines) + "\n")
            print(
                f"[id_trace] report → {path} classification="
                f"{report['diagnosis'].get('classification')}",
                file=sys.__stdout__,
                flush=True,
            )
        return report


def overlay_bbox(image: Image.Image, box: FaceBox | None, label: str = "") -> Image.Image:
    out = image.convert("RGB").copy()
    if box is None:
        return out
    d = ImageDraw.Draw(out)
    d.rectangle([box.x0, box.y0, box.x1, box.y1], outline=(0, 255, 80), width=3)
    if label:
        d.text((box.x0, max(0, box.y0 - 14)), label, fill=(0, 255, 80))
    return out
