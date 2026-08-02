"""Unit tests for the modular InSwap pipeline (no GPU / InsightFace required)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from headswap.inswap.blend import face_ellipse_mask, soft_blend
from headswap.inswap.detect import select_face
from headswap.inswap.engines.base import DetectedFace, SwapEngine, SwapEngineResult
from headswap.inswap.engines import ENGINE_REGISTRY, create_engine
from headswap.inswap.pipeline import InSwapPipeline
from headswap.inswap.viz import difference_map, draw_detections


def _face(bbox, *, area_boost: float = 1.0) -> DetectedFace:
    x0, y0, x1, y1 = bbox
    # Fake raw object so engine can run without InsightFace
    return DetectedFace(
        bbox=(float(x0), float(y0), float(x1), float(y1)),
        kps=np.array(
            [
                [x0 + 0.3 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                [x0 + 0.7 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                [x0 + 0.5 * (x1 - x0), y0 + 0.55 * (y1 - y0)],
                [x0 + 0.35 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
                [x0 + 0.65 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
            ],
            dtype=np.float32,
        ),
        det_score=0.99,
        embedding=np.ones(512, dtype=np.float32) * area_boost,
        raw=object(),
    )


class _FakeEngine(SwapEngine):
    name = "fake"

    def load(self, cache_dir: Path, *, device: str = "cuda") -> None:
        return None

    def swap(self, target_bgr, target_face, source_face, *, paste_back: bool = True):
        out = target_bgr.copy()
        x0, y0, x1, y1 = [int(v) for v in target_face.bbox]
        # Paint a distinct color into the face box (simulates identity change)
        out[y0:y1, x0:x1] = (0, 0, 255)
        h, w = out.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        return SwapEngineResult(
            image_bgr=out,
            swapped_face_bgr=out[y0:y1, x0:x1].copy(),
            paste_mask=mask,
            meta={"engine": "fake"},
        )


class _FakeDetector:
    def __init__(self, body_faces, source_face):
        self._body = body_faces
        self._source = source_face

    def load(self, cache_dir, *, device="cuda"):
        return None

    @property
    def ready(self):
        return True

    def detect(self, image_bgr):
        return list(self._body)

    def get_source_identity(self, source_bgr, *, policy="largest", index=0):
        return self._source


def test_engine_registry_has_inswapper():
    assert "inswapper" in ENGINE_REGISTRY
    eng = create_engine("inswapper")
    assert eng.name == "inswapper"


def test_select_face_policies():
    faces = [
        _face((10, 10, 50, 50)),
        _face((200, 10, 280, 100)),  # larger + righter
        _face((5, 5, 20, 20)),
    ]
    assert select_face(faces, policy="largest") is faces[1]
    assert select_face(faces, policy="rightmost").center_x == faces[1].center_x
    assert select_face(faces, policy="leftmost").center_x == faces[2].center_x
    assert select_face(faces, policy="index", index=2) is faces[2]


def test_soft_blend_preserves_outside_mask():
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    swapped = np.zeros_like(orig)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    out = soft_blend(orig, swapped, mask, strength=1.0)
    outside = mask == 0
    assert np.array_equal(out[outside], orig[outside])
    assert not np.array_equal(out[20:40, 20:40], orig[20:40, 20:40])


def test_pipeline_group_photo_only_selected_changes(tmp_path: Path):
    # Two faces on a solid background
    body = np.full((120, 200, 3), 40, dtype=np.uint8)
    body[20:60, 20:60] = (90, 90, 90)  # face A
    body[20:70, 130:180] = (110, 110, 110)  # face B (larger)
    face_a = _face((20, 20, 60, 60))
    face_b = _face((130, 20, 180, 70))
    source = _face((0, 0, 40, 40))

    pipe = InSwapPipeline(
        cache_dir=tmp_path / "cache",
        engine=_FakeEngine(),
        restorer="none",
        device="cpu",
        color_match_strength=0.0,
        reblend_after_engine=True,
    )
    pipe.detector = _FakeDetector([face_a, face_b], source)  # type: ignore[assignment]
    pipe._loaded = True

    body_pil = Image.fromarray(body[:, :, ::-1])  # BGR→RGB for PIL
    face_pil = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8))

    result = pipe.run(
        body_pil,
        face_pil,
        out_dir=tmp_path / "run",
        face_policy="largest",
        save_intermediates=True,
    )
    assert result.meta["faces_detected"] == 2
    assert result.meta["full_scene_regenerated"] is False
    assert (tmp_path / "run" / "result.png").is_file()
    assert (tmp_path / "run" / "inswapper_only.png").is_file()
    assert (tmp_path / "run" / "intermediates" / "01_detection.png").is_file()
    assert (tmp_path / "run" / "intermediates" / "07_inswapper_blended.png").is_file()
    assert (tmp_path / "run" / "intermediates" / "15_difference.png").is_file()
    assert result.meta.get("krea2_refine_requested") is False

    out = np.asarray(result.image)[:, :, ::-1]  # RGB→BGR
    # Face B region should differ; far background should match original.
    assert not np.array_equal(out[30:50, 140:170], body[30:50, 140:170])
    assert np.array_equal(out[100:110, 90:100], body[100:110, 90:100])


def test_hybrid_krea2_refine_mock(tmp_path: Path):
    """InSwapper + mock Krea2 refine writes head-crop intermediates."""
    body = np.full((120, 160, 3), 40, dtype=np.uint8)
    body[30:80, 50:110] = (100, 100, 100)
    selected = _face((50, 30, 110, 80))
    source = _face((0, 0, 40, 40))

    pipe = InSwapPipeline(
        cache_dir=tmp_path / "cache",
        engine=_FakeEngine(),
        restorer="none",
        device="cpu",
        color_match_strength=0.0,
        krea2_refine=True,
        krea2_force_mock=True,
    )
    pipe.detector = _FakeDetector([selected], source)  # type: ignore[assignment]
    pipe._loaded = True
    # Refiner load still needed for mock Krea2
    pipe.krea2_refiner.load()

    body_pil = Image.fromarray(body[:, :, ::-1])
    face_pil = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8))
    result = pipe.run(
        body_pil,
        face_pil,
        out_dir=tmp_path / "hybrid",
        save_intermediates=True,
        krea2_refine=True,
    )
    assert result.meta.get("krea2_refine_requested") is True
    assert result.meta.get("krea2_refine_applied") is True
    assert result.meta.get("pipeline") == "inswap_hybrid"
    inter = tmp_path / "hybrid" / "intermediates"
    assert (inter / "08_inswapper_result.png").is_file()
    assert (inter / "11_head_crop_for_krea2.png").is_file()
    assert (inter / "12_krea2_refined_head.png").is_file() or (
        inter / "14_final_hybrid.png"
    ).is_file()


def test_hybrid_can_disable_per_run(tmp_path: Path):
    body = np.full((80, 80, 3), 30, dtype=np.uint8)
    body[20:60, 20:60] = 90
    selected = _face((20, 20, 60, 60))
    source = _face((0, 0, 30, 30))
    pipe = InSwapPipeline(
        cache_dir=tmp_path / "cache",
        engine=_FakeEngine(),
        restorer="none",
        device="cpu",
        krea2_refine=True,
        krea2_force_mock=True,
    )
    pipe.detector = _FakeDetector([selected], source)  # type: ignore[assignment]
    pipe._loaded = True
    pipe.krea2_refiner.load()

    body_pil = Image.fromarray(body[:, :, ::-1])
    face_pil = Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8))
    off = pipe.run(
        body_pil,
        face_pil,
        out_dir=tmp_path / "off",
        krea2_refine=False,
        save_intermediates=False,
    )
    assert off.meta.get("krea2_refine_requested") is False
    assert off.meta.get("pipeline") == "inswap"


def test_difference_map_and_overlay():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = a.copy()
    b[8:24, 8:24] = 255
    heat = difference_map(a, b)
    assert heat.shape == a.shape
    faces = [_face((8, 8, 24, 24))]
    overlay = draw_detections(a, faces, selected=faces[0])
    assert overlay.shape == a.shape


def test_face_ellipse_mask_shape():
    f = _face((10, 10, 50, 50))
    m = face_ellipse_mask((80, 80), f)
    assert m.shape == (80, 80)
    assert m.max() == 255
    assert m.min() == 0


def test_create_pipeline_registration():
    from headswap.pipelines import PIPELINES, create_pipeline

    assert "inswapper" in PIPELINES
    pipe = create_pipeline({"pipeline": "inswapper", "device": "cpu", "restorer": "none"})
    assert pipe.name == "inswapper"
