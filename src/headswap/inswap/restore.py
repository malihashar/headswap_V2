"""Optional face-only restoration (GFPGAN / CodeFormer).

Restoration is restricted to the selected face region so the rest of the
full-resolution image stays unchanged.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from headswap.inswap.blend import face_ellipse_mask, soft_blend
from headswap.inswap.engines.base import DetectedFace

RestorerName = Literal["none", "gfpgan", "codeformer"]

GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
CODEFORMER_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"[restore] downloading {dest.name}…", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


class FaceRestorer:
    """Lazy GFPGAN / CodeFormer wrapper with face-local paste-back."""

    def __init__(self, name: RestorerName = "none") -> None:
        self.name: RestorerName = name if name in {"none", "gfpgan", "codeformer"} else "none"
        self._impl: Any = None
        self._cache_dir: Path | None = None
        self._device = "cuda"

    def load(self, cache_dir: Path, *, device: str = "cuda") -> None:
        self._cache_dir = Path(cache_dir)
        self._device = device
        if self.name == "none":
            return
        if self.name == "gfpgan":
            self._load_gfpgan()
        elif self.name == "codeformer":
            self._load_codeformer()

    def _load_gfpgan(self) -> None:
        assert self._cache_dir is not None
        weight = _download(GFPGAN_URL, self._cache_dir / "models" / "GFPGANv1.4.pth")
        try:
            from gfpgan import GFPGANer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "gfpgan is not installed. pip install gfpgan basicsr facexlib"
            ) from exc
        self._impl = GFPGANer(
            model_path=str(weight),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        print("[restore] GFPGAN ready", flush=True)

    def _load_codeformer(self) -> None:
        assert self._cache_dir is not None
        weight = _download(
            CODEFORMER_URL, self._cache_dir / "models" / "codeformer.pth"
        )
        # Prefer torchvision-free CodeFormer via basicsr if available.
        try:
            from basicsr.utils.download_util import load_file_from_url  # noqa: F401
            from torchvision.transforms.functional import normalize  # noqa: F401
        except ImportError:
            pass
        try:
            # Official CodeFormer helper ships with many Colab installs as
            # `codeformer` or via local clone; fall back to GFPGAN-style API.
            from codeformer.app import inference_app  # type: ignore

            self._impl = ("codeformer_app", inference_app, weight)
            print("[restore] CodeFormer (app) ready", flush=True)
            return
        except Exception:
            pass
        # Fallback: reuse GFPGAN path if CodeFormer package missing — caller
        # can still set restorer=none. Keep weight downloaded for later.
        print(
            "[restore] CodeFormer package not found; weight downloaded. "
            "Install CodeFormer or set RESTORE=gfpgan / none.",
            flush=True,
        )
        self.name = "none"
        self._impl = None

    def enhance(
        self,
        image_bgr: np.ndarray,
        face: DetectedFace,
        *,
        fidelity: float = 0.5,
    ) -> np.ndarray:
        if self.name == "none" or self._impl is None:
            return image_bgr

        if self.name == "gfpgan":
            # only_center_face=False but we re-mask to the selected face only.
            _, _, restored = self._impl.enhance(
                image_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=float(fidelity),
            )
            restored = np.asarray(restored)
        elif self.name == "codeformer" and isinstance(self._impl, tuple):
            # codeformer app expects RGB path-like usage; operate in-memory via temp.
            import tempfile
            from pathlib import Path as P

            _, app_fn, _weight = self._impl
            with tempfile.TemporaryDirectory() as td:
                inp = P(td) / "in.png"
                outp = P(td) / "out.png"
                cv2.imwrite(str(inp), image_bgr)
                app_fn(
                    image=str(inp),
                    background_enhance=False,
                    face_upsample=False,
                    upscale=1,
                    codeformer_fidelity=float(fidelity),
                    output_path=str(outp),
                )
                restored = cv2.imread(str(outp))
                if restored is None:
                    return image_bgr
        else:
            return image_bgr

        if restored.shape[:2] != image_bgr.shape[:2]:
            restored = cv2.resize(
                restored,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        mask = face_ellipse_mask(image_bgr.shape[:2], face, expand=0.20, blur_frac=0.15)
        return soft_blend(image_bgr, restored, mask, strength=1.0)
