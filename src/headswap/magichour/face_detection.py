"""Magic Hour face-detection API client.

POST /v1/face-detection  — create a detection job
GET  /v1/face-detection/{id} — poll / retrieve faces[]

Auth: Bearer token from MAGIC_HOUR_API_KEY (never hardcode).

LIMITATION: Magic Hour returns cropped face images (path/url) only — no
bounding boxes or landmarks. It cannot fully replace the current InsightFace /
OpenCV detector for crop_stitch / full_frame geometry without additional work
(e.g. re-detecting each crop in the scene). Use face_detection_backend:
\"magic_hour\" for API parity / face-crop listing; geometry still needs the
current detector unless a follow-up maps MH crops back to scene boxes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from headswap.config import load_project_dotenv

DEFAULT_BASE_URL = "https://api.magichour.ai"
API_KEY_ENV = "MAGIC_HOUR_API_KEY"


class MagicHourFaceDetectionError(RuntimeError):
    """Raised when create/poll/upload fails or the job errors out."""


@dataclass
class MagicHourDetectedFace:
    path: str
    url: str


@dataclass
class MagicHourFaceDetectionResult:
    id: str
    status: str
    credits_charged: int = 0
    faces: list[MagicHourDetectedFace] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def face_count(self) -> int:
        return len(self.faces)


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve API key from arg, process env, or project .env (no hardcoding)."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    load_project_dotenv()
    key = (os.environ.get(API_KEY_ENV) or "").strip()
    if not key:
        raise MagicHourFaceDetectionError(
            f"{API_KEY_ENV} is not set. Copy .env.example to .env and paste your key."
        )
    return key


class MagicHourFaceDetectionClient:
    """Thin wrapper around Magic Hour face-detection create + get."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def create(
        self,
        target_file_path: str,
        *,
        confidence_score: float | None = 0.5,
    ) -> dict[str, Any]:
        """POST /v1/face-detection — returns {id, credits_charged}."""
        body: dict[str, Any] = {
            "assets": {"target_file_path": str(target_file_path)},
        }
        if confidence_score is not None:
            body["confidence_score"] = float(confidence_score)
        resp = self._session.post(
            f"{self.base_url}/v1/face-detection",
            json=body,
            timeout=self.timeout_s,
        )
        return self._json_or_raise(resp, "create face-detection job")

    def get(self, job_id: str) -> MagicHourFaceDetectionResult:
        """GET /v1/face-detection/{id} — poll/retrieve detection result."""
        resp = self._session.get(
            f"{self.base_url}/v1/face-detection/{job_id}",
            timeout=self.timeout_s,
        )
        data = self._json_or_raise(resp, f"get face-detection {job_id}")
        faces = [
            MagicHourDetectedFace(
                path=str(f.get("path") or ""),
                url=str(f.get("url") or ""),
            )
            for f in (data.get("faces") or [])
            if isinstance(f, dict)
        ]
        return MagicHourFaceDetectionResult(
            id=str(data.get("id") or job_id),
            status=str(data.get("status") or ""),
            credits_charged=int(data.get("credits_charged") or 0),
            faces=faces,
            raw=data,
        )

    def wait(
        self,
        job_id: str,
        *,
        poll_interval_s: float = 1.0,
        max_wait_s: float = 120.0,
    ) -> MagicHourFaceDetectionResult:
        """Poll GET until status is complete or error."""
        deadline = time.monotonic() + float(max_wait_s)
        last: MagicHourFaceDetectionResult | None = None
        while time.monotonic() < deadline:
            last = self.get(job_id)
            status = (last.status or "").lower()
            if status == "complete":
                return last
            if status == "error":
                raise MagicHourFaceDetectionError(
                    f"Magic Hour face-detection job {job_id} failed: {last.raw}"
                )
            time.sleep(max(0.1, float(poll_interval_s)))
        raise MagicHourFaceDetectionError(
            f"Magic Hour face-detection job {job_id} timed out after {max_wait_s}s "
            f"(last status={None if last is None else last.status})"
        )

    def detect(
        self,
        target_file_path: str,
        *,
        confidence_score: float | None = 0.5,
        poll_interval_s: float = 1.0,
        max_wait_s: float = 120.0,
    ) -> MagicHourFaceDetectionResult:
        """Create a job and wait for completion."""
        created = self.create(
            target_file_path, confidence_score=confidence_score
        )
        job_id = str(created.get("id") or "").strip()
        if not job_id:
            raise MagicHourFaceDetectionError(
                f"face-detection create returned no id: {created}"
            )
        return self.wait(
            job_id, poll_interval_s=poll_interval_s, max_wait_s=max_wait_s
        )

    def upload_image(
        self,
        image_path: str | Path,
        *,
        extension: str | None = None,
    ) -> str:
        """Upload a local image via POST /v1/files/upload-urls + PUT.

        Returns Magic Hour ``file_path`` suitable for ``target_file_path``.
        """
        path = Path(image_path)
        if not path.is_file():
            raise MagicHourFaceDetectionError(f"Image not found: {path}")
        ext = (extension or path.suffix.lstrip(".") or "png").lower()
        resp = self._session.post(
            f"{self.base_url}/v1/files/upload-urls",
            json={"items": [{"type": "image", "extension": ext}]},
            timeout=self.timeout_s,
        )
        data = self._json_or_raise(resp, "generate upload urls")
        items = data.get("items") or []
        if not items:
            raise MagicHourFaceDetectionError(f"upload-urls returned no items: {data}")
        item = items[0]
        upload_url = str(item.get("upload_url") or "")
        file_path = str(item.get("file_path") or "")
        if not upload_url or not file_path:
            raise MagicHourFaceDetectionError(f"upload-urls missing fields: {item}")
        with path.open("rb") as f:
            put = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": f"image/{ext}"},
                timeout=max(self.timeout_s, 120.0),
            )
        if put.status_code >= 400:
            raise MagicHourFaceDetectionError(
                f"upload PUT failed HTTP {put.status_code}: {put.text[:500]}"
            )
        return file_path

    def detect_local_image(
        self,
        image_path: str | Path,
        *,
        confidence_score: float | None = 0.5,
        poll_interval_s: float = 1.0,
        max_wait_s: float = 120.0,
    ) -> MagicHourFaceDetectionResult:
        """Upload a local image, then run face detection to completion."""
        file_path = self.upload_image(image_path)
        return self.detect(
            file_path,
            confidence_score=confidence_score,
            poll_interval_s=poll_interval_s,
            max_wait_s=max_wait_s,
        )

    @staticmethod
    def _json_or_raise(resp: requests.Response, action: str) -> dict[str, Any]:
        if resp.status_code >= 400:
            detail = resp.text[:800]
            try:
                detail = str(resp.json())
            except Exception:
                pass
            raise MagicHourFaceDetectionError(
                f"Magic Hour {action} failed HTTP {resp.status_code}: {detail}"
            )
        try:
            data = resp.json()
        except Exception as e:
            raise MagicHourFaceDetectionError(
                f"Magic Hour {action} returned non-JSON: {e}"
            ) from e
        if not isinstance(data, dict):
            raise MagicHourFaceDetectionError(
                f"Magic Hour {action} returned non-object JSON: {type(data)}"
            )
        return data


def detect_faces_magic_hour(
    target_file_path: str | None = None,
    *,
    local_image_path: str | Path | None = None,
    confidence_score: float | None = 0.5,
    api_key: str | None = None,
    max_wait_s: float = 120.0,
) -> MagicHourFaceDetectionResult:
    """Convenience entry for the optional ``face_detection_backend: magic_hour``.

    Provide either a Magic Hour ``target_file_path`` (URL or uploaded file_path)
    or a ``local_image_path`` (uploaded automatically).
    """
    client = MagicHourFaceDetectionClient(api_key=api_key)
    if local_image_path is not None:
        return client.detect_local_image(
            local_image_path,
            confidence_score=confidence_score,
            max_wait_s=max_wait_s,
        )
    if not target_file_path:
        raise MagicHourFaceDetectionError(
            "Need target_file_path or local_image_path for Magic Hour detection"
        )
    return client.detect(
        target_file_path,
        confidence_score=confidence_score,
        max_wait_s=max_wait_s,
    )
