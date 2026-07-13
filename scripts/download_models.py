#!/usr/bin/env python3
"""
Download model weights for headswap_V2 via huggingface_hub (no curl/aria2).

Sources are limited to verified official repos (see docs/VALIDATION.md):
  - Black Forest Labs FLUX.2 [klein] 4B (Apache 2.0)
  - Comfy-Org packaging linked from https://docs.comfy.org/tutorials/flux/flux-2-klein
  - Comfy-Org Qwen Image Edit 2511 packaging
  - lightx2v Lightning LoRA (Apache 2.0)
  - Alissonerdx BFS LoRAs (MIT; optional)

Files are cached by huggingface_hub, then placed under:
  {COMFYUI_PATH}/models/{diffusion_models|text_encoders|vae|loras}/

Every required artifact is checked via the Hugging Face API (file present + size)
and a resolve probe (HTTP 302 with X-Linked-Size) before download.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

UA = "headswap-v2-downloader/1.2 (huggingface_hub)"

# Comfy-Org/flux2-klein-4B resolve redirects to this twin (same split_files blobs).
COMFY_KLEIN_4B_TWIN = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"


@dataclass(frozen=True)
class Artifact:
    """One downloadable weight file."""

    url: str
    subdir: str
    filename: str
    repo_id: str
    repo_path: str
    required: bool
    notes: str
    # Optional override when API siblings live on a redirect twin repo.
    download_repo_id: str | None = None


# ---------------------------------------------------------------------------
# Verified required Klein set (matches ComfyUI official docs)
# https://docs.comfy.org/tutorials/flux/flux-2-klein
# ---------------------------------------------------------------------------
KLEIN_REQUIRED: list[Artifact] = [
    Artifact(
        url="https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors",
        subdir="diffusion_models",
        filename="flux-2-klein-4b-fp8.safetensors",
        repo_id="black-forest-labs/FLUX.2-klein-4b-fp8",
        repo_path="flux-2-klein-4b-fp8.safetensors",
        required=True,
        notes="FLUX.2 Klein 4B distilled FP8 diffusion model (Apache 2.0).",
    ),
    Artifact(
        url="https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
        subdir="text_encoders",
        filename="qwen_3_4b.safetensors",
        repo_id="Comfy-Org/flux2-klein-4B",
        repo_path="split_files/text_encoders/qwen_3_4b.safetensors",
        required=True,
        notes="Qwen3-4B text encoder used by Klein 4B ComfyUI workflows.",
        download_repo_id=COMFY_KLEIN_4B_TWIN,
    ),
    Artifact(
        url="https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        subdir="vae",
        filename="flux2-vae.safetensors",
        repo_id="Comfy-Org/flux2-dev",
        repo_path="split_files/vae/flux2-vae.safetensors",
        required=True,
        notes="Shared FLUX.2 VAE used by Klein ComfyUI templates.",
    ),
]

KLEIN_OPTIONAL: list[Artifact] = [
    Artifact(
        url="https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1.1_optional_flux-klein_4b.safetensors",
        subdir="loras",
        filename="bfs_head_v1.1_optional_flux-klein_4b.safetensors",
        repo_id="Alissonerdx/BFS-Best-Face-Swap",
        repo_path="bfs_head_v1.1_optional_flux-klein_4b.safetensors",
        required=False,
        notes="OPTIONAL community head-swap LoRA (MIT). Off by default in configs/klein4b.yaml.",
    ),
    Artifact(
        url="https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors",
        subdir="diffusion_models",
        filename="flux-2-klein-4b.safetensors",
        repo_id="Comfy-Org/flux2-klein-4B",
        repo_path="split_files/diffusion_models/flux-2-klein-4b.safetensors",
        required=False,
        notes="OPTIONAL bf16 distilled UNET (~7.8GB). Prefer FP8 unless you need bf16.",
        download_repo_id=COMFY_KLEIN_4B_TWIN,
    ),
]

QWEN_REQUIRED: list[Artifact] = [
    Artifact(
        url="https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        subdir="diffusion_models",
        filename="qwen_image_edit_2511_fp8mixed.safetensors",
        repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI",
        repo_path="split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        required=True,
        notes="Qwen Image Edit 2511 ComfyUI split diffusion model.",
    ),
    Artifact(
        url="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        subdir="vae",
        filename="qwen_image_vae.safetensors",
        repo_id="Comfy-Org/Qwen-Image_ComfyUI",
        repo_path="split_files/vae/qwen_image_vae.safetensors",
        required=True,
        notes="Qwen Image VAE.",
    ),
    Artifact(
        url="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        subdir="text_encoders",
        filename="qwen_2.5_vl_7b_fp8_scaled.safetensors",
        repo_id="Comfy-Org/Qwen-Image_ComfyUI",
        repo_path="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        required=True,
        notes="Qwen2.5-VL 7B text encoder (fp8 scaled).",
    ),
    Artifact(
        url="https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        subdir="loras",
        filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        repo_path="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        required=True,
        notes="Lightning 4-step LoRA for Qwen Image Edit 2511 (Apache 2.0).",
    ),
    Artifact(
        url="https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
        subdir="loras",
        filename="bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
        repo_id="Alissonerdx/BFS-Best-Face-Swap",
        repo_path="bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
        required=True,
        notes="BFS Head V5 LoRA for Qwen 2511 (MIT).",
    ),
]


def _require_hub():
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            "  pip install -U huggingface_hub\n"
            f"({exc})"
        ) from exc


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def api_file_size(repo_id: str, repo_path: str) -> int | None:
    """Return file size from HF API siblings/tree, or None if missing."""
    model = _http_json(f"https://huggingface.co/api/models/{repo_id}")
    for sib in model.get("siblings") or []:
        if sib.get("rfilename") == repo_path:
            size = sib.get("size")
            if size:
                return int(size)
            lfs = sib.get("lfs") or {}
            if lfs.get("size"):
                return int(lfs["size"])
    parent = repo_path.rsplit("/", 1)[0] if "/" in repo_path else ""
    tree_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    if parent:
        tree_url += "/" + "/".join(urllib.parse.quote(p) for p in parent.split("/"))
    items = _http_json(tree_url)
    name = repo_path.split("/")[-1]
    for item in items:
        if item.get("path", "").endswith(name) or item.get("path") == repo_path:
            size = item.get("size") or (item.get("lfs") or {}).get("size")
            if size:
                return int(size)
    return None


def resolve_probe(url: str) -> tuple[int, int | None]:
    """
    Probe Hugging Face resolve URL without downloading the full file.
    Success means HF returns 302/307 with X-Linked-Size.
    """

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        opener.open(req, timeout=60)
        return 200, None
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        linked = e.headers.get("X-Linked-Size")
        if e.code in (301, 302, 303, 307, 308) and loc:
            if loc.startswith("/"):
                loc = "https://huggingface.co" + loc
                try:
                    opener.open(urllib.request.Request(loc, headers={"User-Agent": UA}), timeout=60)
                except urllib.error.HTTPError as e2:
                    linked = e2.headers.get("X-Linked-Size") or linked
                    return e2.code, int(linked) if linked else None
            return e.code, int(linked) if linked else None
        return e.code, int(linked) if linked else None


def verify_artifact(art: Artifact) -> None:
    size = api_file_size(art.repo_id, art.repo_path)
    if size is None and art.download_repo_id:
        size = api_file_size(art.download_repo_id, art.repo_path)
        if size is None:
            raise RuntimeError(
                f"Missing on HF API: {art.repo_id}:{art.repo_path} "
                f"(also checked {art.download_repo_id})"
            )
        print(f"  api: {art.repo_id} → {art.download_repo_id} :: {art.repo_path} ({size} bytes)")
    elif size is None:
        raise RuntimeError(f"Missing on HF API: {art.repo_id}:{art.repo_path}")
    else:
        print(f"  api: {art.repo_id}:{art.repo_path} ({size} bytes)")

    status, linked = resolve_probe(art.url)
    if status not in (200, 302, 303, 307, 308) or not linked:
        raise RuntimeError(
            f"Resolve probe failed for {art.url} (status={status}, x-linked-size={linked})"
        )
    print(f"  resolve: HTTP {status}, X-Linked-Size={linked}")
    if linked and size and abs(linked - size) > 1024:
        print(f"  warn: size mismatch api={size} linked={linked}")


def _place_into_comfy(cached_path: Path, dest: Path) -> None:
    """Put a cached hub file into the ComfyUI models layout (prefer link over copy)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.stat().st_size > 1_000_000:
            print(f"  exists: {dest} ({dest.stat().st_size if dest.exists() else 'symlink'} bytes)")
            return
        dest.unlink()

    cached_path = cached_path.resolve()
    try:
        os.link(cached_path, dest)
        print(f"  hardlinked → {dest}")
        return
    except OSError:
        pass
    try:
        dest.symlink_to(cached_path)
        print(f"  symlinked → {dest}")
        return
    except OSError:
        pass
    shutil.copy2(cached_path, dest)
    print(f"  copied → {dest}")


def download_artifact(art: Artifact, models_root: Path, revision: str = "main") -> Path:
    """
    Download via huggingface_hub into the HF cache, then place under ComfyUI models/.
    Resumable downloads / local caching are handled by huggingface_hub.
    """
    from huggingface_hub import hf_hub_download

    dest = models_root / art.subdir / art.filename
    if dest.exists() and not dest.is_symlink() and dest.stat().st_size > 1_000_000:
        print(f"  exists: {dest.name} ({dest.stat().st_size} bytes)")
        return dest

    repo_candidates = [art.download_repo_id or art.repo_id]
    if art.download_repo_id and art.repo_id not in repo_candidates:
        repo_candidates.append(art.repo_id)
    elif not art.download_repo_id and art.repo_id == "Comfy-Org/flux2-klein-4B":
        repo_candidates.append(COMFY_KLEIN_4B_TWIN)

    last_err: Exception | None = None
    cached: str | None = None
    for repo_id in repo_candidates:
        print(f"  hf_hub_download: {repo_id} :: {art.repo_path}")
        kwargs = {
            "repo_id": repo_id,
            "filename": art.repo_path,
            "revision": revision,
            "repo_type": "model",
        }
        # resume_download is always-on in recent huggingface_hub; keep for older installs.
        try:
            cached = hf_hub_download(**kwargs, resume_download=True)
        except TypeError:
            # Newer hub removed the kwarg — download is still resumable/cached.
            try:
                cached = hf_hub_download(**kwargs)
            except Exception as exc:
                last_err = exc
                print(f"  retry next repo after: {exc}")
                continue
        except Exception as exc:
            last_err = exc
            print(f"  retry next repo after: {exc}")
            continue
        break

    if not cached:
        raise RuntimeError(f"Failed to download {art.filename}: {last_err}")

    _place_into_comfy(Path(cached), dest)
    return dest


def select_artifacts(kind: str, include_optional: bool) -> list[Artifact]:
    arts: list[Artifact] = []
    if kind in ("klein", "all"):
        arts.extend(KLEIN_REQUIRED)
        if include_optional:
            arts.extend(KLEIN_OPTIONAL)
    if kind in ("qwen", "all"):
        arts.extend(QWEN_REQUIRED)
    return arts


def main() -> int:
    _require_hub()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy", default=os.environ.get("COMFYUI_PATH", "/content/ComfyUI"))
    ap.add_argument("--set", choices=["klein", "qwen", "all"], default="klein")
    ap.add_argument(
        "--include-optional",
        action="store_true",
        help="Also download optional Klein artifacts (BFS LoRA, bf16 UNET).",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate URLs via HF API + resolve probe; do not download.",
    )
    ap.add_argument(
        "--revision",
        default="main",
        help="Hugging Face revision/branch/tag (default: main).",
    )
    args = ap.parse_args()

    arts = select_artifacts(args.set, include_optional=args.include_optional)
    print(f"Validating {len(arts)} artifact(s) for set={args.set} optional={args.include_optional}")
    for art in arts:
        tag = "REQUIRED" if art.required else "OPTIONAL"
        print(f"\n[{tag}] {art.filename}\n  {art.notes}")
        verify_artifact(art)

    if args.verify_only:
        print("\nAll selected URLs verified.")
        return 0

    root = Path(args.comfy) / "models"
    print(f"\nDownloading into {root} (via huggingface_hub cache)")
    for art in arts:
        print(f"\n↓ {art.filename}")
        download_artifact(art, root, revision=args.revision)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
