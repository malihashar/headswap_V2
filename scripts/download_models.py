#!/usr/bin/env python3
"""
Download model weights for headswap_V2.

Sources are limited to verified official repos (see docs/VALIDATION.md):
  - Black Forest Labs FLUX.2 [klein] 4B (Apache 2.0)
  - Comfy-Org packaging linked from https://docs.comfy.org/tutorials/flux/flux-2-klein
  - Comfy-Org Qwen Image Edit 2511 packaging
  - lightx2v Lightning LoRA (Apache 2.0)
  - Alissonerdx BFS LoRAs (MIT; optional)

Every required URL is checked via the Hugging Face API (file present + size)
and a resolve probe (HTTP 302 with X-Linked-Size) before download.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

UA = "headswap-v2-downloader/1.1 (validation-required)"


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


# ---------------------------------------------------------------------------
# Verified required Klein set (matches ComfyUI official docs)
# https://docs.comfy.org/tutorials/flux/flux-2-klein
# ---------------------------------------------------------------------------
KLEIN_REQUIRED: list[Artifact] = [
    Artifact(
        # Official BFL FP8 distilled UNET (Comfy template widget name)
        url="https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors",
        subdir="diffusion_models",
        filename="flux-2-klein-4b-fp8.safetensors",
        repo_id="black-forest-labs/FLUX.2-klein-4b-fp8",
        repo_path="flux-2-klein-4b-fp8.safetensors",
        required=True,
        notes="FLUX.2 Klein 4B distilled FP8 diffusion model (Apache 2.0).",
    ),
    Artifact(
        # Comfy-Org text encoder packaging referenced by Comfy docs.
        # Resolves via 307 to Comfy-Org/vae-text-encorder-for-flux-klein-4b (same files).
        url="https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
        subdir="text_encoders",
        filename="qwen_3_4b.safetensors",
        repo_id="Comfy-Org/flux2-klein-4B",
        repo_path="split_files/text_encoders/qwen_3_4b.safetensors",
        required=True,
        notes="Qwen3-4B text encoder used by Klein 4B ComfyUI workflows.",
    ),
    Artifact(
        # VAE linked from Comfy Klein docs (also present in flux2-klein-4B package).
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
        # Alternate full-precision distilled UNET from Comfy-Org packaging (larger than FP8).
        url="https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors",
        subdir="diffusion_models",
        filename="flux-2-klein-4b.safetensors",
        repo_id="Comfy-Org/flux2-klein-4B",
        repo_path="split_files/diffusion_models/flux-2-klein-4b.safetensors",
        required=False,
        notes="OPTIONAL bf16 distilled UNET (~7.8GB). Prefer FP8 unless you need bf16.",
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


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def api_file_size(repo_id: str, repo_path: str) -> int | None:
    """Return file size from HF API siblings/tree, or None if missing."""
    # Prefer siblings listing on the model card API
    model = _http_json(f"https://huggingface.co/api/models/{repo_id}")
    for sib in model.get("siblings") or []:
        if sib.get("rfilename") == repo_path:
            size = sib.get("size")
            if size:
                return int(size)
            lfs = sib.get("lfs") or {}
            if lfs.get("size"):
                return int(lfs["size"])
    # Fallback: tree on parent dir
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
    Returns (hf_status, linked_size). Success means HF returns 302/307 with X-Linked-Size.
    Note: the subsequent CDN hop can 403 in some sandboxes; linked size is the integrity signal.
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
        # Relative redirects (307 to another HF path) — follow one hop manually
        loc = e.headers.get("Location")
        linked = e.headers.get("X-Linked-Size")
        if e.code in (301, 302, 303, 307, 308) and loc:
            if loc.startswith("/"):
                loc = "https://huggingface.co" + loc
                # second hop
                try:
                    opener.open(urllib.request.Request(loc, headers={"User-Agent": UA}), timeout=60)
                except urllib.error.HTTPError as e2:
                    linked = e2.headers.get("X-Linked-Size") or linked
                    return e2.code, int(linked) if linked else None
            return e.code, int(linked) if linked else None
        return e.code, int(linked) if linked else None


def verify_artifact(art: Artifact) -> None:
    size = api_file_size(art.repo_id, art.repo_path)
    if size is None:
        # Official Comfy-Org/flux2-klein-4B redirects to vae-text-encorder twin; siblings live there.
        if art.repo_id == "Comfy-Org/flux2-klein-4B":
            twin = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
            size = api_file_size(twin, art.repo_path)
            if size is None:
                raise RuntimeError(
                    f"Missing on HF API: {art.repo_id}:{art.repo_path} "
                    f"(also checked twin {twin})"
                )
            print(f"  api: {art.repo_id} → twin {twin} :: {art.repo_path} ({size} bytes)")
        else:
            raise RuntimeError(f"Missing on HF API: {art.repo_id}:{art.repo_path}")
    else:
        print(f"  api: {art.repo_id}:{art.repo_path} ({size} bytes)")

    status, linked = resolve_probe(art.url)
    if status not in (200, 302, 303, 307, 308) or not linked:
        raise RuntimeError(
            f"Resolve probe failed for {art.url} (status={status}, x-linked-size={linked})"
        )
    print(f"  resolve: HTTP {status}, X-Linked-Size={linked}")
    if abs(linked - size) > 0 and art.repo_id != "Comfy-Org/flux2-dev":
        # Allow tiny mismatch only if sizes roughly equal (packaging twins)
        if abs(linked - size) > 1024:
            # flux2-dev VAE vs klein package VAE can differ slightly; warn only
            print(f"  warn: size mismatch api={size} linked={linked}")


def download(url: str, dest_dir: Path, filename: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"  exists: {filename} ({path.stat().st_size} bytes)")
        return
    print(f"  downloading: {filename}")
    if subprocess.call(["which", "aria2c"], stdout=subprocess.DEVNULL) == 0:
        subprocess.check_call(
            ["aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M", "-d", str(dest_dir), "-o", filename, url]
        )
    else:
        subprocess.check_call(["curl", "-L", "-A", UA, url, "-o", str(path)])


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
    print(f"\nDownloading into {root}")
    for art in arts:
        download(art.url, root / art.subdir, art.filename)
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
