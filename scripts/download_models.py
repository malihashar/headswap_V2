#!/usr/bin/env python3
"""
Download model weights for headswap_V2 (Colab-robust).

Architecture:
  1. Never write partial files to Google Drive.
  2. Download into local staging (/content/_hf_dl_staging).
  3. Verify size against scripts/models.json (or refreshed HF API sizes).
  4. Promote the complete file into --store-dir (Drive).
  5. Delete staging copy + hub-cache leftovers for that file (staging keeps in-progress only).
  6. Symlink into {COMFYUI_PATH}/models/<subdir>/.

Backend ladder (auto):
  Hub HTTP (HF_HUB_DISABLE_XET=1) x2 → aria2 resume x2 → fail

A stall watchdog kills any backend that transfers <1 MiB in --stall-window-sec
(default 5 minutes) and advances the ladder.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

UA = "headswap-v2-downloader/2.0 (colab-staging)"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "models.json"
DEFAULT_DRIVE_STORE = Path("/content/drive/MyDrive/headswap_V2/models")
DEFAULT_STAGING = Path("/content/_hf_dl_staging")


class StallError(RuntimeError):
    """Raised when a download backend transfers too few bytes in the stall window."""


@dataclass(frozen=True)
class Artifact:
    filename: str
    size: int
    path: str  # ComfyUI models subdir
    repo_id: str
    repo_path: str
    url: str
    required: bool
    set_name: str
    download_repo_id: str | None = None


def _on_colab() -> bool:
    return Path("/content").exists()


def configure_hub_env(*, disable_xet: bool) -> None:
    """Must run before importing huggingface_hub."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("HF_HUB_DISABLE_XET=1 (classic HTTP path; avoids Colab Xet stalls)")
    else:
        print("Xet left enabled (HF_HUB_DISABLE_XET not set)")


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def api_file_size(repo_id: str, repo_path: str) -> int | None:
    """Return file size from HF API siblings/tree, or None if missing."""
    try:
        model = _http_json(f"https://huggingface.co/api/models/{repo_id}")
    except Exception:
        model = {}
    for sib in model.get("siblings") or []:
        if sib.get("rfilename") == repo_path:
            size = sib.get("size") or (sib.get("lfs") or {}).get("size")
            if size:
                return int(size)
    parent = repo_path.rsplit("/", 1)[0] if "/" in repo_path else ""
    tree_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    if parent:
        tree_url += "/" + "/".join(urllib.parse.quote(p) for p in parent.split("/"))
    try:
        items = _http_json(tree_url)
    except Exception:
        return None
    name = repo_path.split("/")[-1]
    for item in items:
        if item.get("path") == repo_path or item.get("path", "").endswith(name):
            size = item.get("size") or (item.get("lfs") or {}).get("size")
            if size:
                return int(size)
    return None


def resolve_probe(url: str) -> tuple[int, int | None]:
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


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def artifact_from_manifest(filename: str, entry: dict[str, Any]) -> Artifact:
    return Artifact(
        filename=filename,
        size=int(entry["size"]),
        path=str(entry["path"]),
        repo_id=str(entry["repo_id"]),
        repo_path=str(entry["repo_path"]),
        url=str(entry["url"]),
        required=bool(entry.get("required", True)),
        set_name=str(entry.get("set", "all")),
        download_repo_id=entry.get("download_repo_id") or None,
    )


def select_artifacts(
    manifest: dict[str, Any], kind: str, include_optional: bool
) -> list[Artifact]:
    out: list[Artifact] = []
    for filename, entry in manifest.items():
        art = artifact_from_manifest(filename, entry)
        if kind != "all" and art.set_name != kind:
            continue
        if not art.required and not include_optional:
            continue
        out.append(art)
    return out


def refresh_manifest_sizes(manifest: dict[str, Any]) -> dict[str, Any]:
    updated = dict(manifest)
    for filename, entry in updated.items():
        repo = entry.get("download_repo_id") or entry["repo_id"]
        size = api_file_size(repo, entry["repo_path"])
        if size is None and entry.get("download_repo_id"):
            size = api_file_size(entry["repo_id"], entry["repo_path"])
        if size is None:
            print(f"  warn: could not refresh size for {filename}")
            continue
        if int(entry.get("size") or 0) != size:
            print(f"  updated {filename}: {entry.get('size')} → {size}")
        entry = dict(entry)
        entry["size"] = size
        updated[filename] = entry
    return updated


def verify_artifact(art: Artifact) -> None:
    size = api_file_size(art.repo_id, art.repo_path)
    if size is None and art.download_repo_id:
        size = api_file_size(art.download_repo_id, art.repo_path)
    if size is None:
        raise RuntimeError(f"Missing on HF API: {art.repo_id}:{art.repo_path}")
    print(f"  api: {art.repo_id}:{art.repo_path} ({size} bytes)")
    if size != art.size:
        print(f"  warn: manifest size {art.size} != api size {size}")
    status, linked = resolve_probe(art.url)
    if status not in (200, 302, 303, 307, 308) or not linked:
        raise RuntimeError(
            f"Resolve probe failed for {art.url} (status={status}, x-linked-size={linked})"
        )
    print(f"  resolve: HTTP {status}, X-Linked-Size={linked}")


def store_path_for(store_dir: Path, art: Artifact) -> Path:
    return store_dir / art.path / art.filename


def comfy_path_for(comfy: Path, art: Artifact) -> Path:
    return comfy / "models" / art.path / art.filename


def is_complete(path: Path, expected_size: int) -> bool:
    try:
        if not path.exists():
            return False
        return path.stat().st_size == expected_size
    except OSError:
        return False


def wait_until_complete(
    path: Path, expected_size: int, *, attempts: int = 10, delay_sec: float = 1.0
) -> bool:
    """Drive/FUSE can lag on size visibility right after promote — retry briefly."""
    for i in range(attempts):
        if is_complete(path, expected_size):
            return True
        if i + 1 < attempts:
            time.sleep(delay_sec)
    try:
        got = path.stat().st_size if path.exists() else -1
    except OSError:
        got = -1
    print(f"  warn: size check still failing for {path} (got {got}, want {expected_size})")
    return False


def staging_usage_bytes(staging_dir: Path) -> int:
    if not staging_dir.exists():
        return 0
    total = 0
    for p in staging_dir.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _bytes_under(root: Path, patterns: list[str] | None = None) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if patterns and not any(pat in p.name for pat in patterns):
            continue
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def run_with_stall_watch(
    target: Callable[..., None],
    *,
    args: tuple = (),
    kwargs: dict | None = None,
    watch_roots: list[Path],
    watch_name_hints: list[str],
    stall_window_sec: int,
    stall_min_bytes: int,
    poll_sec: float = 5.0,
) -> None:
    """Run target in a child process; kill on stall."""
    kwargs = kwargs or {}
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=target, args=args, kwargs=kwargs)
    proc.start()
    last_bytes = sum(_bytes_under(r, watch_name_hints) for r in watch_roots)
    last_progress = time.monotonic()
    try:
        while proc.is_alive():
            time.sleep(poll_sec)
            cur = sum(_bytes_under(r, watch_name_hints) for r in watch_roots)
            if cur - last_bytes >= stall_min_bytes:
                last_bytes = cur
                last_progress = time.monotonic()
                print(f"  progress: {cur:,} bytes watched")
            elif time.monotonic() - last_progress >= stall_window_sec:
                print(
                    f"  STALL: <{stall_min_bytes} bytes in {stall_window_sec}s "
                    f"(watched={cur:,}); terminating backend"
                )
                proc.terminate()
                proc.join(timeout=15)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
                raise StallError(
                    f"Download stalled (<{stall_min_bytes} B / {stall_window_sec}s)"
                )
        if proc.exitcode not in (0, None):
            raise RuntimeError(f"Download worker exited with code {proc.exitcode}")
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def _hub_worker(
    repo_id: str,
    repo_path: str,
    revision: str,
    hub_cache: str,
    staging_file: str,
    expected_size: int,
) -> None:
    os.environ["HF_HUB_DISABLE_XET"] = os.environ.get("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["HF_HOME"] = str(Path(hub_cache).parent / "hf_home")
    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": repo_id,
        "filename": repo_path,
        "revision": revision,
        "repo_type": "model",
        "cache_dir": hub_cache,
    }
    try:
        cached = hf_hub_download(**kwargs, resume_download=True)
    except TypeError:
        cached = hf_hub_download(**kwargs)
    cached_path = Path(cached)
    if cached_path.stat().st_size != expected_size:
        raise SystemExit(
            f"Hub download size mismatch: got {cached_path.stat().st_size}, "
            f"expected {expected_size}"
        )
    dest = Path(staging_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Copy into flat staging name (still local disk only).
    shutil.copy2(cached_path, dest)
    if dest.stat().st_size != expected_size:
        raise SystemExit("Staging copy size mismatch")


def hub_download_to_staging(
    art: Artifact,
    *,
    staging_dir: Path,
    revision: str,
    stall_window_sec: int,
    stall_min_bytes: int,
) -> Path:
    staging_file = staging_dir / art.filename
    hub_cache = staging_dir / "_hub_cache"
    hub_cache.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    repo_candidates = [art.download_repo_id or art.repo_id]
    if art.download_repo_id and art.repo_id not in repo_candidates:
        repo_candidates.append(art.repo_id)

    last_err: Exception | None = None
    for repo_id in repo_candidates:
        print(f"  hub: {repo_id} :: {art.repo_path}")
        try:
            run_with_stall_watch(
                _hub_worker,
                args=(
                    repo_id,
                    art.repo_path,
                    revision,
                    str(hub_cache),
                    str(staging_file),
                    art.size,
                ),
                watch_roots=[hub_cache, staging_dir],
                watch_name_hints=[art.filename, Path(art.repo_path).name, ".incomplete"],
                stall_window_sec=stall_window_sec,
                stall_min_bytes=stall_min_bytes,
            )
            if is_complete(staging_file, art.size):
                return staging_file
            last_err = RuntimeError("staging file incomplete after hub download")
        except Exception as exc:
            last_err = exc
            print(f"  hub attempt failed: {exc}")
            continue
    raise RuntimeError(f"Hub download failed for {art.filename}: {last_err}")


def _hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    try:
        from huggingface_hub import get_token

        t = get_token()
        return t.strip() if t else None
    except Exception:
        pass
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip() or None
    return None


def aria2_download_to_staging(
    art: Artifact,
    *,
    staging_dir: Path,
    stall_window_sec: int,
    stall_min_bytes: int,
) -> Path:
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c not found; install with: apt-get install -y aria2")

    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_file = staging_dir / art.filename
    token = _hf_token()
    cmd = [
        "aria2c",
        "-c",
        "-x",
        "8",
        "-s",
        "8",
        "-k",
        "1M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "-d",
        str(staging_dir),
        "-o",
        art.filename,
        "--user-agent",
        UA,
    ]
    if token:
        cmd.extend(["--header", f"Authorization: Bearer {token}"])
    cmd.append(art.url)

    print(f"  aria2c: {art.url}")

    # Run aria2 as subprocess with stall watch on staging file growth
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_bytes = staging_file.stat().st_size if staging_file.exists() else 0
    # also count .aria2 control + related
    last_progress = time.monotonic()
    try:
        while True:
            ret = proc.poll()
            cur = 0
            for p in staging_dir.glob(art.filename + "*"):
                if p.is_file():
                    try:
                        cur += p.stat().st_size
                    except OSError:
                        pass
            if cur - last_bytes >= stall_min_bytes:
                last_bytes = cur
                last_progress = time.monotonic()
                print(f"  progress: {cur:,} bytes (aria2)")
            elif time.monotonic() - last_progress >= stall_window_sec:
                print(
                    f"  STALL: <{stall_min_bytes} bytes in {stall_window_sec}s; "
                    "killing aria2c"
                )
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise StallError(
                    f"aria2 stalled (<{stall_min_bytes} B / {stall_window_sec}s)"
                )
            if ret is not None:
                out = proc.stdout.read() if proc.stdout else ""
                if ret != 0:
                    raise RuntimeError(f"aria2c failed ({ret}): {out[-2000:]}")
                break
            time.sleep(5)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    if not is_complete(staging_file, art.size):
        got = staging_file.stat().st_size if staging_file.exists() else 0
        raise RuntimeError(
            f"aria2 completed but size mismatch: got {got}, expected {art.size}"
        )
    return staging_file


def promoting_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".promoting")


def same_filesystem(src: Path, dst_parent: Path) -> bool:
    """True when src and destination parent share a device (rename/move is free)."""
    dst_parent.mkdir(parents=True, exist_ok=True)
    return os.stat(src).st_dev == os.stat(dst_parent).st_dev


def resume_or_clear_promoting(dest: Path, expected_size: int) -> bool:
    """
    Resume an interrupted promote on the store side.

    - complete .promoting (size == expected) → rename to final dest
    - incomplete .promoting (size < expected) → delete it
    Returns True when dest is a complete final file afterward.
    """
    tmp = promoting_path(dest)
    if not tmp.exists() and not tmp.is_symlink():
        return is_complete(dest, expected_size)

    try:
        size = tmp.stat().st_size
    except OSError as exc:
        print(f"  resume: cannot stat {tmp}: {exc} — removing")
        try:
            tmp.unlink()
        except OSError:
            pass
        return is_complete(dest, expected_size)

    if size == expected_size:
        print(f"  resume: complete .promoting ({size} bytes) → {dest.name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            try:
                dest.unlink()
            except OSError as exc:
                print(f"  warn: could not replace existing dest {dest}: {exc}")
        os.replace(tmp, dest)
        return wait_until_complete(dest, expected_size, attempts=5, delay_sec=0.5)

    print(
        f"  resume: incomplete .promoting ({size} bytes < {expected_size}) — deleting {tmp}"
    )
    try:
        tmp.unlink()
    except OSError as exc:
        print(f"  warn: could not delete incomplete .promoting: {exc}")
    return is_complete(dest, expected_size)


def promote_to_store(staging_file: Path, dest: Path, expected_size: int) -> None:
    """
    Move a verified staging file into the persistent store.

    Same filesystem (e.g. Kaggle /kaggle/working staging + models):
      staging → .promoting → final  via os.replace (no extra disk)

    Cross filesystem (e.g. Colab local → Google Drive):
      shutil.copy2(staging → .promoting) then os.replace(.promoting → final)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = promoting_path(dest)

    # Interrupted prior promote: finish or clear before writing again.
    if resume_or_clear_promoting(dest, expected_size):
        print(f"  promote: dest already complete after .promoting resume → {dest}")
        return

    if not staging_file.exists():
        raise RuntimeError(f"staging file missing for promote: {staging_file}")
    if staging_file.stat().st_size != expected_size:
        raise RuntimeError(
            f"staging size mismatch before promote: "
            f"got {staging_file.stat().st_size}, expected {expected_size}"
        )

    try:
        if same_filesystem(staging_file, dest.parent):
            print(
                f"  promote: same filesystem (dev match) — "
                f"rename {staging_file.name} → {tmp.name} → {dest.name}"
            )
            # Atomic on the same device; peeks at ~0 extra space.
            os.replace(staging_file, tmp)
            if tmp.stat().st_size != expected_size:
                raise RuntimeError(
                    f"promote size mismatch after rename: "
                    f"got {tmp.stat().st_size}, expected {expected_size}"
                )
            os.replace(tmp, dest)
        else:
            print(
                f"  promote: cross-filesystem — "
                f"copy2 {staging_file} → {tmp} then rename to {dest.name}"
            )
            if tmp.exists():
                tmp.unlink()
            shutil.copy2(staging_file, tmp)
            if tmp.stat().st_size != expected_size:
                raise RuntimeError(
                    f"promote size mismatch after copy2: "
                    f"got {tmp.stat().st_size}, expected {expected_size}"
                )
            os.replace(tmp, dest)
    except Exception:
        # Leave a complete .promoting for resume; drop incomplete ones.
        if tmp.exists():
            try:
                if tmp.stat().st_size == expected_size:
                    print(f"  promote: left complete .promoting for resume: {tmp}")
                else:
                    print(f"  promote: removing incomplete .promoting after error: {tmp}")
                    tmp.unlink()
            except OSError:
                pass
        raise

    if not wait_until_complete(dest, expected_size, attempts=10, delay_sec=0.5):
        raise RuntimeError(f"promote finished but dest incomplete: {dest}")
    print(f"  promoted → {dest} ({dest.stat().st_size} bytes)")


def _force_unlink(path: Path) -> tuple[bool, int]:
    """Delete a file; return (ok, bytes_freed)."""
    if not path.exists() and not path.is_symlink():
        return False, 0
    try:
        size = path.stat().st_size if path.exists() else 0
    except OSError:
        size = 0
    print(f"  cleanup DELETE file: {path} ({size} bytes)")
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        # Older Python
        try:
            path.unlink()
        except FileNotFoundError:
            return False, 0
    except OSError as exc:
        print(f"  cleanup ERROR unlink {path}: {exc}")
        return False, 0
    if path.exists() or path.is_symlink():
        print(f"  cleanup ERROR still exists after unlink: {path}")
        return False, 0
    print(f"  cleanup OK deleted: {path}")
    return True, size


def _force_rmtree(path: Path) -> tuple[bool, int]:
    """Recursively delete a directory; log every file removed."""
    if not path.exists() and not path.is_symlink():
        print(f"  cleanup skip missing dir: {path}")
        return False, 0
    if path.is_symlink() or path.is_file():
        return _force_unlink(path)
    freed = 0
    print(f"  cleanup DELETE tree: {path}")
    # Delete files deepest-first with explicit logging
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            root_p = Path(root)
            for name in files:
                ok, n = _force_unlink(root_p / name)
                if ok:
                    freed += n
            for name in dirs:
                d = root_p / name
                print(f"  cleanup DELETE dir: {d}")
                try:
                    d.rmdir()
                    print(f"  cleanup OK rmdir: {d}")
                except OSError as exc:
                    print(f"  cleanup ERROR rmdir {d}: {exc}")
        print(f"  cleanup DELETE dir: {path}")
        path.rmdir()
        print(f"  cleanup OK rmdir: {path}")
    except OSError as exc:
        print(f"  cleanup ERROR rmtree {path}: {exc}; falling back to shutil.rmtree")
        shutil.rmtree(path, ignore_errors=False)
    if path.exists():
        print(f"  cleanup ERROR tree still exists: {path}")
        return False, freed
    print(f"  cleanup OK tree gone: {path}")
    return True, freed


def cleanup_staging_for_artifact(
    art: Artifact,
    staging_dir: Path,
    *,
    staging_file: Path | None = None,
) -> None:
    """
    Remove completed staging payloads for one artifact.

    Must run immediately after a successful promote. Deletes:
      - the explicit staging_file path
      - aria2 sidecars (filename*)
      - entire staging/_hub_cache and staging/hf_home
    """
    staging_dir = staging_dir.resolve()
    before = staging_usage_bytes(staging_dir)
    print(f"  cleanup BEGIN staging_dir={staging_dir}")
    print(f"  cleanup staging usage before: {before} bytes ({before / (1024**3):.2f} GiB)")

    targets: list[Path] = []
    if staging_file is not None:
        targets.append(Path(staging_file))
    targets.append(staging_dir / art.filename)
    # aria2 / partial siblings next to the flat file
    targets.extend(sorted(staging_dir.glob(f"{art.filename}*")))

    seen: set[Path] = set()
    freed = 0
    for p in targets:
        try:
            rp = p.resolve() if p.exists() or p.is_symlink() else p
        except OSError:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        if p.is_dir() and not p.is_symlink():
            continue
        ok, n = _force_unlink(p)
        if ok:
            freed += n

    # Always wipe disposable hub caches under THIS staging_dir after promote.
    for tree_name in ("_hub_cache", "hf_home"):
        tree = staging_dir / tree_name
        ok, n = _force_rmtree(tree)
        if ok:
            freed += n

    after = staging_usage_bytes(staging_dir)
    print(f"  cleanup staging usage after:  {after} bytes ({after / (1024**3):.2f} GiB)")
    print(
        f"  cleanup END for {art.filename}: "
        f"freed~{freed / (1024**3):.2f} GiB, delta={(before - after) / (1024**3):.2f} GiB"
    )
    leftover = staging_dir / art.filename
    if leftover.exists() or leftover.is_symlink():
        raise RuntimeError(
            f"cleanup failed: staging file still present: {leftover} "
            f"({leftover.stat().st_size if leftover.exists() else '?'} bytes)"
        )
    hub = staging_dir / "_hub_cache"
    if hub.exists():
        raise RuntimeError(f"cleanup failed: _hub_cache still present: {hub}")


def link_into_comfy(store_file: Path, comfy_file: Path) -> None:
    comfy_file.parent.mkdir(parents=True, exist_ok=True)
    if comfy_file.is_symlink() or comfy_file.exists():
        try:
            if comfy_file.resolve() == store_file.resolve() and is_complete(
                comfy_file, store_file.stat().st_size
            ):
                print(f"  comfy link ok: {comfy_file}")
                return
        except OSError:
            pass
        if comfy_file.is_symlink() or comfy_file.exists():
            comfy_file.unlink()
    try:
        os.symlink(store_file, comfy_file)
        print(f"  symlinked → {comfy_file}")
        return
    except OSError:
        pass
    try:
        os.link(store_file, comfy_file)
        print(f"  hardlinked → {comfy_file}")
        return
    except OSError:
        pass
    shutil.copy2(store_file, comfy_file)
    print(f"  copied → {comfy_file}")


def _finish_promoted_artifact(
    art: Artifact,
    *,
    staging_file: Path | None,
    staging_dir: Path,
    store_file: Path,
    comfy_file: Path,
) -> Path:
    """
    After a successful promote (shutil.copy2 into the store):
      try:    create Comfy symlink/link/copy from the store file
      finally: always delete staging .safetensors + _hub_cache

    Cleanup runs even if symlink creation or other post-promote steps fail.
    """
    print(f"  finish: store={store_file}")
    print(f"  finish: staging_dir={staging_dir.resolve()}")
    if staging_file is not None:
        print(f"  finish: staging_file={staging_file}")

    link_err: Exception | None = None
    try:
        link_into_comfy(store_file, comfy_file)
    except Exception as exc:
        link_err = exc
        print(f"  warn: post-promote link step failed: {exc}")
    finally:
        # Always delete staging copy left by promote's shutil.copy2.
        cleanup_staging_for_artifact(
            art,
            staging_dir,
            staging_file=staging_file or (staging_dir / art.filename),
        )

    if link_err is not None:
        # Staging is already gone; retry link once from store only.
        print("  retrying Comfy link after staging cleanup")
        link_into_comfy(store_file, comfy_file)
    return store_file


def ensure_artifact(
    art: Artifact,
    *,
    store_dir: Path,
    staging_dir: Path,
    comfy: Path,
    backend: str,
    revision: str,
    stall_window_sec: int,
    stall_min_bytes: int,
) -> Path:
    store_file = store_path_for(store_dir, art)
    comfy_file = comfy_path_for(comfy, art)
    staging_dir = Path(staging_dir)

    # Resume interrupted cross-FS promotes left as dest.safetensors.promoting
    if resume_or_clear_promoting(store_file, art.size):
        print(f"  store complete via .promoting resume ({art.size} bytes)")
        return _finish_promoted_artifact(
            art,
            staging_file=staging_dir / art.filename,
            staging_dir=staging_dir,
            store_file=store_file,
            comfy_file=comfy_file,
        )

    if is_complete(store_file, art.size) or wait_until_complete(store_file, art.size, attempts=3):
        print(f"  store complete ({art.size} bytes) — skip download")
        return _finish_promoted_artifact(
            art,
            staging_file=staging_dir / art.filename,
            staging_dir=staging_dir,
            store_file=store_file,
            comfy_file=comfy_file,
        )

    if store_file.exists():
        print(
            f"  store incomplete/wrong size "
            f"(got {store_file.stat().st_size if store_file.exists() else 0}, "
            f"want {art.size}) — re-download via staging"
        )
        try:
            store_file.unlink()
        except OSError:
            pass

    # Ladder: Hub HTTP ×2 → aria2 ×2 (subset when --backend hub|aria2)
    steps: list[tuple[str, Callable[[], Path]]] = []
    hub_n = 2 if backend in ("auto", "hub") else 0
    aria_n = 2 if backend in ("auto", "aria2") else 0
    for _ in range(hub_n):
        steps.append(
            (
                "hub",
                lambda: hub_download_to_staging(
                    art,
                    staging_dir=staging_dir,
                    revision=revision,
                    stall_window_sec=stall_window_sec,
                    stall_min_bytes=stall_min_bytes,
                ),
            )
        )
    for _ in range(aria_n):
        steps.append(
            (
                "aria2",
                lambda: aria2_download_to_staging(
                    art,
                    staging_dir=staging_dir,
                    stall_window_sec=stall_window_sec,
                    stall_min_bytes=stall_min_bytes,
                ),
            )
        )

    last_err: Exception | None = None
    for i, (name, fn) in enumerate(steps, 1):
        # Another attempt (or prior flaky promote) may already have completed the store.
        if resume_or_clear_promoting(store_file, art.size) or is_complete(
            store_file, art.size
        ) or wait_until_complete(store_file, art.size, attempts=3):
            print(f"  store already complete before attempt {i} — cleaning staging")
            return _finish_promoted_artifact(
                art,
                staging_file=staging_dir / art.filename,
                staging_dir=staging_dir,
                store_file=store_file,
                comfy_file=comfy_file,
            )

        print(f"  attempt {i}/{len(steps)} backend={name}")
        staging_file: Path | None = None
        promoted_ok = False
        link_err: Exception | None = None
        try:
            staging_file = fn()
            if not is_complete(staging_file, art.size):
                raise RuntimeError("staging incomplete after backend returned")
            # Same-FS: rename staging→.promoting→final (no 2× disk).
            # Cross-FS: copy2 into .promoting then rename (Colab→Drive).
            promote_to_store(staging_file, store_file, expected_size=art.size)
            if not wait_until_complete(store_file, art.size, attempts=15, delay_sec=1.0):
                raise RuntimeError(
                    f"store incomplete after promote: {store_file}"
                )
            promoted_ok = True
            try:
                link_into_comfy(store_file, comfy_file)
            except Exception as exc:
                link_err = exc
                print(f"  warn: symlink/link after promote failed: {exc}")
        except Exception as exc:
            last_err = exc
            print(f"  attempt {i} failed: {exc}")
            if resume_or_clear_promoting(store_file, art.size) or is_complete(
                store_file, art.size
            ) or wait_until_complete(store_file, art.size, attempts=5, delay_sec=1.0):
                promoted_ok = True
            else:
                continue
        finally:
            # Runs after successful promote even if link_into_comfy fails.
            # Same-FS rename may already have removed staging_file; cleanup is idempotent.
            if promoted_ok:
                print("  finally: delete leftover staging after promote")
                cleanup_staging_for_artifact(
                    art,
                    staging_dir,
                    staging_file=staging_file or (staging_dir / art.filename),
                )

        if promoted_ok:
            if link_err is not None:
                print("  retrying Comfy link after staging cleanup")
                link_into_comfy(store_file, comfy_file)
            return store_file

    raise RuntimeError(f"All download attempts failed for {art.filename}: {last_err}")


def default_store_dir(comfy: Path) -> Path:
    env = os.environ.get("HEADSWAP_MODEL_STORE")
    if env:
        return Path(env)
    if Path("/content/drive/MyDrive").exists():
        return DEFAULT_DRIVE_STORE
    # Kaggle: keep store on the same volume as staging (/kaggle/working).
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/models")
    return comfy / "models"


def default_staging_dir() -> Path:
    env = os.environ.get("HEADSWAP_STAGING_DIR")
    if env:
        return Path(env)
    if _on_colab():
        return DEFAULT_STAGING
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/_hf_dl_staging")
    return Path("/tmp/headswap_hf_staging")


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
    ap.add_argument("--revision", default="main")
    ap.add_argument(
        "--store-dir",
        default=None,
        help="Persistent complete-model store (Drive). Partials never go here.",
    )
    ap.add_argument(
        "--staging-dir",
        default=None,
        help="Local staging for downloads (default /content/_hf_dl_staging on Colab).",
    )
    ap.add_argument(
        "--backend",
        choices=["auto", "hub", "aria2"],
        default="auto",
        help="auto: Hub HTTP x2 then aria2 x2 (default).",
    )
    ap.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to models.json manifest.",
    )
    ap.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Update sizes in the manifest from the live HF API, then save.",
    )
    ap.add_argument(
        "--stall-window-sec",
        type=int,
        default=int(os.environ.get("HEADSWAP_STALL_WINDOW_SEC", "300")),
        help="Seconds without enough progress before killing a backend (default 300).",
    )
    ap.add_argument(
        "--stall-min-bytes",
        type=int,
        default=int(os.environ.get("HEADSWAP_STALL_MIN_BYTES", str(1_048_576))),
        help="Minimum bytes required per stall window (default 1 MiB).",
    )
    ap.add_argument(
        "--disable-xet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force HF_HUB_DISABLE_XET (default: on when /content exists).",
    )
    args = ap.parse_args()

    disable_xet = (
        True
        if args.disable_xet is True
        else False
        if args.disable_xet is False
        else (
            os.environ.get("HEADSWAP_DISABLE_XET", "").lower() in {"1", "true", "yes", "on"}
            or _on_colab()
            or os.environ.get("HF_HUB_DISABLE_XET") == "1"
        )
    )
    # Default Colab: disable Xet. Non-Colab: disable unless explicitly enabled.
    if args.disable_xet is None and not _on_colab():
        if os.environ.get("HF_HUB_DISABLE_XET") is None and not os.environ.get(
            "HEADSWAP_DISABLE_XET"
        ):
            disable_xet = True  # prefer reliable HTTP everywhere for large files
    configure_hub_env(disable_xet=disable_xet)

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    if args.refresh_manifest:
        print(f"Refreshing sizes into {manifest_path}")
        manifest = refresh_manifest_sizes(manifest)
        save_manifest(manifest_path, manifest)
        print("Manifest saved.")

    arts = select_artifacts(manifest, args.set, include_optional=args.include_optional)
    if not arts:
        raise SystemExit(f"No artifacts selected for set={args.set}")

    print(
        f"Selected {len(arts)} artifact(s) set={args.set} optional={args.include_optional} "
        f"backend={args.backend}"
    )
    for art in arts:
        tag = "REQUIRED" if art.required else "OPTIONAL"
        print(f"\n[{tag}] {art.filename} ({art.size} bytes)")
        verify_artifact(art)

    if args.verify_only:
        print("\nAll selected URLs verified.")
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            "  pip install -U huggingface_hub\n"
            f"({exc})"
        ) from exc

    comfy = Path(args.comfy)
    store_dir = Path(args.store_dir) if args.store_dir else default_store_dir(comfy)
    staging_dir = Path(args.staging_dir) if args.staging_dir else default_staging_dir()
    staging_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStore (complete only): {store_dir}")
    print(f"Staging (partials ok): {staging_dir}")
    print(f"ComfyUI:               {comfy}")

    for art in arts:
        print(f"\n↓ {art.filename}")
        ensure_artifact(
            art,
            store_dir=store_dir,
            staging_dir=staging_dir,
            comfy=comfy,
            backend=args.backend,
            revision=args.revision,
            stall_window_sec=args.stall_window_sec,
            stall_min_bytes=args.stall_min_bytes,
        )

    # Final sweep: drop any leftover staging copies of store-complete models
    for art in arts:
        if is_complete(store_path_for(store_dir, art), art.size):
            cleanup_staging_for_artifact(art, staging_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    # Required for spawn watchdog children on some platforms.
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
