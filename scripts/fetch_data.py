#!/usr/bin/env python3
"""
fetch_data.py — Download the external datasets this project needs.

Two datasets are required, both publicly available:

  1. LibriSpeech test-clean  (~346 MB download, ~1.2 GB extracted)
     Clean read speech used as the target speaker and the interferer
     speakers by generate_synthetic_dataset.py.
     Source: https://www.openslr.org/12/

  2. EasyCom Device_ATFs.h5  (~37 MB)
     Measured array transfer functions for the 6-mic AR-glasses array
     (1020 directions on a sphere). Optional: without it,
     generate_synthetic_dataset.py falls back to pyroomacoustics ISM
     simulation. With it, spatial cues are far more realistic.
     Source: https://github.com/facebookresearch/EasyComDataset
     Fetched straight from the git-lfs blob store — no clone, no git needed.

Both downloads are idempotent — an already-complete dataset is detected and
skipped, so re-running this script is cheap and safe.

Usage
-----
    python scripts/fetch_data.py                 # fetch both into ./data
    python scripts/fetch_data.py --only librispeech
    python scripts/fetch_data.py --only atf
    python scripts/fetch_data.py --data-dir /path/to/data
    python scripts/fetch_data.py --force         # re-download even if present

After it finishes it prints the exact generate/eval commands to run next,
with the correct paths already filled in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

# ── Sources ──────────────────────────────────────────────────────────────────

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
# openslr.org/12 publishes the size (346M) but no checksum for the archives,
# so integrity is verified by exact byte count (from HTTP HEAD) plus a
# successful gzip/tar decode, which together catch any truncated or corrupt
# transfer. No MD5 is asserted here because none could be verified against an
# authoritative source.
LIBRISPEECH_SIZE = 346_663_984  # bytes, verified via HTTP HEAD

EASYCOM_REPO = "https://github.com/facebookresearch/EasyComDataset.git"
# Path inside the EasyCom repo. Mirrored verbatim under --data-dir so the
# layout matches what generate_synthetic_dataset.py's docstring references.
ATF_REL_PATH = Path("Calibration/Array Transfer Functions/Device_ATFs.h5")

# The ATF file is stored in git-lfs. We resolve it through the LFS batch API
# and download the blob directly, rather than cloning the repo.
#
# `git clone --filter=blob:none` + `git lfs pull --include=<one file>` looks
# like the obvious approach but is unusably slow here: `git lfs pull` runs
# `git ls-tree -r --full-tree` over the whole tree, and in a blobless clone
# every one of those tree objects has to be lazily re-fetched from the
# server first. On this repo that stalls for tens of minutes while
# transferring far more than the 37 MB we actually want. The batch API needs
# exactly two requests and no local git state at all.
#
# The pointer (oid + size) is committed here because it identifies one
# immutable blob — the oid IS the sha256 of the content, so a successful
# download that matches it is self-verifying.
ATF_LFS_URL = ("https://github.com/facebookresearch/EasyComDataset.git"
               "/info/lfs/objects/batch")
ATF_LFS_OID = "96b7862abd6963e15eccbaa7af1c413c64566d9146f76ed7cad656088ef98a73"
ATF_MIN_SIZE = 38_866_568  # bytes, from the git-lfs pointer


# ── Small helpers ────────────────────────────────────────────────────────────

def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _download(url: str, dest: Path, expect_size: int | None = None) -> None:
    """Stream `url` to `dest` with a progress line, via a .part temp file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  Downloading {url}")
    print(f"           -> {dest}")

    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", expect_size or 0))
        done = 0
        chunk = 1 << 20  # 1 MiB
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {_human(done)} / {_human(total)}  ({pct:5.1f}%)",
                          end="", flush=True)
                else:
                    print(f"\r  {_human(done)}", end="", flush=True)
    print()

    if expect_size and tmp.stat().st_size != expect_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Size mismatch: got {tmp.stat().st_size} B, expected {expect_size} B. "
            f"Download was truncated — re-run to retry.")

    tmp.rename(dest)


# ── LibriSpeech ──────────────────────────────────────────────────────────────

def _librispeech_ok(root: Path) -> bool:
    """
    True if `root` looks like a complete LibriSpeech test-clean tree.

    generate_synthetic_dataset.py requires numeric speaker directories
    directly under test-clean/, each holding .flac utterances. test-clean
    has 40 speakers; require a healthy majority so a half-extracted tree is
    not mistaken for a complete one.
    """
    if not root.is_dir():
        return False
    spk_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
    if len(spk_dirs) < 30:
        return False
    # Spot-check that at least one speaker actually holds audio.
    return any(next(d.rglob("*.flac"), None) is not None for d in spk_dirs[:5])


def fetch_librispeech(data_dir: Path, force: bool) -> Path:
    dest_root = data_dir / "LibriSpeech" / "test-clean"

    print("\n[1/2] LibriSpeech test-clean")
    if not force and _librispeech_ok(dest_root):
        n = len([d for d in dest_root.iterdir() if d.is_dir() and d.name.isdigit()])
        print(f"  Already present ({n} speakers) — skipping.")
        return dest_root

    data_dir.mkdir(parents=True, exist_ok=True)
    tarball = data_dir / "test-clean.tar.gz"

    if force or not tarball.exists() or tarball.stat().st_size != LIBRISPEECH_SIZE:
        _download(LIBRISPEECH_URL, tarball, expect_size=LIBRISPEECH_SIZE)
    else:
        print(f"  Tarball already downloaded ({_human(tarball.stat().st_size)}).")

    size = tarball.stat().st_size
    if size != LIBRISPEECH_SIZE:
        raise RuntimeError(
            f"Size mismatch for {tarball}: got {size} B, "
            f"expected {LIBRISPEECH_SIZE} B. Delete it and re-run.")
    print(f"  Size OK ({_human(size)}).")

    # tarfile decodes the gzip stream as it extracts, so a corrupt transfer
    # that somehow matched on size would still fail loudly here.
    print("  Extracting...")
    # The tarball contains a top-level 'LibriSpeech/test-clean/...' path, so
    # extracting at data_dir produces exactly dest_root.
    with tarfile.open(tarball, "r:gz") as tf:
        # Guard against path traversal in archive members.
        base = data_dir.resolve()
        for member in tf.getmembers():
            target = (base / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tf.extractall(data_dir)

    if not _librispeech_ok(dest_root):
        raise RuntimeError(f"Extraction finished but {dest_root} looks incomplete.")

    n = len([d for d in dest_root.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"  Done — {n} speakers at {dest_root}")
    print(f"  (You may delete {tarball} to reclaim {_human(LIBRISPEECH_SIZE)}.)")
    return dest_root


# ── EasyCom ATF ──────────────────────────────────────────────────────────────

def _atf_ok(path: Path) -> bool:
    """
    True if `path` is the real 37 MB HDF5 file rather than a git-lfs pointer.

    An un-smudged LFS pointer is a ~130-byte text file starting with
    'version https://git-lfs...', which h5py would reject with a confusing
    error much later, so catch it here.
    """
    if not path.is_file() or path.stat().st_size < ATF_MIN_SIZE:
        return False
    with open(path, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def fetch_atf(data_dir: Path, force: bool) -> Path | None:
    dest = data_dir / "EasyCom" / ATF_REL_PATH

    print("\n[2/2] EasyCom Device_ATFs.h5")
    if not force and _atf_ok(dest):
        print(f"  Already present ({_human(dest.stat().st_size)}) — skipping.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: ask the LFS batch API where the blob lives. Returns a
    # short-lived pre-signed URL.
    print("  Resolving git-lfs object...")
    req = urllib.request.Request(
        ATF_LFS_URL,
        data=json.dumps({
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": ATF_LFS_OID, "size": ATF_MIN_SIZE}],
        }).encode(),
        headers={"Accept": "application/vnd.git-lfs+json",
                 "Content-Type": "application/vnd.git-lfs+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)

    try:
        obj = payload["objects"][0]
        if "error" in obj:
            raise RuntimeError(f"LFS server error: {obj['error']}")
        href = obj["actions"]["download"]["href"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Unexpected LFS batch response ({e}): {payload}") from None

    # Step 2: download the blob itself.
    _download(href, dest, expect_size=ATF_MIN_SIZE)

    # The LFS oid is the sha256 of the content, so this is a real integrity
    # check rather than just a size check.
    print("  Verifying sha256...")
    digest = _sha256(dest)
    if digest != ATF_LFS_OID:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch\n    expected {ATF_LFS_OID}\n    got      {digest}")

    if not _atf_ok(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file is not a valid HDF5 file.")

    print(f"  Done — {_human(dest.stat().st_size)} at {dest}")
    return dest


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("Usage")[0].strip())
    p.add_argument("--data-dir", default="data", metavar="DIR",
                   help="Where to place the downloaded datasets")
    p.add_argument("--only", choices=["librispeech", "atf"], default=None,
                   help="Fetch only one of the two datasets")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the dataset is already present")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    print(f"Data directory: {data_dir}")

    libri = atf = None
    try:
        if args.only in (None, "librispeech"):
            libri = fetch_librispeech(data_dir, args.force)
        if args.only in (None, "atf"):
            atf = fetch_atf(data_dir, args.force)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    # ── Next steps, with real paths filled in ────────────────────────────
    print("\n" + "=" * 74)
    print("Datasets ready. Next steps:")
    print("=" * 74)

    if libri:
        atf_gen = f" \\\n      --atf-path '{data_dir / 'EasyCom' / ATF_REL_PATH}'" if atf else ""
        atf_eval = f" \\\n      --atf '{data_dir / 'EasyCom' / ATF_REL_PATH}'" if atf else ""
        print("\n  # 1. Generate a small smoke-test dataset (~1 min)")
        print(f"  .venv/bin/python generate_synthetic_dataset.py \\\n"
              f"      --librispeech '{libri}' \\\n"
              f"      --out ./synthetic_dataset \\\n"
              f"      --scenarios white_noise directional_mid \\\n"
              f"      --snrs 0 10 \\\n"
              f"      --duration 10{atf_gen}")
        print("\n  # 2. Evaluate it (beamformer only, DeepFilterNet skipped)")
        print(f"  .venv/bin/python eval_synthetic_2.py \\\n"
              f"      --dataset ./synthetic_dataset \\\n"
              f"      --no-denoise \\\n"
              f"      --modes raw_mic oracle_gaze srp \\\n"
              f"      --jobs 4{atf_eval}")

    if atf is None and args.only != "librispeech":
        print("\n  NOTE: Device_ATFs.h5 was not fetched. That is fine — omit")
        print("        --atf-path/--atf and the ISM room simulation is used instead.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
