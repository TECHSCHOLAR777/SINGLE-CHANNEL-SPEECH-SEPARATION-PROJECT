#!/usr/bin/env python3
"""
CALM-Sep Phase 0 — Step 1: Download raw datasets.

Downloads LibriSpeech (openslr.org) and WHAM! noise (S3) automatically,
then extracts archives in-place.  DNS-4 requires a manual step (see below).

Usage
-----
    python scripts/download_step1.py --output-dir ~/Downloads/calmsep-raw

What gets downloaded
--------------------
  LibriSpeech train-clean-100   6.3 GB
  LibriSpeech train-clean-360  23.1 GB
  LibriSpeech dev-clean         337 MB
  LibriSpeech test-clean        346 MB
  WHAM! noise                  ~17.1 GB
  ─────────────────────────────────────
  Total auto-download           ~47 GB

DNS-4 is printed as instructions (requires Microsoft registration / azcopy).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# ─────────────────────────────────────────────────────────────────────────────
# Dataset manifest
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Download:
    name: str
    url: str
    filename: str
    size_bytes: int  # approximate — used for ETA only
    extract: bool = True
    extract_subdir: str = ""  # target subfolder inside output_dir
    md5: str = ""  # optional integrity check


LIBRISPEECH_BASE = "https://www.openslr.org/resources/12"
WHAM_URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7" ".s3.amazonaws.com/wham_noise.zip"

AUTO_DOWNLOADS: list[Download] = [
    Download(
        name="LibriSpeech dev-clean",
        url=f"{LIBRISPEECH_BASE}/dev-clean.tar.gz",
        filename="dev-clean.tar.gz",
        size_bytes=337_926_546,
        extract_subdir="LibriSpeech",
    ),
    Download(
        name="LibriSpeech test-clean",
        url=f"{LIBRISPEECH_BASE}/test-clean.tar.gz",
        filename="test-clean.tar.gz",
        size_bytes=346_663_984,
        extract_subdir="LibriSpeech",
    ),
    Download(
        name="LibriSpeech train-clean-100",
        url=f"{LIBRISPEECH_BASE}/train-clean-100.tar.gz",
        filename="train-clean-100.tar.gz",
        size_bytes=6_387_309_499,
        extract_subdir="LibriSpeech",
    ),
    Download(
        name="LibriSpeech train-clean-360",
        url=f"{LIBRISPEECH_BASE}/train-clean-360.tar.gz",
        filename="train-clean-360.tar.gz",
        size_bytes=23_049_477_885,
        extract_subdir="LibriSpeech",
    ),
    Download(
        name="WHAM! noise",
        url=WHAM_URL,
        filename="wham_noise.zip",
        size_bytes=17_116_602_368,
        extract_subdir="wham_noise",
    ),
]

TOTAL_BYTES = sum(d.size_bytes for d in AUTO_DOWNLOADS)

# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds == float("inf"):
        return "calculating…"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _bar(fraction: float, width: int = 30) -> str:
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# aria2c fast-path (16 parallel connections — bypasses server throttling)
# ─────────────────────────────────────────────────────────────────────────────


def _aria2c_available() -> bool:
    return shutil.which("aria2c") is not None


def _download_with_aria2c(url: str, dest: Path, label: str) -> int:
    """Use aria2c with 16 connections for maximum speed. Returns bytes downloaded."""
    before = dest.stat().st_size if dest.exists() else 0
    cmd = [
        "aria2c",
        "--file-allocation=none",  # skip pre-allocation on macOS
        "-x",
        "16",  # 16 connections per server
        "-s",
        "16",  # 16 segments
        "-k",
        "1M",  # 1 MB chunk size
        "--continue=true",  # resume partial downloads
        "--console-log-level=notice",
        "--summary-interval=2",
        "-d",
        str(dest.parent),
        "-o",
        dest.name,
        url,
    ]
    print(f"  ⚡ aria2c  16 connections  →  {dest.name}")
    print(f"     {url}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        after = dest.stat().st_size if dest.exists() else 0
        return max(after - before, 0)
    raise RuntimeError(f"aria2c failed (exit {result.returncode}) for {dest.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Download with live progress (urllib fallback)
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK = 1 << 17  # 128 KB


def _download_file(
    url: str,
    dest: Path,
    expected_bytes: int,
    label: str,
    session_start: float,
    session_downloaded_before: int,
    session_total: int,
) -> int:
    """
    Download url → dest.
    Uses aria2c (16 connections) if available, else urllib single-connection.
    Returns number of bytes downloaded in this call.
    """
    if _aria2c_available():
        return _download_with_aria2c(url, dest, label)

    # ── urllib fallback (single connection) ───────────────────────────────────
    resume_pos = dest.stat().st_size if dest.exists() else 0

    headers: dict[str, str] = {}
    if resume_pos:
        headers["Range"] = f"bytes={resume_pos}-"
        print(f"  ↩  Resuming from {_fmt_size(resume_pos)}")

    req = Request(url, headers=headers)
    try:
        resp = urlopen(req, timeout=60)
    except URLError as e:
        raise RuntimeError(f"Failed to open {url}: {e}") from e

    # Content-Length of the *remaining* bytes.
    content_length = int(resp.headers.get("Content-Length", 0))
    total_file = resume_pos + content_length if content_length else expected_bytes

    mode = "ab" if resume_pos else "wb"
    downloaded_this_call = 0
    t_file_start = time.monotonic()

    with dest.open(mode) as fh:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            downloaded_this_call += len(chunk)

            # ── per-file progress ──────────────────────────────────────────
            file_done = resume_pos + downloaded_this_call
            file_frac = min(file_done / max(total_file, 1), 1.0)
            elapsed_file = time.monotonic() - t_file_start
            speed = downloaded_this_call / max(elapsed_file, 0.001)
            remaining_file = (total_file - file_done) / max(speed, 1)

            # ── session-level progress ─────────────────────────────────────
            session_done = session_downloaded_before + downloaded_this_call
            session_frac = min(session_done / max(session_total, 1), 1.0)
            elapsed_session = time.monotonic() - session_start
            session_speed = session_done / max(elapsed_session, 0.001)
            session_remaining = (session_total - session_done) / max(session_speed, 1)

            line = (
                f"\r  {label[:28]:<28}  "
                f"[{_bar(file_frac, 20)}] {file_frac*100:5.1f}%  "
                f"{_fmt_size(int(speed))}/s  ETA file: {_fmt_eta(remaining_file):<12}"
                f"  |  Overall: {session_frac*100:5.1f}%  ETA total: {_fmt_eta(session_remaining)}"
            )
            sys.stdout.write(line)
            sys.stdout.flush()

    sys.stdout.write("\n")
    return downloaded_this_call


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────


def _extract(archive: Path, dest_dir: Path, label: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(archive.suffixes)
    print(f"  ⟳  Extracting {archive.name} …", end=" ", flush=True)
    t0 = time.monotonic()

    if suffix in (".tar.gz", ".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir)
    elif suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    else:
        print(f"Unknown archive format: {suffix} — skipping extraction.")
        return

    elapsed = time.monotonic() - t0
    print(f"done in {elapsed:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# MD5 check
# ─────────────────────────────────────────────────────────────────────────────


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# DNS-4 instructions
# ─────────────────────────────────────────────────────────────────────────────

DNS4_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  DNS-4 — Manual download required                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

DNS-4 noise data is hosted on Microsoft Azure Blob Storage and requires
azcopy (Microsoft's tool).

  Step A — Install azcopy
    macOS:   brew install azcopy
    Linux:   wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
             tar -xf azcopy.tar.gz && sudo mv azcopy*/azcopy /usr/local/bin/

  Step B — Download the noise clips (no account needed, public SAS URL)
    azcopy copy \\
      "<DNS_CHALLENGE_NOISE_SAS_URL>" \\
      {output_dir}/dns4_raw/DNS-Challenge_Noise.zip \\
      --recursive

    Replace <DNS_CHALLENGE_NOISE_SAS_URL> with the full URL from:
    https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-4.sh
    (it is the blob URL ending in DNS-Challenge_Noise.zip?<SAS_TOKEN>)

  Step C — Once downloaded, run the stratification script:
    python data/prepare_dns4.py \\
      --out-dir {output_dir}/dns4-subset \\
      --target-gb 20 \\
      --materialize

  Alternatively: skip DNS-4 and train the noise adapter on DEMAND only.
  DEMAND noise (~10 GB, already downloaded) covers a wide SNR range and
  is sufficient for a working system.  DNS-4 adds diversity but is optional.

"""

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _upfront_summary(output_dir: Path) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         CALM-Sep Phase 0 — Step 1: Dataset Download             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Output directory : {output_dir}")
    print()
    print(f"  {'Dataset':<32}  {'Size':>8}  Status")
    print(f"  {'─'*32}  {'─'*8}  {'─'*12}")

    already_bytes = 0
    for d in AUTO_DOWNLOADS:
        archive = output_dir / d.filename
        if archive.exists():
            sz = archive.stat().st_size
            if sz >= d.size_bytes * 0.99:
                status = "✓ already done"
                already_bytes += d.size_bytes
            else:
                status = f"partial ({_fmt_size(sz)})"
                already_bytes += sz
        else:
            status = "pending"
        print(f"  {d.name:<32}  {_fmt_size(d.size_bytes):>8}  {status}")

    remaining = max(TOTAL_BYTES - already_bytes, 0)
    print()
    print(f"  Total to download : {_fmt_size(remaining)} " f"(of {_fmt_size(TOTAL_BYTES)} total)")
    print()

    # ETA at typical Mac broadband speeds
    speeds = {
        "10 Mbps  (slow WiFi) ": 10 * 1e6 / 8,
        "50 Mbps  (avg WiFi)  ": 50 * 1e6 / 8,
        "100 Mbps (fast WiFi) ": 100 * 1e6 / 8,
        "200 Mbps (fibre)     ": 200 * 1e6 / 8,
    }
    print("  Estimated download time (remaining data only):")
    for label, bps in speeds.items():
        print(f"    {label}  →  {_fmt_eta(remaining / bps)}")
    print()
    print("  + extraction adds ~10–20 min on top of download time.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CALM-Sep Phase 0 datasets (LibriSpeech + WHAM!).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path.home() / "Downloads" / "calmsep-raw",
        help="Root directory for downloaded and extracted datasets. "
        "(default: ~/Downloads/calmsep-raw)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download archives but do not extract them.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[
            "librispeech-dev",
            "librispeech-test",
            "librispeech-100",
            "librispeech-360",
            "wham",
        ],
        help="Download only the listed items (useful for partial runs).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by --only
    downloads = AUTO_DOWNLOADS
    if args.only:
        key_map = {
            "librispeech-dev": "LibriSpeech dev-clean",
            "librispeech-test": "LibriSpeech test-clean",
            "librispeech-100": "LibriSpeech train-clean-100",
            "librispeech-360": "LibriSpeech train-clean-360",
            "wham": "WHAM! noise",
        }
        wanted = {key_map[k] for k in args.only}
        downloads = [d for d in AUTO_DOWNLOADS if d.name in wanted]

    _upfront_summary(output_dir)

    # Compute already-downloaded bytes for accurate session-level ETA.
    session_total = sum(d.size_bytes for d in downloads)
    session_done_before = 0
    for d in downloads:
        p = output_dir / d.filename
        if p.exists():
            session_done_before += min(p.stat().st_size, d.size_bytes)

    session_start = time.monotonic()
    cumulative = session_done_before

    for idx, dl in enumerate(downloads, 1):
        archive = output_dir / dl.filename
        already_done = archive.exists() and archive.stat().st_size >= dl.size_bytes * 0.99

        print(f"[{idx}/{len(downloads)}] {dl.name}")
        print(f"  URL  : {dl.url}")
        print(f"  File : {archive}")

        if already_done:
            print(
                f"  ✓  Already complete ({_fmt_size(archive.stat().st_size)}), skipping download."
            )
            cumulative += dl.size_bytes
        else:
            print(f"  Size : ~{_fmt_size(dl.size_bytes)}")
            try:
                downloaded = _download_file(
                    url=dl.url,
                    dest=archive,
                    expected_bytes=dl.size_bytes,
                    label=dl.name,
                    session_start=session_start,
                    session_downloaded_before=cumulative,
                    session_total=session_total,
                )
                cumulative += downloaded
                actual = archive.stat().st_size
                print(f"  ✓  Download complete ({_fmt_size(actual)})")
            except Exception as exc:
                print(f"\n  ✗  Download failed: {exc}")
                print("     The partial file is kept — re-run to resume.")
                sys.exit(1)

        # Extraction
        extract_dir = output_dir / dl.extract_subdir if dl.extract_subdir else output_dir
        if not args.skip_extract and dl.extract:
            _extract(archive, extract_dir, dl.name)
        print()

    # Session summary
    elapsed = time.monotonic() - session_start
    print("═" * 70)
    print("  Step 1 complete!")
    print(f"  Total time  : {_fmt_eta(elapsed)}")
    print(f"  Output root : {output_dir}")
    print()
    print("  Directory layout:")
    print(f"    {output_dir}/")
    print("    ├── LibriSpeech/")
    print("    │   ├── train-clean-100/")
    print("    │   ├── train-clean-360/")
    print("    │   ├── dev-clean/")
    print("    │   └── test-clean/")
    print("    └── wham_noise/")
    print("        ├── tr/   (train noise)")
    print("        ├── cv/   (validation noise)")
    print("        └── tt/   (test noise)")
    print()

    # DNS-4 instructions
    print(DNS4_INSTRUCTIONS.format(output_dir=output_dir))

    print("═" * 70)
    print("  Ready for Step 2:")
    print()
    print("  python data/prepare_librispeech_8k.py \\")
    print(f"    --input-dir  {output_dir}/LibriSpeech \\")
    print("    --output-dir ~/Desktop/calmsep-8k/librispeech-8k")
    print()
    print("  python data/prepare_noise_staging.py \\")
    print(f"    --wham-dir   {output_dir}/wham_noise \\")
    print("    --output-dir ~/Desktop/calmsep-8k/noise")
    print("═" * 70)


if __name__ == "__main__":
    main()
