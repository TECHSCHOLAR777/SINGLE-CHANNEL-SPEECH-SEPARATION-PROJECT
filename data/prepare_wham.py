"""
Prepare the WHAM! noise corpus for CA-MoSE — Phase 1 (P1-A3).

WHAM! (WSJ0 Hipster Ambient Mixtures) ships its ambient-noise recordings as a
single freely downloadable archive, ``wham_noise.zip``.  This is the only piece
CA-MoSE augmentation needs: ``AugmentationPipeline`` (data/augmentation.py) adds
noise by sampling WAVs from a ``wham_dir`` via ``rglob("*.wav")``, and
``prepare_sparselibrimix.py --wham-noise-dir`` uses the same directory to build
the ``mix_noisy`` mixtures.

This script downloads the archive, extracts it, verifies the ``tr/cv/tt`` split
layout, and prints the per-split noise directories to hand to those consumers.

Scope
-----
WHAM! noise only.  WHAMR! (reverberant) is generated from WSJ0, which is
LDC-licensed and cannot be auto-downloaded, so it is intentionally out of scope
here and handled separately.

Usage
-----
    python data/prepare_wham.py --output-dir /path/to/datasets
    python data/prepare_wham.py --output-dir /path/to/datasets \\
        --wham-noise-url https://mirror.example/wham_noise.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# ── URLs and constants ────────────────────────────────────────────────────────

# Canonical WHAM! noise archive (same source used by the Asteroid WHAM recipe).
# Overridable via --wham-noise-url in case the bucket URL changes.
WHAM_NOISE_URL = (
    "https://my-bucket-a8b4b49c25c811e9a7e29cec32478954.s3.amazonaws.com/wham_noise.zip"
)

# The archive extracts into a top-level ``wham_noise/`` directory holding the
# WSJ0-aligned train / cross-validation / test noise splits.
EXTRACTED_DIRNAME = "wham_noise"
NOISE_SPLITS: list[str] = ["tr", "cv", "tt"]

# ── Download ──────────────────────────────────────────────────────────────────


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_wham_noise(output_dir: Path, url: str = WHAM_NOISE_URL) -> Path:
    """
    Download and extract the WHAM! noise archive into output_dir.

    Returns the path to the extracted ``wham_noise`` directory (the one holding
    the tr/cv/tt splits).  Skips the download and/or extraction if the expected
    output already exists and is non-empty.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = output_dir / EXTRACTED_DIRNAME
    if extracted.is_dir() and any(extracted.iterdir()):
        print(f"  [skip] wham_noise already at {extracted}")
        return extracted

    archive = output_dir / "wham_noise.zip"
    if not archive.exists():
        print("  [download] wham_noise.zip")
        print(f"    {url}")
        urlretrieve(url, str(archive), reporthook=_report_progress)
        print()
    else:
        print(f"  [skip download] archive exists: {archive.name}")

    print(f"  [extract] {archive.name} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(str(output_dir))

    if not extracted.is_dir():
        raise RuntimeError(
            f"Extraction did not produce expected directory: {extracted}\n"
            "The WHAM! archive may have a different internal layout."
        )
    print(f"  [ok] {extracted}")
    return extracted


# ── Layout verification ───────────────────────────────────────────────────────


def verify_layout(wham_noise_dir: Path, splits: list[str] | None = None) -> None:
    """
    Verify wham_noise_dir contains the expected split directories with WAV files.

    Checks that each of ``tr``, ``cv``, ``tt`` exists under wham_noise_dir and
    holds at least one ``.wav`` file.  Raises RuntimeError listing every split
    that is missing or empty.
    """
    splits = splits or NOISE_SPLITS

    problems: list[str] = []
    for split in splits:
        split_dir = wham_noise_dir / split
        if not split_dir.is_dir():
            problems.append(f"  {split_dir}  (missing)")
        elif not any(split_dir.glob("*.wav")):
            problems.append(f"  {split_dir}  (no .wav files)")

    if problems:
        raise RuntimeError(
            "WHAM! layout verification failed:\n"
            + "\n".join(problems)
            + "\n\nRe-run prepare_wham.py or check the downloaded archive."
        )

    print("  Layout OK:")
    for split in splits:
        n_wavs = len(list((wham_noise_dir / split).glob("*.wav")))
        print(f"    {wham_noise_dir / split}  [{n_wavs} noise clips]")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify the WHAM! noise corpus for CA-MoSE augmentation "
            "(P1-A3, WHAM! only)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root directory for the download. wham_noise lands in <DIR>/wham_noise/.",
    )
    parser.add_argument(
        "--wham-noise-url",
        type=str,
        default=WHAM_NOISE_URL,
        metavar="URL",
        help="Override the WHAM! noise archive URL (default: canonical S3 bucket).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()

    print("=" * 60)
    print("CA-MoSE  |  Prepare WHAM! noise  |  Phase 1 (P1-A3)")
    print("=" * 60)
    print(f"  output dir:  {output_dir}")
    print(f"  source:      {args.wham_noise_url}")
    print()

    print("Step 1 / 2  Download WHAM! noise")
    wham_noise_dir = download_wham_noise(output_dir, url=args.wham_noise_url)
    print()

    print("Step 2 / 2  Verify layout")
    verify_layout(wham_noise_dir)
    print()

    print("=" * 60)
    print("Done.  Use these noise directories for augmentation:")
    print()
    print(f"  train noise:  {wham_noise_dir / 'tr'}")
    print(f"  test noise:   {wham_noise_dir / 'tt'}")
    print()
    print("Wire the training noise into AugmentationConfig:")
    print(f'  AugmentationConfig(wham_dir="{wham_noise_dir / "tr"}")')
    print()
    print("Or build SparseLibriMix mix_noisy with the test split:")
    print("  python data/prepare_sparselibrimix.py --output-dir ... \\")
    print(f"      --wham-noise-dir {wham_noise_dir / 'tt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
