"""
Prepare LibriSpeech with dual sample rates for CALM-Sep (BLUEPRINT §7.1).

All training mixtures use 8 kHz (locked by the frozen checkpoint). Keep 16 kHz
copies for band-recovery targets and DNSMOS evaluation.

This script resamples existing LibriSpeech FLAC trees in-place under a mirrored
output layout::

    {out_root}/8k/{split}/...   # 8 kHz for mixing / separation
    {out_root}/16k/{split}/...  # 16 kHz for band recovery / DNSMOS

Usage::

    python data/prepare_librispeech_8k.py --librispeech-root /data/LibriSpeech \\
        --out-root data/librispeech_dual
    python data/prepare_librispeech_8k.py --librispeech-root /data/LibriSpeech \\
        --out-root data/librispeech_dual --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

CALMSEP_SR: int = 8_000
BAND_RECOVERY_SR: int = 16_000

DEFAULT_SPLITS: tuple[str, ...] = (
    "train-clean-100",
    "dev-clean",
    "test-clean",
)


@dataclass
class ResampleStats:
    """Summary of a dual-rate preparation run."""

    n_files: int
    n_skipped: int
    out_root: str
    splits: list[str]
    dry_run: bool


def _resample_mono(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling (CPU-only, no torchaudio dependency)."""
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    duration = wav.shape[0] / orig_sr
    n_out = int(round(duration * target_sr))
    x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def _output_paths(out_root: Path, librispeech_root: Path, flac: Path) -> tuple[Path, Path]:
    rel = flac.relative_to(librispeech_root)
    out_8k = out_root / "8k" / rel.with_suffix(".wav")
    out_16k = out_root / "16k" / rel.with_suffix(".wav")
    return out_8k, out_16k


def prepare_librispeech_dual(
    librispeech_root: Path,
    out_root: Path,
    *,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    dry_run: bool = False,
    max_files: int | None = None,
) -> ResampleStats:
    """
    Resample LibriSpeech FLAC files to 8 kHz and copy/resample to 16 kHz.

    Idempotent: skips files whose 8 kHz and 16 kHz outputs already exist.

    Args:
        librispeech_root: Root of extracted LibriSpeech (contains train-clean-100/, etc.).
        out_root: Dual-rate output root.
        splits: Subsets to process.
        dry_run: When True, only count files and print the plan.
        max_files: Optional cap for smoke testing.

    Returns:
        ResampleStats summary.
    """
    if not librispeech_root.is_dir():
        raise FileNotFoundError(f"LibriSpeech root not found: {librispeech_root}")

    n_files = 0
    n_skipped = 0

    for split in splits:
        split_dir = librispeech_root / split
        if not split_dir.is_dir():
            print(f"  [skip] split not found: {split_dir}")
            continue

        flacs = sorted(split_dir.rglob("*.flac"))
        if max_files is not None:
            flacs = flacs[:max_files]

        for flac in flacs:
            out_8k, out_16k = _output_paths(out_root, librispeech_root, flac)
            if out_8k.exists() and out_16k.exists():
                n_skipped += 1
                continue

            if dry_run:
                n_files += 1
                continue

            wav, sr = sf.read(str(flac), dtype="float32", always_2d=True)
            mono = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]

            wav_16k = _resample_mono(mono, sr, BAND_RECOVERY_SR)
            wav_8k = _resample_mono(mono, sr, CALMSEP_SR)

            out_8k.parent.mkdir(parents=True, exist_ok=True)
            out_16k.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_8k), wav_8k, CALMSEP_SR, subtype="FLOAT")
            sf.write(str(out_16k), wav_16k, BAND_RECOVERY_SR, subtype="FLOAT")
            n_files += 1

    stats = ResampleStats(
        n_files=n_files,
        n_skipped=n_skipped,
        out_root=str(out_root),
        splits=list(splits),
        dry_run=dry_run,
    )

    if not dry_run:
        manifest = out_root / "dual_rate_manifest.json"
        manifest.write_text(json.dumps(asdict(stats), indent=2, sort_keys=True), encoding="utf-8")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech at 8 kHz + 16 kHz")
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        required=True,
        help="Root of extracted LibriSpeech corpus",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/librispeech_dual"),
        help="Dual-rate output directory",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="LibriSpeech splits to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files only; do not write WAVs",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap files per split (smoke testing)",
    )
    args = parser.parse_args()

    try:
        stats = prepare_librispeech_dual(
            args.librispeech_root,
            args.out_root,
            splits=tuple(args.splits),
            dry_run=args.dry_run,
            max_files=args.max_files,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    mode = "dry-run" if stats.dry_run else "written"
    print(
        f"LibriSpeech dual-rate prep ({mode}): "
        f"{stats.n_files} files, {stats.n_skipped} skipped -> {stats.out_root}"
    )


if __name__ == "__main__":
    main()
