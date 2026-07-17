"""
Resample LibriSpeech from 16 kHz to 8 kHz for the CALM-Sep pipeline (Dev A, P0-A2).

The frozen SR-CorrNet var-2-5 checkpoint is locked to 8 kHz (STFT window 128,
hop 64). All training and evaluation audio must be at this rate before it enters
the pipeline. This script does the one-time conversion and writes a manifest
that prepare_but_reverbdb.py and fixed_eval_generator.py depend on to enumerate
speakers and files.

Speaker IDs are parsed from LibriSpeech filenames ({speaker}-{chapter}-{utt}.flac)
and from directory names (which duplicate the speaker ID in the corpus layout).
Both match because LibriSpeech is internally consistent, but the directory name
is used as the authoritative source to avoid any filename-parsing edge cases.

Transcription (.txt) files are copied unchanged: they reference utterance IDs
that remain stable regardless of sample rate, so no path editing is needed.

Usage
-----
    python data/prepare_librispeech_8k.py \\
        --input-dir /data/LibriSpeech \\
        --output-dir /data/LibriSpeech_8k \\
        --splits train-clean-100 train-clean-360 dev-clean test-clean
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

SRC_SR: int = 16_000
DST_SR: int = 8_000
# GCD(16000, 8000) = 8000, so up=1, down=2
_UP: int = 1
_DOWN: int = 2


def _speaker_id_from_dir(speaker_dir: Path) -> str:
    """LibriSpeech speaker directories are named by speaker ID (an integer string)."""
    return speaker_dir.name


def _resample_file(src: Path, dst: Path) -> float:
    """
    Read src, resample 16 kHz → 8 kHz, write dst.

    Returns duration in seconds at the output rate. Using resample_poly with
    up=1/down=2 is exact (integer ratio, no windowing artefacts from
    scipy.signal.resample's DFT-based method).
    """
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if sr != SRC_SR:
        raise ValueError(
            f"{src}: expected {SRC_SR} Hz source, found {sr} Hz. "
            "Pass only unmodified LibriSpeech files to this script."
        )
    # Mono: mean over channels (most LibriSpeech files are already mono)
    mono = audio.mean(axis=1)
    out = resample_poly(mono, _UP, _DOWN).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out, DST_SR, subtype="PCM_16")
    return float(len(out)) / DST_SR


def _copy_txt(src: Path, dst: Path) -> None:
    """Copy a transcription file, creating parent dirs as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def process_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
) -> dict[str, object]:
    """
    Resample all FLAC files in one LibriSpeech split.

    Returns a manifest entry for the split recording speaker IDs, file counts,
    and total duration.
    """
    src_split = input_dir / split
    if not src_split.is_dir():
        raise FileNotFoundError(
            f"split directory not found: {src_split}\n"
            "Check --input-dir and --splits."
        )
    dst_split = output_dir / split

    # Collect all FLAC files in {speaker}/{chapter}/ layout
    flac_files = sorted(src_split.rglob("*.flac"))
    if not flac_files:
        raise RuntimeError(
            f"No FLAC files found under {src_split}. "
            "This does not look like a LibriSpeech split."
        )

    total_duration_s = 0.0
    file_count = 0
    speaker_ids: set[str] = set()

    for src in tqdm(flac_files, desc=f"{split}", unit="file"):
        # Layout: split/speaker_id/chapter_id/file.flac
        relative = src.relative_to(src_split)
        dst = dst_split / relative.with_suffix(".wav")

        # Record speaker from the first component of the relative path
        parts = relative.parts
        if len(parts) >= 1:
            speaker_ids.add(parts[0])

        if dst.exists():
            # Idempotent: skip already-converted files, but add their duration
            info = sf.info(str(dst))
            total_duration_s += info.duration
            file_count += 1
            continue

        duration = _resample_file(src, dst)
        total_duration_s += duration
        file_count += 1

    # Copy .txt transcription files unchanged
    for txt in sorted(src_split.rglob("*.txt")):
        relative = txt.relative_to(src_split)
        dst_txt = dst_split / relative
        if not dst_txt.exists():
            _copy_txt(txt, dst_txt)

    return {
        "split": split,
        "speaker_ids": sorted(speaker_ids),
        "file_count": file_count,
        "total_duration_s": round(total_duration_s, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample LibriSpeech 16 kHz → 8 kHz for the CALM-Sep pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, metavar="DIR",
                        help="Root of the 16 kHz LibriSpeech corpus.")
    parser.add_argument("--output-dir", required=True, type=Path, metavar="DIR",
                        help="Destination root. Mirror layout is written here.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100", "train-clean-360", "dev-clean", "test-clean"],
        metavar="SPLIT",
        help="LibriSpeech splits to process.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"ERROR: --input-dir does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Resampling {SRC_SR} Hz → {DST_SR} Hz\n")

    manifest_entries: list[dict] = []

    for split in args.splits:
        print(f"--- {split} ---")
        entry = process_split(input_dir, output_dir, split)
        manifest_entries.append(entry)
        hours = entry["total_duration_s"] / 3600.0
        print(
            f"  {entry['file_count']} files, "
            f"{hours:.2f} h, "
            f"{len(entry['speaker_ids'])} speakers\n"
        )

    manifest_path = output_dir / "manifest_8k.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Split':<25} {'Files':>8} {'Hours':>8} {'Speakers':>10}")
    print("-" * 56)
    total_files = 0
    total_hours = 0.0
    for entry in manifest_entries:
        h = entry["total_duration_s"] / 3600.0
        print(
            f"{entry['split']:<25} {entry['file_count']:>8} {h:>8.2f} "
            f"{len(entry['speaker_ids']):>10}"
        )
        total_files += entry["file_count"]
        total_hours += h
    print("-" * 56)
    print(f"{'TOTAL':<25} {total_files:>8} {total_hours:>8.2f}")


if __name__ == "__main__":
    main()
