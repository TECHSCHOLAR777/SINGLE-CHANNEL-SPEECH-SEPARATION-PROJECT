"""Slice the local 8 kHz dataset into a Kaggle-uploadable subset.

Kaggle datasets are size-capped, and the full local corpus is roughly 176,000
files. This produces a representative subset, around 900 MB uncompressed:

    <out>/
      librispeech-8k/
        train-clean-100/   N speakers, capped utterances per speaker
        manifest_8k.json   counts updated, speaker IDs preserved
      rirs/                the whole bank, 12 KB per response
      noise/wham/          a random sample of noise clips

Speaker IDs are preserved exactly, because the held-out split logic keys on
them. Slicing must not silently change which speakers count as held out.

Run from the project root:

    python scripts/slice_for_kaggle.py --out-dir build/calmsep-kaggle

Then package and upload:

    cd build && zip -r calmsep-kaggle.zip calmsep-kaggle/
    kaggle datasets create -p calmsep-kaggle-meta/

The dataset this produced is `rishig777/calmsep-8k-slice`, which the Stage 4
training log records as its audio input.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

DEFAULT_SRC_ROOT = Path("data/calmsep-8k")
DEFAULT_OUT_ROOT = Path("build/calmsep-kaggle")
DEFAULT_SEED = 42
DEFAULT_TRAIN_SPEAKERS = 100
DEFAULT_MAX_UTTERANCES = 30
DEFAULT_NOISE_CLIPS = 2_000


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def slice_speech(
    src_root: Path,
    out_root: Path,
    rng: random.Random,
    n_speakers: int,
    max_utterances: int,
) -> tuple[list[str], int]:
    """Copy a capped number of utterances for a sampled set of speakers.

    Returns:
        The chosen speaker IDs, sorted, and the number of files copied.
    """
    train_src = src_root / "librispeech-8k" / "train-clean-100"
    if not train_src.is_dir():
        raise FileNotFoundError(f"no train-clean-100 under {src_root / 'librispeech-8k'}")

    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(train_src.rglob("*.wav")):
        speaker = f.parts[f.parts.index("train-clean-100") + 1]
        by_speaker[speaker].append(f)

    if not by_speaker:
        raise FileNotFoundError(f"no .wav files under {train_src}")

    chosen = sorted(rng.sample(sorted(by_speaker), min(n_speakers, len(by_speaker))))

    copied = 0
    for speaker in chosen:
        files = by_speaker[speaker]
        rng.shuffle(files)
        for f in files[:max_utterances]:
            rel = f.relative_to(src_root / "librispeech-8k")
            copy_file(f, out_root / "librispeech-8k" / rel)
            copied += 1
    return chosen, copied


def rewrite_manifest(
    src_root: Path,
    out_root: Path,
    chosen_speakers: list[str],
    file_count: int,
) -> Path:
    """Copy the speech manifest with the train-clean-100 entry updated.

    dev-clean and test-clean entries are left untouched so the held-out split
    behaves identically on the slice. train-clean-360 is zeroed because none of
    it is included.
    """
    manifest_src = src_root / "librispeech-8k" / "manifest_8k.json"
    if not manifest_src.is_file():
        raise FileNotFoundError(f"no manifest at {manifest_src}")

    manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    for entry in manifest:
        if entry.get("split") == "train-clean-100":
            entry["speaker_ids"] = chosen_speakers
            entry["file_count"] = file_count
        elif entry.get("split") == "train-clean-360":
            entry["speaker_ids"] = []
            entry["file_count"] = 0

    out_path = out_root / "librispeech-8k" / "manifest_8k.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def slice_rirs(src_root: Path, out_root: Path, rng: random.Random, limit: int | None) -> int:
    """Copy the RIR bank. The whole bank is small enough to keep by default."""
    rir_src = src_root / "rirs"
    rir_dst = out_root / "rirs"
    rir_dst.mkdir(parents=True, exist_ok=True)

    bank = rir_src / "bank.json"
    if bank.is_file():
        shutil.copy2(bank, rir_dst / "bank.json")

    files = sorted(rir_src.glob("*.wav"))
    if limit is not None and limit < len(files):
        files = sorted(rng.sample(files, limit))
    for f in files:
        shutil.copy2(f, rir_dst / f.name)
    return len(files)


def slice_noise(src_root: Path, out_root: Path, rng: random.Random, n_clips: int) -> int:
    """Copy a random sample of noise clips, with the noise manifest alongside."""
    noise_src = src_root / "noise" / "wham"
    noise_dst = out_root / "noise" / "wham"
    noise_dst.mkdir(parents=True, exist_ok=True)

    manifest = src_root / "noise" / "noise_manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, out_root / "noise" / "noise_manifest.json")

    files = sorted(noise_src.glob("*.wav"))
    selected = sorted(rng.sample(files, min(n_clips, len(files))))
    for f in selected:
        shutil.copy2(f, noise_dst / f.name)
    return len(selected)


def summarise(out_root: Path) -> tuple[int, int]:
    """Return the total byte size and file count of the slice."""
    files = [f for f in out_root.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files), len(files)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT, help="Local 8 kHz dataset root")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT, help="Where to write the slice")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed, recorded for reproducibility")
    p.add_argument("--n-speakers", type=int, default=DEFAULT_TRAIN_SPEAKERS)
    p.add_argument("--max-utterances", type=int, default=DEFAULT_MAX_UTTERANCES)
    p.add_argument("--n-noise", type=int, default=DEFAULT_NOISE_CLIPS)
    p.add_argument(
        "--n-rirs",
        type=int,
        default=None,
        help="Sample this many RIRs. Default keeps the whole bank, which is small.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    out_root = args.out_dir

    print(f"[1/4] Slicing train-clean-100 from {args.src_root}")
    speakers, copied = slice_speech(
        args.src_root, out_root, rng, args.n_speakers, args.max_utterances
    )
    print(f"      {len(speakers)} speakers, {copied} files copied")

    print("[2/4] Rewriting the speech manifest")
    manifest_path = rewrite_manifest(args.src_root, out_root, speakers, copied)
    print(f"      written to {manifest_path}")

    print("[3/4] Copying the RIR bank")
    n_rirs = slice_rirs(args.src_root, out_root, rng, args.n_rirs)
    print(f"      {n_rirs} RIR files copied")

    print("[4/4] Sampling noise clips")
    n_noise = slice_noise(args.src_root, out_root, rng, args.n_noise)
    print(f"      {n_noise} noise files copied")

    total_bytes, file_count = summarise(out_root)
    print("\nSummary")
    print(f"  output    : {out_root}")
    print(f"  seed      : {args.seed}")
    print(f"  size      : {total_bytes / 1e9:.2f} GB")
    print(f"  files     : {file_count:,}")
    print("\nNext:")
    print(f"  zip -r {out_root.name}.zip {out_root}/")
    print("  kaggle datasets create -p calmsep-kaggle-meta/")


if __name__ == "__main__":
    main()
