"""
Prepare Libri3Mix for CA-MoSE Phase 0.

Downloads LibriSpeech train-clean-100 / dev-clean / test-clean, clones the
LibriMix generation repo, generates Libri3Mix at 16 kHz / max mode / mix_both
type for the dev and test splits, verifies the output layout expected by
discover_librimix_samples, and prints the data_root value to paste into
configs/baseline.yaml.

Note on the train split
-----------------------
Libri3Mix train-360 requires LibriSpeech train-clean-360 (not train-clean-100).
We skip it here so Phase 0 never touches that ~25 GB download.  The M0
milestone only needs the test split.  The training data will be addressed in
Phase 2 — set --include-train and ensure train-clean-360 is present.

Usage
-----
    python data/prepare_librimix.py --output-dir /path/to/datasets
    python data/prepare_librimix.py --output-dir /path/to/datasets \\
        --librispeech-dir /path/to/existing/librispeech
    python data/prepare_librimix.py --output-dir /path/to/datasets \\
        --include-train --librispeech-dir /path/to/librispeech_with_360
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# ── URLs and constants ────────────────────────────────────────────────────────

LIBRISPEECH_URLS: dict[str, str] = {
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}

LIBRIMIX_REPO_URL = "https://github.com/JorisCos/LibriMix"

# CSV files to include when generating without the train split.
_DEV_TEST_CSVS = ["mixture_dev_mix_both.csv", "mixture_test_mix_both.csv"]
# LibriMix ships metadata for BOTH train-100 and train-360. The script only
# downloads LibriSpeech train-clean-100, so defaulting the train CSV to
# train-360 guaranteed a generation crash: every source path in that CSV points
# at train-clean-360 speakers that are not on disk. Keep train-360 available for
# whoever wants to pay the 23 GB download, but default to the split we actually
# fetch.
TRAIN_CSV_BY_SPLIT: dict[str, str] = {
    "train-100": "mixture_train-100_mix_both.csv",
    "train-360": "mixture_train-360_mix_both.csv",
}
LIBRISPEECH_URL_BY_TRAIN_SPLIT: dict[str, str] = {
    "train-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
}
DEFAULT_TRAIN_SPLIT = "train-100"

# Subsets that must exist for verify_layout to pass (train is best-effort).
REQUIRED_SUBSETS = ["dev", "test"]
STREAM_DIRS = ["mix_both", "s1", "s2", "s3"]

# Candidate names LibriMix may use for the training split.
_TRAIN_CANDIDATES = ["train-360", "train-100", "train"]

# ── Download helpers ──────────────────────────────────────────────────────────


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_librispeech(librispeech_dir: Path, extra_splits: dict[str, str] | None = None) -> None:
    """
    Download and extract LibriSpeech splits into librispeech_dir.

    Skips any split whose extracted directory already exists and is non-empty.
    Tarballs extract into librispeech_dir/LibriSpeech/{split}/.
    """
    librispeech_dir.mkdir(parents=True, exist_ok=True)

    urls = dict(LIBRISPEECH_URLS)
    if extra_splits:
        urls.update(extra_splits)

    for split, url in urls.items():
        extracted = librispeech_dir / "LibriSpeech" / split
        if extracted.exists() and any(extracted.iterdir()):
            print(f"  [skip] {split} already at {extracted}")
            continue

        tarball = librispeech_dir / f"{split}.tar.gz"
        if not tarball.exists():
            print(f"  [download] {split}")
            print(f"    {url}")
            urllib.request.urlretrieve(url, str(tarball), reporthook=_report_progress)
            print()
        else:
            print(f"  [skip download] tarball exists: {tarball.name}")

        print(f"  [extract] {tarball.name} ...")
        with tarfile.open(tarball, "r:gz") as tar:
            if sys.version_info >= (3, 12):
                tar.extractall(str(librispeech_dir), filter="data")
            else:
                tar.extractall(str(librispeech_dir))

        if not extracted.exists():
            raise RuntimeError(
                f"Extraction did not produce expected directory: {extracted}\n"
                "The LibriSpeech tarball may have a different internal layout."
            )
        print(f"  [ok] {extracted}")


# ── Repo cloning ──────────────────────────────────────────────────────────────


def clone_librimix(tools_dir: Path) -> Path:
    """
    Shallow-clone the LibriMix repo into tools_dir/LibriMix.

    Returns the path to the cloned repo.  Skips if already present.
    """
    repo_dir = tools_dir / "LibriMix"
    if repo_dir.is_dir() and (repo_dir / "README.md").exists():
        print(f"  [skip] LibriMix repo already at {repo_dir}")
        return repo_dir

    tools_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [clone] {LIBRIMIX_REPO_URL}")
    subprocess.run(
        ["git", "clone", "--depth", "1", LIBRIMIX_REPO_URL, str(repo_dir)],
        check=True,
    )
    print(f"  [ok] {repo_dir}")
    return repo_dir


# ── Generation ────────────────────────────────────────────────────────────────


def _find_generation_script(librimix_repo: Path) -> Path:
    """Locate create_librimix_from_metadata.py inside the cloned repo."""
    candidates = [
        librimix_repo / "create_librimix_from_metadata.py",
        librimix_repo / "scripts" / "create_librimix_from_metadata.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Cannot find create_librimix_from_metadata.py in {librimix_repo}. "
        "The LibriMix repo structure may have changed — check the clone."
    )


def _make_filtered_metadata(
    librimix_repo: Path,
    work_dir: Path,
    include_train: bool,
    train_split: str = DEFAULT_TRAIN_SPLIT,
) -> Path:
    """
    Copy LibriMix metadata CSVs into a private directory.

    When include_train is False (default), the train-360 CSV is excluded so
    the generator never attempts to read train-clean-360 speakers.  The
    caller passes this filtered directory as --metadata_path to the script.
    """
    src_dir = librimix_repo / "metadata" / "Libri3Mix"
    dst_dir = work_dir / "metadata_filtered" / "Libri3Mix"
    dst_dir.mkdir(parents=True, exist_ok=True)

    csvs = list(_DEV_TEST_CSVS)
    if include_train:
        csvs.append(TRAIN_CSV_BY_SPLIT[train_split])

    for csv_name in csvs:
        src = src_dir / csv_name
        dst = dst_dir / csv_name
        if not dst.exists():
            if not src.exists():
                raise FileNotFoundError(
                    f"Expected metadata CSV not found: {src}\n"
                    "Re-clone the LibriMix repo or check --librimix-repo."
                )
            shutil.copy2(str(src), str(dst))

    return work_dir / "metadata_filtered"


def generate_librimix(
    librimix_repo: Path,
    librispeech_dir: Path,
    output_dir: Path,
    *,
    include_train: bool = False,
    train_split: str = DEFAULT_TRAIN_SPLIT,
) -> None:
    """
    Run the LibriMix generation script to produce Libri3Mix.

    Generates: 16 kHz, max mode, mix_both type, 3 sources.
    By default generates dev and test only (include_train=False).
    Skips entirely if test/mix_both already contains WAV files.
    """
    libri3mix_out = output_dir / "Libri3Mix"
    test_mix = libri3mix_out / "wav16k" / "max" / "test" / "mix_both"
    if test_mix.is_dir() and any(test_mix.glob("*.wav")):
        print(f"  [skip] Libri3Mix test split already at {libri3mix_out}")
        return

    script = _find_generation_script(librimix_repo)
    filtered_meta = _make_filtered_metadata(
        librimix_repo, output_dir / "tools", include_train, train_split
    )

    # LibriSpeech tarballs extract into librispeech_dir/LibriSpeech/
    ls_root = librispeech_dir / "LibriSpeech"
    if not ls_root.is_dir():
        ls_root = librispeech_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = "dev + test" + (f" + {train_split}" if include_train else "")
    print(f"  [generate] Libri3Mix ({splits}) -> {libri3mix_out}")
    print(f"    script:      {script.name}")
    print(f"    librispeech: {ls_root}")
    print(f"    metadata:    {filtered_meta}")
    print("    This may take 30–90 minutes depending on disk speed.")

    cmd = [
        sys.executable,
        str(script),
        "--librispeech_path",
        str(ls_root),
        "--metadata_path",
        str(filtered_meta),
        "--librimix_path",
        str(output_dir),
        "--n_src",
        "3",
        "--freqs",
        "16000",
        "--modes",
        "max",
        "--types",
        "mix_both",
    ]
    subprocess.run(cmd, check=True)
    print(f"  [ok] {libri3mix_out}")


# ── Train alias ───────────────────────────────────────────────────────────────


def _create_directory_alias(src: Path, dst: Path) -> None:
    """
    Create a directory alias dst -> src.

    Tries os.symlink first.  On Windows where symlinks need elevated rights,
    falls back to a junction point (mklink /J) which does not.
    """
    try:
        os.symlink(src, dst, target_is_directory=True)
        print(f"  [symlink] train -> {src.name}")
        return
    except (OSError, NotImplementedError):
        pass

    if sys.platform == "win32":
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                check=True,
                capture_output=True,
            )
            print(f"  [junction] train -> {src.name}")
            return
        except subprocess.CalledProcessError:
            pass

    print(
        f"  [warn] Could not create 'train' alias for '{src.name}'.\n"
        f"         Manually link if you need the train split:\n"
        f"           {src}  ->  {dst}"
    )


def _ensure_train_alias(wav_root: Path) -> None:
    """
    Create wav_root/train as an alias to whichever train split LibriMix generated.

    discover_librimix_samples uses subset='train' so this alias is needed when
    callers pass subset='train'.  The alias is best-effort: absent train data
    is not an error for the M0 baseline (which only uses the test subset).
    """
    train_link = wav_root / "train"
    if train_link.exists() or train_link.is_symlink():
        return

    for candidate in _TRAIN_CANDIDATES:
        src = wav_root / candidate
        if src.is_dir():
            _create_directory_alias(src, train_link)
            return

    print(
        "  [info] No train split found (looked for train-360, train-100, train).\n"
        "         The dev and test splits are all that is needed for the M0 baseline."
    )


# ── Layout verification ───────────────────────────────────────────────────────


def verify_layout(data_root: Path) -> None:
    """
    Verify that data_root matches the layout expected by discover_librimix_samples.

    Checks:
        {data_root}/wav16k/max/{subset}/{stream_dir}/
    for subset in [dev, test] (and train if present) and
    stream_dir in [mix_both, s1, s2, s3].

    Raises RuntimeError listing every missing directory if anything is absent.
    """
    wav_root = data_root / "wav16k" / "max"
    if not wav_root.is_dir():
        raise RuntimeError(
            f"wav16k/max/ not found under {data_root}.\n"
            "Did the LibriMix generation step complete successfully?"
        )

    _ensure_train_alias(wav_root)

    subsets_to_check = list(REQUIRED_SUBSETS)
    if (wav_root / "train").is_dir():
        subsets_to_check = ["train"] + subsets_to_check

    missing: list[str] = []
    for subset in subsets_to_check:
        for stream_dir in STREAM_DIRS:
            d = wav_root / subset / stream_dir
            if not d.is_dir():
                missing.append(f"  {d}")

    if missing:
        raise RuntimeError(
            "Layout verification failed — missing directories:\n"
            + "\n".join(missing)
            + "\n\nRe-run prepare_librimix.py or check the LibriMix generation output."
        )

    print("  Layout OK:")
    for subset in subsets_to_check:
        print(f"    {data_root}/wav16k/max/{subset}/  [mix_both, s1, s2, s3]")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download LibriSpeech, generate Libri3Mix, and verify the layout "
            "for CA-MoSE Phase 0."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root directory for all generated data. Libri3Mix lands in <DIR>/Libri3Mix/.",
    )
    parser.add_argument(
        "--librispeech-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=("Where LibriSpeech will be downloaded. " "Defaults to <output-dir>/librispeech."),
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also generate a Libri3Mix train split (see --train-split).",
    )
    parser.add_argument(
        "--train-split",
        choices=sorted(TRAIN_CSV_BY_SPLIT),
        default=DEFAULT_TRAIN_SPLIT,
        help=(
            "Which train split to generate with --include-train. "
            "train-100 (default) uses LibriSpeech train-clean-100, already downloaded. "
            "train-360 pulls an extra 23 GB of LibriSpeech."
        ),
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    librispeech_dir: Path = (
        args.librispeech_dir.resolve() if args.librispeech_dir else output_dir / "librispeech"
    )

    print("=" * 60)
    print("CA-MoSE  |  Prepare Libri3Mix  |  Phase 0")
    print("=" * 60)
    print(f"  output dir:      {output_dir}")
    print(f"  librispeech dir: {librispeech_dir}")
    if args.include_train:
        print("  train split:     yes (requires train-clean-360)")
    else:
        print("  train split:     no  (use --include-train for Phase 2)")
    print()

    print("Step 1 / 4  Download LibriSpeech")
    extra = {}
    if args.include_train and args.train_split in LIBRISPEECH_URL_BY_TRAIN_SPLIT:
        extra[args.train_split.replace("train-", "train-clean-")] = LIBRISPEECH_URL_BY_TRAIN_SPLIT[
            args.train_split
        ]
    download_librispeech(librispeech_dir, extra_splits=extra)
    print()

    print("Step 2 / 4  Clone LibriMix repo")
    librimix_repo = clone_librimix(output_dir / "tools")
    print()

    print("Step 3 / 4  Generate Libri3Mix")
    generate_librimix(
        librimix_repo,
        librispeech_dir,
        output_dir,
        include_train=args.include_train,
        train_split=args.train_split,
    )
    print()

    print("Step 4 / 4  Verify layout")
    data_root = output_dir / "Libri3Mix"
    verify_layout(data_root)
    print()

    print("=" * 60)
    print("Done.  Set this in configs/baseline.yaml:")
    print()
    print(f'  data_root: "{data_root}"')
    print()
    print("Then score the frozen backbone on one split:")
    print(f"  python scripts/run_baseline.py --data-root {data_root}/Libri2Mix --max-samples 30")
    print("=" * 60)


if __name__ == "__main__":
    main()
