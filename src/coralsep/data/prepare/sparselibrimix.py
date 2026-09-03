"""
Prepare SparseLibriMix (test-only) for CoRAL-Sep, Phase 3 (P3-A2).

SparseLibriMix is the sparse-overlap evaluation set: the same speakers mixed at
six controlled overlap ratios {0, 0.2, 0.4, 0.6, 0.8, 1.0}.  It powers the
Flagship-1 sparse-overlap SI-SDRi curve (P5-C1) and the L2 evaluation tier.

This script downloads LibriSpeech test-clean, clones popcornell/SparseLibriMix,
runs its generation script for each (n_src, overlap) configuration, verifies the
output layout, and prints the data_root.

WHAM! noise
-----------
The upstream generator adds WHAM! noise only when a noise directory is given.
WHAM! acquisition is a separate task (P1-A3); pass an existing noise directory
via --wham-noise-dir to also produce the ``mix_noisy`` mixtures.  Without it the
clean ``mix_clean`` mixtures (all that the sparse-overlap curve needs) are still
generated.

Usage
-----
    python src/coralsep/data/prepare_sparselibrimix.py --output-dir /path/to/datasets
    python src/coralsep/data/prepare_sparselibrimix.py --output-dir /path/to/datasets \\
        --librispeech-dir /path/to/existing/librispeech
    python src/coralsep/data/prepare_sparselibrimix.py --output-dir /path/to/datasets \\
        --wham-noise-dir /path/to/wham_noise/tt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# ── URLs and constants ────────────────────────────────────────────────────────

LIBRISPEECH_TEST_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SPARSELIBRIMIX_REPO_URL = "https://github.com/popcornell/SparseLibriMix"

# The six overlap ratios, as strings that match the metadata folder names
# (sparse_{n}_{ratio}); "0" and "1" are integer-named upstream, not "0.0"/"1.0".
OVERLAP_RATIOS: list[str] = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
N_SRC_VALUES: list[int] = [2, 3]
DEFAULT_FREQ = 16000

# ── Download ──────────────────────────────────────────────────────────────────


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_librispeech_test(librispeech_dir: Path) -> Path:
    """
    Download and extract LibriSpeech test-clean into librispeech_dir.

    Returns the path to the extracted test-clean directory (the one containing
    speaker-id subfolders), which is what make_mixtures.py expects.
    Skips the download/extract if it already exists and is non-empty.
    """
    librispeech_dir.mkdir(parents=True, exist_ok=True)
    extracted = librispeech_dir / "LibriSpeech" / "test-clean"
    if extracted.is_dir() and any(extracted.iterdir()):
        print(f"  [skip] test-clean already at {extracted}")
        return extracted

    tarball = librispeech_dir / "test-clean.tar.gz"
    if not tarball.exists():
        print("  [download] test-clean")
        print(f"    {LIBRISPEECH_TEST_URL}")
        urllib.request.urlretrieve(LIBRISPEECH_TEST_URL, str(tarball), reporthook=_report_progress)
        print()
    else:
        print(f"  [skip download] tarball exists: {tarball.name}")

    print(f"  [extract] {tarball.name} ...")
    with tarfile.open(tarball, "r:gz") as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(str(librispeech_dir), filter="data")
        else:
            tar.extractall(str(librispeech_dir))

    if not extracted.is_dir():
        raise RuntimeError(
            f"Extraction did not produce expected directory: {extracted}\n"
            "The LibriSpeech tarball may have a different internal layout."
        )
    print(f"  [ok] {extracted}")
    return extracted


# ── Repo cloning ──────────────────────────────────────────────────────────────


def clone_sparselibrimix(tools_dir: Path) -> Path:
    """
    Shallow-clone the SparseLibriMix repo into tools_dir/SparseLibriMix.

    Returns the path to the cloned repo.  Skips if already present.
    """
    repo_dir = tools_dir / "SparseLibriMix"
    if repo_dir.is_dir() and (repo_dir / "metadata").is_dir():
        print(f"  [skip] SparseLibriMix repo already at {repo_dir}")
        return repo_dir

    tools_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [clone] {SPARSELIBRIMIX_REPO_URL}")
    subprocess.run(
        ["git", "clone", "--depth", "1", SPARSELIBRIMIX_REPO_URL, str(repo_dir)],
        check=True,
    )
    print(f"  [ok] {repo_dir}")
    return repo_dir


# ── Generation ────────────────────────────────────────────────────────────────


def _find_make_mixtures_script(repo: Path) -> Path:
    """Locate scripts/make_mixtures.py inside the cloned repo."""
    candidates = [
        repo / "scripts" / "make_mixtures.py",
        repo / "make_mixtures.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Cannot find make_mixtures.py in {repo}. "
        "The SparseLibriMix repo structure may have changed, check the clone."
    )


def _metadata_json(repo: Path, n_src: int, ratio: str) -> Path:
    """Path to the metadata.json for one (n_src, overlap ratio) configuration."""
    return repo / "metadata" / f"sparse_{n_src}_{ratio}" / "metadata.json"


def generate_sparselibrimix(
    repo: Path,
    librispeech_test_dir: Path,
    data_root: Path,
    *,
    freq: int = DEFAULT_FREQ,
    n_src_values: list[int] | None = None,
    ratios: list[str] | None = None,
    wham_noise_dir: Path | None = None,
) -> None:
    """
    Run make_mixtures.py for every (n_src, overlap ratio) configuration.

    Writes each set to data_root/sparse_{n}_{ratio}/wav{freq}/.  A configuration
    is skipped if its mix_clean directory already contains WAV files.  WHAM!
    noise (and the mix_noisy/noise outputs) is added only when wham_noise_dir is
    given.
    """
    n_src_values = n_src_values or N_SRC_VALUES
    ratios = ratios or OVERLAP_RATIOS
    script = _find_make_mixtures_script(repo)

    for n_src in n_src_values:
        for ratio in ratios:
            out_dir = data_root / f"sparse_{n_src}_{ratio}" / f"wav{freq}"
            mix_clean = out_dir / "mix_clean"
            if mix_clean.is_dir() and any(mix_clean.glob("*.wav")):
                print(f"  [skip] sparse_{n_src}_{ratio} already at {out_dir}")
                continue

            metadata = _metadata_json(repo, n_src, ratio)
            if not metadata.exists():
                raise FileNotFoundError(
                    f"Metadata not found: {metadata}\n"
                    "Re-clone the SparseLibriMix repo or check --output-dir."
                )

            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [generate] sparse_{n_src}_{ratio} (wav{freq}) -> {out_dir}")

            cmd = [
                sys.executable,
                str(script),
                str(metadata),
                str(librispeech_test_dir),
                str(out_dir),
                "--rate",
                str(freq),
            ]
            if wham_noise_dir is not None:
                cmd.extend(["--noise_dir", str(wham_noise_dir)])

            subprocess.run(cmd, check=True)
            print(f"  [ok] {out_dir}")


# ── Layout verification ───────────────────────────────────────────────────────


def _expected_stream_dirs(n_src: int, expect_noise: bool) -> list[str]:
    dirs = ["mix_clean"] + [f"s{i}" for i in range(1, n_src + 1)]
    if expect_noise:
        dirs += ["mix_noisy", "noise"]
    return dirs


def verify_layout(
    data_root: Path,
    *,
    freq: int = DEFAULT_FREQ,
    n_src_values: list[int] | None = None,
    ratios: list[str] | None = None,
    expect_noise: bool = False,
) -> None:
    """
    Verify data_root matches the SparseLibriMix layout.

    Checks data_root/sparse_{n}_{ratio}/wav{freq}/{stream_dir}/ for every
    configuration.  Raises RuntimeError listing all missing directories.
    """
    n_src_values = n_src_values or N_SRC_VALUES
    ratios = ratios or OVERLAP_RATIOS

    missing: list[str] = []
    for n_src in n_src_values:
        for ratio in ratios:
            base = data_root / f"sparse_{n_src}_{ratio}" / f"wav{freq}"
            for stream_dir in _expected_stream_dirs(n_src, expect_noise):
                d = base / stream_dir
                if not d.is_dir():
                    missing.append(f"  {d}")

    if missing:
        raise RuntimeError(
            "Layout verification failed, missing directories:\n"
            + "\n".join(missing)
            + "\n\nRe-run prepare_sparselibrimix.py or check the generation output."
        )

    print("  Layout OK:")
    for n_src in n_src_values:
        streams = ", ".join(_expected_stream_dirs(n_src, expect_noise))
        print(f"    {data_root}/sparse_{n_src}_{{{','.join(ratios)}}}/wav{freq}/  [{streams}]")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_n_src(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download LibriSpeech test-clean, generate SparseLibriMix at six "
            "overlap ratios, and verify the layout for CoRAL-Sep (P3-A2)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root directory for generated data. SparseLibriMix lands in <DIR>/SparseLibriMix/.",
    )
    parser.add_argument(
        "--librispeech-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Where LibriSpeech test-clean is/goes. Defaults to <output-dir>/librispeech.",
    )
    parser.add_argument(
        "--wham-noise-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Existing WHAM! noise directory (e.g. wham_noise/tt). If given, also "
        "generates mix_noisy. Omit for clean mixtures only.",
    )
    parser.add_argument(
        "--freqs",
        type=int,
        default=DEFAULT_FREQ,
        metavar="HZ",
        help="Sample rate to generate (default: 16000).",
    )
    parser.add_argument(
        "--n-src",
        type=str,
        default="2,3",
        metavar="LIST",
        help="Comma-separated speaker counts to generate (default: 2,3).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    librispeech_dir: Path = (
        args.librispeech_dir.resolve() if args.librispeech_dir else output_dir / "librispeech"
    )
    wham_noise_dir: Path | None = args.wham_noise_dir.resolve() if args.wham_noise_dir else None
    n_src_values = _parse_n_src(args.n_src)
    data_root = output_dir / "SparseLibriMix"

    print("=" * 60)
    print("CoRAL-Sep  |  Prepare SparseLibriMix  |  Phase 3 (P3-A2)")
    print("=" * 60)
    print(f"  output dir:      {output_dir}")
    print(f"  librispeech dir: {librispeech_dir}")
    print(f"  wham noise:      {wham_noise_dir or 'none (clean mixtures only)'}")
    print(f"  sample rate:     {args.freqs}")
    print(f"  speaker counts:  {n_src_values}")
    print(f"  overlap ratios:  {OVERLAP_RATIOS}")
    print()

    print("Step 1 / 4  Download LibriSpeech test-clean")
    librispeech_test = download_librispeech_test(librispeech_dir)
    print()

    print("Step 2 / 4  Clone SparseLibriMix repo")
    repo = clone_sparselibrimix(output_dir / "tools")
    print(
        "  [note] SparseLibriMix generation may need its own requirements:\n"
        f"         pip install -r {repo / 'requirements.txt'}"
    )
    print()

    print("Step 3 / 4  Generate SparseLibriMix")
    generate_sparselibrimix(
        repo,
        librispeech_test,
        data_root,
        freq=args.freqs,
        n_src_values=n_src_values,
        wham_noise_dir=wham_noise_dir,
    )
    print()

    print("Step 4 / 4  Verify layout")
    verify_layout(
        data_root,
        freq=args.freqs,
        n_src_values=n_src_values,
        expect_noise=wham_noise_dir is not None,
    )
    print()

    print("=" * 60)
    print("Done.  SparseLibriMix data_root:")
    print()
    print(f"  {data_root}")
    print()
    print("Load samples tagged with overlap ratio via:")
    print("  from data.sparselibrimix import discover_sparse_samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
