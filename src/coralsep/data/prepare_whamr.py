"""
Prepare real WHAMR! (reverberant-noisy) for CoRAL-Sep — Phase 1 (P1-A3, gated).

WHAMR! reverberates WSJ0 speech and adds WHAM! noise.  WSJ0 is distributed by
the LDC under licence and cannot be auto-downloaded, and the WHAMR! generation
scripts are released from the WHAM! website rather than a clean git clone.  This
preparer therefore *orchestrates* generation rather than fetching everything:
the user supplies the paths to WSJ0 and the official WHAMR! scripts, and this
runs them against our already-downloaded WHAM! noise (data/prepare_wham.py).

If either WSJ0 or the WHAMR! scripts are absent, generation is skipped with a
clear "deferred" message instead of failing — mirroring how prepare_librimix.py
treats the train-360 split.  For a license-free reverberant-noisy eval set that
needs none of this, use data/make_reverb_eval.py instead.

Usage
-----
    # Deferred (no WSJ0): prints guidance and exits cleanly.
    python src/coralsep/data/prepare_whamr.py --output-dir /data/WHAMR

    # Real generation (requires an LDC WSJ0 licence + the WHAMR! scripts):
    python src/coralsep/data/prepare_whamr.py --output-dir /data/WHAMR \\
        --wsj0-dir /data/wsj0 --whamr-scripts-dir /data/whamr_scripts \\
        --wham-noise-dir /data/wham_noise
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# The official WHAMR! generation entry point (from the wham.whisper.ai scripts).
# Exposed as a constant so it can be adjusted if upstream renames it.
WHAMR_SCRIPT_NAME = "create_wham_from_scratch.py"

DEFAULT_FREQ = 16000
# WHAMR! is 2-speaker (built on wsj0-2mix).
_WHAMR_STREAMS = ["mix_both", "s1", "s2"]


def _deferred_note(reason: str) -> None:
    print(f"  [deferred] {reason}")
    print(
        "  WHAMR! needs WSJ0 (LDC licence) + the official WHAMR! scripts.\n"
        "  For a license-free reverberant-noisy eval set instead, run:\n"
        "    python src/coralsep/data/make_reverb_eval.py --librimix-root <Libri3Mix> \\\n"
        "        --out-root <out> --wham-noise-dir <wham_noise/tt>"
    )


def find_whamr_script(scripts_dir: Path) -> Path:
    """Locate the WHAMR! generation script inside scripts_dir."""
    candidates = [
        scripts_dir / WHAMR_SCRIPT_NAME,
        scripts_dir / "scripts" / WHAMR_SCRIPT_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Cannot find {WHAMR_SCRIPT_NAME} under {scripts_dir}. "
        "Point --whamr-scripts-dir at the official WHAMR! scripts."
    )


def generate_whamr(
    scripts_dir: Path,
    wsj0_dir: Path,
    wham_noise_dir: Path,
    output_dir: Path,
    *,
    freq: int = DEFAULT_FREQ,
) -> Path:
    """
    Run the official WHAMR! generation script.

    Invokes the upstream ``create_wham_from_scratch.py`` with the WSJ0 root, our
    WHAM! noise root, and the output directory.  Skips if the test split already
    contains WAVs.  Returns the WHAMR! data root.
    """
    whamr_out = output_dir / "WHAMR"
    test_mix = whamr_out / f"wav{freq // 1000}k" / "max" / "tt" / "mix_both"
    if test_mix.is_dir() and any(test_mix.glob("*.wav")):
        print(f"  [skip] WHAMR! test split already at {whamr_out}")
        return whamr_out

    script = find_whamr_script(scripts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [generate] WHAMR! -> {whamr_out}")
    print(f"    script:     {script}")
    print(f"    wsj0:       {wsj0_dir}")
    print(f"    wham noise: {wham_noise_dir}")

    cmd = [
        sys.executable,
        str(script),
        "--wsj0-root",
        str(wsj0_dir),
        "--wham-noise-root",
        str(wham_noise_dir),
        "--output-dir",
        str(whamr_out),
    ]
    subprocess.run(cmd, check=True)
    print(f"  [ok] {whamr_out}")
    return whamr_out


def verify_layout(whamr_root: Path, *, freq: int = DEFAULT_FREQ) -> None:
    """
    Verify the WHAMR! test split matches the expected layout.

    Checks {whamr_root}/wav16k/max/tt/{mix_both,s1,s2} each hold WAVs.  Raises
    RuntimeError listing any that are missing or empty.
    """
    base = whamr_root / f"wav{freq // 1000}k" / "max" / "tt"

    problems: list[str] = []
    for stream in _WHAMR_STREAMS:
        d = base / stream
        if not d.is_dir():
            problems.append(f"  {d}  (missing)")
        elif not any(d.glob("*.wav")):
            problems.append(f"  {d}  (no .wav files)")

    if problems:
        raise RuntimeError(
            "WHAMR! layout verification failed:\n"
            + "\n".join(problems)
            + "\n\nCheck the WHAMR! generation output."
        )

    n = len(list((base / "mix_both").glob("*.wav")))
    print(f"  Layout OK: {base}  [mix_both, s1, s2] — {n} mixtures")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare real WHAMR! (requires WSJ0 + WHAMR! scripts; deferred otherwise).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True, type=Path, metavar="DIR")
    parser.add_argument(
        "--wsj0-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="LDC WSJ0 root. Absent → generation is deferred.",
    )
    parser.add_argument(
        "--whamr-scripts-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory holding the official WHAMR! generation scripts.",
    )
    parser.add_argument(
        "--wham-noise-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="WHAM! noise root (from data/prepare_wham.py).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()

    print("=" * 60)
    print("CoRAL-Sep  |  Prepare WHAMR!  |  Phase 1 (P1-A3, gated)")
    print("=" * 60)
    print(f"  output dir:   {output_dir}")
    print()

    if args.wsj0_dir is None:
        _deferred_note("no --wsj0-dir supplied")
        return
    if args.whamr_scripts_dir is None:
        _deferred_note("no --whamr-scripts-dir supplied")
        return
    if args.wham_noise_dir is None:
        _deferred_note("no --wham-noise-dir supplied (run data/prepare_wham.py first)")
        return

    print("Step 1 / 2  Generate WHAMR!")
    whamr_root = generate_whamr(
        args.whamr_scripts_dir.resolve(),
        args.wsj0_dir.resolve(),
        args.wham_noise_dir.resolve(),
        output_dir,
    )
    print()

    print("Step 2 / 2  Verify layout")
    verify_layout(whamr_root)
    print()

    print("=" * 60)
    print(f"Done.  WHAMR! data root: {whamr_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
