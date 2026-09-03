"""
Build a reverberant-noisy evaluation set from Libri3Mix, CoRAL-Sep Phase 1 (P1-A3, WHAMR! alt).

WHAMR! itself is generated from WSJ0, which is LDC-licensed and cannot be
auto-downloaded (see data/prepare_whamr.py for the gated, WSJ0-required path).
This module is the license-free alternative that serves the same L3 evaluation
need ("reverberant + noisy", MASTER_PROJECT tier L3 / task P4-C2): it applies
our own two-stage augmentation (pyroomacoustics RIR reverb + WHAM! noise) to the
clean Libri3Mix test set, once, with a fixed seed, and writes a frozen eval set.

The clean reference stems are copied through unchanged, so SI-SDRi is still
computed against the original clean sources: only the mixture is degraded.

Output layout mirrors LibriMix so the existing ``discover_librimix_samples``
loader reads it with no changes::

    {out_root}/wav16k/max/{subset}/mix_both/{uid}.wav   (reverb + noise)
    {out_root}/wav16k/max/{subset}/s1/{uid}.wav         (clean stem)
    {out_root}/wav16k/max/{subset}/s2/{uid}.wav
    {out_root}/wav16k/max/{subset}/s3/{uid}.wav

Caveat: the reverb comes from our simulated RIRs, not MERL's, so numbers are
reproducible but not directly comparable to published WHAMR! results.  Use
data/prepare_whamr.py when literature-parity WHAMR! is required.

Usage
-----
    python src/coralsep/data/make_reverb_eval.py --librimix-root /data/Libri3Mix \\
        --out-root /data/ReverbNoisyLibri3Mix --wham-noise-dir /data/wham_noise/tt
    python src/coralsep/data/make_reverb_eval.py --librimix-root /data/Libri3Mix \\
        --out-root /data/ReverbLibri3Mix     # reverb only, no noise
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import soundfile as sf

from coralsep.data.augmentation import AugmentationConfig, AugmentationPipeline
from coralsep.data.mixer_stub import MixtureSample, discover_librimix_samples

DEFAULT_FREQ = 16000
Augmentor = Callable[[MixtureSample], MixtureSample]


def _build_default_augmentor(wham_noise_dir: Path | None, seed: int) -> AugmentationPipeline:
    """Construct the reverb (+ optional noise) augmentor used for the eval set."""
    import numpy as np

    config = AugmentationConfig(
        rir_prob=1.0,
        noise_prob=1.0 if wham_noise_dir is not None else 0.0,
        wham_dir=wham_noise_dir,
    )
    return AugmentationPipeline(config, rng=np.random.default_rng(seed))


def _write_sample(test_dir: Path, sample: MixtureSample, freq: int) -> None:
    """Write one augmented mixture + its clean stems into the LibriMix layout."""
    uid = sample.utterance_id
    mix_dir = test_dir / "mix_both"
    mix_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(mix_dir / f"{uid}.wav"), sample.mixture, freq, subtype="FLOAT")

    for spk_idx in range(sample.references.shape[0]):
        stem_dir = test_dir / f"s{spk_idx + 1}"
        stem_dir.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(stem_dir / f"{uid}.wav"),
            sample.references[spk_idx],
            freq,
            subtype="FLOAT",
        )


def build_reverb_noisy_eval(
    librimix_root: Path,
    out_root: Path,
    *,
    subset: str = "test",
    wham_noise_dir: Path | None = None,
    freq: int = DEFAULT_FREQ,
    seed: int = 0,
    max_samples: int | None = None,
    augmentor: Augmentor | None = None,
) -> Path:
    """
    Apply reverb (+ optional WHAM! noise) to a Libri3Mix subset and write it out.

    Loads the clean subset via ``discover_librimix_samples``, augments each
    mixture with a fixed-seed augmentor (references kept clean), and writes the
    result to out_root in LibriMix layout.  Idempotent: skips entirely if the
    output mix_both directory already contains WAVs.  Returns out_root, ready to
    pass back to ``discover_librimix_samples``.

    The augmentor is injectable for testing; by default a reverb(+noise)
    AugmentationPipeline is built from wham_noise_dir and seed.
    """
    out_test_dir = out_root / f"wav{freq // 1000}k" / "max" / subset
    mix_dir = out_test_dir / "mix_both"
    if mix_dir.is_dir() and any(mix_dir.glob("*.wav")):
        print(f"  [skip] reverb-noisy eval already at {out_root}")
        return out_root

    samples = discover_librimix_samples(librimix_root, subset=subset, max_samples=max_samples)
    if not samples:
        raise RuntimeError(
            f"No Libri3Mix samples found under {librimix_root} (subset={subset}). "
            "Run data/prepare_librimix.py first."
        )

    if augmentor is None:
        augmentor = _build_default_augmentor(wham_noise_dir, seed)

    kind = "reverb + noise" if wham_noise_dir is not None else "reverb only"
    print(f"  [build] {len(samples)} samples ({kind}, seed={seed}) -> {out_test_dir}")
    for sample in samples:
        _write_sample(out_test_dir, augmentor(sample), freq)
    print(f"  [ok] {out_root}")
    return out_root


def verify_layout(out_root: Path, *, subset: str = "test", freq: int = DEFAULT_FREQ) -> None:
    """
    Verify the written eval set matches the LibriMix layout the loader expects.

    Checks {out_root}/wav16k/max/{subset}/{mix_both,s1,s2} each hold WAVs (s3 is
    optional so 2-speaker inputs also verify).  Raises RuntimeError otherwise.
    """
    base = out_root / f"wav{freq // 1000}k" / "max" / subset
    required = ["mix_both", "s1", "s2"]

    problems: list[str] = []
    for stream in required:
        d = base / stream
        if not d.is_dir():
            problems.append(f"  {d}  (missing)")
        elif not any(d.glob("*.wav")):
            problems.append(f"  {d}  (no .wav files)")

    if problems:
        raise RuntimeError(
            "Reverb-noisy eval layout verification failed:\n"
            + "\n".join(problems)
            + "\n\nRe-run make_reverb_eval.py or check the source Libri3Mix."
        )

    n = len(list((base / "mix_both").glob("*.wav")))
    print(f"  Layout OK: {base}  [mix_both, s1, s2(, s3)], {n} mixtures")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reverberant-noisy Libri3Mix eval set (license-free WHAMR! "
            "alternative for the L3 tier)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--librimix-root",
        required=True,
        type=Path,
        metavar="DIR",
        help="Existing Libri3Mix data root (from data/prepare_librimix.py).",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        type=Path,
        metavar="DIR",
        help="Where the reverb-noisy eval set is written.",
    )
    parser.add_argument(
        "--wham-noise-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="WHAM! noise split (e.g. wham_noise/tt). Omit for reverb-only.",
    )
    parser.add_argument("--subset", type=str, default="test", metavar="NAME")
    parser.add_argument("--seed", type=int, default=0, metavar="N")
    parser.add_argument("--max-samples", type=int, default=None, metavar="N")
    args = parser.parse_args()

    librimix_root: Path = args.librimix_root.resolve()
    out_root: Path = args.out_root.resolve()
    wham_noise_dir: Path | None = args.wham_noise_dir.resolve() if args.wham_noise_dir else None

    print("=" * 60)
    print("CoRAL-Sep  |  Reverberant-noisy eval set  |  Phase 1 (L3 / WHAMR! alt)")
    print("=" * 60)
    print(f"  librimix root:  {librimix_root}")
    print(f"  out root:       {out_root}")
    print(f"  wham noise:     {wham_noise_dir or 'none (reverb only)'}")
    print(f"  subset / seed:  {args.subset} / {args.seed}")
    print()

    print("Step 1 / 2  Build augmented eval set")
    build_reverb_noisy_eval(
        librimix_root,
        out_root,
        subset=args.subset,
        wham_noise_dir=wham_noise_dir,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print()

    print("Step 2 / 2  Verify layout")
    verify_layout(out_root, subset=args.subset)
    print()

    print("=" * 60)
    print("Done.  Load this eval set with the standard loader:")
    print(f'  discover_librimix_samples("{out_root}", subset="{args.subset}")')
    print("=" * 60)


if __name__ == "__main__":
    main()
