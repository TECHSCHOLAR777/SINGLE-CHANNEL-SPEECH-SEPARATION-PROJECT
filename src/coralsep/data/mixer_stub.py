"""
Minimal mixer stub for Phase 0 baseline runs.

Dev A will replace this with the full dynamic mixer (`data/mixer.py`).
This stub loads pre-mixed Libri3Mix test files from disk and yields
(mixture, reference_stems, sample_rate) tuples for the baseline runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class MixtureSample:
    """A single mixture with ground-truth clean stems."""

    mixture: np.ndarray
    """Mono mixture waveform, shape [T]."""

    references: np.ndarray
    """Clean source waveforms, shape [N, T]."""

    sample_rate: int
    utterance_id: str


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    return audio[:, 0], sr


def discover_librimix_samples(
    data_root: str | Path,
    subset: str = "test",
    max_samples: int | None = None,
    sample_rate: int = 16000,
    mode: str = "max",
) -> list[MixtureSample]:
    """
    Discover LibriNMix samples (N=2..5) from a standard LibriMix directory layout.

    The number of speakers is detected automatically by probing which sN/
    directories exist under the subset folder, so the same function works for
    Libri2Mix, Libri3Mix, Libri4Mix, and Libri5Mix without any extra arguments.

    Expected layout:
        {data_root}/wav{sr}k/{mode}/{subset}/mix_both/   # always present
        {data_root}/wav{sr}k/{mode}/{subset}/s1/         # N >= 1
        ...
        {data_root}/wav{sr}k/{mode}/{subset}/s5/         # N == 5

    Args:
        data_root: Root of the LibriMix dataset.
        subset: Split name ('train', 'dev', 'test').
        max_samples: Cap the number of returned samples.
        sample_rate: 8000 or 16000. Selects the wav8k or wav16k tree. The
            recorded evaluations of this project use 8000; the default is 16000
            for backwards compatibility with the Phase 0 callers.
        mode: 'min' or 'max'. Selects the LibriMix truncation mode. The recorded
            evaluations use 'min'.

    Returns:
        List of MixtureSample objects.

    Raises:
        ValueError: for an unsupported sample rate or mode.
        FileNotFoundError: if the mixture directory is absent, naming the path
            that was looked for.
    """
    if sample_rate not in (8000, 16000):
        raise ValueError(f"sample_rate must be 8000 or 16000, got {sample_rate}")
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    root = Path(data_root)
    subset_dir = root / f"wav{sample_rate // 1000}k" / mode / subset
    mix_dir = subset_dir / "mix_both"
    if not mix_dir.exists():
        raise FileNotFoundError(
            f"LibriMix mix directory not found: {mix_dir}\n"
            "Download LibriNMix and set data_root in configs/baseline.yaml."
        )

    mix_files = sorted(mix_dir.glob("*.wav"))
    if max_samples is not None:
        mix_files = mix_files[:max_samples]

    # Auto-detect speaker count from which sN/ dirs exist (N=1..5).
    # Only raise if there are mix files but no stem dirs (corrupted dataset).
    max_n = sum(1 for i in range(1, 6) if (subset_dir / f"s{i}").is_dir())
    if mix_files and max_n < 1:
        raise FileNotFoundError(
            f"No speaker stem directories (s1/..s5/) found under {subset_dir}"
        )

    samples: list[MixtureSample] = []
    for mix_path in mix_files:
        uid = mix_path.stem
        refs: list[np.ndarray] = []
        sr: int | None = None
        for spk_idx in range(1, max_n + 1):
            ref_path = subset_dir / f"s{spk_idx}" / f"{uid}.wav"
            if not ref_path.exists():
                break
            ref, ref_sr = _load_wav(ref_path)
            if sr is None:
                sr = ref_sr
            refs.append(ref)

        if not refs or sr is None:
            continue

        mixture, mix_sr = _load_wav(mix_path)
        if mix_sr != sr:
            raise ValueError(f"Sample rate mismatch for {uid}: mix={mix_sr}, ref={sr}")

        min_len = min(len(mixture), *(len(r) for r in refs))
        mixture = mixture[:min_len]
        refs_arr = np.stack([r[:min_len] for r in refs], axis=0)

        samples.append(
            MixtureSample(
                mixture=mixture.astype(np.float32),
                references=refs_arr.astype(np.float32),
                sample_rate=sr,
                utterance_id=uid,
            )
        )

    return samples


def iter_mixtures(
    data_root: str | Path,
    subset: str = "test",
    max_samples: int | None = None,
    sample_rate: int = 16000,
    mode: str = "max",
) -> list[MixtureSample]:
    """Alias for discover_librimix_samples (baseline runner entry point)."""
    return discover_librimix_samples(
        data_root,
        subset=subset,
        max_samples=max_samples,
        sample_rate=sample_rate,
        mode=mode,
    )
