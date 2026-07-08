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
) -> list[MixtureSample]:
    """
    Discover Libri3Mix samples from a standard LibriMix directory layout.

    Expected layout (Libri3Mix, 16 kHz, max):
        {data_root}/wav16k/max/{subset}/mix_both/
        {data_root}/wav16k/max/{subset}/s1/
        {data_root}/wav16k/max/{subset}/s2/
        {data_root}/wav16k/max/{subset}/s3/

    Args:
        data_root: Root of the LibriMix dataset.
        subset: Split name ('train', 'dev', 'test').
        max_samples: Cap the number of returned samples.

    Returns:
        List of MixtureSample objects.
    """
    root = Path(data_root)
    mix_dir = root / "wav16k" / "max" / subset / "mix_both"
    if not mix_dir.exists():
        raise FileNotFoundError(
            f"Libri3Mix mix directory not found: {mix_dir}\n"
            "Download Libri3Mix and set data_root in configs/baseline.yaml."
        )

    mix_files = sorted(mix_dir.glob("*.wav"))
    if max_samples is not None:
        mix_files = mix_files[:max_samples]

    samples: list[MixtureSample] = []
    for mix_path in mix_files:
        uid = mix_path.stem
        refs: list[np.ndarray] = []
        sr: int | None = None
        for spk_idx in (1, 2, 3):
            ref_path = root / "wav16k" / "max" / subset / f"s{spk_idx}" / f"{uid}.wav"
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
) -> list[MixtureSample]:
    """Alias for discover_librimix_samples (baseline runner entry point)."""
    return discover_librimix_samples(data_root, subset=subset, max_samples=max_samples)
