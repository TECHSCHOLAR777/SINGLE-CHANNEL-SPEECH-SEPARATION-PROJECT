"""
SparseLibriMix loader for CoRAL-Sep evaluation.

Walks a generated SparseLibriMix tree (see data/prepare_sparselibrimix.py) and
yields mixtures tagged with their overlap ratio, so the sparse-overlap SI-SDRi
curve (P5-C1) can group results by overlap.  The overlap ratio is encoded in the
directory name (sparse_{n}_{ratio}), not in a per-mixture file.

Layout consumed:
    {data_root}/sparse_{n_src}_{ratio}/wav{freq}/mix_clean/{uid}.wav
    {data_root}/sparse_{n_src}_{ratio}/wav{freq}/s1/{uid}.wav  ... sN
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coralsep.data.mixer_stub import _load_wav


@dataclass
class SparseSample:
    """One SparseLibriMix mixture with clean stems and its overlap ratio."""

    mixture: np.ndarray
    """Mono mixture waveform, shape [T]."""

    references: np.ndarray
    """Clean source waveforms, shape [N, T]."""

    sample_rate: int
    utterance_id: str
    overlap_ratio: float
    """Overlap proportion in [0, 1], parsed from the sparse_{n}_{ratio} folder."""

    n_src: int


def _parse_config_dir(name: str) -> tuple[int, float]:
    """
    Parse a 'sparse_{n_src}_{ratio}' directory name into (n_src, overlap_ratio).

    Examples: 'sparse_3_0.2' -> (3, 0.2); 'sparse_2_0' -> (2, 0.0);
    'sparse_3_1' -> (3, 1.0).
    """
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "sparse":
        raise ValueError(f"Not a SparseLibriMix config directory: {name!r}")
    return int(parts[1]), float(parts[2])


def discover_sparse_samples(
    data_root: str | Path,
    *,
    n_src: int,
    overlap_ratio: float | None = None,
    freq: int = 16_000,
    mixture: str = "mix_clean",
    max_samples: int | None = None,
) -> list[SparseSample]:
    """
    Discover SparseLibriMix samples for one speaker count.

    Args:
        data_root: Root containing sparse_{n}_{ratio}/ folders.
        n_src: Speaker count (2 or 3).
        overlap_ratio: If given, only this ratio; otherwise all ratios found.
        freq: Sample-rate subfolder (wav{freq}).
        mixture: Which mixture folder to load ('mix_clean' or 'mix_noisy').
        max_samples: Cap the number of returned samples per ratio.

    Returns:
        List of SparseSample, each tagged with its overlap ratio.

    Raises:
        FileNotFoundError: If data_root has no matching config directories.
    """
    root = Path(data_root)
    config_dirs = sorted(d for d in root.glob(f"sparse_{n_src}_*") if d.is_dir())
    if not config_dirs:
        raise FileNotFoundError(
            f"No SparseLibriMix config directories (sparse_{n_src}_*) under {root}.\n"
            "Run data/prepare_sparselibrimix.py first."
        )

    samples: list[SparseSample] = []
    for config_dir in config_dirs:
        parsed_n, ratio = _parse_config_dir(config_dir.name)
        if parsed_n != n_src:
            continue
        if overlap_ratio is not None and abs(ratio - overlap_ratio) > 1e-9:
            continue

        mix_dir = config_dir / f"wav{freq}" / mixture
        if not mix_dir.is_dir():
            continue

        mix_files = sorted(mix_dir.glob("*.wav"))
        if max_samples is not None:
            mix_files = mix_files[:max_samples]

        for mix_path in mix_files:
            sample = _load_one(config_dir, freq, mixture, mix_path, n_src, ratio)
            if sample is not None:
                samples.append(sample)

    return samples


def _load_one(
    config_dir: Path,
    freq: int,
    mixture: str,
    mix_path: Path,
    n_src: int,
    ratio: float,
) -> SparseSample | None:
    uid = mix_path.stem
    wav_root = config_dir / f"wav{freq}"

    refs: list[np.ndarray] = []
    sr: int | None = None
    for spk_idx in range(1, n_src + 1):
        ref_path = wav_root / f"s{spk_idx}" / f"{uid}.wav"
        if not ref_path.exists():
            return None
        ref, ref_sr = _load_wav(ref_path)
        if sr is None:
            sr = ref_sr
        refs.append(ref)

    if sr is None:
        return None

    mix, mix_sr = _load_wav(mix_path)
    if mix_sr != sr:
        raise ValueError(f"Sample rate mismatch for {uid}: mix={mix_sr}, ref={sr}")

    min_len = min(len(mix), *(len(r) for r in refs))
    mix = mix[:min_len]
    refs_arr = np.stack([r[:min_len] for r in refs], axis=0)

    return SparseSample(
        mixture=mix.astype(np.float32),
        references=refs_arr.astype(np.float32),
        sample_rate=sr,
        utterance_id=uid,
        overlap_ratio=ratio,
        n_src=n_src,
    )
