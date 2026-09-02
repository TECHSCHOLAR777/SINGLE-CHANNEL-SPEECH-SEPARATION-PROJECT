"""
PyTorch Dataset wrapper for DynamicMixer — P0-INT3.

Bridges the on-the-fly DynamicMixer with any PyTorch training or evaluation
loop that expects a torch.utils.data.Dataset yielding MixtureSample objects.
The training loop (P2) and the baseline runner (models/baseline_runner.py) are
the primary consumers.

Overlap curriculum is threaded through __getitem__ via progress = idx / (n-1),
so OverlapScheduler automatically drives sparser overlap as training advances
without any loop-level bookkeeping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from coralsep.data.mixer import DynamicMixer
from coralsep.data.mixer_stub import MixtureSample
from coralsep.data.overlap_scheduler import OverlapScheduler


class DynamicMixDataset(Dataset):
    """
    Map-style Dataset wrapping DynamicMixer; each index produces one MixtureSample.

    Parameters
    ----------
    source_files:
        Paths to clean single-speaker WAV/FLAC files (LibriSpeech-style names).
    n_samples:
        Virtual dataset length — how many mixes to draw per epoch.  Each
        ``__getitem__`` call draws a new random mix regardless of index, so
        this is a budget, not a fixed pool.
    allowed_n:
        Speaker counts to mix.  Defaults to ``[2, 3]`` (the common eval case).
    db_min, db_max:
        Per-speaker volume offset range in dB (passed to DynamicMixer).
    train_speaker_ids:
        Restrict training pool to these speakers.
    test_speaker_ids:
        Reserve these speakers for ``split='test'``.
    overlap_scheduler:
        When provided, ``__getitem__(idx)`` threads ``progress = idx / (n-1)``
        to ``DynamicMixer.mix()``, applying the curriculum schedule.
    split:
        Pool split passed to ``DynamicMixer.mix()`` ('train', 'test', or any).
    seed:
        Seed for the DynamicMixer RNG.  Two datasets with the same seed and
        same source files will produce identical item sequences.
    """

    def __init__(
        self,
        source_files: list[Path | str],
        n_samples: int,
        *,
        allowed_n: list[int] | None = None,
        db_min: float = 0.0,
        db_max: float = 5.0,
        train_speaker_ids: set[str] | None = None,
        test_speaker_ids: set[str] | None = None,
        overlap_scheduler: OverlapScheduler | None = None,
        split: str = "train",
        seed: int = 0,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
        self._n = n_samples
        self._split = split
        self._overlap_scheduler = overlap_scheduler
        self._mixer = DynamicMixer(
            source_files=source_files,
            allowed_n=allowed_n if allowed_n is not None else [2, 3],
            db_min=db_min,
            db_max=db_max,
            train_speaker_ids=train_speaker_ids,
            test_speaker_ids=test_speaker_ids,
            rng=np.random.default_rng(seed),
            overlap_scheduler=overlap_scheduler,
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> MixtureSample:
        progress = idx / max(self._n - 1, 1)
        return self._mixer.mix(split=self._split, progress=progress)


def collate_mixture_samples(batch: list[MixtureSample]) -> dict[str, torch.Tensor]:
    """
    Collate a list of MixtureSample objects into padded batched tensors.

    Mixtures and references are zero-padded to the longest item in the batch
    along the time axis so all samples can be stacked into a single tensor.
    References are padded along N (speaker count) too, so batches with varying
    speaker counts are handled — the max N in the batch is used.

    Returns
    -------
    dict with keys:
      "mixture"     : float32 Tensor [B, T]
      "references"  : float32 Tensor [B, N, T]
      "n_speakers"  : int64 Tensor [B] — actual speaker count per item
      "sample_rate" : int64 Tensor [B]
    """
    max_t = max(s.mixture.shape[0] for s in batch)
    max_n = max(s.references.shape[0] for s in batch)

    mix_list: list[torch.Tensor] = []
    ref_list: list[torch.Tensor] = []
    n_spk_list: list[int] = []
    sr_list: list[int] = []

    for s in batch:
        t = s.mixture.shape[0]
        mix_pad = torch.zeros(max_t, dtype=torch.float32)
        mix_pad[:t] = torch.from_numpy(s.mixture)
        mix_list.append(mix_pad)

        n, t_ref = s.references.shape
        ref_pad = torch.zeros(max_n, max_t, dtype=torch.float32)
        ref_pad[:n, :t_ref] = torch.from_numpy(s.references)
        ref_list.append(ref_pad)

        n_spk_list.append(n)
        sr_list.append(s.sample_rate)

    return {
        "mixture": torch.stack(mix_list),
        "references": torch.stack(ref_list),
        "n_speakers": torch.tensor(n_spk_list, dtype=torch.int64),
        "sample_rate": torch.tensor(sr_list, dtype=torch.int64),
    }
