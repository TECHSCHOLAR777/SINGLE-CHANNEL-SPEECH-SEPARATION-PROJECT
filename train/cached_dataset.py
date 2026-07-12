"""
Cached frozen-expert dataset for CA-MoSE training (P2 unlock).

The experts (MossFormer2, the expensive TF expert, REAL-M, ECAPA) are frozen.
Re-running them every epoch is the dominant training cost and buys nothing, so
`scripts/build_train_cache.py` runs them **once** over a fixed set of mixtures
and writes their outputs to sharded ``.pt`` files. This module reads those
shards back into ``TrainBatch`` objects and feeds ``CAMoSETrainer`` with zero
experts loaded.

Shard format (one file ``shard_00000.pt`` ... per chunk)::

    {
        "version": CACHE_VERSION,
        "sample_rate": 16000,
        "samples": [ {<sample dict>}, ... ],
    }

Each sample dict stores fp16 waveforms (halves disk/IO) plus small metadata::

    mixture              [T]      fp16
    references           [N, T]   fp16
    moss_streams         [K, T]   fp16   (padded to target speakers)
    sr_streams           [K, T]   fp16   (Hungarian-aligned to moss order)
    true_count           scalar   fp32
    quality_db           scalar   fp32   (REAL-M min SI-SNR, the gate signal)
    sr_confidence        [K]      fp16
    moss_mask_entropy    [K]      fp16
    trivial_mask         [S]      fp16   (per ~1s segment silence flag)
    stream_embeddings    [K, D]   fp16   (ECAPA of moss streams)
    reference_embeddings [N, D]   fp16   (ECAPA of references)

All samples in a cache share the same ``T``, ``K`` and ``N`` (the builder crops
to a fixed segment length and pads streams to ``target_speakers``), so a plain
``DataLoader`` can stack them without a custom collate beyond
``_collate_train_batch``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from train.trainer import TrainBatch, _collate_train_batch

CACHE_VERSION = 1
SHARD_GLOB = "shard_*.pt"


def _to_f16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float16).contiguous()


def save_cache_shard(path: str | Path, samples: list[dict], sample_rate: int = 16000) -> None:
    """Write one shard of cached sample dicts to ``path``."""
    payload = {
        "version": CACHE_VERSION,
        "sample_rate": int(sample_rate),
        "samples": samples,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def make_sample_dict(
    *,
    mixture: torch.Tensor,
    references: torch.Tensor,
    moss_streams: torch.Tensor,
    sr_streams: torch.Tensor,
    true_count: float,
    quality_db: float,
    sr_confidence: torch.Tensor,
    moss_mask_entropy: torch.Tensor,
    trivial_mask: torch.Tensor,
    stream_embeddings: torch.Tensor | None,
    reference_embeddings: torch.Tensor | None,
) -> dict:
    """Assemble one cache sample dict, downcasting waveforms to fp16."""
    sample = {
        "mixture": _to_f16(mixture),
        "references": _to_f16(references),
        "moss_streams": _to_f16(moss_streams),
        "sr_streams": _to_f16(sr_streams),
        "true_count": float(true_count),
        "quality_db": float(quality_db),
        "sr_confidence": _to_f16(sr_confidence),
        "moss_mask_entropy": _to_f16(moss_mask_entropy),
        "trivial_mask": _to_f16(trivial_mask),
    }
    sample["stream_embeddings"] = None if stream_embeddings is None else _to_f16(stream_embeddings)
    sample["reference_embeddings"] = (
        None if reference_embeddings is None else _to_f16(reference_embeddings)
    )
    return sample


def _sample_to_batch(sample: dict) -> TrainBatch:
    """Reconstruct a single-sample TrainBatch (fp32) from a cache dict."""

    def f32(key: str) -> torch.Tensor:
        return sample[key].to(torch.float32)

    stream_emb = sample.get("stream_embeddings")
    ref_emb = sample.get("reference_embeddings")
    return TrainBatch(
        mixture=f32("mixture"),
        references=f32("references"),
        true_count=torch.tensor(float(sample["true_count"])),
        trivial_mask=f32("trivial_mask"),
        moss_streams=f32("moss_streams"),
        sr_streams=f32("sr_streams"),
        quality_scores_db=torch.tensor(float(sample["quality_db"])),
        sr_confidence=f32("sr_confidence"),
        moss_mask_entropy=f32("moss_mask_entropy"),
        stream_embeddings=None if stream_emb is None else stream_emb.to(torch.float32),
        reference_embeddings=None if ref_emb is None else ref_emb.to(torch.float32),
    )


class CachedExpertDataset(Dataset):
    """
    Map-style dataset over cached frozen-expert outputs.

    Parameters
    ----------
    cache_dir:
        Directory containing ``shard_*.pt`` files written by
        ``scripts/build_train_cache.py`` (or ``save_cache_shard``).
    keep_shards_in_memory:
        When True, every shard is loaded eagerly and held in RAM (fine for the
        small smoke caches). When False (default), one shard is memoised at a
        time — enough for sequential access and light shuffling.
    """

    def __init__(self, cache_dir: str | Path, keep_shards_in_memory: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        shards = sorted(self.cache_dir.glob(SHARD_GLOB))
        if not shards:
            raise FileNotFoundError(
                f"No {SHARD_GLOB} shards in {self.cache_dir}. "
                "Run scripts/build_train_cache.py first."
            )
        self._shard_paths = shards
        self._keep = keep_shards_in_memory
        self._mem: dict[int, list[dict]] = {}
        self._last_idx: int | None = None
        self._last_samples: list[dict] | None = None

        # Build a flat index: global i -> (shard_idx, local_idx).
        self._index: list[tuple[int, int]] = []
        self.sample_rate = 16000
        for si, path in enumerate(shards):
            payload = torch.load(path, map_location="cpu")
            self._check_version(payload, path)
            self.sample_rate = int(payload.get("sample_rate", 16000))
            n = len(payload["samples"])
            self._index.extend((si, li) for li in range(n))
            if self._keep:
                self._mem[si] = payload["samples"]

    @staticmethod
    def _check_version(payload: dict, path: Path) -> None:
        v = payload.get("version")
        if v != CACHE_VERSION:
            raise ValueError(
                f"Cache shard {path} has version {v}, expected {CACHE_VERSION}. "
                "Rebuild the cache with the current build_train_cache.py."
            )

    def _load_shard(self, shard_idx: int) -> list[dict]:
        if self._keep:
            return self._mem[shard_idx]
        if self._last_idx == shard_idx and self._last_samples is not None:
            return self._last_samples
        payload = torch.load(self._shard_paths[shard_idx], map_location="cpu")
        samples = payload["samples"]
        self._last_idx = shard_idx
        self._last_samples = samples
        return samples

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> TrainBatch:
        shard_idx, local_idx = self._index[idx]
        samples = self._load_shard(shard_idx)
        return _sample_to_batch(samples[local_idx])


def cached_train_loader(
    cache_dir: str | Path,
    batch_size: int = 4,
    shuffle: bool = True,
    keep_shards_in_memory: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader over a cached expert-output directory."""
    ds = CachedExpertDataset(cache_dir, keep_shards_in_memory=keep_shards_in_memory)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_train_batch,
        num_workers=num_workers,
    )
