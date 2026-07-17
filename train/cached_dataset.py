"""
Cached-expert training dataset (Dev C).

Consumes the shards written by `scripts/build_train_cache.py` and yields
`TrainBatch` objects, so `CAMoSETrainer` trains without touching a frozen
expert. Waveforms are stored fp16 on disk and cast back to fp32 on load, which
halves cache size at no cost to a training loop that runs in fp32 anyway.

    from train.cached_dataset import cached_train_loader

    loader = cached_train_loader("/kaggle/input/camose-cache/train", batch_size=8)
    for batch in loader:
        breakdown, n_escalated = trainer.train_step(batch)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from train.trainer import TrainBatch, _collate_train_batch

_WAVE_KEYS = ("mixture", "references", "moss_streams", "sr_streams")


class CachedExpertDataset(Dataset):
    """
    Lazily-loaded shards of precomputed expert outputs.

    Shards are loaded on first access and held, so a run over the dataset pages
    each shard in exactly once. With the default 256-record shards a cache of a
    few thousand crops sits comfortably in RAM; if it does not, lower
    --shard-size at build time rather than fighting this class.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"no manifest.json in {self.cache_dir}. "
                "Point this at a directory written by scripts/build_train_cache.py."
            )
        self.manifest = json.loads(manifest_path.read_text())

        self.shard_paths = sorted(self.cache_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"no shard_*.pt in {self.cache_dir}")

        self._cache: dict[int, list[dict]] = {}
        self._index: list[tuple[int, int]] = []
        size = int(self.manifest["shard_size"])
        total = int(self.manifest["records"])
        for shard_i in range(len(self.shard_paths)):
            in_shard = min(size, total - shard_i * size)
            self._index.extend((shard_i, j) for j in range(max(in_shard, 0)))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_speakers(self) -> int:
        return int(self.manifest["num_speakers"])

    def _shard(self, i: int) -> list[dict]:
        if i not in self._cache:
            self._cache[i] = torch.load(self.shard_paths[i], map_location="cpu", weights_only=False)
        return self._cache[i]

    def __getitem__(self, idx: int) -> TrainBatch:
        shard_i, offset = self._index[idx]
        record = self._shard(shard_i)[offset]
        waves = {key: record[key].float() for key in _WAVE_KEYS}
        return TrainBatch(
            mixture=waves["mixture"],
            references=waves["references"],
            true_count=record["true_count"],
            trivial_mask=record["trivial_mask"],
            moss_streams=waves["moss_streams"],
            sr_streams=waves["sr_streams"],
            quality_scores_db=record["quality_scores_db"],
            sr_confidence=record["sr_confidence"],
            moss_mask_entropy=record["moss_mask_entropy"],
        )


def cached_train_loader(
    cache_dir: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader over a cache directory, collated into batched TrainBatch."""
    dataset = CachedExpertDataset(cache_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_train_batch,
    )
