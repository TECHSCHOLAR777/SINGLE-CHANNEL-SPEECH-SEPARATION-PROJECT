"""Tests for the expert cache: helpers, round-trip, and a real training step."""

from __future__ import annotations

# import the pure helpers from the build script without running its CLI
import importlib.util
import json
import pathlib

import numpy as np
import pytest
import torch

from models.cascade_gate import CascadeGate
from train.cached_dataset import CachedExpertDataset, cached_train_loader
from train.losses import CompositeLoss
from train.trainer import CAMoSETrainable, CAMoSETrainer

_spec = importlib.util.spec_from_file_location(
    "build_train_cache",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "build_train_cache.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
mask_entropy = _mod.mask_entropy
trivial_mask = _mod.trivial_mask

SR = 16000
T = 4 * SR
K = 3
N_SEG = 4


class TestMaskEntropy:
    def test_clean_separation_scores_low(self):
        """Disjoint streams own their frames, so the implied mask is confident."""
        streams = np.zeros((2, 1000), dtype=np.float32)
        streams[0, :500] = 1.0
        streams[1, 500:] = 1.0
        h = mask_entropy(streams)
        assert h.shape == (2,)
        assert np.all(h < 0.05)

    def test_ambivalent_separation_scores_high(self):
        """Two identical streams: the mask cannot tell them apart, m = 0.5."""
        streams = np.ones((2, 1000), dtype=np.float32)
        h = mask_entropy(streams)
        assert np.all(h > 0.95)

    def test_silence_does_not_produce_nan(self):
        h = mask_entropy(np.zeros((3, 1000), dtype=np.float32))
        assert np.all(np.isfinite(h))


class TestTrivialMask:
    def test_silent_segment_is_trivial(self):
        refs = np.zeros((K, T), dtype=np.float32)
        assert np.all(trivial_mask(refs, SR, N_SEG) == 1.0)

    def test_single_active_speaker_is_trivial(self):
        refs = np.zeros((K, T), dtype=np.float32)
        refs[0] = np.random.default_rng(0).standard_normal(T).astype(np.float32)
        assert np.all(trivial_mask(refs, SR, N_SEG) == 1.0)

    def test_overlapping_speakers_are_not_trivial(self):
        rng = np.random.default_rng(0)
        refs = rng.standard_normal((K, T)).astype(np.float32)
        assert np.all(trivial_mask(refs, SR, N_SEG) == 0.0)

    def test_mask_length_matches_n_seg(self):
        refs = np.random.default_rng(0).standard_normal((K, T)).astype(np.float32)
        for n in (2, 4, 8):
            assert trivial_mask(refs, SR, n).shape == (n,)


def _write_cache(tmp_path, records: int = 6, shard_size: int = 4):
    """Write a cache in exactly the format build_train_cache.py produces."""
    rng = np.random.default_rng(0)
    cache = tmp_path / "cache"
    cache.mkdir()

    shards, shard = 0, []
    for i in range(records):
        refs = (rng.standard_normal((K, T)) * 0.1).astype(np.float32)
        mix = refs.sum(axis=0)
        shard.append(
            {
                "utterance_id": f"utt{i}",
                "start": 0,
                "mixture": torch.from_numpy(mix).half(),
                "references": torch.from_numpy(refs).half(),
                "moss_streams": torch.from_numpy(refs + 0.01).half(),
                "sr_streams": torch.from_numpy(refs + 0.005).half(),
                "quality_scores_db": torch.tensor(8.0 if i % 2 else 15.0),
                "sr_confidence": torch.full((K,), 0.9),
                "moss_mask_entropy": torch.full((K,), 0.3),
                "trivial_mask": torch.zeros(N_SEG),
                "true_count": torch.tensor(float(K)),
            }
        )
        if len(shard) >= shard_size:
            torch.save(shard, cache / f"shard_{shards:05d}.pt")
            shards, shard = shards + 1, []
    if shard:
        torch.save(shard, cache / f"shard_{shards:05d}.pt")
        shards += 1

    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "subset": "train",
                "num_speakers": K,
                "sample_rate": SR,
                "crop_samples": T,
                "n_seg": N_SEG,
                "records": records,
                "shards": shards,
                "shard_size": shard_size,
            }
        )
    )
    return cache


class TestCachedDataset:
    def test_length_spans_all_shards(self, tmp_path):
        ds = CachedExpertDataset(_write_cache(tmp_path, records=6, shard_size=4))
        assert len(ds) == 6
        assert ds.num_speakers == K

    def test_fp16_on_disk_comes_back_fp32(self, tmp_path):
        ds = CachedExpertDataset(_write_cache(tmp_path))
        item = ds[0]
        for tensor in (item.mixture, item.references, item.moss_streams, item.sr_streams):
            assert tensor.dtype == torch.float32
        assert item.references.shape == (K, T)
        assert item.mixture.shape == (T,)

    def test_missing_manifest_raises_clearly(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            CachedExpertDataset(empty)

    def test_loader_collates_into_a_batch(self, tmp_path):
        loader = cached_train_loader(_write_cache(tmp_path), batch_size=3, shuffle=False)
        batch = next(iter(loader))
        assert batch.mixture.shape == (3, T)
        assert batch.references.shape == (3, K, T)
        assert batch.sr_streams.shape == (3, K, T)
        assert batch.quality_scores_db.shape == (3,)


def test_cache_drives_a_real_training_step(tmp_path):
    """The whole point: cached expert outputs train the heads, no expert loaded."""
    torch.manual_seed(0)
    loader = cached_train_loader(_write_cache(tmp_path, records=4, shard_size=4), batch_size=2)
    trainer = CAMoSETrainer(
        model=CAMoSETrainable(),
        gate=CascadeGate(tau=12.0, signal="min"),
        loss_fn=CompositeLoss(),
        device="cpu",
    )
    before = [p.detach().clone() for p in trainer.model.parameters()]

    batch = next(iter(loader))
    breakdown, n_escalated = trainer.train_step(batch)

    assert torch.isfinite(breakdown.total)
    assert n_escalated >= 0
    params_after = trainer.model.parameters()
    changed = any(
        not torch.equal(b, a.detach()) for b, a in zip(before, params_after, strict=False)
    )
    assert changed, "training step changed no parameters"
