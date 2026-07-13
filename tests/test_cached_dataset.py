"""
Cache round-trip + real train_step on cached data (P2 unlock verification).

No expert weights are downloaded: synthetic sample dicts are written through the
real shard format, read back by CachedExpertDataset, and fed into a real
CAMoSETrainer.train_step, asserting parameters actually move and the loss stays
finite. Also unit-tests the pure numeric helpers in build_train_cache.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from scripts.build_train_cache import (
    _crop_or_pad,
    _fit_k,
    align_expensive_to_cheap,
    mask_entropy_proxy,
    trivial_mask_proxy,
)
from train.cached_dataset import (
    CachedExpertDataset,
    cached_train_loader,
    make_sample_dict,
    save_cache_shard,
)
from train.trainer import build_trainer_from_config

SR = 16000
K = 3
T = 12000  # 0.75 s — small but real segment


def _synthetic_sample(seed: int, k: int = K, n: int = K, t: int = T) -> dict:
    rng = np.random.default_rng(seed)
    time = np.arange(t, dtype=np.float32) / SR
    refs = np.stack(
        [np.sin(2 * np.pi * f * time).astype(np.float32) for f in (300.0, 500.0, 700.0)[:n]],
        axis=0,
    )
    mixture = refs.sum(axis=0)
    moss = refs + rng.normal(0, 0.02, size=(k, t)).astype(np.float32)
    sr_streams = refs + rng.normal(0, 0.01, size=(k, t)).astype(np.float32)
    # 192-dim: matches real ECAPA-TDNN output (speechbrain/spkrec-ecapa-voxceleb)
    # and CompositeLoss's real default embedding_dim.
    emb = rng.normal(size=(k, 192)).astype(np.float32)
    ref_emb = emb + rng.normal(0, 0.05, size=(n, 192)).astype(np.float32)
    return make_sample_dict(
        mixture=torch.from_numpy(mixture),
        references=torch.from_numpy(refs),
        moss_streams=torch.from_numpy(moss),
        sr_streams=torch.from_numpy(sr_streams),
        true_count=float(n),
        quality_db=float(rng.uniform(6.0, 14.0)),
        sr_confidence=torch.full((k,), 0.8),
        moss_mask_entropy=torch.full((k,), 0.3),
        trivial_mask=torch.zeros(3),
        stream_embeddings=torch.from_numpy(emb),
        reference_embeddings=torch.from_numpy(ref_emb),
    )


def _build_cache(tmp_path, n_samples: int = 8, shard_size: int = 3):
    shard: list[dict] = []
    idx = 0
    for i in range(n_samples):
        shard.append(_synthetic_sample(i))
        if len(shard) >= shard_size:
            save_cache_shard(tmp_path / f"shard_{idx:05d}.pt", shard, SR)
            shard = []
            idx += 1
    if shard:
        save_cache_shard(tmp_path / f"shard_{idx:05d}.pt", shard, SR)


def test_cache_roundtrip_shapes(tmp_path):
    _build_cache(tmp_path, n_samples=8, shard_size=3)
    ds = CachedExpertDataset(tmp_path)
    assert len(ds) == 8
    batch = ds[0]
    assert batch.mixture.shape == (T,)
    assert batch.references.shape == (K, T)
    assert batch.moss_streams.shape == (K, T)
    assert batch.sr_streams.shape == (K, T)
    assert batch.mixture.dtype == torch.float32  # promoted back from fp16
    assert batch.stream_embeddings is not None and batch.stream_embeddings.shape == (K, 192)


def test_train_step_moves_params(tmp_path):
    _build_cache(tmp_path, n_samples=8, shard_size=4)
    trainer = build_trainer_from_config({"realm": {"quality_threshold_tau": 12.0}}, device="cpu")
    before = copy.deepcopy(trainer.model.state_dict())

    loader = cached_train_loader(tmp_path, batch_size=4, shuffle=False)
    metrics = trainer.train_epoch(loader)

    assert np.isfinite(metrics.loss)
    assert 0.0 <= metrics.escalation_rate <= 1.0
    after = trainer.model.state_dict()
    moved = any(not torch.equal(before[k], after[k]) for k in before)
    assert moved, "no trainable parameter changed after a cached train epoch"


def test_missing_cache_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        CachedExpertDataset(tmp_path)


def test_mask_entropy_in_unit_range():
    rng = np.random.default_rng(0)
    streams = rng.normal(size=(3, 4000)).astype(np.float32)
    ent = mask_entropy_proxy(streams)
    assert ent.shape == (3,)
    assert np.all(ent >= 0.0) and np.all(ent <= 1.0 + 1e-5)


def test_trivial_mask_flags_silence():
    sr = 16000
    loud = np.random.default_rng(0).normal(0, 1.0, size=sr).astype(np.float32)
    silent = np.zeros(sr, dtype=np.float32)
    mixture = np.concatenate([loud, silent])
    mask = trivial_mask_proxy(mixture, sr, seg_seconds=1.0)
    assert mask.shape == (2,)
    assert mask[0] == 0.0 and mask[1] == 1.0


def test_align_recovers_permutation():
    rng = np.random.default_rng(1)
    moss = rng.normal(size=(3, 2000)).astype(np.float32)
    perm = [2, 0, 1]
    sr = moss[perm]  # expensive expert emits a shuffled order
    aligned = align_expensive_to_cheap(moss, sr)
    # After alignment each row should match the corresponding moss row best.
    for i in range(3):
        assert np.corrcoef(aligned[i], moss[i])[0, 1] > 0.99


def test_fit_k_pad_and_truncate():
    x = np.ones((2, 100), dtype=np.float32)
    assert _fit_k(x, 3).shape == (3, 100)
    assert _fit_k(np.ones((5, 100), dtype=np.float32), 3).shape == (3, 100)
    assert _crop_or_pad(np.ones((3, 50), dtype=np.float32), 80).shape == (3, 80)
    assert _crop_or_pad(np.ones((3, 120), dtype=np.float32), 80).shape == (3, 80)


def test_evaluate_cache_untrained_reports(tmp_path):
    from scripts.evaluate_cascade import evaluate_cache

    _build_cache(tmp_path, n_samples=8, shard_size=4)
    report = evaluate_cache(tmp_path, checkpoint=None, device="cpu", batch_size=4)

    assert report["n_samples"] == 8
    assert 0.0 <= report["escalation_rate"] <= 1.0
    for k in ("cascade", "mossformer2", "expensive"):
        assert np.isfinite(report["si_sdri_db"][k])
    assert isinstance(report["cascade_beats_single_expert"], bool)
