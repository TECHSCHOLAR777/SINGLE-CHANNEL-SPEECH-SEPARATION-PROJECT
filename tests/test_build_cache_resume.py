"""
Prove build_train_cache.py resumes correctly after a hard interruption.

Simulates a Kaggle session dying mid cache-build (a real risk: cell 4 can run
for hours) by raising KeyboardInterrupt partway through the frozen-expert loop,
then re-running build_cache and checking every sample ends up written exactly
once — no duplicates, no gaps, no reprocessing of already-flushed shards.

All experts are faked (no weights, no network) so this runs on CPU in CI.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import scripts.build_train_cache as btc
from data.mixer_stub import MixtureSample
from train.cached_dataset import CachedExpertDataset

SR = 16000
SEG_SECONDS = 0.05
SEG_LEN = int(SR * SEG_SECONDS)
K = 3
N_SAMPLES = 10
SHARD_SIZE = 3


def _synthetic_samples(n: int) -> list[MixtureSample]:
    """Each sample's mixture is a constant equal to its index / 100 — a unique,
    round-trip-safe marker recoverable from the cached quality_db field."""
    out = []
    for i in range(n):
        val = np.float32(i / 100.0)
        mixture = np.full(SEG_LEN, val, dtype=np.float32)
        refs = np.stack([mixture] * K, axis=0)
        out.append(
            MixtureSample(mixture=mixture, references=refs, sample_rate=SR, utterance_id=f"u{i}")
        )
    return out


def _fake_result(streams: np.ndarray, confidence: float = 0.9) -> SimpleNamespace:
    meta = [SimpleNamespace(embedding=None, confidence=confidence) for _ in range(streams.shape[0])]
    return SimpleNamespace(streams=streams, metadata=meta)


class _FakeExpert:
    """Stands in for both MossFormer2Expert and SRCorrNetExpert."""

    is_available = True

    def __init__(self, **kwargs) -> None:
        self.k = kwargs.get("target_speakers") or kwargs.get("num_speakers") or K

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SimpleNamespace:
        streams = np.stack([mixture] * self.k, axis=0).astype(np.float32)
        return _fake_result(streams)


class _FakeRealm:
    def __init__(self, **kwargs) -> None:
        pass

    def estimate(
        self, mixture: np.ndarray, streams: np.ndarray, sample_rate: int
    ) -> SimpleNamespace:
        # Encode the sample's identity into the cached quality_db field so the
        # test can verify exactly which samples ended up in the cache.
        return SimpleNamespace(min_sisnr_db=float(mixture[0]) * 100.0)


class _FakeEcapa:
    def __init__(self, **kwargs) -> None:
        pass

    def embed_stream(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)


class _CrashAt:
    """Raises KeyboardInterrupt the Nth time a wrapped expert's separate() is
    called, simulating a session dying mid-sample (not a per-sample failure —
    build_cache's per-sample try/except only catches Exception, not
    KeyboardInterrupt, so this propagates and aborts the run, same as a real
    Kaggle timeout would)."""

    def __init__(self, inner: _FakeExpert, crash_call_index: int) -> None:
        self.inner = inner
        self.crash_call_index = crash_call_index
        self.calls = 0

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SimpleNamespace:
        self.calls += 1
        if self.calls == self.crash_call_index:
            raise KeyboardInterrupt("simulated session death")
        return self.inner.separate(mixture, sample_rate)


def _args(out_dir, no_resume: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        librimix_root=None,
        subset="train-100",
        mix_dir="mix_clean",
        mode="min",
        dynamic_source_glob="unused",
        allowed_n=[K],
        limit=N_SAMPLES,
        target_speakers=K,
        srcorrnet_repo=None,
        srcorrnet_checkpoint=None,
        srcorrnet_hf_model="fake/model",
        out_dir=str(out_dir),
        shard_size=SHARD_SIZE,
        segment_seconds=SEG_SECONDS,
        trivial_seg_seconds=1.0,
        sample_rate=SR,
        device="cpu",
        seed=0,
        no_resume=no_resume,
    )


def _patch_experts(monkeypatch, moss_wrapper=None, sr_wrapper=None) -> None:
    monkeypatch.setattr(
        "models.experts.mossformer2.MossFormer2Expert",
        moss_wrapper or (lambda **kw: _FakeExpert(**kw)),
    )
    monkeypatch.setattr(
        "models.experts.srcorrnet.SRCorrNetExpert",
        sr_wrapper or (lambda **kw: _FakeExpert(**kw)),
    )
    monkeypatch.setattr("models.realm_quality.REALMQualityEstimator", _FakeRealm)
    monkeypatch.setattr("models.experts.embeddings.ECAPAEmbedder", _FakeEcapa)
    monkeypatch.setattr(btc, "load_dynamic_samples", lambda *a, **k: _synthetic_samples(N_SAMPLES))


def test_resume_after_simulated_crash_writes_every_sample_exactly_once(tmp_path, monkeypatch):
    out_dir = tmp_path / "cache"

    crash_moss = _FakeExpert(target_speakers=K)
    crasher = _CrashAt(crash_moss, crash_call_index=7)  # after 6 samples flushed, mid-shard-3
    _patch_experts(monkeypatch, moss_wrapper=lambda **kw: crasher)

    with pytest.raises(KeyboardInterrupt):
        btc.build_cache(_args(out_dir))

    progress = btc._load_progress(out_dir)
    assert progress["next_index"] == 6  # last flush point (shards of size 3: 0-2, 3-5)
    assert sorted(p.name for p in out_dir.glob("shard_*.pt")) == [
        "shard_00000.pt",
        "shard_00001.pt",
    ]

    # Resume: fresh (non-crashing) experts, same deterministic sample source.
    _patch_experts(monkeypatch)
    manifest = btc.build_cache(_args(out_dir))

    assert manifest["n_written"] == N_SAMPLES
    assert manifest["n_skipped"] == 0

    ds = CachedExpertDataset(out_dir)
    assert len(ds) == N_SAMPLES

    seen_indices = set()
    for i in range(len(ds)):
        batch = ds[i]
        idx = round(float(batch.quality_scores_db) / 100.0 * 100)  # recover encoded index
        seen_indices.add(idx)
    assert seen_indices == set(range(N_SAMPLES))  # every original sample present, none duplicated


def test_no_resume_flag_restarts_from_scratch(tmp_path, monkeypatch):
    out_dir = tmp_path / "cache"
    _patch_experts(monkeypatch)

    btc.build_cache(_args(out_dir))
    assert len(list(out_dir.glob("shard_*.pt"))) == 4  # 3+3+3+1

    manifest = btc.build_cache(_args(out_dir, no_resume=True))
    assert manifest["n_written"] == N_SAMPLES  # reprocessed all 10, not skipped
