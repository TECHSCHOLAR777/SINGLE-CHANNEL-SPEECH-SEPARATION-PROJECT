"""
Mixed-N cache build: variable speaker counts (2..5) into a fixed-K cache.

The clean all-3-speaker eval showed the cascade can't beat SR-CorrNet (P2-INT4)
because the cheap 2-speaker model can't contribute and the fusion has nothing to
fix. The real P2-INT4 attempt needs a MIXED 2-5 speaker cache. This test proves
the build loop:
  * accepts allowed_n = [2,3,4,5] and produces samples with varying true_count,
  * pads/truncates every sample's streams to the same fixed K (so the cache
    stacks), and
  * targets the expensive expert at each sample's TRUE count (var-2-5spk gets
    n_spks = n_true, not a fixed K).

All experts are faked — CPU only, no weights, no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import scripts.build_train_cache as btc
from data.mixer_stub import MixtureSample
from train.cached_dataset import CachedExpertDataset

SR = 16000
SEG_SECONDS = 0.05
SEG_LEN = int(SR * SEG_SECONDS)
K = 5  # fixed cache width = max speakers
N_PER_SAMPLE = [2, 3, 4, 5, 2, 3, 4, 5]
N_SAMPLES = len(N_PER_SAMPLE)


def _mixed_samples(_n: int) -> list[MixtureSample]:
    out = []
    for i, n in enumerate(N_PER_SAMPLE):
        rng = np.random.default_rng(i)
        refs = rng.standard_normal((n, SEG_LEN)).astype(np.float32)
        mixture = refs.sum(axis=0)
        out.append(
            MixtureSample(
                mixture=mixture, references=refs, sample_rate=SR, utterance_id=f"u{i}_n{n}"
            )
        )
    return out


class _FakeMoss:
    """Cheap expert: always emits K streams (residual-padded), like the real one."""

    def __init__(self, **kw) -> None:
        self.k = kw.get("target_speakers") or K

    def separate(self, mixture, sample_rate):
        streams = np.stack([mixture.astype(np.float32)] * self.k, axis=0)
        meta = [SimpleNamespace(embedding=None, confidence=0.9) for _ in range(self.k)]
        return SimpleNamespace(streams=streams, metadata=meta)


class _FakeSrCorrNet:
    """Expensive expert: emits exactly self.num_speakers streams (variable-N)."""

    seen_counts: list[int] = []
    is_available = True

    def __init__(self, **kw) -> None:
        self.num_speakers = kw.get("num_speakers") or K

    def separate(self, mixture, sample_rate):
        n = int(self.num_speakers)
        _FakeSrCorrNet.seen_counts.append(n)  # record what each sample targeted
        streams = np.stack([mixture.astype(np.float32)] * n, axis=0)
        meta = [SimpleNamespace(embedding=None, confidence=0.8) for _ in range(n)]
        return SimpleNamespace(streams=streams, metadata=meta)


class _FakeRealm:
    def __init__(self, **kw) -> None:
        pass

    def estimate(self, mixture, streams, sample_rate):
        return SimpleNamespace(min_sisnr_db=7.0)


class _FakeEcapa:
    def __init__(self, **kw) -> None:
        pass

    def embed_stream(self, wav, sample_rate):
        return np.zeros(4, dtype=np.float32)


def _args(out_dir) -> SimpleNamespace:
    return SimpleNamespace(
        librimix_root=None,
        subset="train-100",
        mix_dir="mix_clean",
        mode="min",
        dynamic_source_glob="unused",
        allowed_n=[2, 3, 4, 5],
        limit=N_SAMPLES,
        target_speakers=K,
        srcorrnet_repo=None,
        srcorrnet_checkpoint=None,
        srcorrnet_hf_model="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk",
        out_dir=str(out_dir),
        shard_size=4,
        segment_seconds=SEG_SECONDS,
        trivial_seg_seconds=1.0,
        sample_rate=SR,
        device="cpu",
        seed=0,
        no_resume=True,
    )


def test_mixed_n_build_fixed_k_varying_true_count(tmp_path, monkeypatch):
    _FakeSrCorrNet.seen_counts = []
    monkeypatch.setattr("models.experts.mossformer2.MossFormer2Expert", _FakeMoss)
    monkeypatch.setattr("models.experts.srcorrnet.SRCorrNetExpert", _FakeSrCorrNet)
    monkeypatch.setattr("models.realm_quality.REALMQualityEstimator", _FakeRealm)
    monkeypatch.setattr("models.experts.embeddings.ECAPAEmbedder", _FakeEcapa)
    monkeypatch.setattr(btc, "load_dynamic_samples", lambda *a, **k: _mixed_samples(N_SAMPLES))

    out_dir = tmp_path / "cache_mixed"
    manifest = btc.build_cache(_args(out_dir))

    assert manifest["n_written"] == N_SAMPLES
    assert manifest["n_skipped"] == 0
    assert manifest["allowed_n"] == [2, 3, 4, 5]
    assert manifest["target_speakers"] == K

    # The expensive expert was targeted at each sample's TRUE count, in order.
    assert _FakeSrCorrNet.seen_counts == N_PER_SAMPLE

    ds = CachedExpertDataset(out_dir)
    assert len(ds) == N_SAMPLES

    seen_true_counts = []
    for i in range(len(ds)):
        b = ds[i]
        # Every sample padded to the SAME fixed K, regardless of true count.
        assert b.moss_streams.shape[0] == K
        assert b.sr_streams.shape[0] == K
        seen_true_counts.append(int(round(float(b.true_count))))

    # true_count preserves the real per-sample speaker count (2..5).
    assert seen_true_counts == N_PER_SAMPLE
    assert set(seen_true_counts) == {2, 3, 4, 5}
