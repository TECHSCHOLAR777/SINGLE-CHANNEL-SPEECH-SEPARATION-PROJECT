"""CPU smoke for counting, confidence, band recovery."""

from __future__ import annotations

import numpy as np
import torch

from models.band_recovery import BandRecoveryHead, apply_band_recovery, zero_pad_8k_to_16k
from models.confidence import ConfidenceSubsystem, inter_stage_consistency
from models.counting import CountingSubsystem, residual_energy_fraction


def test_counting_vote1_and_fuse():
    cs = CountingSubsystem(enabled_sweep=False)
    p_k = torch.tensor([[0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]])
    n = cs.vote1_from_pk(p_k)
    assert n == 2
    dec = cs.decide(p_k, [0.6, 0.2, 0.1, 0.1])
    assert 2 <= dec.n_hat <= 5
    assert dec.posterior.shape == (4,)


def test_residual_energy():
    mix = np.random.randn(1000).astype(np.float32)
    streams = np.stack([mix * 0.5, mix * 0.5], axis=0)
    r = residual_energy_fraction(mix, streams)
    assert 0.0 <= r < 0.5


def test_confidence_smoke():
    conf = ConfidenceSubsystem()
    p_k = np.array([0.0, 0.9, 0.8, 0.1, 0.1, 0.1, 0.0], dtype=np.float32)
    streams = np.random.randn(2, 1600).astype(np.float32) * 0.05
    mix = streams.sum(0)
    out = conf(p_k=p_k, streams=streams, mixture=mix)
    assert len(out.per_stream) == 2
    assert 0.0 <= out.completeness <= 1.0


def test_inter_stage_consistency():
    a = np.random.randn(2, 4, 8).astype(np.float32)
    scores = inter_stage_consistency(a, a)
    assert len(scores) == 2
    assert all(s > 0.99 for s in scores)


def test_band_recovery_shapes():
    head = BandRecoveryHead()
    streams = np.random.randn(2, 1600).astype(np.float32) * 0.05
    mix16 = zero_pad_8k_to_16k(streams.sum(0))
    result = apply_band_recovery(streams, mix16, head)
    assert result.waveforms_16k.shape[0] == 2
    assert result.waveforms_16k.shape[1] > streams.shape[1]


def test_band_recovery_disabled_bypass():
    head = BandRecoveryHead()
    head.enabled = False
    streams = np.random.randn(2, 800).astype(np.float32) * 0.05
    mix16 = zero_pad_8k_to_16k(streams.sum(0))
    result = apply_band_recovery(streams, mix16, head)
    assert result.applied == [False, False]
