"""CPU smoke tests for counting, confidence, band recovery (BLUEPRINT §5.6, §5.7, §5.9)."""

from __future__ import annotations

import torch

from coralsep.models.band_recovery import BandRecoveryHead
from coralsep.models.confidence import (
    StreamConfidenceHead,
    inter_stage_consistency,
    residual_energy_fraction,
)
from coralsep.models.counting import (
    ThreeVoteCounter,
    attractor_confidence,
    count_from_attractors,
)


def test_count_from_attractors_clamp():
    probs = torch.tensor([[0.0, 0.9, 0.9, 0.1, 0.1, 0.1, 0.0]])
    assert count_from_attractors(probs) == 2
    probs5 = torch.tensor([[0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0]])
    assert count_from_attractors(probs5) == 5
    probs0 = torch.tensor([[0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]])
    assert count_from_attractors(probs0) == 2


def test_three_vote_counter_n_clamped():
    counter = ThreeVoteCounter()
    probs = torch.tensor([[0.0, 0.9, 0.9, 0.1, 0.1, 0.1, 0.0]])
    prior = torch.tensor([[0.6, 0.2, 0.1, 0.1]])
    mix = torch.randn(8000)
    sep = torch.randn(5, 8000)
    result = counter.estimate(probs, prior, mix, sep)
    assert 2 <= result["n_est"] <= 5
    assert result["n_v1"] in {2, 3, 4, 5}


def test_residual_energy_fraction_range():
    mix = torch.randn(8000)
    streams = torch.stack([mix * 0.5, mix * 0.5])
    r = residual_energy_fraction(mix, streams)
    assert 0.0 <= r <= 1.0


def test_attractor_confidence_range():
    probs = torch.tensor([[0.0, 0.9, 0.85, 0.1, 0.1, 0.1, 0.0]])
    conf = attractor_confidence(probs, n_est=2)
    assert 0.0 <= conf <= 1.0


def test_confidence_head_output_shape():
    head = StreamConfidenceHead()
    feats = torch.rand(1, 3, 3)
    out = head(feats)
    assert out.shape == (1, 3)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_inter_stage_consistency_no_dec():
    e0 = torch.randn(1, 5, 65, 128)
    score = inter_stage_consistency(e0, [], stream_idx=0)
    assert score == 1.0


def test_band_recovery_head_forward_shape():
    head = BandRecoveryHead()
    K, T, F_8k, F_high = 2, 24, 65, 129
    mag_8k = torch.rand(K, F_8k, T)
    mag_high = torch.rand(K, F_high, T)
    mask = head(mag_8k, mag_high)
    assert mask.shape[0] == K
    assert mask.shape[1] == F_high
    assert (mask >= 0.0).all() and (mask <= 1.0).all()
