"""CPU smoke tests for condition analyzer and gate network (BLUEPRINT §5.4, §5.5)."""

from __future__ import annotations

import numpy as np
import torch

from coralsep.models.condition import (
    CORALSEP_SR,
    Level2Analyzer,
    level1_features,
    level1_tensor,
)
from coralsep.models.gate import GateNetwork, GateSmoother


def test_level1_features_on_random_audio():
    wav = torch.from_numpy(np.random.randn(8000).astype(np.float32) * 0.05)
    feats = level1_features(wav)
    assert "snr_est_db" in feats
    assert "voiced_density" in feats
    assert "codec_bw_ratio" in feats
    assert 0.0 <= feats["voiced_density"] <= 1.0


def test_level1_tensor_shape():
    wav = torch.randn(8000)
    t = level1_tensor(wav)
    assert t.shape == (4,)
    assert t.dtype == torch.float32


def test_level2_analyzer_output_shapes():
    analyzer = Level2Analyzer()
    e0 = torch.randn(1, 10, 65, 128)
    out = analyzer(e0)
    assert "count_probs" in out
    assert out["count_probs"].shape[-1] == 4
    assert "t60_s" in out


def test_gate_network_output_range():
    g = GateNetwork()
    cond = torch.randn(1, 10)
    gates = g(cond)
    assert gates.shape == (1, 3)
    assert (gates >= 0.0).all()
    assert (gates <= 1.5).all()


def test_gate_dict_keys():
    g = GateNetwork()
    cond = torch.randn(10)
    d = g.gate_dict(cond)
    assert set(d.keys()) == {"reverb", "noise", "codec"}
    assert all(0.0 <= v <= 1.5 for v in d.values())


def test_gate_l1_penalty_non_negative():
    g = GateNetwork()
    cond = torch.randn(4, 10)
    gates = g(cond)
    loss = g.l1_penalty(gates)
    assert loss.item() >= 0.0


def test_gate_smoother_ema_dampens_jump():
    smoother = GateSmoother(alpha=0.7)
    g1 = {"reverb": 1.0, "noise": 0.0, "codec": 0.5}
    g2 = {"reverb": 0.0, "noise": 1.0, "codec": 0.5}
    smoother.smooth(g1)
    s2 = smoother.smooth(g2)
    assert s2["reverb"] < 1.0
    assert s2["noise"] > 0.0
