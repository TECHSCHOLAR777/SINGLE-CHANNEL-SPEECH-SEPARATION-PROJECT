"""CPU smoke tests for condition analyzer and gate."""

from __future__ import annotations

import numpy as np
import torch

from models.condition import ConditionAnalyzer, ConditionVector
from models.gate import GateMLP


def test_level1_on_random_audio():
    wav = np.random.randn(8000).astype(np.float32) * 0.05
    a = ConditionAnalyzer()
    l1 = a.forward_level1(wav, 8000)
    assert "snr_db" in l1
    assert "voiced_density" in l1
    assert 0.0 <= l1["voiced_density"] <= 1.0


def test_level2_shapes():
    a = ConditionAnalyzer()
    e0 = torch.randn(1, 10, 65, 128)
    l1 = {"snr_db": 10.0, "voiced_density": 0.5, "codec_class": "none", "codec_class_idx": 0, "codec_bitrate_bps": 0.0}
    l2 = a.forward_level2(e0, l1)
    assert len(l2["count_prior"]) == 4
    assert l2["t60_s"] >= 0.0


def test_gate_per_adapter_and_ema():
    g = GateMLP(mode="per_adapter", n_layers=17)
    cond = ConditionVector(snr_db=0.0, t60_s=0.5, voiced_density=0.4)
    g1 = g(cond, apply_ema=True)
    g2 = g(cond, apply_ema=True)
    assert set(g1) == {"reverb", "noise", "codec"}
    assert all(0.0 <= v <= 1.5 for v in g1.values())
    # EMA should be defined on second call
    assert set(g2) == set(g1)


def test_gate_sparsity_loss():
    g = GateMLP(mode="per_adapter")
    t = torch.ones(1, 3)
    loss = g.sparsity_loss(t)
    assert float(loss) > 0
