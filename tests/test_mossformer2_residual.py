"""Tests for MossFormer2 residual padding (unblocks fusion on 3-spk mixtures)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.experts.mossformer2 import MossFormer2Expert
from models.fusion import CRRRFusionHead

T = 8000
pad = MossFormer2Expert._pad_to_target


def _mix_and_streams() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a, b, c = (rng.standard_normal(T).astype(np.float32) * 0.1 for _ in range(3))
    mixture = a + b + c
    emitted = np.stack([a, b])  # MossFormer2_SS_16K only ever gives 2
    return mixture, emitted


def test_no_padding_when_target_is_none():
    mixture, emitted = _mix_and_streams()
    streams, conf, synth = pad(emitted, [1.0, 1.0], mixture, None)
    assert streams.shape[0] == 2
    assert synth == [None, None]


def test_no_padding_when_model_already_covers_target():
    mixture, emitted = _mix_and_streams()
    streams, conf, synth = pad(emitted, [1.0, 1.0], mixture, 2)
    assert streams.shape[0] == 2
    assert synth == [None, None]


def test_residual_recovers_the_missing_speaker():
    """The third stream is mixture minus the two the model found: speaker C."""
    rng = np.random.default_rng(0)
    a, b, c = (rng.standard_normal(T).astype(np.float32) * 0.1 for _ in range(3))
    mixture = a + b + c
    streams, conf, synth = pad(np.stack([a, b]), [1.0, 1.0], mixture, 3)

    assert streams.shape == (3, T)
    assert synth == [None, None, "residual"]
    assert conf[2] == 0.0  # marked untrustworthy
    np.testing.assert_allclose(streams[2], c, atol=1e-5)


def test_gap_larger_than_one_zero_fills_the_rest():
    mixture, emitted = _mix_and_streams()
    streams, conf, synth = pad(emitted, [1.0, 1.0], mixture, 5)
    assert streams.shape[0] == 5
    assert synth == [None, None, "residual", "zero", "zero"]
    assert np.all(streams[3] == 0.0)
    assert np.all(streams[4] == 0.0)


def test_padded_streams_unblock_the_fusion_head():
    """The whole point: without padding, CRRRFusionHead raises on shape mismatch."""
    mixture, emitted = _mix_and_streams()
    head = CRRRFusionHead()
    mix_t = torch.from_numpy(mixture).unsqueeze(0)  # [1, T]
    sr = torch.randn(1, 3, T)  # SR-CorrNet gives 3

    moss_raw = torch.from_numpy(emitted).unsqueeze(0)  # [1, 2, T]
    with pytest.raises(ValueError, match="stream shape mismatch"):
        head(sr, moss_raw, mix_t, torch.ones(1, 3), torch.ones(1, 3), torch.ones(1, 3))

    padded, _, _ = pad(emitted, [1.0, 1.0], mixture, 3)
    moss_padded = torch.from_numpy(padded).unsqueeze(0)  # [1, 3, T]
    out = head(sr, moss_padded, mix_t, torch.ones(1, 3), torch.ones(1, 3), torch.ones(1, 3))
    assert out.fused_streams.shape == (1, 3, T)
    assert torch.isfinite(out.fused_streams).all()


def test_residual_energy_is_a_real_escalation_signal():
    """A clean 2-speaker mix leaves near-zero residual; a 3-speaker mix does not."""
    rng = np.random.default_rng(1)
    a, b, c = (rng.standard_normal(T).astype(np.float32) * 0.1 for _ in range(3))

    two_spk = a + b
    _, _, _ = pad(np.stack([a, b]), [1.0, 1.0], two_spk, 3)
    clean_residual = two_spk - np.stack([a, b]).sum(axis=0)

    three_spk = a + b + c
    dirty_residual = three_spk - np.stack([a, b]).sum(axis=0)

    assert np.sqrt(np.mean(clean_residual**2)) < 1e-6
    assert np.sqrt(np.mean(dirty_residual**2)) > 0.05
