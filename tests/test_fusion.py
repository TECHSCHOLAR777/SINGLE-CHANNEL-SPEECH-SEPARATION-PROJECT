"""Tests for models/fusion.py (skipped when torch unavailable)."""

import pytest

torch = pytest.importorskip("torch")

from models.fusion import CRRRFusionHead, compute_fusion_gate_features  # noqa: E402


def test_fusion_output_shape() -> None:
    head = CRRRFusionHead(hidden_channels=128)
    b, k, t = 2, 3, 2000
    sr = torch.randn(b, k, t)
    moss = torch.randn(b, k, t)
    mix = torch.randn(b, t)
    out = head(
        sr_streams=sr,
        moss_streams=moss,
        mixture=mix,
        sr_confidence=torch.full((b, k), 0.9),
        moss_mask_entropy=torch.full((b, k), 0.2),
        scene_weight_tf=torch.full((b, k), 0.6),
    )
    assert out.fused_streams.shape == (b, k, t)
    assert out.residual.shape == (b, k, t)
    assert out.alpha.shape == (b, k, 1)
    assert torch.all(out.alpha >= 0) and torch.all(out.alpha <= 1)


def test_fusion_formula_residual_path() -> None:
    head = CRRRFusionHead(hidden_channels=64)
    sr = torch.zeros(1, 2, 512)
    moss = torch.zeros(1, 2, 512)
    mix = torch.zeros(1, 512)
    out = head(
        sr_streams=sr,
        moss_streams=moss,
        mixture=mix,
        sr_confidence=torch.ones(1, 2) * 0.5,
        moss_mask_entropy=torch.ones(1, 2) * 0.5,
        scene_weight_tf=torch.ones(1, 2) * 0.5,
    )
    expected = sr + out.alpha * out.residual
    assert torch.allclose(out.fused_streams, expected)


def test_parameter_budget_near_one_million() -> None:
    head = CRRRFusionHead(hidden_channels=256)
    n = head.parameter_count()
    assert 5.0e5 < n < 1.5e6


def test_gate_features_four_channels() -> None:
    b, k, t = 1, 2, 256
    feats = compute_fusion_gate_features(
        torch.randn(b, k, t),
        torch.randn(b, k, t),
        torch.randn(b, t),
        torch.ones(b, k) * 0.8,
        torch.ones(b, k) * 0.3,
        torch.ones(b, k) * 0.5,
    )
    assert feats.shape == (b, k, 4)


def test_fusion_backward() -> None:
    head = CRRRFusionHead(hidden_channels=64)
    sr = torch.randn(2, 3, 1024, requires_grad=True)
    moss = torch.randn(2, 3, 1024)
    mix = torch.randn(2, 1024)
    out = head(
        sr_streams=sr,
        moss_streams=moss,
        mixture=mix,
        sr_confidence=torch.full((2, 3), 0.7),
        moss_mask_entropy=torch.full((2, 3), 0.4),
        scene_weight_tf=torch.full((2, 3), 0.6),
    )
    out.fused_streams.mean().backward()
    assert sr.grad is not None
