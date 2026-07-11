"""Tests for models/router.py (skipped when torch is unavailable in the env)."""

import pytest

torch = pytest.importorskip("torch")

from models.router import TwoLevelRouter, load_balance_loss, null_sparsity_loss  # noqa: E402


def test_weights_shape_and_simplex() -> None:
    router = TwoLevelRouter(feature_dim=64, num_experts=3)
    feats = torch.randn(2, 5, 64)
    w = router(feats)
    assert w.shape == (2, 5, 3)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 5), atol=1e-5)


def test_parameter_budget_near_half_million() -> None:
    router = TwoLevelRouter()
    assert 3.0e5 < router.parameter_count() < 7.0e5


def test_load_balance_loss_prefers_uniform() -> None:
    uniform = torch.full((4, 6, 3), 1.0 / 3.0)
    collapsed = torch.zeros(4, 6, 3)
    collapsed[..., 0] = 1.0
    assert load_balance_loss(uniform) < 1e-6
    assert load_balance_loss(collapsed) > load_balance_loss(uniform)


def test_null_sparsity_loss_rewards_matching_target() -> None:
    good = torch.zeros(2, 4, 3)
    good[..., 2] = 0.99
    trivial = torch.ones(2, 4)
    bad = torch.zeros(2, 4, 3)
    bad[..., 2] = 0.01
    assert null_sparsity_loss(good, trivial, null_index=2) < null_sparsity_loss(
        bad, trivial, null_index=2
    )


def test_router_is_trainable() -> None:
    router = TwoLevelRouter(feature_dim=8, num_experts=3, hidden_dim=16)
    feats = torch.randn(3, 4, 8)
    loss = load_balance_loss(router(feats)) + router(feats).mean()
    loss.backward()
    grads = [p.grad for p in router.parameters() if p.grad is not None]
    assert len(grads) > 0
