"""Tests for train/losses.py (skipped when torch unavailable)."""

import pytest

torch = pytest.importorskip("torch")

from models.router import TwoLevelRouter, load_balance_loss, null_sparsity_loss  # noqa: E402
from train.losses import (  # noqa: E402
    CompositeLoss,
    LossWeights,
    MultiResolutionSTFTLoss,
    SpeakerConsistencyLoss,
    count_bce_loss,
    neg_si_sdr,
    pit_si_sdr_loss,
    residual_regularization_loss,
)


def test_neg_si_sdr_perfect_match_is_low() -> None:
    ref = torch.sin(torch.linspace(0, 8, 256))
    loss = neg_si_sdr(ref, ref)
    assert float(loss) < -20.0


def test_pit_si_sdr_resolves_permutation() -> None:
    t = 256
    time = torch.linspace(0, 1, t)
    ref = torch.stack(
        [
            torch.sin(2 * torch.pi * 3.0 * time),
            torch.sin(2 * torch.pi * 5.0 * time) + 0.5 * time,
            torch.sin(2 * torch.pi * 7.0 * time) ** 2,
        ]
    )
    est = ref[[1, 2, 0]]
    loss_wrong_order = (
        neg_si_sdr(est[0], ref[0]) + neg_si_sdr(est[1], ref[1]) + neg_si_sdr(est[2], ref[2])
    )
    loss_right = pit_si_sdr_loss(ref.unsqueeze(0), ref.unsqueeze(0))
    assert float(loss_right) < float(loss_wrong_order / 3.0)


def test_multi_res_stft_zero_on_identical() -> None:
    loss_fn = MultiResolutionSTFTLoss(fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,))
    x = torch.randn(2, 3, 512)
    assert float(loss_fn(x, x)) < 1e-6


def test_residual_regularization_penalizes_large_residual() -> None:
    small = torch.ones(2, 3, 100) * 0.01
    large = torch.ones(2, 3, 100) * 2.0
    assert residual_regularization_loss(large) > residual_regularization_loss(small)


def test_count_bce_loss_decreases_with_correct_class() -> None:
    logits = torch.zeros(4, 5)
    logits[:, 2] = 5.0
    targets = torch.tensor([3.0, 3.0, 3.0, 3.0])
    good = count_bce_loss(logits, targets)
    bad_logits = torch.zeros(4, 5)
    bad = count_bce_loss(bad_logits, targets)
    assert float(good) < float(bad)


def test_speaker_consistency_loss_trainable() -> None:
    loss_fn = SpeakerConsistencyLoss(embedding_dim=32)
    z = torch.randn(4, 3, 32)
    y = z + 0.01 * torch.randn(4, 3, 32)
    loss = loss_fn(z, y)
    loss.backward()
    assert any(p.grad is not None for p in loss_fn.parameters())


def test_composite_loss_default_embedding_dim_matches_real_ecapa() -> None:
    """
    Regression: CompositeLoss's default embedding_dim must match the REAL
    ECAPA-TDNN output (speechbrain/spkrec-ecapa-voxceleb = 192-dim), the
    embedder MossFormer2Expert/ECAPAEmbedder actually use. The old hardcoded
    default of 64 passed every CI test (which all used synthetic 64-dim
    embeddings) but crashed the first real Kaggle training run with a
    "shapes cannot be multiplied (24x192 and 64x64)" error the moment real
    ECAPA embeddings reached the speaker-consistency loss.
    """
    b, k, d = 2, 3, 192
    composite = CompositeLoss(LossWeights())  # real default, no override
    z = torch.randn(b, k, d)
    y = torch.randn(b, k, d)
    loss = composite.speaker_loss(z, y)
    assert torch.isfinite(loss)


def test_composite_loss_all_seven_terms() -> None:
    b, k, t = 2, 3, 512
    estimates = torch.randn(b, k, t)
    references = torch.randn(b, k, t)
    router = TwoLevelRouter(feature_dim=16, num_experts=3, hidden_dim=32)
    weights = router(torch.randn(b, 4, 16))
    composite = CompositeLoss(LossWeights(), embedding_dim=64)  # must match embeddings below
    breakdown = composite(
        estimates=estimates,
        references=references,
        count_logits=torch.randn(b, 5),
        true_counts=torch.tensor([3.0, 3.0]),
        router_weights=weights,
        trivial_mask=torch.zeros(b, 4),
        null_index=2,
        fusion_residual=torch.randn(b, k, t) * 0.1,
        stream_embeddings=torch.randn(b, k, 64),
        reference_embeddings=torch.randn(b, k, 64),
    )
    assert breakdown.total.ndim == 0
    assert len(breakdown.weighted) == 7
    assert breakdown.weighted["si_sdr_upit"] != 0.0
    breakdown.total.backward()


def test_router_aux_losses_match_exports() -> None:
    w = torch.softmax(torch.randn(3, 5, 3), dim=-1)
    trivial = (torch.rand(3, 5) > 0.5).float()
    assert load_balance_loss(w).ndim == 0
    assert null_sparsity_loss(w, trivial, null_index=2).ndim == 0
