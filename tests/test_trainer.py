"""Tests for train/trainer.py (skipped when torch unavailable)."""

import pytest

torch = pytest.importorskip("torch")

from train.trainer import (  # noqa: E402
    CAMoSETrainable,
    build_trainer_from_config,
    run_self_test,
    synth_train_loader,
    SyntheticTrainDataset,
    _collate_train_batch,
)


def test_trainable_module_forward_shapes() -> None:
    model = CAMoSETrainable(feature_dim=32, num_experts=3)
    mix = torch.randn(2, 8000)
    scene = model.scene_analyzer(mix)
    assert scene["segment_features"].ndim == 3
    assert scene["count_logits"].shape[0] == 2
    router_w = model.router(scene["segment_features"])
    assert router_w.shape[-1] == 3


def test_trainer_forward_batch_produces_loss() -> None:
    trainer = build_trainer_from_config({"realm": {"quality_threshold_tau": 12.0}})
    ds = SyntheticTrainDataset(n_samples=2, t=2000, seed=1)
    batch = _collate_train_batch([ds[0], ds[1]])
    out = trainer.forward_batch(batch)
    assert out.loss_breakdown is not None
    assert out.estimates.shape == batch.references.shape


def test_trainer_train_step_updates_weights() -> None:
    trainer = build_trainer_from_config({"realm": {"quality_threshold_tau": 12.0}})
    before = next(trainer.model.parameters()).detach().clone()
    ds = SyntheticTrainDataset(n_samples=1, t=2000, seed=2)
    batch = _collate_train_batch([ds[0]])
    trainer.train_step(batch)
    after = next(trainer.model.parameters()).detach()
    assert not torch.allclose(before, after)


def test_self_test_runs_and_finite_loss() -> None:
    result = run_self_test(epochs=1, device="cpu")
    assert result["epochs"] == 1
    assert result["history"][0]["loss"] == pytest.approx(result["history"][0]["loss"])
    assert result["trainable_params"] > 0


def test_synth_loader_batching() -> None:
    loader = synth_train_loader(n_samples=8, batch_size=4, seed=0)
    batch = next(iter(loader))
    assert batch.mixture.shape[0] == 4
    assert batch.references.shape[1] == 3


def test_escalation_mask_respected() -> None:
    trainer = build_trainer_from_config({"realm": {"quality_threshold_tau": 20.0}})
    ds = SyntheticTrainDataset(n_samples=1, t=1500, seed=3)
    batch = _collate_train_batch([ds[0]])
    batch.quality_scores_db = torch.tensor([8.0])
    out = trainer.forward_batch(batch)
    assert out.escalated_mask.all()
    assert out.fusion_residual is not None
