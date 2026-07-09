"""Tests for the stop-classifier: numpy features always, torch parts when available."""

import numpy as np
import pytest

from models.stop_classifier import FEATURE_NAMES, compute_stop_features

RNG = np.random.default_rng(seed=3)


def test_feature_vector_order_and_shape() -> None:
    mix = RNG.standard_normal(8000)
    feats = compute_stop_features(
        mixture=mix,
        accepted_stems=np.zeros((0, 8000)),
        candidate_stem=mix * 0.5,
        vad_speech_prob=0.7,
        min_embedding_distance=1.0,
        attractor_stop_logit=-0.4,
    )
    assert feats.shape == (len(FEATURE_NAMES),)
    assert feats[1] == pytest.approx(0.7)
    assert feats[4] == pytest.approx(-0.4)


def test_perfect_extraction_drives_residual_features_to_zero() -> None:
    a, b = RNG.standard_normal((2, 8000))
    mix = a + b
    feats = compute_stop_features(
        mixture=mix,
        accepted_stems=a[None, :],
        candidate_stem=b,
        vad_speech_prob=0.0,
        min_embedding_distance=1.0,
    )
    assert feats[0] == pytest.approx(0.0, abs=1e-9)  # residual energy ratio
    assert feats[3] == pytest.approx(0.0, abs=1e-6)  # mixture consistency error


def test_unexplained_speaker_leaves_high_residual() -> None:
    a, b, c = RNG.standard_normal((3, 8000))
    mix = a + b + c
    feats = compute_stop_features(
        mixture=mix,
        accepted_stems=a[None, :],
        candidate_stem=b,
        vad_speech_prob=0.9,
        min_embedding_distance=0.8,
    )
    assert feats[0] > 0.1  # speaker c is still in the residual


class TestTorchParts:
    torch = pytest.importorskip("torch")

    def test_forward_shape_and_param_budget(self) -> None:
        import torch

        from models.stop_classifier import StopClassifier

        clf = StopClassifier()
        logits = clf(torch.randn(6, len(FEATURE_NAMES)))
        assert logits.shape == (6,)
        assert 1.5e5 < clf.parameter_count() < 5.0e5

    def test_temperature_calibration_and_roundtrip(self, tmp_path) -> None:
        import torch

        from models.stop_classifier import StopClassifier
        from train.train_stop_classifier import synth_features, train

        x, y = synth_features(n=1200, seed=0)
        model, metrics = train(
            x, y, epochs=15, lr=1e-3, batch_size=128, val_fraction=0.2, seed=0, device="cpu"
        )
        assert metrics["val_accuracy"] > 0.9
        assert metrics["temperature"] > 0.0

        path = tmp_path / "stop.pt"
        model.save(path)
        reloaded = StopClassifier.load(path)
        with torch.no_grad():
            assert torch.allclose(
                model.predict_proba(x[:8]), reloaded.predict_proba(x[:8]), atol=1e-5
            )
