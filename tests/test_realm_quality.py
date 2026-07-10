"""Tests for REAL-M blind quality estimator (mocked)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from models.realm_quality import REALMQualityEstimator


def test_realm_estimate_per_stream_db() -> None:
    sr = 16000
    t = 4000
    mixture = np.random.randn(t).astype(np.float32)
    streams = np.random.randn(3, t).astype(np.float32)

    mock_model = MagicMock()
    raw = torch.tensor([0.15, 0.20, 0.18])
    mock_model.estimate_batch.return_value = raw
    mock_model.gettrue_snrrange.return_value = raw * 10.0
    mock_model.eval.return_value = mock_model

    estimator = REALMQualityEstimator(device="cpu")
    estimator._model = mock_model

    quality = estimator.estimate(mixture, streams, sample_rate=sr)

    assert len(quality.sisnr_db_per_stream) == 3
    assert quality.mean_sisnr_db == pytest.approx(1.7667, rel=1e-3)
    assert quality.min_sisnr_db == 1.5
    mock_model.estimate_batch.assert_called_once()


def test_realm_lazy_load_starts_uninitialized() -> None:
    estimator = REALMQualityEstimator(device="cpu")
    assert estimator._model is None
