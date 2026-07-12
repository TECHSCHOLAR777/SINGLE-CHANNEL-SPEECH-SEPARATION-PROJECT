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
    streams = np.random.randn(2, t).astype(np.float32)  # REAL-M is a 2-source model

    mock_model = MagicMock()
    raw = torch.tensor([0.15, 0.20])
    mock_model.estimate_batch.return_value = raw
    mock_model.gettrue_snrrange.return_value = raw * 10.0
    mock_model.eval.return_value = mock_model

    estimator = REALMQualityEstimator(device="cpu")
    estimator._model = mock_model

    quality = estimator.estimate(mixture, streams, sample_rate=sr)

    assert len(quality.sisnr_db_per_stream) == 2
    assert quality.mean_sisnr_db == pytest.approx(1.75, rel=1e-3)
    assert quality.min_sisnr_db == 1.5
    mock_model.estimate_batch.assert_called_once()
    # The predictions handed to REAL-M must carry exactly 2 sources.
    _mix_arg, pred_arg = mock_model.estimate_batch.call_args.args
    assert pred_arg.shape[-1] == 2


def test_realm_reduces_three_streams_to_two_before_the_model() -> None:
    # MossFormer2 residual-padded to 3 must not reach the 2-source estimator as
    # 3 sources — that is the torch.cat "size 3 vs size 2" crash from the Kaggle
    # cache build. The wrapper reduces to the 2 highest-energy streams first.
    sr = 16000
    t = 4000
    mixture = np.random.randn(t).astype(np.float32)
    loud_a = (np.random.randn(t) * 5.0).astype(np.float32)
    loud_b = (np.random.randn(t) * 4.0).astype(np.float32)
    residual_pad = np.zeros(t, dtype=np.float32)  # low-energy synthetic pad
    streams = np.stack([loud_a, residual_pad, loud_b], axis=0)

    mock_model = MagicMock()
    raw = torch.tensor([0.1, 0.1])
    mock_model.estimate_batch.return_value = raw
    mock_model.gettrue_snrrange.return_value = raw * 10.0
    mock_model.eval.return_value = mock_model

    estimator = REALMQualityEstimator(device="cpu")
    estimator._model = mock_model
    quality = estimator.estimate(mixture, streams, sample_rate=sr)

    assert len(quality.sisnr_db_per_stream) == 2
    _mix_arg, pred_arg = mock_model.estimate_batch.call_args.args
    assert pred_arg.shape[-1] == 2  # the zero pad was dropped, not passed


def test_reduce_to_two_shapes() -> None:
    reduce = REALMQualityEstimator._reduce_to_two
    assert reduce(np.random.randn(2, 100).astype(np.float32)).shape == (2, 100)
    assert reduce(np.random.randn(3, 100).astype(np.float32)).shape == (2, 100)
    assert reduce(np.random.randn(5, 100).astype(np.float32)).shape == (2, 100)
    assert reduce(np.random.randn(1, 100).astype(np.float32)).shape == (2, 100)
    with pytest.raises(ValueError):
        reduce(np.zeros((0, 100), dtype=np.float32))


def test_realm_lazy_load_starts_uninitialized() -> None:
    estimator = REALMQualityEstimator(device="cpu")
    assert estimator._model is None
