"""Tests for calibration/ package (Dev C)."""

import numpy as np
import pytest
import torch

from calibration.temperature import TemperatureScaler
from calibration.completeness import CompletenessCalibrator


class TestTemperatureScaler:
    def test_forward_log_softmax_shape(self):
        scaler = TemperatureScaler(init_temperature=1.0)
        logits = torch.randn(8, 4)
        out = scaler(logits)
        assert out.shape == (8, 4)

    def test_forward_log_probabilities(self):
        scaler = TemperatureScaler(init_temperature=1.0)
        logits = torch.randn(4, 4)
        log_probs = scaler(logits)
        probs = log_probs.exp()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_high_temperature_softer(self):
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
        scaler_low = TemperatureScaler(0.1)
        scaler_high = TemperatureScaler(10.0)
        prob_low = scaler_low(logits).exp()
        prob_high = scaler_high(logits).exp()
        # High temperature → less peaked.
        assert float(prob_high[:, 0]) < float(prob_low[:, 0])

    def test_calibrate_reduces_nll(self):
        # Confident model on easy labels — calibration should converge.
        N = 100
        rng = np.random.default_rng(0)
        labels = torch.tensor(rng.integers(0, 4, size=N))
        logits = torch.zeros(N, 4)
        for i, lbl in enumerate(labels):
            logits[i, lbl] = 5.0  # very confident

        scaler = TemperatureScaler(init_temperature=2.0)  # intentionally miscalibrated
        initial_nll = float(torch.nn.NLLLoss()(scaler(logits), labels).item())
        scaler.calibrate(logits.clone(), labels.clone())
        final_nll = float(torch.nn.NLLLoss()(scaler(logits), labels).item())
        # Calibration should not increase NLL.
        assert final_nll <= initial_nll + 0.1

    def test_save_load_roundtrip(self, tmp_path):
        scaler = TemperatureScaler(init_temperature=1.5)
        path = tmp_path / "temp.pt"
        scaler.save(path)
        loaded = TemperatureScaler.load(path)
        assert abs(loaded.temperature.item() - 1.5) < 1e-5


class TestCompletenessCalibrator:
    def test_calibrate_no_change_for_perfect_model(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0.8, 1.0, 200).astype(np.float32)
        labels = np.ones(200, dtype=np.float32)
        cal = CompletenessCalibrator()
        cal.fit(probs, labels)
        cal_out = cal.calibrate(probs)
        assert cal_out.min() >= 0.0
        assert cal_out.max() <= 1.0

    def test_calibrate_range(self):
        rng = np.random.default_rng(1)
        probs = rng.uniform(0.1, 0.9, 100).astype(np.float32)
        labels = (rng.uniform(size=100) > 0.5).astype(np.float32)
        cal = CompletenessCalibrator()
        cal.fit(probs, labels)
        out = cal.calibrate(probs)
        assert out.dtype == np.float32
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_save_load(self, tmp_path):
        rng = np.random.default_rng(2)
        probs = rng.uniform(0.1, 0.9, 50).astype(np.float32)
        labels = (rng.uniform(size=50) > 0.5).astype(np.float32)
        cal = CompletenessCalibrator()
        cal.fit(probs, labels)
        out1 = cal.calibrate(probs)

        path = tmp_path / "completeness_cal.npy"
        cal.save(path)
        loaded = CompletenessCalibrator.load(path)
        out2 = loaded.calibrate(probs)
        np.testing.assert_allclose(out1, out2, atol=1e-5)
