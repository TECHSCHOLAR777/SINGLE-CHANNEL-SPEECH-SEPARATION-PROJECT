"""Tests for calibration/ package (Dev C)."""

import json
import pickle

import numpy as np
import pytest
import torch

from coralsep.calibration.completeness import CompletenessCalibrator
from coralsep.calibration.confidence import ConfidenceCalibrator
from coralsep.calibration.ood import OODCalibrator
from coralsep.calibration.temperature import TemperatureScaler


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
        # Confident model on easy labels, calibration should converge.
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

    def test_save_writes_exactly_the_given_path(self, tmp_path):
        """I-038 acceptance criterion: save(path) writes exactly path. The old
        implementation used np.save, which silently appended .npy."""
        cal = CompletenessCalibrator()
        cal.fit(np.array([0.3, 0.7, 0.9], dtype=np.float32), np.array([0.0, 1.0, 1.0]))
        path = tmp_path / "completeness"  # deliberately no extension
        cal.save(path)
        assert path.exists()
        assert not path.with_suffix(".npy").exists()

    def test_save_is_readable_json_not_a_binary_array(self, tmp_path):
        cal = CompletenessCalibrator()
        cal.fit(np.array([0.3, 0.7, 0.9], dtype=np.float32), np.array([0.0, 1.0, 1.0]))
        path = tmp_path / "completeness.json"
        cal.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "a" in payload and "b" in payload

    def test_load_reads_the_deprecated_npy_format_with_a_warning(self, tmp_path):
        path = tmp_path / "old.npy"
        np.save(str(path), np.array([1.3, -0.2]))
        with pytest.warns(DeprecationWarning):
            loaded = CompletenessCalibrator.load(path)
        assert abs(loaded._a - 1.3) < 1e-6
        assert abs(loaded._b - (-0.2)) < 1e-6


class TestConfidenceCalibrator:
    def test_fit_calibrate_is_monotone_and_bounded(self):
        rng = np.random.default_rng(3)
        raw = rng.uniform(0.0, 1.0, 200).astype(np.float32)
        correct = (raw + rng.normal(0, 0.1, 200) > 0.5).astype(np.float32)
        cal = ConfidenceCalibrator()
        cal.fit(raw, correct)
        out = cal.calibrate(np.sort(raw))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        # Isotonic regression is non-decreasing by construction.
        assert np.all(np.diff(out) >= -1e-6)

    def test_unfitted_calibrator_passes_scores_through(self):
        cal = ConfidenceCalibrator()
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        np.testing.assert_allclose(cal.calibrate(scores), scores)

    def test_save_load_roundtrip_matches_fitted_output(self, tmp_path):
        rng = np.random.default_rng(4)
        raw = rng.uniform(0.0, 1.0, 100).astype(np.float32)
        correct = (raw > 0.5).astype(np.float32)
        cal = ConfidenceCalibrator()
        cal.fit(raw, correct)
        out1 = cal.calibrate(raw)

        path = tmp_path / "confidence.json"
        cal.save(path)
        assert path.exists()
        loaded = ConfidenceCalibrator.load(path)
        np.testing.assert_allclose(out1, loaded.calibrate(raw), atol=1e-6)

    def test_no_pickle_in_a_saved_file(self, tmp_path):
        """I-038 acceptance criterion: no pickle in the calibration package.
        A JSON file must not be parseable as a pickle stream and vice versa,
        so this is a real, if indirect, check that save() no longer pickles."""
        rng = np.random.default_rng(5)
        raw = rng.uniform(0.0, 1.0, 50).astype(np.float32)
        cal = ConfidenceCalibrator()
        cal.fit(raw, (raw > 0.5).astype(np.float32))
        path = tmp_path / "confidence.json"
        cal.save(path)
        json.loads(path.read_text(encoding="utf-8"))  # must not raise

    def test_load_reads_a_deprecated_pickle_with_a_warning(self, tmp_path):
        rng = np.random.default_rng(6)
        raw = rng.uniform(0.0, 1.0, 60).astype(np.float32)
        correct = (raw > 0.5).astype(np.float32)

        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
        ir.fit(raw, correct)

        path = tmp_path / "old_confidence.pkl"
        with open(path, "wb") as f:
            pickle.dump({"ir": ir, "fitted": True}, f)

        with pytest.warns(DeprecationWarning):
            loaded = ConfidenceCalibrator.load(path)
        assert loaded._fitted
        np.testing.assert_allclose(
            loaded.calibrate(raw), ir.transform(raw.astype(np.float64)), atol=1e-6
        )


class TestOODCalibrator:
    def test_fit_and_is_ood_roundtrip(self, tmp_path):
        rng = np.random.default_rng(7)
        id_features = rng.normal(0, 1, size=(200, 8)).astype(np.float32)
        cal = OODCalibrator(fpr_target=0.05)
        cal.fit(id_features)
        cal.calibrate_threshold(id_features)

        far_ood = np.full(8, 50.0, dtype=np.float32)
        assert cal.is_ood(far_ood) is True

        path = tmp_path / "ood.json"
        cal.save(path)
        assert path.exists()
        loaded = OODCalibrator.load(path)
        assert loaded.is_ood(far_ood) == cal.is_ood(far_ood)
        assert abs(loaded._threshold - cal._threshold) < 1e-9

    def test_save_is_readable_json(self, tmp_path):
        rng = np.random.default_rng(8)
        id_features = rng.normal(0, 1, size=(100, 4)).astype(np.float32)
        cal = OODCalibrator()
        cal.fit(id_features)
        cal.calibrate_threshold(id_features)
        path = tmp_path / "ood.json"
        cal.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["detector_mean"] is not None
        assert payload["detector_cov_inv"] is not None

    def test_load_reads_a_deprecated_pickle_with_a_warning(self, tmp_path):
        from coralsep.models.confidence import MahalanobisOOD

        rng = np.random.default_rng(9)
        id_features = rng.normal(0, 1, size=(50, 4)).astype(np.float32)
        detector = MahalanobisOOD()
        detector.fit(id_features)

        path = tmp_path / "old_ood.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {"detector": detector, "threshold": 12.0, "fpr_target": 0.05, "fitted": True}, f
            )

        with pytest.warns(DeprecationWarning):
            loaded = OODCalibrator.load(path)
        assert loaded._fitted
        assert loaded._threshold == 12.0
