"""
Per-stream confidence calibration (Dev C).

Maps the raw StreamConfidenceHead scores to well-calibrated probabilities
using isotonic regression (non-parametric, monotone).

After calibration:
  ECE (Expected Calibration Error) should be < 0.05 on the held-out eval set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ConfidenceCalibrator:
    """
    Isotonic regression calibrator for per-stream confidence scores.

    Fit on a held-out set of (raw_score, correct) pairs where `correct=1`
    means the stream's SI-SDR improvement was positive.
    """

    def __init__(self) -> None:
        self._x_thresholds: np.ndarray | None = None
        self._y_calibrated: np.ndarray | None = None
        self._fitted = False

    def fit(self, raw_scores: np.ndarray, is_correct: np.ndarray) -> None:
        """
        Fit isotonic regression.

        Args:
            raw_scores: (N,) float32 scores in [0, 1].
            is_correct: (N,) binary labels (1 = stream improved quality).
        """
        from sklearn.isotonic import IsotonicRegression

        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        is_correct = np.asarray(is_correct, dtype=np.float64)
        ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
        ir.fit(raw_scores, is_correct)
        self._x_thresholds = raw_scores
        self._y_calibrated = ir.transform(raw_scores)
        self._ir = ir
        self._fitted = True

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Map raw scores to calibrated probabilities.

        Args:
            raw_scores: (N,) or scalar float32 in [0, 1].

        Returns:
            (N,) calibrated probabilities in [0, 1].
        """
        if not self._fitted:
            return np.asarray(raw_scores, dtype=np.float32)
        return self._ir.transform(np.asarray(raw_scores, dtype=np.float64)).astype(np.float32)

    def expected_calibration_error(
        self, raw_scores: np.ndarray, is_correct: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute ECE on a validation set."""
        cal = self.calibrate(raw_scores)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(cal)
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            mask = (cal >= lo) & (cal < hi)
            if mask.sum() == 0:
                continue
            frac_pos = float(is_correct[mask].mean())
            mean_conf = float(cal[mask].mean())
            ece += mask.sum() / n * abs(frac_pos - mean_conf)
        return float(ece)

    def save(self, path: str | Path) -> None:
        import pickle

        with open(str(path), "wb") as f:
            pickle.dump({"ir": self._ir, "fitted": self._fitted}, f)

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceCalibrator:
        import pickle

        with open(str(path), "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._ir = state["ir"]
        obj._fitted = state["fitted"]
        return obj
