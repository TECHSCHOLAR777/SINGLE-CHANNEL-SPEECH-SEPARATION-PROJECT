"""
Per-stream confidence calibration (Dev C).

Maps the raw StreamConfidenceHead scores to well-calibrated probabilities
using isotonic regression (non-parametric, monotone).

After calibration:
  ECE (Expected Calibration Error) should be < 0.05 on the held-out eval set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ConfidenceCalibrator:
    """
    Isotonic regression calibrator for per-stream confidence scores.

    Fit on a held-out set of (raw_score, correct) pairs where `correct=1`
    means the stream's SI-SDR improvement was positive.

    Serialization (I-038): only the fitted step function's breakpoints
    (X_thresholds_, y_thresholds_) are kept after fit, as plain float
    arrays, not the fitted scikit-learn estimator object. calibrate()
    reimplements the estimator's own transform via linear interpolation over
    those breakpoints (numpy.interp clips to the boundary y-value outside the
    range by default, matching IsotonicRegression's out_of_bounds="clip").
    This means calibrate() never needs scikit-learn or a pickled object at
    inference time, only at fit time, and a saved calibrator is a JSON file
    of two float arrays instead of a pickled object graph.
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
        # The deduplicated, sorted step-function breakpoints scikit-learn
        # itself interpolates over, not the raw (possibly repeated,
        # unsorted) input scores.
        self._x_thresholds = np.asarray(ir.X_thresholds_, dtype=np.float64)
        self._y_calibrated = np.asarray(ir.y_thresholds_, dtype=np.float64)
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
        x = np.asarray(raw_scores, dtype=np.float64)
        return np.interp(x, self._x_thresholds, self._y_calibrated).astype(np.float32)

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
        """Write exactly `path`, as JSON. No pickle: see the class docstring."""
        payload = {
            "format": "coralsep-confidence-calibrator-v2",
            "fitted": self._fitted,
            "x_thresholds": (self._x_thresholds.tolist() if self._x_thresholds is not None else []),
            "y_calibrated": (self._y_calibrated.tolist() if self._y_calibrated is not None else []),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceCalibrator:
        """Load a calibrator saved by `save`.

        Also reads the old pickled format for one release (I-038's
        deprecation window): a file that fails to parse as JSON is retried
        as a pickle of {"ir": <fitted IsotonicRegression>, "fitted": bool},
        and its breakpoints are extracted the same way `fit` now does.
        """
        raw = Path(path).read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            import pickle
            import warnings

            warnings.warn(
                f"{path} is in the deprecated pickled ConfidenceCalibrator format. "
                "Re-save it with the current save() to get the JSON format.",
                DeprecationWarning,
                stacklevel=2,
            )
            state = pickle.loads(raw)  # noqa: S301 - explicit, logged legacy-format fallback
            obj = cls()
            obj._fitted = state["fitted"]
            ir = state["ir"]
            obj._x_thresholds = np.asarray(ir.X_thresholds_, dtype=np.float64)
            obj._y_calibrated = np.asarray(ir.y_thresholds_, dtype=np.float64)
            return obj

        obj = cls()
        obj._fitted = payload["fitted"]
        obj._x_thresholds = np.asarray(payload["x_thresholds"], dtype=np.float64)
        obj._y_calibrated = np.asarray(payload["y_calibrated"], dtype=np.float64)
        return obj
