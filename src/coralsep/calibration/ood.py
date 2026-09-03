"""
Mahalanobis OOD detector fit/load wrapper (Dev C).

The detector estimates the in-distribution manifold from E(0) features
collected on the training set. At inference time it computes the Mahalanobis
distance of each utterance's E(0) to the fitted Gaussian; utterances beyond
a calibrated threshold are flagged as OOD.

This wraps models.confidence.MahalanobisOOD to add persistent fit/load
and threshold calibration against a held-out ID/OOD set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OODCalibrator:
    """
    Wrapper around MahalanobisOOD that adds threshold calibration.

    A threshold τ is chosen such that FPR on in-distribution data ≤ 5%.
    """

    def __init__(self, fpr_target: float = 0.05) -> None:
        self.fpr_target = fpr_target
        self._detector = None
        self._threshold: float = float("inf")
        self._fitted = False

    def fit(self, features: np.ndarray) -> None:
        """
        Fit the Mahalanobis detector on in-distribution E(0) features.

        Args:
            features: (N, D) float32 feature matrix. Each row is the
                flattened mean pooled E(0) for one utterance.
        """
        from coralsep.models.confidence import MahalanobisOOD

        self._detector = MahalanobisOOD()
        self._detector.fit(features)
        self._fitted = True

    def calibrate_threshold(self, id_features: np.ndarray) -> float:
        """
        Set threshold τ at the (1 - fpr_target) quantile of in-distribution
        Mahalanobis distances.

        Args:
            id_features: (N, D) in-distribution holdout features.

        Returns:
            The calibrated threshold τ.
        """
        if not self._fitted or self._detector is None:
            raise RuntimeError("Call fit() before calibrate_threshold()")
        distances = self._mahalanobis_distances(id_features)
        self._threshold = float(np.quantile(distances, 1.0 - self.fpr_target))
        return self._threshold

    def is_ood(self, features: np.ndarray) -> bool:
        """
        Return True when the Mahalanobis distance exceeds the threshold.

        Args:
            features: (D,) or (1, D) feature vector.

        Returns:
            True if OOD.
        """
        if not self._fitted or self._detector is None:
            return False
        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        dist = float(self._mahalanobis_distances(features)[0])
        return dist > self._threshold

    def discount(self, features: np.ndarray) -> float:
        """
        Return a discount factor in [0, 1] (1.0 = in-distribution, 0.0 = far OOD).

        Sigmoid-shaped decay around the threshold:
            discount = sigmoid(-(d - τ) / σ)
        where σ = 0.1 × τ.
        """
        if not self._fitted or self._detector is None:
            return 1.0
        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        d = float(self._mahalanobis_distances(features)[0])
        tau = self._threshold
        sigma = max(0.1 * tau, 1e-6)
        return float(1.0 / (1.0 + np.exp((d - tau) / sigma)))

    def _mahalanobis_distances(self, features: np.ndarray) -> np.ndarray:
        """(N, D) → (N,) Mahalanobis distances."""
        if self._detector is None:
            return np.zeros(len(features))
        discounts = np.array([self._detector.discount(f) for f in features])
        # discount ≈ 1 at centre, 0 at far OOD. Invert to get distance proxy.
        eps = 1e-7
        discounts = np.clip(discounts, eps, 1 - eps)
        return -np.log(discounts)

    def save(self, path: str | Path) -> None:
        import pickle

        state = {
            "detector": self._detector,
            "threshold": self._threshold,
            "fpr_target": self.fpr_target,
            "fitted": self._fitted,
        }
        with open(str(path), "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> OODCalibrator:
        import pickle

        with open(str(path), "rb") as f:
            state = pickle.load(f)
        obj = cls(fpr_target=state["fpr_target"])
        obj._detector = state["detector"]
        obj._threshold = state["threshold"]
        obj._fitted = state["fitted"]
        return obj
