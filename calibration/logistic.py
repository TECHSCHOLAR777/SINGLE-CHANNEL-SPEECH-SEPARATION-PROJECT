"""Logistic calibrators for confidence / completeness / counting fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LogisticCalibrator:
    """Binary logistic regression: sigmoid(w·x + b)."""

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.1,
        steps: int = 500,
        l2: float = 1e-3,
    ) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.ndim != 2:
            raise ValueError("features must be 2-D")
        n, d = x.shape
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        for _ in range(steps):
            logits = x @ w + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            err = probs - y
            w -= lr * ((x.T @ err) / n + l2 * w)
            b -= lr * float(err.mean())
        self.weights = w
        self.bias = float(b)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("calibrator not fitted")
        x = np.asarray(features, dtype=np.float64)
        logits = x @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))

    def save(self, path: str | Path) -> None:
        if self.weights is None:
            raise RuntimeError("calibrator not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"weights": self.weights.tolist(), "bias": self.bias}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LogisticCalibrator:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.weights = np.asarray(data["weights"], dtype=np.float64)
        obj.bias = float(data["bias"])
        return obj
