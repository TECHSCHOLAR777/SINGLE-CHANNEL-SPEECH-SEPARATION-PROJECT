"""Temperature scaling for count posteriors (BLUEPRINT §8.5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TemperatureScaler:
    """Scalar temperature T for softmax(logits / T)."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        grid: np.ndarray | None = None,
    ) -> float:
        """Fit T by minimizing NLL on a grid search (held-out data only)."""
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if grid is None:
            grid = np.concatenate(
                [np.linspace(0.05, 1.0, 20), np.linspace(1.0, 5.0, 21)]
            )
        best_t, best_nll = 1.0, float("inf")
        for t in grid:
            nll = self._nll(logits, labels, float(t))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
        self.temperature = best_t
        return best_t

    @staticmethod
    def _nll(logits: np.ndarray, labels: np.ndarray, t: float) -> float:
        scaled = logits / max(t, 1e-6)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        probs = exp / exp.sum(axis=1, keepdims=True)
        idx = np.arange(len(labels))
        return float(-np.log(probs[idx, labels] + 1e-12).mean())

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float64) / max(self.temperature, 1e-6)
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        exp = np.exp(scaled)
        return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"temperature": self.temperature}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(temperature=float(data["temperature"]))
