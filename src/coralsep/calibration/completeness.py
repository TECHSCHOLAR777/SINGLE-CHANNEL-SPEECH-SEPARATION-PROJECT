"""
Completeness calibration (Dev C).

Calibrates the output of CompletenessHead (P(all speakers captured)) using
Platt scaling (logistic regression on top of the raw head output).

A well-calibrated completeness score lets downstream consumers decide
when to escalate to a more expensive expert or flag for human review.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class CompletenessCalibrator:
    """
    Platt-scaling calibrator for CompletenessHead outputs.

    Fits a linear layer (a, b) such that:
        P_cal(complete) = sigmoid(a * logit + b)

    where logit = log(p / (1-p)) and p is the raw CompletenessHead output.
    """

    def __init__(self) -> None:
        self._a: float = 1.0
        self._b: float = 0.0
        self._fitted = False

    def fit(
        self,
        raw_probs: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 500,
    ) -> float:
        """
        Fit Platt scaling.

        Args:
            raw_probs: (N,) CompletenessHead sigmoid outputs in (0, 1).
            labels: (N,) binary labels (1 = all speakers captured).
            lr: Learning rate.
            max_iter: Gradient-descent iterations.

        Returns:
            Final BCE loss.
        """
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7).astype(np.float32)
        logits = np.log(raw_probs / (1 - raw_probs))

        x = torch.from_numpy(logits).unsqueeze(1)
        y = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)

        layer = nn.Linear(1, 1)
        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.fill_(0.0)

        opt = torch.optim.LBFGS(
            layer.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        bce = nn.BCEWithLogitsLoss()

        def _closure():
            opt.zero_grad()
            loss = bce(layer(x), y)
            loss.backward()
            return loss

        opt.step(_closure)

        self._a = float(layer.weight.item())
        self._b = float(layer.bias.item())
        self._fitted = True

        with torch.no_grad():
            final_loss = float(bce(layer(x), y).item())
        return final_loss

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to raw probabilities.

        Returns:
            Calibrated probabilities in (0, 1).
        """
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7).astype(np.float32)
        logits = np.log(raw_probs / (1 - raw_probs))
        cal_logits = self._a * logits + self._b
        return (1.0 / (1.0 + np.exp(-cal_logits))).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Write exactly `path`, as JSON (I-038). The old np.save(str(path),
        ...) silently appended .npy to whatever path was given, so the file
        actually written was never the one the caller asked for."""
        payload = {
            "format": "coralsep-completeness-calibrator-v2",
            "a": self._a,
            "b": self._b,
            "fitted": self._fitted,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CompletenessCalibrator:
        """Load a calibrator saved by `save`.

        Also reads the old .npy array format for one release (I-038's
        deprecation window): [a, b], with fitted implied True.
        """
        p = Path(path)
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            import warnings

            warnings.warn(
                f"{path} is in the deprecated .npy CompletenessCalibrator format. "
                "Re-save it with the current save() to get the JSON format.",
                DeprecationWarning,
                stacklevel=2,
            )
            ab = np.load(str(p))
            obj = cls()
            obj._a = float(ab[0])
            obj._b = float(ab[1])
            obj._fitted = True
            return obj

        obj = cls()
        obj._a = float(payload["a"])
        obj._b = float(payload["b"])
        obj._fitted = payload["fitted"]
        return obj
