"""
Temperature scaling for speaker count posterior (Dev C).

After training, the model's count logits may be over- or under-confident.
Temperature scaling applies a single learned scalar T > 0 to the logits
before softmax:

    p(N) = softmax(logits / T)

T > 1 → softer distribution (less confident)
T < 1 → sharper distribution (more confident)

Calibration fitting minimises negative log-likelihood on a held-out set.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """
    Scalar temperature applied to count logits (N ∈ {2, 3, 4, 5}).

    Wraps the CountFusion head for calibrated inference without retraining.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling.

        Args:
            logits: (B, 4) raw logits for N ∈ {2, 3, 4, 5}.

        Returns:
            (B, 4) temperature-scaled log-softmax probabilities.
        """
        return torch.log_softmax(logits / self.temperature.clamp(min=1e-3), dim=-1)

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> float:
        """
        Fit temperature T by minimising NLL on validation logits.

        Args:
            logits: (N_val, 4) raw logits.
            labels: (N_val,) int64 labels in {0, 1, 2, 3} (→ N ∈ {2, 3, 4, 5}).
            lr: Adam learning rate.
            max_iter: Optimisation iterations.

        Returns:
            Final NLL loss value.
        """
        self.train()
        optimizer = torch.optim.LBFGS(
            self.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        nll = nn.NLLLoss()

        def _closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        self.eval()
        with torch.no_grad():
            final_loss = float(nll(self.forward(logits), labels).item())
        return final_loss

    def save(self, path: str | Path) -> None:
        """Write exactly `path`, as JSON (I-038), for the same format as the
        other three calibrators. A single scalar never needed torch.save's
        pickle-based container format."""
        payload = {
            "format": "coralsep-temperature-scaler-v2",
            "temperature": self.temperature.item(),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        """Load a scaler saved by `save`.

        Also reads the old torch.save format for one release (I-038's
        deprecation window): {"temperature": float}.
        """
        p = Path(path)
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            temperature = float(payload["temperature"])
        except (UnicodeDecodeError, json.JSONDecodeError):
            import warnings

            warnings.warn(
                f"{path} is in the deprecated torch.save TemperatureScaler format. "
                "Re-save it with the current save() to get the JSON format.",
                DeprecationWarning,
                stacklevel=2,
            )
            state = torch.load(str(p), map_location="cpu", weights_only=True)
            temperature = float(state["temperature"])

        obj = cls(init_temperature=temperature)
        obj.eval()
        return obj
