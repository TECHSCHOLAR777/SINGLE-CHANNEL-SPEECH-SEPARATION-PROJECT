"""
Learned stop-classifier for speaker counting (Dev C, Phase 3).

A small MLP that decides "do more speakers remain?" from four signals plus
the attractor stop logit (MASTER_PROJECT section 4.5): residual energy ratio,
VAD speech probability on the residual, minimum speaker-embedding distance of
the newest stem to accepted stems, and mixture-consistency reconstruction
error. After training, temperature scaling calibrates the probabilities so
the demo's confidence badge and the calibration curve are honest.

Feature extraction lives in ``models/counting_features.py`` (P3-B1); this
module owns the MLP, calibration, and checkpoint contract.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.counting_features import _EPS, FEATURE_NAMES, compute_stop_features

__all__ = [
    "FEATURE_NAMES",
    "StopClassifier",
    "compute_stop_features",
]


class StopClassifier(nn.Module):
    """
    MLP over stop features producing P(more speakers remain).

    Args:
        in_features: Input feature count; must equal len(FEATURE_NAMES) unless
            a custom feature set is deliberately configured.
        hidden_dims: Hidden layer widths. Default sized near the 0.3M
            parameter budget from MASTER_PROJECT 5.2.
        dropout: Dropout probability between hidden layers.
    """

    def __init__(
        self,
        in_features: int = len(FEATURE_NAMES),
        hidden_dims: tuple[int, ...] = (448, 448),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)
        self.in_features = in_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Raw logits [B] for P(continue peeling)."""
        if features.ndim != 2 or features.shape[1] != self.in_features:
            raise ValueError(
                f"expected [B, {self.in_features}] features, got {tuple(features.shape)}"
            )
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Calibrated P(continue) in [0, 1], temperature applied."""
        self.eval()
        return torch.sigmoid(self.forward(features) / self.temperature.clamp_min(_EPS))

    def fit_temperature(self, val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
        """
        Post-hoc temperature scaling on held-out logits (Guo et al., 2017).

        Optimizes a single scalar with LBFGS to minimize BCE, leaving decision
        boundaries unchanged while making confidences honest. Chosen over
        isotonic regression to avoid an sklearn dependency and to keep the
        calibrated model a single torch checkpoint.

        Args:
            val_logits: [N] raw logits on a held-out split.
            val_labels: [N] binary continue/stop labels.

        Returns:
            The fitted temperature value.
        """
        # The temperature scalar must live on the same device as the logits, or
        # the closure's `logits / temp` raises a CUDA-vs-CPU device mismatch
        # (the LBFGS/temperature step ran fine in CPU CI but broke on the Kaggle
        # GPU run).
        logits = val_logits.detach()
        labels = val_labels.detach().float().to(logits.device)
        temp = nn.Parameter(torch.ones(1, device=logits.device))
        optimizer = torch.optim.LBFGS([temp], lr=0.05, max_iter=100)
        bce = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = bce(logits / temp.clamp_min(_EPS), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature.copy_(temp.detach().clamp_min(_EPS))
        return float(self.temperature.item())

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "in_features": self.in_features,
            "feature_names": list(FEATURE_NAMES),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> StopClassifier:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = cls(in_features=payload["in_features"], **kwargs)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
