"""
Learned stop-classifier for speaker counting (Dev C, Phase 3).

A small MLP that decides "do more speakers remain?" from four signals plus
the attractor stop logit (MASTER_PROJECT section 4.5): residual energy ratio,
VAD speech probability on the residual, minimum speaker-embedding distance of
the newest stem to accepted stems, and mixture-consistency reconstruction
error. After training, temperature scaling calibrates the probabilities so
the demo's confidence badge and the calibration curve are honest.

compute_stop_features is the reference (numpy) feature implementation; the
Dev B feature module may supersede it, but the ordering and semantics defined
here are the frozen interface the trained classifier depends on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_EPS = 1e-8

FEATURE_NAMES: tuple[str, ...] = (
    "residual_energy_ratio",
    "vad_speech_prob",
    "min_embedding_distance",
    "mixture_consistency_error",
    "attractor_stop_logit",
)
"""Frozen feature order. Index positions are part of the checkpoint contract."""


def compute_stop_features(
    mixture: np.ndarray,
    accepted_stems: np.ndarray,
    candidate_stem: np.ndarray,
    vad_speech_prob: float,
    min_embedding_distance: float,
    attractor_stop_logit: float = 0.0,
) -> np.ndarray:
    """
    Build the feature vector for one stop decision.

    Args:
        mixture: [T] original mixture waveform.
        accepted_stems: [K, T] stems accepted so far (K may be 0).
        candidate_stem: [T] the newly extracted stem under consideration.
        vad_speech_prob: Speech probability of the residual after removing
            accepted stems plus the candidate, from the VAD adapter, in [0, 1].
        min_embedding_distance: Minimum cosine distance between the candidate
            stem's speaker embedding and all accepted stems' embeddings
            (1.0 when no stems are accepted yet).
        attractor_stop_logit: Raw stop logit from the attractor-based expert
            when available; 0.0 otherwise.

    Returns:
        Feature vector [len(FEATURE_NAMES)] float32, in FEATURE_NAMES order.
    """
    mix = np.asarray(mixture, dtype=np.float64)
    stems = np.atleast_2d(np.asarray(accepted_stems, dtype=np.float64))
    if stems.size == 0:
        stems = np.zeros((0, mix.shape[0]))
    cand = np.asarray(candidate_stem, dtype=np.float64)

    explained = stems.sum(axis=0) + cand
    residual = mix - explained
    mix_energy = float(np.dot(mix, mix)) + _EPS

    residual_energy_ratio = float(np.dot(residual, residual)) / mix_energy
    mixture_consistency_error = float(np.linalg.norm(residual)) / (
        float(np.linalg.norm(mix)) + _EPS
    )

    return np.asarray(
        [
            residual_energy_ratio,
            float(vad_speech_prob),
            float(min_embedding_distance),
            mixture_consistency_error,
            float(attractor_stop_logit),
        ],
        dtype=np.float32,
    )


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
        temp = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temp], lr=0.05, max_iter=100)
        bce = nn.BCEWithLogitsLoss()
        logits = val_logits.detach()
        labels = val_labels.detach().float()

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
