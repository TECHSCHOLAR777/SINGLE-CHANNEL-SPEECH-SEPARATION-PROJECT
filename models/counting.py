"""Speaker counting subsystem (BLUEPRINT §5.7).

Vote 1: attractor p_k (slots 1–5)
Vote 2: condition-analyzer count prior
Vote 3: bounded residual sweep (max 3 candidates in [2,5]) when uncertain
Fusion: logistic regression over vote features, temperature-calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CountDecision:
    n_hat: int
    posterior: np.ndarray  # (4,) over N=2..5
    margin: float
    sweep_triggered: bool
    vote1_n: int
    vote2_n: int


class CountingFusion(nn.Module):
    """Logistic fusion over vote features → logits for N∈{2,3,4,5}."""

    def __init__(self, in_dim: int = 12, temperature: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 4)
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features) / max(self.temperature, 1e-6)

    def set_temperature(self, t: float) -> None:
        self.temperature = float(t)


class CountingSubsystem:
    """Full counting pipeline with optional residual sweep callback."""

    def __init__(
        self,
        prob_thres: float = 0.5,
        uncertainty_margin: float = 0.2,
        fusion: CountingFusion | None = None,
        enabled_sweep: bool = True,
    ) -> None:
        self.prob_thres = prob_thres
        self.uncertainty_margin = uncertainty_margin
        self.fusion = fusion or CountingFusion()
        self.enabled_sweep = enabled_sweep

    @staticmethod
    def vote1_from_pk(p_k: torch.Tensor | np.ndarray, prob_thres: float = 0.5) -> int:
        if isinstance(p_k, torch.Tensor):
            probs = p_k.detach().cpu().numpy()
        else:
            probs = np.asarray(p_k)
        if probs.ndim == 2:
            probs = probs[0]
        n = int((probs[1:6] > prob_thres).sum())
        return int(np.clip(n, 2, 5)) if n >= 2 else max(n, 2) if n > 0 else 2

    @staticmethod
    def vote2_from_prior(count_prior: list[float] | np.ndarray) -> int:
        prior = np.asarray(count_prior, dtype=np.float64)
        return int(np.argmax(prior)) + 2

    def build_features(
        self,
        p_k: np.ndarray,
        count_prior: np.ndarray,
        residual_energies: dict[int, float] | None = None,
    ) -> np.ndarray:
        probs = np.asarray(p_k, dtype=np.float64).reshape(-1)
        if probs.shape[0] >= 7:
            slot = probs[1:6]
        else:
            slot = np.pad(probs, (0, max(0, 5 - probs.shape[0])))[:5]
        prior = np.asarray(count_prior, dtype=np.float64).reshape(-1)[:4]
        if prior.shape[0] < 4:
            prior = np.pad(prior, (0, 4 - prior.shape[0]))
        residual_vec = np.zeros(3, dtype=np.float64)
        if residual_energies:
            for i, n in enumerate(sorted(residual_energies)[:3]):
                residual_vec[i] = residual_energies[n]
        return np.concatenate([slot, prior, residual_vec]).astype(np.float32)

    def fuse(
        self,
        p_k: torch.Tensor | np.ndarray,
        count_prior: list[float] | np.ndarray,
        residual_energies: dict[int, float] | None = None,
    ) -> tuple[int, np.ndarray, float]:
        feats = self.build_features(
            np.asarray(p_k.detach().cpu() if isinstance(p_k, torch.Tensor) else p_k),
            np.asarray(count_prior),
            residual_energies,
        )
        with torch.no_grad():
            logits = self.fusion(torch.from_numpy(feats).unsqueeze(0))
            post = torch.softmax(logits, dim=-1)[0].numpy()
        order = np.argsort(-post)
        n_hat = int(order[0]) + 2
        margin = float(post[order[0]] - post[order[1]]) if len(order) > 1 else 1.0
        return n_hat, post, margin

    def decide(
        self,
        p_k: torch.Tensor | np.ndarray,
        count_prior: list[float] | np.ndarray,
        residual_fn: Callable[[int], float] | None = None,
    ) -> CountDecision:
        v1 = self.vote1_from_pk(p_k, self.prob_thres)
        v2 = self.vote2_from_prior(count_prior)
        n_hat, post, margin = self.fuse(p_k, count_prior)
        sweep = False
        residual_energies: dict[int, float] = {}
        if self.enabled_sweep and margin < self.uncertainty_margin and residual_fn is not None:
            sweep = True
            mode = n_hat
            candidates = sorted({int(np.clip(mode + d, 2, 5)) for d in (-1, 0, 1)})
            for n in candidates:
                residual_energies[n] = float(residual_fn(n))
            # Prefer lower residual energy among candidates.
            best_n = min(residual_energies, key=residual_energies.get)  # type: ignore[arg-type]
            n_hat, post, margin = self.fuse(p_k, count_prior, residual_energies)
            # Blend: if residual strongly prefers, take it.
            if residual_energies[best_n] < 0.85 * residual_energies.get(mode, residual_energies[best_n]):
                n_hat = best_n
        return CountDecision(
            n_hat=int(np.clip(n_hat, 2, 5)),
            posterior=post,
            margin=margin,
            sweep_triggered=sweep,
            vote1_n=v1,
            vote2_n=v2,
        )


def residual_energy_fraction(
    mixture: np.ndarray,
    streams: list[np.ndarray] | np.ndarray,
) -> float:
    """Energy of mixture minus sum of streams, normalized by mixture energy."""
    mix = np.asarray(mixture, dtype=np.float64)
    if isinstance(streams, np.ndarray) and streams.ndim == 2:
        recon = streams.sum(axis=0)
    else:
        recon = np.sum([np.asarray(s, dtype=np.float64) for s in streams], axis=0)
    n = min(mix.shape[0], recon.shape[0])
    residual = mix[:n] - recon[:n]
    num = float(np.dot(residual, residual))
    den = float(np.dot(mix[:n], mix[:n])) + 1e-10
    return num / den
