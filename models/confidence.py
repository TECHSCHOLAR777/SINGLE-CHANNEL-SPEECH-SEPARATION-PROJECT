"""Per-stream confidence, completeness, and OOD discount (BLUEPRINT §5.8)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from data.vad_features import voiced_frame_density
from models.counting import residual_energy_fraction


@dataclass
class ConfidenceOutput:
    per_stream: list[float]
    completeness: float
    ood_flag: bool
    ood_distance: float


class LogisticHead(nn.Module):
    """Small logistic model over hand-crafted features."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def inter_stage_consistency(
    stage_a: torch.Tensor | np.ndarray,
    stage_b: torch.Tensor | np.ndarray,
) -> list[float]:
    """Per-stream correlation of magnitude maps between last two decoder stages."""
    if isinstance(stage_a, torch.Tensor):
        a = stage_a.detach().float().cpu().numpy()
        b = stage_b.detach().float().cpu().numpy()
    else:
        a = np.asarray(stage_a, dtype=np.float32)
        b = np.asarray(stage_b, dtype=np.float32)
    # Expected (K, T, F, C) or (K, T, F)
    if a.ndim == 4:
        a = np.linalg.norm(a, axis=-1)
        b = np.linalg.norm(b, axis=-1)
    k = a.shape[0]
    scores: list[float] = []
    for i in range(k):
        x = a[i].reshape(-1)
        y = b[i].reshape(-1)
        x = x - x.mean()
        y = y - y.mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
        scores.append(float(np.dot(x, y) / denom))
    return scores


class ConfidenceHead(nn.Module):
    """Calibrated per-stream confidence from p_k, consistency, quality proxy."""

    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticHead(3)

    def features(
        self,
        p_k_i: float,
        consistency: float,
        quality: float,
    ) -> torch.Tensor:
        return torch.tensor([p_k_i, consistency, quality], dtype=torch.float32)

    def forward_scores(self, feats: torch.Tensor) -> torch.Tensor:
        return self.model(feats)


class CompletenessHead(nn.Module):
    """Calibrated completeness from residual energy, VAD residual, attractor mass."""

    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticHead(3)

    def forward_scores(self, feats: torch.Tensor) -> torch.Tensor:
        return self.model(feats)


class OODDetector:
    """Mahalanobis OOD flag on condition vectors."""

    def __init__(self, percentile: float = 99.0) -> None:
        self.percentile = percentile
        self.mean: np.ndarray | None = None
        self.cov_inv: np.ndarray | None = None
        self.threshold: float = float("inf")

    def fit(self, vectors: np.ndarray) -> None:
        x = np.asarray(vectors, dtype=np.float64)
        self.mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False) + 1e-6 * np.eye(x.shape[1])
        self.cov_inv = np.linalg.inv(cov)
        dists = self.distances(x)
        self.threshold = float(np.percentile(dists, self.percentile))

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        if self.mean is None or self.cov_inv is None:
            return np.zeros(len(vectors))
        d = np.asarray(vectors, dtype=np.float64) - self.mean
        return np.einsum("ni,ij,nj->n", d, self.cov_inv, d)

    def score(self, vector: np.ndarray) -> tuple[bool, float]:
        dist = float(self.distances(np.asarray(vector).reshape(1, -1))[0])
        return dist > self.threshold, dist


class ConfidenceSubsystem:
    """End-to-end confidence + completeness + OOD."""

    def __init__(
        self,
        ood_discount: float = 0.7,
        quality_proxy: Callable[[np.ndarray, int], float] | None = None,
    ) -> None:
        self.conf_head = ConfidenceHead()
        self.comp_head = CompletenessHead()
        self.ood = OODDetector()
        self.ood_discount = ood_discount
        self.quality_proxy = quality_proxy

    def __call__(
        self,
        *,
        p_k: np.ndarray,
        streams: np.ndarray,
        mixture: np.ndarray,
        dec_stages: dict[int, object] | None = None,
        condition_vec: np.ndarray | None = None,
        sample_rate: int = 8000,
    ) -> ConfidenceOutput:
        probs = np.asarray(p_k, dtype=np.float64).reshape(-1)
        slot = probs[1:6] if probs.shape[0] >= 6 else probs
        k = streams.shape[0]
        consistency = [0.5] * k
        if dec_stages and len(dec_stages) >= 2:
            keys = sorted(dec_stages)
            try:
                consistency = inter_stage_consistency(dec_stages[keys[-2]], dec_stages[keys[-1]])
                if len(consistency) < k:
                    consistency = (consistency + [0.5] * k)[:k]
            except Exception:
                consistency = [0.5] * k

        per_stream: list[float] = []
        with torch.no_grad():
            for i in range(k):
                pk_i = float(slot[i]) if i < len(slot) else 0.5
                q = 0.5
                if self.quality_proxy is not None:
                    try:
                        q = float(self.quality_proxy(streams[i], sample_rate))
                    except Exception:
                        q = 0.5
                feats = self.conf_head.features(pk_i, consistency[i], q).unsqueeze(0)
                # Untrained head ≈ 0.5; blend with raw features for usable defaults.
                raw = 0.5 * pk_i + 0.3 * max(consistency[i], 0.0) + 0.2 * q
                model_s = float(self.conf_head.forward_scores(feats).item())
                per_stream.append(float(np.clip(0.5 * model_s + 0.5 * raw, 0.0, 1.0)))

        resid = residual_energy_fraction(mixture, streams)
        try:
            vad_resid = float(voiced_frame_density(mixture - streams.sum(axis=0)[: mixture.shape[0]], sample_rate))
        except Exception:
            vad_resid = resid
        attractor_mass = float(slot[:k].sum()) if len(slot) else 0.0
        with torch.no_grad():
            cfeats = torch.tensor([[resid, vad_resid, attractor_mass]], dtype=torch.float32)
            comp_model = float(self.comp_head.forward_scores(cfeats).item())
        # High residual / VAD → low completeness.
        comp_raw = float(np.clip(1.0 - 0.6 * resid - 0.4 * vad_resid, 0.0, 1.0))
        completeness = float(np.clip(0.5 * comp_model + 0.5 * comp_raw, 0.0, 1.0))

        ood_flag = False
        ood_dist = 0.0
        if condition_vec is not None and self.ood.mean is not None:
            ood_flag, ood_dist = self.ood.score(condition_vec)
            if ood_flag:
                per_stream = [s * self.ood_discount for s in per_stream]
                completeness *= self.ood_discount

        return ConfidenceOutput(
            per_stream=per_stream,
            completeness=completeness,
            ood_flag=ood_flag,
            ood_distance=ood_dist,
        )
