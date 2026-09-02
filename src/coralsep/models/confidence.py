"""
Per-stream confidence, completeness, and OOD scoring (Dev B, P3-B3).

Three composite scores are computed for every separation result:

1. Per-stream confidence  (calibrated, per-speaker):
   • p_k from attractor probabilities (primary signal)
   • Inter-stage consistency: cosine similarity between E(0) embeddings
     and decoder-stage features (high if the model is consistent about
     which content belongs to this speaker)
   • DNSMOS: blind quality (rewards clean, penalises artefacts)

2. Completeness probability  (one per clip, not per stream):
   • Residual energy fraction: how much energy is in mixture − sum(streams)
   • SileroVAD on the residual: voiced residual energy suggests a missed speaker
   • Attractor mass: fraction of total attractor probability not assigned
     to any detected speaker

3. OOD Mahalanobis discount  (scalar, multiplied into confidence):
   • Mahalanobis distance of the Level-1 feature vector from the training
     distribution (fit at calibration time, stored in calibration/)
   • Clips above threshold shrink confidence proportionally
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-10


# ---------------------------------------------------------------------------
# 1. Per-stream confidence
# ---------------------------------------------------------------------------


class StreamConfidenceHead(nn.Module):
    """
    Logistic regression fusing p_k, consistency, and DNSMOS into per-stream confidence.

    Input features (3-D per stream):
      0: p_k (attractor probability for this slot)
      1: inter-stage cosine consistency (mean over layers)
      2: DNSMOS OVRL score normalised to [0, 1] (1.0 if unavailable)

    Output: scalar confidence in [0, 1].
    Trained on held-out calibration data (P3-C3).
    """

    def __init__(self, in_features: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, K, 3) per-stream features.
        Returns:
            (B, K) confidence scores in [0, 1].
        """
        return torch.sigmoid(self.linear(features)).squeeze(-1)


def inter_stage_consistency(
    e0: torch.Tensor,
    dec_features: list[torch.Tensor],
    stream_idx: int,
) -> float:
    """
    Mean cosine similarity between E(0) and each decoder-stage feature for stream k.

    Args:
        e0: (1, T, F, D) encoder output.
        dec_features: List of 4 tensors, each (B, K, T, F, D) from dec_block hooks.
        stream_idx: Which output stream (0-indexed).

    Returns:
        Mean cosine similarity in [-1, 1]; higher = more consistent.
    """
    if not dec_features:
        return 1.0

    # Pool E(0): (D,)
    e0_pooled = e0.squeeze(0).mean(dim=(0, 1))  # (D,)

    sims: list[float] = []
    for dec in dec_features:
        # dec: (B, K, T, F, D)
        if dec.ndim == 5 and dec.shape[1] > stream_idx:
            dec_stream = dec[0, stream_idx].mean(dim=(0, 1))  # (D,)
            sim = F.cosine_similarity(e0_pooled.unsqueeze(0), dec_stream.unsqueeze(0)).item()
            sims.append(float(sim))

    return float(np.mean(sims)) if sims else 1.0


def stream_confidence_features(
    probs: torch.Tensor,
    e0: torch.Tensor,
    dec_features: list[torch.Tensor],
    n_streams: int,
    dnsmos_scores: list[float] | None = None,
) -> torch.Tensor:
    """
    Build (K, 3) feature matrix for StreamConfidenceHead.

    Args:
        probs: (1, 7) attractor probability vector.
        e0: (1, T, F, D) encoder output.
        dec_features: Decoder stage features from hooks.
        n_streams: Number of active streams (N_est).
        dnsmos_scores: List of K DNSMOS OVRL scores, or None.

    Returns:
        (K, 3) float32 tensor.
    """
    p = probs.squeeze()
    speaker_probs = p[1:6]  # slots 1-5

    rows: list[list[float]] = []
    for k in range(n_streams):
        pk = float(speaker_probs[k].item()) if k < len(speaker_probs) else 0.5
        consistency = inter_stage_consistency(e0, dec_features, k)
        dnsmos = (dnsmos_scores[k] / 5.0) if (dnsmos_scores and k < len(dnsmos_scores)) else 1.0
        rows.append([pk, consistency, float(np.clip(dnsmos, 0.0, 1.0))])

    return torch.tensor(rows, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 2. Completeness
# ---------------------------------------------------------------------------


def residual_energy_fraction(mixture: torch.Tensor, streams: torch.Tensor) -> float:
    """
    Fraction of total energy remaining in the residual (mixture − sum of streams).

    A high fraction suggests a missed speaker or severe model failure.

    Args:
        mixture: (T,) mixture waveform.
        streams: (K, T) separated streams.

    Returns:
        Float in [0, 1].
    """
    mix = mixture.float().squeeze()
    reconst = streams.float().sum(0)
    residual = mix - reconst
    resid_energy = float(residual.pow(2).sum().item())
    total_energy = float(mix.pow(2).sum().item())
    return float(np.clip(resid_energy / max(total_energy, _EPS), 0.0, 1.0))


def attractor_mass_unassigned(probs: torch.Tensor, n_est: int) -> float:
    """
    Fraction of total attractor probability mass NOT assigned to any detected speaker.

    Values near 1.0 mean the model thinks there are more speakers than we counted.
    Values near 0.0 mean all probability mass is accounted for.
    """
    p = probs.squeeze()[1:6]  # (5,) speaker slots
    total_mass = float(p.sum().item())
    assigned_mass = float(p[:n_est].sum().item())
    unassigned = max(0.0, total_mass - assigned_mass)
    return float(np.clip(unassigned / max(total_mass, _EPS), 0.0, 1.0))


class CompletenessHead(nn.Module):
    """
    Logistic regression predicting P(all speakers captured).

    Input features (3-D):
      0: residual energy fraction (low = good)
      1: voiced density of residual (low = good)
      2: attractor mass fraction unassigned (low = good)

    Output: scalar completeness probability in [0, 1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 3) completeness features.
        Returns:
            (B,) completeness probability in [0, 1].
        """
        return torch.sigmoid(self.linear(features)).squeeze(-1)


def completeness_features(
    mixture: torch.Tensor,
    streams: torch.Tensor,
    probs: torch.Tensor,
    n_est: int,
    voiced_residual_fn: object | None = None,
) -> torch.Tensor:
    """
    Build (3,) feature vector for CompletenessHead.

    Args:
        mixture: (T,) mixture.
        streams: (K, T) separated.
        probs: (1, 7) attractor probabilities.
        n_est: Estimated speaker count.
        voiced_residual_fn: Optional function(waveform) → float voiced density.

    Returns:
        (3,) float32 tensor.
    """
    resid_frac = residual_energy_fraction(mixture, streams)

    residual = mixture.float().squeeze() - streams.float().sum(0)
    if voiced_residual_fn is not None:
        voiced_resid = float(voiced_residual_fn(residual))
    else:
        from coralsep.models.condition import voiced_density_energy
        voiced_resid = voiced_density_energy(residual)

    unassigned = attractor_mass_unassigned(probs, n_est)

    return torch.tensor([resid_frac, voiced_resid, unassigned], dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. OOD Mahalanobis discount
# ---------------------------------------------------------------------------


class MahalanobisOOD:
    """
    Scores how out-of-distribution a condition vector is relative to training.

    Fit at calibration time on held-out clean+augmented data. Stored alongside
    the calibration models in calibration/ood.npz.
    """

    def __init__(
        self,
        mean: np.ndarray | None = None,
        cov_inv: np.ndarray | None = None,
        threshold: float = 10.0,
    ) -> None:
        self.mean = mean
        self.cov_inv = cov_inv
        self.threshold = threshold
        self._is_fit = mean is not None and cov_inv is not None

    def fit(self, features: np.ndarray) -> None:
        """
        Fit mean and precision matrix from training-distribution features.

        Args:
            features: (N, D) feature matrix (e.g. Level-1 features from training).
        """
        self.mean = features.mean(axis=0)
        cov = np.cov(features, rowvar=False)
        self.cov_inv = np.linalg.pinv(cov)
        self._is_fit = True

    def distance(self, x: np.ndarray) -> float:
        """Mahalanobis distance of feature vector x from training distribution."""
        if not self._is_fit:
            return 0.0
        diff = x - self.mean  # type: ignore[operator]
        return float(np.sqrt(diff @ self.cov_inv @ diff))  # type: ignore[operator]

    def discount(self, x: np.ndarray) -> float:
        """
        Return a multiplicative discount in [0, 1] for confidence.

        1.0 at d=0, decaying to 0.0 at d=threshold.
        """
        d = self.distance(x)
        return float(np.clip(1.0 - d / max(self.threshold, _EPS), 0.0, 1.0))

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, cov_inv=self.cov_inv, threshold=self.threshold)

    @classmethod
    def load(cls, path: str) -> "MahalanobisOOD":
        data = np.load(path)
        return cls(
            mean=data["mean"],
            cov_inv=data["cov_inv"],
            threshold=float(data["threshold"]),
        )
