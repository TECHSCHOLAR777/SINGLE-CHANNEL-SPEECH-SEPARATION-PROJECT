"""
Three-vote speaker counting subsystem (Dev B, P3-B2).

The SR-CorrNet model produces K0=5 output streams but only N of them
contain a real speaker. Counting combines three evidence sources:

Vote 1, Attractor probabilities p_k (BLUEPRINT §4.3):
    p_k = softmax(model output "pres["probs"]")[k], k=1..5
    Active speakers: slots where p_k > 0.5 (hardcoded in AttractorSplit).
    This is the primary estimate; directly from the model's learned priors.

Vote 2, E(0) count prior (Level-2 analyzer):
    CountPriorMLP from models/condition.py, trained on free count labels.
    Provides a prior before seeing the separated streams.

Vote 3, Bounded residual sweep (BLUEPRINT §4.3):
    Only runs when Vote 1 and Vote 2 disagree.
    Tests N_mode−1, N_mode, N_mode+1 (clipped to [2,5]) by computing SI-SDR
    with N streams against the mixture (using decoder only, no full re-pass).
    Keeps the N that maximises SI-SDR.

Fusion: a logistic regression model trained on held-out data combines the
three votes into a final count estimate. Falls back to Vote 1 (p_k) alone
when the fusion model is not calibrated yet.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_N_MIN = 2
_N_MAX = 5
_PROB_THRESH = 0.5


# ---------------------------------------------------------------------------
# Vote 1: attractor probabilities
# ---------------------------------------------------------------------------


def count_from_attractors(probs: torch.Tensor, threshold: float = _PROB_THRESH) -> int:
    """
    Count active speakers from attractor probability vector.

    Args:
        probs: (1, 7) or (7,) probability vector from pres["probs"].
               Slots 1–5 correspond to speakers 1–5 (slot 0 and 6 unused).
        threshold: Activation threshold (hardcoded 0.5 in AttractorSplit).

    Returns:
        Integer in {2, 3, 4, 5} (clipped).
    """
    p = probs.squeeze()
    if p.ndim == 0 or p.shape[0] < 5:
        return _N_MIN
    # Slots 1-5 are speaker existence probabilities.
    speaker_probs = p[1:6]  # shape (5,)
    n = int((speaker_probs > threshold).sum().item())
    return int(np.clip(n, _N_MIN, _N_MAX))


def attractor_confidence(probs: torch.Tensor, n_est: int, threshold: float = _PROB_THRESH) -> float:
    """
    Confidence of the attractor count estimate.

    High confidence: all active slots well above threshold AND all inactive
    slots well below it. Low confidence: ambiguous probabilities near 0.5.
    """
    p = probs.squeeze()[1:6]  # (5,) speaker slots
    active_margin = float(p[:n_est].min().item()) - threshold if n_est > 0 else 0.0
    if n_est < _N_MAX:
        inactive_margin = threshold - float(p[n_est:].max().item())
    else:
        inactive_margin = threshold
    return float(np.clip((active_margin + inactive_margin) / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Vote 3: residual sweep
# ---------------------------------------------------------------------------


def residual_sweep_sisdr(
    mixture: torch.Tensor,
    separated: torch.Tensor,
    candidates: list[int],
) -> int:
    """
    Pick the count N that maximises cardinality-aware SI-SDR vs the mixture.

    This uses `separated` which already has K0=5 streams; we evaluate the
    energy in the first N streams vs the mixture as a proxy for the "best N".

    Args:
        mixture: (1, T) or (T,) mixture waveform.
        separated: (K0, T) separated streams (K0=5).
        candidates: List of N values to try.

    Returns:
        The N in candidates with the highest proxy SI-SDR.
    """
    mix = mixture.squeeze()
    sep = separated.float()
    best_n, best_score = candidates[0], -float("inf")

    for n in candidates:
        n = int(np.clip(n, 1, sep.shape[0]))
        # Reconstruct: sum of first n streams
        reconst = sep[:n].sum(0)
        # SI-SDR of reconstruction vs mixture (we want them to match)
        score = _si_sdr(reconst, mix)
        if score > best_score:
            best_score = score
            best_n = n

    return int(np.clip(best_n, _N_MIN, _N_MAX))


def _si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> float:
    eps = 1e-10
    s_target = (estimate * target).sum() / (target.pow(2).sum() + eps) * target
    noise = estimate - s_target
    return float((s_target.pow(2).sum() / (noise.pow(2).sum() + eps)).log10().item() * 10.0)


# ---------------------------------------------------------------------------
# Fusion logistic regression
# ---------------------------------------------------------------------------


class CountFusion(nn.Module):
    """
    Logistic regression fusing 3 vote signals into a count estimate.

    Input features (7-D):
      v1: Vote-1 integer count / 5.0  (normalised)
      v1_conf: Vote-1 attractor confidence
      v2_0, v2_1, v2_2, v2_3: Vote-2 count-prior probabilities for {2,3,4,5}
      v3: Vote-3 integer count / 5.0  (or v1 when sweep was skipped)

    Output: 4-class logits for N ∈ {2,3,4,5}.
    Trained on held-out evaluation data (P3-C3).
    """

    def __init__(self, in_features: int = 7) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 7) vote feature vector.
        Returns:
            (B, 4) logits for N ∈ {2,3,4,5}.
        """
        return self.linear(features)

    def count_estimate(self, features: torch.Tensor) -> torch.Tensor:
        """Return (B,) count estimates in {2,3,4,5}."""
        return self.forward(features).argmax(dim=-1) + 2


def build_vote_features(
    probs: torch.Tensor,
    count_prior_probs: torch.Tensor,
    n_sweep: int,
    threshold: float = _PROB_THRESH,
) -> torch.Tensor:
    """
    Build the 7-D feature vector for CountFusion.

    Args:
        probs: (1,7) or (7,) attractor probabilities.
        count_prior_probs: (4,) or (1,4) softmax probs for N∈{2,3,4,5}.
        n_sweep: Vote-3 count from residual sweep.

    Returns:
        (7,) float32 feature tensor.
    """
    n_v1 = count_from_attractors(probs, threshold)
    conf_v1 = attractor_confidence(probs, n_v1, threshold)
    v2 = count_prior_probs.squeeze()[:4]  # (4,)
    return torch.tensor(
        [n_v1 / 5.0, conf_v1, *v2.tolist(), n_sweep / 5.0],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Full three-vote counter
# ---------------------------------------------------------------------------


class ThreeVoteCounter:
    """
    Orchestrates all three votes and produces a final count estimate.

    Designed to run without the fusion model during early training
    (falls back to Vote 1 when fusion is not loaded).
    """

    def __init__(
        self,
        fusion: CountFusion | None = None,
        threshold: float = _PROB_THRESH,
        sweep_disagreement_only: bool = True,
    ) -> None:
        self.fusion = fusion
        self.threshold = threshold
        self.sweep_disagreement_only = sweep_disagreement_only

    def estimate(
        self,
        probs: torch.Tensor,
        count_prior_probs: torch.Tensor,
        mixture: torch.Tensor,
        separated: torch.Tensor,
    ) -> dict[str, object]:
        """
        Run all three votes and fuse them.

        Args:
            probs: (1,7) attractor probabilities (pres["probs"]).
            count_prior_probs: (1,4) or (4,) from CountPriorMLP.
            mixture: (T,) or (1,T) mixture waveform.
            separated: (K0, T) all K0=5 output streams.

        Returns:
            Dict with: n_est (int), n_v1 (int), n_v2 (int), n_v3 (int),
            confidence (float), vote_features (Tensor[7]).
        """
        n_v1 = count_from_attractors(probs, self.threshold)
        n_v2 = int(count_prior_probs.squeeze().argmax().item()) + 2

        # Vote 3: sweep only when v1 and v2 disagree (BLUEPRINT §4.3).
        if not self.sweep_disagreement_only or n_v1 != n_v2:
            mode = n_v1  # use v1 as the mode
            candidates = sorted(
                set(int(np.clip(c, _N_MIN, _N_MAX)) for c in [mode - 1, mode, mode + 1])
            )
            n_v3 = residual_sweep_sisdr(mixture, separated, candidates)
        else:
            n_v3 = n_v1

        vote_feats = build_vote_features(probs, count_prior_probs, n_v3, self.threshold)

        if self.fusion is not None:
            n_est = int(self.fusion.count_estimate(vote_feats.unsqueeze(0)).item())
        else:
            n_est = n_v1  # fallback: trust attractors

        confidence = attractor_confidence(probs, n_est, self.threshold)

        return {
            "n_est": int(np.clip(n_est, _N_MIN, _N_MAX)),
            "n_v1": n_v1,
            "n_v2": n_v2,
            "n_v3": n_v3,
            "confidence": confidence,
            "vote_features": vote_feats,
        }
