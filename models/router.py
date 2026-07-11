"""
Two-level adaptive router (Dev C, Phase 2).

Consumes Scene Analyzer feature vectors and produces per-expert weights
(w_TD, w_TF, w_NULL) via sigmoid gating at two granularities: one
sequence-level gate over the mean-pooled utterance features (global acoustic
condition) and one segment-level gate per window (local refinement). Sigmoid
rather than softmax so several experts can be simultaneously active on
ambiguous inputs (MASTER_PROJECT section 4.4).

Ships with the two auxiliary losses the master doc specifies: a load-balance
loss preventing router collapse onto one expert, and a null-sparsity loss
teaching the null expert to absorb trivial regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_EPS = 1e-8


class TwoLevelRouter(nn.Module):
    """
    Sequence-level plus segment-level sigmoid gating over experts.

    Args:
        feature_dim: Dimension of Scene Analyzer features per segment.
        num_experts: Number of routable experts including the null expert.
        hidden_dim: Hidden width of both gate MLPs. Default sized so total
            parameters land near the 0.5M budget from MASTER_PROJECT 5.2.
        null_index: Which expert index is the null pass-through.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_experts: int = 3,
        hidden_dim: int = 384,
        null_index: int = 2,
    ) -> None:
        super().__init__()
        if not 0 <= null_index < num_experts:
            raise ValueError("null_index must be a valid expert index")
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.null_index = null_index

        def gate() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_experts),
            )

        self.sequence_gate = gate()
        self.segment_gate = gate()

    def forward(self, segment_features: torch.Tensor) -> torch.Tensor:
        """
        Compute per-segment expert weights.

        Args:
            segment_features: [B, S, F] Scene Analyzer features per segment
                (S segments of 1 to 2 seconds each).

        Returns:
            Weights [B, S, E], nonnegative, summing to 1 over experts. The
            combination is elementwise: sigmoid(sequence logits) broadcast
            over segments, multiplied by sigmoid(segment logits), then
            renormalized, so the global condition scales what local gates
            propose.
        """
        if segment_features.ndim != 3:
            raise ValueError(f"expected [B, S, F], got shape {tuple(segment_features.shape)}")
        pooled = segment_features.mean(dim=1)  # [B, F]
        seq_w = torch.sigmoid(self.sequence_gate(pooled)).unsqueeze(1)  # [B, 1, E]
        seg_w = torch.sigmoid(self.segment_gate(segment_features))  # [B, S, E]
        combined = seq_w * seg_w
        return combined / (combined.sum(dim=-1, keepdim=True) + _EPS)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """
    Penalize router collapse onto a subset of experts.

    Implemented as the squared coefficient of variation of mean expert usage
    across the batch: zero when usage is perfectly uniform, growing as any
    expert monopolizes routing. Simple, differentiable, and matches the
    stated purpose in MASTER_PROJECT 7.1 (keep all experts active).

    Args:
        weights: [B, S, E] router output.

    Returns:
        Scalar loss tensor.
    """
    usage = weights.mean(dim=(0, 1))  # [E]
    return usage.var(unbiased=False) / (usage.mean() ** 2 + _EPS)


def null_sparsity_loss(
    weights: torch.Tensor, trivial_mask: torch.Tensor, null_index: int
) -> torch.Tensor:
    """
    Teach the null expert to own trivial regions and stay out of speech.

    Binary cross-entropy between the null expert's weight and a per-segment
    triviality target (1 where the segment is silence or single-speaker, per
    the data pipeline's overlap labels).

    Args:
        weights: [B, S, E] router output.
        trivial_mask: [B, S] float targets in {0, 1}.
        null_index: Index of the null expert in the weight vector.

    Returns:
        Scalar loss tensor.
    """
    null_w = weights[..., null_index].clamp(_EPS, 1.0 - _EPS)
    target = trivial_mask.float()
    return nn.functional.binary_cross_entropy(null_w, target)
