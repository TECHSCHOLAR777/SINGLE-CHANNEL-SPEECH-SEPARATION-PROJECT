"""
Confidence-Routed Residual Refinement fusion head (Dev B, Phase 2).

Fuses aligned expensive-expert (SR-CorrNet) and cheap-expert (MossFormer2)
streams on escalated inputs:

    s_fused_k = s_SR_k + alpha_k(t) * R_theta(s_SR_k, s_Moss_k, mixture)

Alpha is a per-stream sigmoid gate over SR confidence, MossFormer2 mask
entropy, a blind local SI-SDRi proxy, and scene/router weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


@dataclass
class FusionOutput:
    """Fusion head output bundle for loss computation and inference.

    Attributes:
        fused_streams: Corrected waveforms [B, K, T].
        residual: Per-stream correction R_theta [B, K, T].
        alpha: Per-stream gate values in (0, 1) [B, K, 1].
    """

    fused_streams: torch.Tensor
    residual: torch.Tensor
    alpha: torch.Tensor


def _si_sdr_proxy(estimate: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    """
    Differentiable blind SI-SDR proxy per stream (no references).

    Returns values in roughly dB scale for gating; used as local quality hint.
    """
    est = estimate - estimate.mean(dim=-1, keepdim=True)
    mix = mixture - mixture.mean(dim=-1, keepdim=True)
    mix_energy = (mix * mix).sum(dim=-1, keepdim=True).clamp_min(_EPS)
    scale = (est * mix).sum(dim=-1, keepdim=True) / mix_energy
    target = scale * mix
    residual = est - target
    num = (target * target).sum(dim=-1)
    den = (residual * residual).sum(dim=-1).clamp_min(_EPS)
    return 10.0 * torch.log10((num + _EPS) / den)


def compute_fusion_gate_features(
    sr_streams: torch.Tensor,
    moss_streams: torch.Tensor,
    mixture: torch.Tensor,
    sr_confidence: torch.Tensor,
    moss_mask_entropy: torch.Tensor,
    scene_weight_tf: torch.Tensor,
) -> torch.Tensor:
    """
    Build per-stream alpha-gate inputs.

    Args:
        sr_streams: [B, K, T] aligned expensive-expert outputs.
        moss_streams: [B, K, T] cheap-expert outputs.
        mixture: [B, T] original mixture.
        sr_confidence: [B, K] per-stream SR-CorrNet confidence in [0, 1].
        moss_mask_entropy: [B, K] MossFormer2 mask entropy proxy.
        scene_weight_tf: [B, K] time-frequency expert routing weight.

    Returns:
        Gate feature tensor [B, K, 4].
    """
    b, k, _ = sr_streams.shape
    mix_exp = mixture.unsqueeze(1).expand(b, k, -1)
    local_proxy = _si_sdr_proxy(sr_streams, mix_exp)
    local_norm = torch.tanh(local_proxy / 20.0)

    return torch.stack(
        [
            sr_confidence.clamp(0.0, 1.0),
            moss_mask_entropy.clamp(0.0, 1.0),
            local_norm,
            scene_weight_tf.clamp(0.0, 1.0),
        ],
        dim=-1,
    )


class FusionAlphaGate(nn.Module):
    """Sigmoid gate producing alpha_k from four fusion features per stream."""

    def __init__(self, n_gate_features: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_gate_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gate_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_features: [B, K, 4].

        Returns:
            Alpha values [B, K, 1] in (0, 1).
        """
        logits = self.mlp(gate_features)
        return torch.sigmoid(logits)


class ResidualCorrector(nn.Module):
    """Small 1-D conv network R_theta producing per-sample corrections."""

    def __init__(self, in_channels: int = 4, hidden_channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=7, padding=3),
        )

    def forward(self, sr_streams: torch.Tensor, moss_streams: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr_streams: [B, K, T].
            moss_streams: [B, K, T].

        Returns:
            Residual correction [B, K, T].
        """
        b, k, t = sr_streams.shape
        x = torch.stack([sr_streams, moss_streams, sr_streams - moss_streams, sr_streams * moss_streams], dim=2)
        x = x.reshape(b * k, 4, t)
        residual = self.net(x)
        if residual.shape[-1] != t:
            residual = F.interpolate(residual, size=t, mode="linear", align_corners=False)
        return residual.reshape(b, k, t)


class CRRRFusionHead(nn.Module):
    """
    Confidence-Routed Residual Refinement fusion head (~1M parameters).

    Applied only on escalated inputs after Hungarian alignment.
    """

    def __init__(self, hidden_channels: int = 256) -> None:
        super().__init__()
        self.alpha_gate = FusionAlphaGate()
        self.residual_net = ResidualCorrector(hidden_channels=hidden_channels)

    def forward(
        self,
        sr_streams: torch.Tensor,
        moss_streams: torch.Tensor,
        mixture: torch.Tensor,
        sr_confidence: torch.Tensor,
        moss_mask_entropy: torch.Tensor,
        scene_weight_tf: torch.Tensor,
    ) -> FusionOutput:
        """
        Fuse aligned expert outputs.

        Args:
            sr_streams: [B, K, T] aligned SR-CorrNet streams.
            moss_streams: [B, K, T] MossFormer2 streams (same speaker order).
            mixture: [B, T] input mixture.
            sr_confidence: [B, K] per-stream confidence.
            moss_mask_entropy: [B, K] mask entropy proxy.
            scene_weight_tf: [B, K] TF expert routing weight.

        Returns:
            FusionOutput with fused streams, residual, and alpha gate.
        """
        if sr_streams.shape != moss_streams.shape:
            raise ValueError(
                f"stream shape mismatch: sr {tuple(sr_streams.shape)} "
                f"vs moss {tuple(moss_streams.shape)}"
            )
        gate_feats = compute_fusion_gate_features(
            sr_streams,
            moss_streams,
            mixture,
            sr_confidence,
            moss_mask_entropy,
            scene_weight_tf,
        )
        alpha = self.alpha_gate(gate_feats)
        residual = self.residual_net(sr_streams, moss_streams)
        fused = sr_streams + alpha * residual
        return FusionOutput(fused_streams=fused, residual=residual, alpha=alpha)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
