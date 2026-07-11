"""
Scene Analyzer stub (Dev B interim, pending Dev A P2-A1).

Lightweight log-mel feature extractor producing segment features for the
router and coarse speaker-count logits. Sized smaller than the full 1.5M
spec so P2 training can proceed; Dev A replaces with the full analyzer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


class SceneAnalyzer(nn.Module):
    """
    Minimal scene feature extractor for cascade training.

    Args:
        feature_dim: Per-segment feature width consumed by the router.
        n_mels: Log-mel bins.
        max_speakers: Upper bound for coarse count logits.
        segment_samples: Audio samples per segment at 16 kHz.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        n_mels: int = 64,
        max_speakers: int = 5,
        segment_samples: int = 32000,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.max_speakers = max_speakers
        self.segment_samples = segment_samples
        self.n_mels = n_mels

        self.mel = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
        )
        self.segment_encoder = nn.GRU(
            input_size=64,
            hidden_size=feature_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.count_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, max_speakers),
        )

    def _to_log_mel_proxy(self, mixture: torch.Tensor) -> torch.Tensor:
        """Cheap mel proxy: strided conv features from raw waveform [B, T]."""
        x = mixture.unsqueeze(1)
        return self.mel(x).transpose(1, 2)

    def forward(self, mixture: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extract scene features from mono mixtures.

        Args:
            mixture: [B, T] waveforms.

        Returns:
            Dict with:
                segment_features: [B, S, F] for the router.
                count_logits: [B, max_speakers] coarse speaker-count logits.
                scene_weights: [B, 3] mean router hints (w_TD, w_TF, w_NULL proxy).
        """
        if mixture.ndim != 2:
            raise ValueError(f"expected mixture [B, T], got {tuple(mixture.shape)}")

        b, t = mixture.shape
        n_seg = max(1, math.ceil(t / self.segment_samples))
        pad_len = n_seg * self.segment_samples
        if t < pad_len:
            mixture = F.pad(mixture, (0, pad_len - t))
        elif t > pad_len:
            mixture = mixture[:, :pad_len]

        chunks = mixture.view(b, n_seg, self.segment_samples)
        feats: list[torch.Tensor] = []
        for s in range(n_seg):
            mel = self._to_log_mel_proxy(chunks[:, s, :])
            enc, _ = self.segment_encoder(mel)
            feats.append(enc.mean(dim=1))
        segment_features = torch.stack(feats, dim=1)

        pooled = segment_features.mean(dim=1)
        count_logits = self.count_head(pooled)

        reverb_proxy = torch.std(mixture, dim=-1, keepdim=True)
        noise_floor = torch.quantile(mixture.abs(), 0.1, dim=-1, keepdim=True)
        flatness = torch.std(mixture, dim=-1, keepdim=True) / (
            mixture.abs().mean(dim=-1, keepdim=True) + _EPS
        )
        scene_raw = torch.cat([reverb_proxy, noise_floor, flatness], dim=-1)
        scene_weights = torch.softmax(scene_raw, dim=-1)

        return {
            "segment_features": segment_features,
            "count_logits": count_logits,
            "scene_weights": scene_weights,
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
