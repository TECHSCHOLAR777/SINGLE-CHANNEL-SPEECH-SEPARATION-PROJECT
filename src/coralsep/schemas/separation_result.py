"""
Shared separation result schema.

Defines the standard object returned by every expert wrapper, the fusion head,
and the full CoRAL-Sep pipeline. All modules must use this schema, never redefine
ad hoc result types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class StreamMetadata:
    """Per-stream metadata attached to a separation result."""

    expert_source: str = ""
    """Which expert produced this stream (e.g. 'mossformer2', 'srcorrnet', 'fused')."""

    confidence: float = 1.0
    """Calibrated confidence score in [0, 1]."""

    embedding: np.ndarray | None = None
    """Optional speaker-identity embedding (e.g. ECAPA-TDNN), shape [D]."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Extension slot for expert-specific fields (attractor vectors, mask entropy, etc.)."""


@dataclass
class SeparationResult:
    """
    Standard separation output consumed by alignment, fusion, evaluation, and demo.

    Attributes:
        streams: Separated waveforms, shape [K, T] as float32 numpy array.
        sample_rate: Audio sample rate in Hz (project standard: 16000).
        speaker_count: Estimated number of active speakers K_hat.
        metadata: Per-stream metadata list, length K.
        mixture: Optional original mixture waveform [T], for quality estimation.
        escalated: Whether the expensive expert was invoked (cascade path).
        expert_used: Primary expert label for non-escalated (cheap-only) path.
    """

    streams: np.ndarray
    sample_rate: int
    speaker_count: int
    metadata: list[StreamMetadata] = field(default_factory=list)
    mixture: np.ndarray | None = None
    escalated: bool = False
    expert_used: str = ""
    # CoRAL-Sep extension fields (optional; default to inert values)
    gate_vector: dict[str, float] = field(default_factory=dict)
    completeness_prob: float = 1.0
    ood_flag: bool = False
    attractor_probs: np.ndarray | None = None
    encoder_e0: np.ndarray | None = None
    decoder_features: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.streams = np.asarray(self.streams, dtype=np.float32)
        if self.streams.ndim != 2:
            raise ValueError(f"streams must be 2-D [K, T], got shape {self.streams.shape}")
        if self.speaker_count != self.streams.shape[0]:
            raise ValueError(
                f"speaker_count ({self.speaker_count}) must match streams.shape[0] "
                f"({self.streams.shape[0]})"
            )
        if len(self.metadata) == 0:
            self.metadata = [
                StreamMetadata(expert_source=self.expert_used, confidence=1.0)
                for _ in range(self.speaker_count)
            ]
        elif len(self.metadata) != self.speaker_count:
            raise ValueError(
                f"metadata length ({len(self.metadata)}) must match speaker_count "
                f"({self.speaker_count})"
            )

    @property
    def num_streams(self) -> int:
        return int(self.streams.shape[0])

    @property
    def duration_sec(self) -> float:
        return float(self.streams.shape[1]) / float(self.sample_rate)

    def to_torch(self) -> torch.Tensor:
        """Return streams as a PyTorch tensor [K, T]."""
        return torch.from_numpy(self.streams)

    @classmethod
    def from_torch(
        cls,
        streams: torch.Tensor,
        sample_rate: int,
        expert_used: str = "",
        mixture: np.ndarray | torch.Tensor | None = None,
    ) -> SeparationResult:
        """Construct from a PyTorch tensor [K, T] or [B, K, T] (batch size 1 only)."""
        if streams.ndim == 3:
            if streams.shape[0] != 1:
                raise ValueError("from_torch supports batch size 1 only")
            streams = streams[0]
        arr = streams.detach().cpu().numpy().astype(np.float32)
        mixture_arr: np.ndarray | None = None
        if mixture is not None:
            mixture_arr = np.asarray(mixture, dtype=np.float32).squeeze()
        return cls(
            streams=arr,
            sample_rate=sample_rate,
            speaker_count=arr.shape[0],
            mixture=mixture_arr,
            expert_used=expert_used,
        )
