"""
Shared separation result schema for CALM-Sep (and legacy CA-MoSE fields).

All modules must use this schema — never redefine ad hoc result types.
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
    confidence: float = 1.0
    embedding: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SeparationResult:
    """
    Standard separation output consumed by alignment, evaluation, and demo.

    CALM-Sep fields (p_k, gate_vector, completeness, ood_flag, ...) default to
    None/False so legacy callers remain valid.
    """

    streams: np.ndarray
    sample_rate: int
    speaker_count: int
    metadata: list[StreamMetadata] = field(default_factory=list)
    mixture: np.ndarray | None = None
    escalated: bool = False
    expert_used: str = ""
    # CALM-Sep extensions (BLUEPRINT §6.5)
    p_k: np.ndarray | None = None
    gate_vector: dict[str, Any] | list[float] | None = None
    completeness: float | None = None
    ood_flag: bool = False
    condition_estimates: dict[str, Any] | None = None
    count_posterior: np.ndarray | dict[int, float] | None = None

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
        if self.p_k is not None:
            self.p_k = np.asarray(self.p_k, dtype=np.float32)

    @property
    def num_streams(self) -> int:
        return int(self.streams.shape[0])

    @property
    def duration_sec(self) -> float:
        return float(self.streams.shape[1]) / float(self.sample_rate)

    def to_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.streams)

    def to_report_dict(self) -> dict[str, Any]:
        """JSON-serializable report (waveforms excluded)."""
        post = self.count_posterior
        if isinstance(post, np.ndarray):
            post = {int(i + 2): float(v) for i, v in enumerate(post.tolist())}
        return {
            "speaker_count": self.speaker_count,
            "sample_rate": self.sample_rate,
            "expert_used": self.expert_used,
            "completeness": self.completeness,
            "ood_flag": self.ood_flag,
            "p_k": None if self.p_k is None else self.p_k.reshape(-1).tolist(),
            "gate_vector": self.gate_vector,
            "condition_estimates": self.condition_estimates,
            "count_posterior": post,
            "stream_confidence": [m.confidence for m in self.metadata],
        }

    @classmethod
    def from_torch(
        cls,
        streams: torch.Tensor,
        sample_rate: int,
        expert_used: str = "",
        mixture: np.ndarray | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SeparationResult:
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
            **kwargs,
        )
