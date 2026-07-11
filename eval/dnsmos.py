"""
DNSMOS reference-free quality hook (Dev C, P0-C5).

Interface stub for the L5 / no-reference evaluation tier: real recordings
where no clean stems exist, so SI-SDRi is impossible and quality is scored
by Microsoft's DNSMOS P.835 model (predicted SIG / BAK / OVRL mean opinion
scores, 1 to 5).

This module freezes the interface now so RunRecord rows and report queries
can carry DNSMOS fields from day one. The actual ONNX inference activates
when the model file is configured: download `sig_bak_ovrl.onnx` from the
Microsoft DNS-Challenge repository, place it locally, and set
`eval.dnsmos.model_path` in configs/devc.yaml. Until then every call is
availability-gated and fails loudly with instructions rather than returning
fake numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DNSMOS_SAMPLE_RATE = 16000
"""DNSMOS P.835 operates on 16 kHz input."""

_DOWNLOAD_HINT = (
    "DNSMOS model not configured. Download sig_bak_ovrl.onnx from the "
    "microsoft/DNS-Challenge repository (DNSMOS/DNSMOS directory), then set "
    "eval.dnsmos.model_path in configs/devc.yaml."
)


class DnsmosScorer:
    """
    Reference-free quality scorer, availability-gated.

    Args:
        model_path: Path to the DNSMOS P.835 ONNX file, or None (unavailable).
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._session = None

    @property
    def is_available(self) -> bool:
        """True when a model file is configured and exists on disk."""
        return self.model_path is not None and self.model_path.exists()

    def score(self, waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
        """
        Score one waveform; requires the model to be available.

        Args:
            waveform: [T] mono waveform.
            sample_rate: Sample rate in Hz; must be DNSMOS_SAMPLE_RATE
                (callers resample upstream, matching the pipeline convention).

        Returns:
            Dict with keys "sig", "bak", "ovrl" on the 1-5 MOS scale.

        Raises:
            RuntimeError: When the model is not configured (with download
                instructions), so absent scores can never be mistaken for
                real ones.
        """
        if not self.is_available:
            raise RuntimeError(_DOWNLOAD_HINT)
        if sample_rate != DNSMOS_SAMPLE_RATE:
            raise ValueError(f"DNSMOS expects {DNSMOS_SAMPLE_RATE} Hz, got {sample_rate}")
        # Activation plan (kept out of scope for the stub, documented for the
        # implementer): onnxruntime session on self.model_path, 9.01 s
        # segments with hop, per-segment inference, polynomial MOS mapping,
        # mean over segments. Tracked in PROJECT_TODO as the P0-C5 upgrade.
        raise NotImplementedError(
            "DNSMOS inference not yet implemented; interface frozen, activation pending"
        )

    def score_or_none(self, waveform: np.ndarray, sample_rate: int) -> dict[str, float] | None:
        """Graceful variant for batch eval loops: None when unavailable, never fake."""
        if not self.is_available:
            return None
        return self.score(waveform, sample_rate)
