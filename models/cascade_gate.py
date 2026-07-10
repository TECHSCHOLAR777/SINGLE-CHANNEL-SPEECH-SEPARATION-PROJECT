"""
Cascade gate controller (Dev B, Phase 2).

Compares REAL-M blind quality estimates against threshold tau and decides
whether to escalate to the expensive expert. Uses the conservative
min-stream SI-SNR signal by default so borderline inputs escalate rather
than being accepted at marginal quality (MASTER_PROJECT section 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from models.realm_quality import QualityEstimate

GateSignal = Literal["min", "mean"]


@dataclass
class CascadeDecision:
    """Outcome of a single cascade gate evaluation.

    Attributes:
        escalate: True when the expensive expert should run.
        quality_score_db: Blind SI-SNR score (dB) used for the decision.
        threshold_tau: Quality threshold in dB.
        signal: Which REAL-M aggregate was compared to tau.
    """

    escalate: bool
    quality_score_db: float
    threshold_tau: float
    signal: GateSignal


class CascadeGate:
    """
    Binary escalate-or-accept gate driven by REAL-M quality scores.

    Args:
        tau: Quality threshold in dB. Scores below tau escalate.
        signal: REAL-M aggregate to compare — ``min`` (conservative) or
            ``mean`` across streams.
    """

    def __init__(self, tau: float = 12.0, signal: GateSignal = "min") -> None:
        if signal not in ("min", "mean"):
            raise ValueError(f"signal must be 'min' or 'mean', got {signal!r}")
        self.tau = float(tau)
        self.signal: GateSignal = signal

    def quality_score(self, quality: QualityEstimate) -> float:
        """Return the configured aggregate blind SI-SNR in dB."""
        if self.signal == "min":
            return quality.min_sisnr_db
        return quality.mean_sisnr_db

    def decide(self, quality: QualityEstimate) -> CascadeDecision:
        """
        Evaluate whether the cheap expert output is good enough.

        Escalation occurs when quality_score < tau (strictly below threshold).
        """
        score = self.quality_score(quality)
        return CascadeDecision(
            escalate=score < self.tau,
            quality_score_db=score,
            threshold_tau=self.tau,
            signal=self.signal,
        )

    def should_escalate(self, quality: QualityEstimate) -> bool:
        """Convenience wrapper returning only the escalate boolean."""
        return self.decide(quality).escalate
