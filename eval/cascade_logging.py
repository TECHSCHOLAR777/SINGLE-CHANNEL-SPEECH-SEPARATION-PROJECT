"""
Runtime escalation logging (Dev C, Phase 2, P2-B2).

Bridges live cascade decisions into the JSONL RunLog so the existing
``escalation_rate`` / ``aggregate_by`` queries in ``eval/reporting.py``
have a real producer. One RunRecord per mixture, written at the moment
the pipeline finishes that mixture.

Kept in its own module (not reporting.py) so runtime producers and offline
report consumers never collide in a merge.
"""

from __future__ import annotations

import uuid
from typing import Any

from eval.reporting import RunLog, RunRecord
from models.cascade_gate import CascadeDecision
from schemas.separation_result import SeparationResult


def build_cascade_record(
    result: SeparationResult,
    decision: CascadeDecision | None,
    *,
    system: str,
    condition: str = "unknown",
    tier: str = "L1",
    n_true: int = -1,
    mean_si_sdri: float = float("nan"),
    mean_si_sdr: float = float("nan"),
    count_confidence: float = float("nan"),
    latency_sec: float = float("nan"),
    overlap_ratio: float = float("nan"),
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RunRecord:
    """
    Assemble one RunRecord from a pipeline pass.

    Args:
        result: Final SeparationResult for the mixture (post-fusion if
            escalated). ``result.escalated`` is the source of truth for the
            escalation flag; ``decision`` adds the gate's reasoning.
        decision: CascadeDecision from the gate, or None on paths that never
            consulted the gate (e.g. forced-cheap baseline runs).
        system: System label ("cascade", "cascade+fusion", "mossformer2", ...).
        condition / tier / n_true / metric kwargs: Standard RunRecord fields;
            metrics may be NaN for blind (reference-free) runs.
        run_id: Stable id if the caller has one; random hex otherwise.
        extra: Additional payload merged into RunRecord.extra.

    Returns:
        RunRecord ready for RunLog.append.
    """
    payload: dict[str, Any] = {
        "expert_used": result.expert_used,
        "sample_rate": result.sample_rate,
        "duration_sec": result.duration_sec,
    }
    if decision is not None:
        payload["gate_quality_score_db"] = decision.quality_score_db
        payload["gate_threshold_tau"] = decision.threshold_tau
        payload["gate_signal"] = decision.signal
        payload["gate_escalate"] = decision.escalate
    if extra:
        payload.update(extra)

    escalated = result.escalated or (decision.escalate if decision is not None else False)

    return RunRecord(
        run_id=run_id or uuid.uuid4().hex[:12],
        system=system,
        condition=condition,
        tier=tier,
        n_true=int(n_true),
        n_estimated=int(result.speaker_count),
        mean_si_sdri=mean_si_sdri,
        mean_si_sdr=mean_si_sdr,
        escalated=bool(escalated),
        count_confidence=count_confidence,
        latency_sec=latency_sec,
        overlap_ratio=overlap_ratio,
        extra=payload,
    )


class CascadeRunLogger:
    """
    Thin stateful wrapper: construct once per evaluation run, call
    ``log`` per mixture. Keeps system/condition/tier defaults so call
    sites stay one line.

    Example:
        logger = CascadeRunLogger("runs/eval.jsonl", system="cascade+fusion",
                                  condition="noisy", tier="L2")
        ...
        logger.log(result, decision, n_true=3, mean_si_sdri=8.4,
                   latency_sec=t_elapsed)
    """

    def __init__(
        self,
        path: str,
        *,
        system: str,
        condition: str = "unknown",
        tier: str = "L1",
    ) -> None:
        self.run_log = RunLog(path)
        self.system = system
        self.condition = condition
        self.tier = tier

    def log(
        self,
        result: SeparationResult,
        decision: CascadeDecision | None = None,
        **overrides: Any,
    ) -> RunRecord:
        """Build and append one record; returns it for inspection/tests."""
        kwargs: dict[str, Any] = {
            "system": self.system,
            "condition": self.condition,
            "tier": self.tier,
        }
        kwargs.update(overrides)
        record = build_cascade_record(result, decision, **kwargs)
        self.run_log.append(record)
        return record
