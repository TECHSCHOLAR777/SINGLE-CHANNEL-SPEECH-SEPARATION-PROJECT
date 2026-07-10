"""Tests for models/cascade_gate.py."""

from models.cascade_gate import CascadeGate
from models.realm_quality import QualityEstimate


def _quality(min_db: float, mean_db: float | None = None) -> QualityEstimate:
    per = [min_db, min_db + 1.0, min_db + 0.5]
    return QualityEstimate(
        sisnr_db_per_stream=per,
        mean_sisnr_db=mean_db if mean_db is not None else sum(per) / len(per),
        min_sisnr_db=min_db,
    )


def test_escalate_when_below_tau() -> None:
    gate = CascadeGate(tau=12.0, signal="min")
    decision = gate.decide(_quality(10.0))
    assert decision.escalate is True
    assert decision.quality_score_db == 10.0


def test_accept_when_above_tau() -> None:
    gate = CascadeGate(tau=12.0, signal="min")
    decision = gate.decide(_quality(14.0))
    assert decision.escalate is False


def test_mean_signal_mode() -> None:
    gate = CascadeGate(tau=12.0, signal="mean")
    q = _quality(min_db=8.0, mean_db=13.0)
    assert gate.decide(q).escalate is False
    assert gate.quality_score(q) == 13.0


def test_boundary_at_tau_does_not_escalate() -> None:
    gate = CascadeGate(tau=12.0)
    assert gate.should_escalate(_quality(12.0)) is False
