"""Tests for eval/cascade_logging.py (P2-B2)."""

from __future__ import annotations

import numpy as np

from eval.cascade_logging import CascadeRunLogger, build_cascade_record
from eval.reporting import RunLog, escalation_rate
from models.cascade_gate import CascadeDecision
from schemas.separation_result import SeparationResult


def _result(escalated: bool, k: int = 2, expert: str = "mossformer2") -> SeparationResult:
    return SeparationResult(
        streams=np.zeros((k, 1600), dtype=np.float32),
        sample_rate=16000,
        speaker_count=k,
        escalated=escalated,
        expert_used=expert,
    )


def _decision(escalate: bool, score: float = 8.0, tau: float = 12.0) -> CascadeDecision:
    return CascadeDecision(
        escalate=escalate, quality_score_db=score, threshold_tau=tau, signal="min"
    )


def test_record_carries_gate_reasoning():
    rec = build_cascade_record(
        _result(escalated=True, expert="fused"),
        _decision(escalate=True, score=7.3),
        system="cascade+fusion",
        condition="noisy",
        tier="L2",
        n_true=2,
        mean_si_sdri=9.1,
    )
    assert rec.escalated is True
    assert rec.system == "cascade+fusion"
    assert rec.n_estimated == 2
    assert rec.extra["gate_quality_score_db"] == 7.3
    assert rec.extra["gate_threshold_tau"] == 12.0
    assert rec.extra["gate_signal"] == "min"
    assert rec.extra["expert_used"] == "fused"


def test_escalation_flag_or_of_result_and_decision():
    # result says escalated even if decision object missing
    rec = build_cascade_record(_result(escalated=True), None, system="cascade")
    assert rec.escalated is True
    # decision says escalate even if result flag was not set by caller
    rec = build_cascade_record(_result(escalated=False), _decision(True), system="cascade")
    assert rec.escalated is True
    # neither
    rec = build_cascade_record(_result(escalated=False), _decision(False, score=15.0), system="cascade")
    assert rec.escalated is False


def test_no_decision_path_omits_gate_fields():
    rec = build_cascade_record(_result(escalated=False), None, system="mossformer2")
    assert "gate_quality_score_db" not in rec.extra
    assert rec.extra["expert_used"] == "mossformer2"


def test_logger_roundtrip_and_escalation_rate_query(tmp_path):
    path = tmp_path / "runs.jsonl"
    logger = CascadeRunLogger(str(path), system="cascade", condition="clean", tier="L1")

    logger.log(_result(escalated=True), _decision(True), n_true=2)
    logger.log(_result(escalated=False), _decision(False, score=14.0), n_true=2)
    logger.log(_result(escalated=True), _decision(True, score=6.0), n_true=3, tier="L2")
    logger.log(_result(escalated=False), _decision(False, score=13.0), n_true=3, tier="L2")

    records = RunLog(str(path)).load()
    assert len(records) == 4

    rates = {row["tier"]: row for row in escalation_rate(records, by=("tier",))}
    assert rates["L1"]["escalation_rate"] == 0.5
    assert rates["L2"]["escalation_rate"] == 0.5


def test_overrides_win_over_defaults(tmp_path):
    logger = CascadeRunLogger(str(tmp_path / "r.jsonl"), system="cascade")
    rec = logger.log(_result(False), None, system="baseline", condition="reverb", latency_sec=0.42)
    assert rec.system == "baseline"
    assert rec.condition == "reverb"
    assert rec.latency_sec == 0.42


def test_json_roundtrip_preserves_extra(tmp_path):
    path = tmp_path / "r.jsonl"
    logger = CascadeRunLogger(str(path), system="cascade")
    logger.log(_result(True), _decision(True), extra={"peel_iters": 3})
    loaded = RunLog(str(path)).load()[0]
    assert loaded.extra["peel_iters"] == 3
    assert loaded.extra["gate_escalate"] is True
