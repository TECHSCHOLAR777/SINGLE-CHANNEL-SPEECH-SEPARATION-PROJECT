"""Tests for eval/reporting.py: run log, aggregations, calibration, curves."""

from pathlib import Path

import numpy as np
import pytest

from eval.reporting import (
    RunLog,
    RunRecord,
    aggregate_by,
    breakpoint_curve,
    calibration_curve,
    counting_report,
    escalation_rate,
    overlap_curve,
    to_markdown_table,
)


def make_record(**kw) -> RunRecord:
    base = dict(
        run_id="r",
        system="cascade",
        condition="clean",
        tier="L1",
        n_true=3,
        n_estimated=3,
        mean_si_sdri=15.0,
    )
    base.update(kw)
    return RunRecord(**base)


def test_runlog_roundtrip(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "log.jsonl")
    log.append(make_record(run_id="a", escalated=True))
    log.append(make_record(run_id="b", mean_si_sdri=10.0))
    loaded = log.load()
    assert [r.run_id for r in loaded] == ["a", "b"]
    assert loaded[0].escalated is True


def test_aggregate_by_groups_and_stats() -> None:
    records = [
        make_record(tier="L1", mean_si_sdri=10.0),
        make_record(tier="L1", mean_si_sdri=20.0),
        make_record(tier="L2", mean_si_sdri=8.0),
    ]
    rows = aggregate_by(records, keys=("tier",))
    by_tier = {r["tier"]: r for r in rows}
    assert by_tier["L1"]["mean_si_sdri_mean"] == pytest.approx(15.0)
    assert by_tier["L1"]["count"] == 2
    assert by_tier["L2"]["count"] == 1


def test_escalation_rate() -> None:
    records = [
        make_record(tier="L1", escalated=False),
        make_record(tier="L1", escalated=True),
        make_record(tier="L3", escalated=True),
    ]
    rows = {r["tier"]: r for r in escalation_rate(records)}
    assert rows["L1"]["escalation_rate"] == pytest.approx(0.5)
    assert rows["L3"]["escalation_rate"] == pytest.approx(1.0)


def test_counting_report() -> None:
    records = [make_record(n_true=3, n_estimated=3), make_record(n_true=4, n_estimated=3)]
    rep = counting_report(records, count_range=(2, 5))
    assert rep["accuracy"] == pytest.approx(0.5)
    assert rep["confusion"][2][1] == 1  # true 4 estimated 3
    assert rep["labels"] == [2, 3, 4, 5]


def test_calibration_perfectly_calibrated_low_ece() -> None:
    rng = np.random.default_rng(7)
    conf = rng.uniform(0.05, 0.95, size=5000)
    correct = rng.uniform(size=5000) < conf
    calib = calibration_curve(conf.tolist(), correct.tolist(), n_bins=10)
    assert calib["ece"] < 0.05
    assert len(calib["bin_centers"]) == 10


def test_calibration_overconfident_high_ece() -> None:
    conf = [0.95] * 200
    correct = [i < 100 for i in range(200)]  # 50 percent right at 95 confident
    calib = calibration_curve(conf, correct, n_bins=10)
    assert calib["ece"] > 0.3


def test_calibration_validates_inputs() -> None:
    with pytest.raises(ValueError):
        calibration_curve([1.5], [True])
    with pytest.raises(ValueError):
        calibration_curve([], [])


def test_overlap_and_breakpoint_curves() -> None:
    records = [
        make_record(overlap_ratio=0.2, mean_si_sdri=18.0),
        make_record(overlap_ratio=1.0, mean_si_sdri=12.0),
        make_record(n_true=5, mean_si_sdri=7.0),
    ]
    oc = overlap_curve(records, system="cascade")
    assert [r["overlap_ratio"] for r in oc] == [0.2, 1.0]
    bc = breakpoint_curve(records, system="cascade")
    assert bc[-1]["n_speakers"] == 5


def test_markdown_table_renders() -> None:
    rows = [{"tier": "L1", "mean": 15.5, "count": 3}]
    md = to_markdown_table(rows)
    assert "| tier |" in md and "15.50" in md
