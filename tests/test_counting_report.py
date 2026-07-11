"""Tests for the P3 counting artifact generator."""

from __future__ import annotations

import csv
import json

import pytest

from eval.counting_report import generate_counting_report
from eval.reporting import RunRecord


def _record(
    run_id: str,
    true: int,
    estimated: int,
    confidence: float = float("nan"),
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        system="cascade",
        condition="clean",
        tier="L1",
        n_true=true,
        n_estimated=estimated,
        mean_si_sdri=10.0,
        count_confidence=confidence,
    )


def test_generate_counting_report_writes_all_artifacts(tmp_path) -> None:
    records = [
        _record("a", 2, 2, 0.9),
        _record("b", 3, 2, 0.7),
        _record("c", 4, 4, 0.8),
        _record("d", 5, 5, 0.95),
    ]
    artifacts = generate_counting_report(records, tmp_path, n_bins=5)

    for path in (
        artifacts.summary_json,
        artifacts.confusion_csv,
        artifacts.calibration_csv,
        artifacts.markdown,
        artifacts.confusion_svg,
        artifacts.calibration_svg,
    ):
        assert path is not None and path.exists() and path.stat().st_size > 0

    summary = json.loads(artifacts.summary_json.read_text())
    assert summary["n_records"] == 4
    assert summary["counting"]["accuracy"] == pytest.approx(0.75)
    assert summary["calibration"]["ece"] >= 0.0

    with artifacts.confusion_csv.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["true\\estimated", "2", "3", "4", "5"]
    assert rows[2][1:] == ["1", "0", "0", "0"]  # true 3 estimated 2
    assert "Exact count accuracy" in artifacts.markdown.read_text()
    assert "<svg" in artifacts.confusion_svg.read_text()


def test_missing_confidences_skip_calibration_artifacts(tmp_path) -> None:
    records = [_record("a", 2, 2), _record("b", 3, 2)]
    artifacts = generate_counting_report(records, tmp_path)
    assert artifacts.calibration_csv is None
    assert artifacts.calibration_svg is None
    summary = json.loads(artifacts.summary_json.read_text())
    assert summary["calibration"] is None
    assert "No finite count-confidence" in artifacts.markdown.read_text()


def test_empty_records_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        generate_counting_report([], tmp_path)
