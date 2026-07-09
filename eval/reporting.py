"""
Run logging and report generation (Dev C: Phase 0 reporting, Phase 2 escalation
instrumentation, Phase 3 counting reports, Phase 5 curves).

Every evaluation run writes one RunRecord row to a JSONL log. Every table and
figure in the final report is then a query over those rows, never a hand-built
artifact. This module owns the queries: per-tier aggregation, escalation rate,
count confusion, calibration curve with expected calibration error, the
sparse-overlap curve, and markdown rendering for the report.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import count_accuracy, count_confusion_matrix


@dataclass
class RunRecord:
    """One evaluation run of one mixture through one system configuration.

    Attributes:
        run_id: Unique identifier for this row.
        system: System label, e.g. "sepformer", "cascade", "cascade+fusion".
        condition: Acoustic condition slice, e.g. "clean", "noisy", "reverb",
            "sparse", "codec", "real_room".
        tier: Evaluation tier label from MASTER_PROJECT section 1.4 (L1..L4).
        n_true: True speaker count.
        n_estimated: System-estimated speaker count.
        mean_si_sdri: Mean SI-SDRi (dB) over reference streams.
        mean_si_sdr: Mean SI-SDR (dB) over reference streams.
        escalated: Whether the expensive expert ran (cascade path).
        count_confidence: Calibrated confidence of the count decision, [0, 1].
        latency_sec: Wall-clock inference time for this mixture.
        overlap_ratio: Overlap proportion of the mixture, [0, 1], if known.
        extra: Extension slot (per-stream lists, peel positions, etc.).
        timestamp: Unix time the row was written.
    """

    run_id: str
    system: str
    condition: str
    tier: str
    n_true: int
    n_estimated: int
    mean_si_sdri: float
    mean_si_sdr: float = float("nan")
    escalated: bool = False
    count_confidence: float = float("nan")
    latency_sec: float = float("nan")
    overlap_ratio: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class RunLog:
    """Append-only JSONL log of RunRecords, the single source of report data."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RunRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json_line() + "\n")

    def load(self) -> list[RunRecord]:
        if not self.path.exists():
            return []
        records: list[RunRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(RunRecord.from_dict(json.loads(line)))
        return records


def aggregate_by(
    records: Sequence[RunRecord],
    keys: Sequence[str],
    value: str = "mean_si_sdri",
) -> list[dict[str, Any]]:
    """
    Group records by the given attribute names and aggregate one numeric field.

    Args:
        records: Rows to aggregate.
        keys: RunRecord attribute names to group by, e.g. ("system", "tier").
        value: Numeric attribute to aggregate.

    Returns:
        One dict per group with the key fields plus mean/std/count of value,
        sorted by the key tuple for stable report ordering.
    """
    groups: dict[tuple, list[float]] = {}
    for r in records:
        k = tuple(getattr(r, key) for key in keys)
        v = float(getattr(r, value))
        if not np.isnan(v):
            groups.setdefault(k, []).append(v)
    rows: list[dict[str, Any]] = []
    for k in sorted(groups, key=str):
        vals = np.asarray(groups[k], dtype=np.float64)
        row = dict(zip(keys, k, strict=True))
        row.update(
            {
                f"{value}_mean": float(vals.mean()),
                f"{value}_std": float(vals.std(ddof=0)),
                "count": int(vals.size),
            }
        )
        rows.append(row)
    return rows


def escalation_rate(records: Sequence[RunRecord], by: Sequence[str] = ("tier",)) -> list[dict]:
    """
    Fraction of runs that escalated to the expensive expert, per group.

    The cascade's headline instrumentation (MASTER_PROJECT section 4.3): the
    expected-cost claim is only checkable if this number is measured per tier.
    """
    groups: dict[tuple, list[bool]] = {}
    for r in records:
        k = tuple(getattr(r, key) for key in by)
        groups.setdefault(k, []).append(bool(r.escalated))
    rows = []
    for k in sorted(groups, key=str):
        flags = groups[k]
        row = dict(zip(by, k, strict=True))
        row.update({"escalation_rate": float(np.mean(flags)), "count": len(flags)})
        rows.append(row)
    return rows


def counting_report(
    records: Sequence[RunRecord],
    count_range: tuple[int, int],
) -> dict[str, Any]:
    """
    Count accuracy plus confusion matrix from run records.

    Returns:
        Dict with "accuracy", "confusion" (list of lists, rows true N), and
        "labels" (the N values each row/column represents).
    """
    true = [r.n_true for r in records]
    est = [r.n_estimated for r in records]
    mat = count_confusion_matrix(true, est, count_range)
    lo, hi = count_range
    return {
        "accuracy": count_accuracy(true, est),
        "confusion": mat.tolist(),
        "labels": list(range(lo, hi + 1)),
    }


def calibration_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Reliability diagram data plus Expected Calibration Error.

    Args:
        confidences: Predicted confidence per decision, in [0, 1].
        correct: Whether each decision was actually correct.
        n_bins: Equal-width bins over [0, 1].

    Returns:
        Dict with bin_centers, bin_confidence (mean predicted), bin_accuracy
        (empirical), bin_counts, and ece. Empty bins carry NaN accuracy and
        zero count rather than being dropped, so plots stay aligned.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=bool)
    if conf.shape != corr.shape:
        raise ValueError("confidences and correct must have equal length")
    if conf.size == 0:
        raise ValueError("empty calibration inputs")
    if np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidences must lie in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)

    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            bin_conf[b] = float(conf[mask].mean())
            bin_acc[b] = float(corr[mask].mean())

    nonempty = counts > 0
    ece = float(
        np.sum(np.abs(bin_acc[nonempty] - bin_conf[nonempty]) * counts[nonempty]) / conf.size
    )
    return {
        "bin_centers": centers.tolist(),
        "bin_confidence": bin_conf.tolist(),
        "bin_accuracy": bin_acc.tolist(),
        "bin_counts": counts.tolist(),
        "ece": ece,
    }


def overlap_curve(records: Sequence[RunRecord], system: str) -> list[dict[str, Any]]:
    """SI-SDRi versus overlap ratio for one system: the SparseLibriMix figure data."""
    filtered = [r for r in records if r.system == system and not np.isnan(r.overlap_ratio)]
    groups: dict[float, list[float]] = {}
    for r in filtered:
        groups.setdefault(round(float(r.overlap_ratio), 2), []).append(r.mean_si_sdri)
    return [
        {
            "overlap_ratio": k,
            "mean_si_sdri": float(np.mean(v)),
            "std": float(np.std(v)),
            "count": len(v),
        }
        for k, v in sorted(groups.items())
    ]


def breakpoint_curve(records: Sequence[RunRecord], system: str) -> list[dict[str, Any]]:
    """SI-SDRi versus true speaker count for one system: the break-point figure data."""
    filtered = [r for r in records if r.system == system]
    groups: dict[int, list[float]] = {}
    for r in filtered:
        groups.setdefault(int(r.n_true), []).append(r.mean_si_sdri)
    return [
        {
            "n_speakers": k,
            "mean_si_sdri": float(np.mean(v)),
            "std": float(np.std(v)),
            "count": len(v),
        }
        for k, v in sorted(groups.items())
    ]


def to_markdown_table(rows: Sequence[dict[str, Any]], float_fmt: str = "{:.2f}") -> str:
    """Render aggregation rows as a GitHub-flavored markdown table for the report."""
    if not rows:
        return "(no data)"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        cells = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                cells.append("nan" if np.isnan(v) else float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run fn and return (result, elapsed_seconds); the latency hook for RunRecord."""
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream raw dicts from a JSONL file (feature logs, external run logs)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
