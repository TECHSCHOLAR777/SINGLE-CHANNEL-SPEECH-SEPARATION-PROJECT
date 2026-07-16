"""
Statistical helpers for CALM-Sep evaluation (BLUEPRINT §9.1).

Bootstrap confidence intervals, Wilcoxon signed-rank tests, and calibration /
reliability (ECE) utilities. All functions are numpy/scipy only for CPU smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    alpha: float = 0.05
    n_resamples: int = 10_000


def bootstrap_ci(
    values: np.ndarray | list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ConfidenceInterval:
    """
    Utterance-level bootstrap 95% CI for the mean (BLUEPRINT §9.1).

    Args:
        values: Per-utterance metric values.
        n_resamples: Number of bootstrap resamples (default 10,000).
        alpha: Significance level; default 0.05 → 95% CI.
        seed: RNG seed for reproducibility.

    Returns:
        ConfidenceInterval with mean and [low, high] bounds.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("values must be non-empty")

    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = sample.mean()

    low = float(np.percentile(means, 100 * alpha / 2))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return ConfidenceInterval(
        mean=float(arr.mean()),
        low=low,
        high=high,
        alpha=alpha,
        n_resamples=n_resamples,
    )


def mean_ci(values: np.ndarray | list[float], **kwargs) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) shorthand."""
    ci = bootstrap_ci(values, **kwargs)
    return ci.mean, ci.low, ci.high


@dataclass
class WilcoxonResult:
    """Wilcoxon signed-rank test result."""

    statistic: float
    p_value: float
    n_pairs: int
    significant: bool


def wilcoxon_signed_rank(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    alpha: float = 0.05,
) -> WilcoxonResult:
    """
    Paired Wilcoxon signed-rank test on per-utterance deltas (a - b).

    BLUEPRINT §9.1: claim a difference only when p < 0.05 and the bootstrap
    interval excludes zero (apply both checks at the call site).

    Args:
        a: Metric values for system A.
        b: Metric values for system B (same length as a).
        alpha: Significance threshold.

    Returns:
        WilcoxonResult with statistic, p-value, and significance flag.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"a and b must have the same shape, got {x.shape} vs {y.shape}")
    if x.size == 0:
        raise ValueError("empty input arrays")

    diffs = x - y
    if np.allclose(diffs, 0.0):
        return WilcoxonResult(statistic=0.0, p_value=1.0, n_pairs=x.size, significant=False)

    stat, p = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
    return WilcoxonResult(
        statistic=float(stat),
        p_value=float(p),
        n_pairs=x.size,
        significant=float(p) < alpha,
    )


@dataclass
class ReliabilityBin:
    """One bin in a reliability diagram."""

    bin_low: float
    bin_high: float
    mean_confidence: float
    accuracy: float
    count: int


@dataclass
class CalibrationReport:
    """Expected calibration error and per-bin reliability data."""

    ece: float
    n_bins: int
    bins: list[ReliabilityBin]


def expected_calibration_error(
    confidences: np.ndarray | list[float],
    accuracies: np.ndarray | list[float] | list[bool],
    n_bins: int = 10,
) -> CalibrationReport:
    """
    Compute ECE and reliability bins (BLUEPRINT §2.9, §9.5).

    Args:
        confidences: Predicted probabilities in [0, 1].
        accuracies: Binary correctness labels (0/1 or bool) per item.
        n_bins: Number of equal-width confidence bins.

    Returns:
        CalibrationReport with ECE and per-bin statistics.
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    acc = np.asarray(accuracies, dtype=np.float64).ravel()
    if conf.shape != acc.shape:
        raise ValueError("confidences and accuracies must have the same length")
    if conf.size == 0:
        raise ValueError("empty inputs")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    n = conf.size

    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)

        count = int(mask.sum())
        if count == 0:
            bins.append(ReliabilityBin(lo, hi, mean_confidence=float("nan"), accuracy=float("nan"), count=0))
            continue

        mean_conf = float(conf[mask].mean())
        bin_acc = float(acc[mask].mean())
        weight = count / n
        ece += weight * abs(bin_acc - mean_conf)
        bins.append(
            ReliabilityBin(
                bin_low=lo,
                bin_high=hi,
                mean_confidence=mean_conf,
                accuracy=bin_acc,
                count=count,
            )
        )

    return CalibrationReport(ece=float(ece), n_bins=n_bins, bins=bins)


def reliability_curve_data(report: CalibrationReport) -> dict[str, list[float]]:
    """Extract plotting-friendly reliability curve arrays."""
    valid = [b for b in report.bins if b.count > 0]
    return {
        "mean_confidence": [b.mean_confidence for b in valid],
        "accuracy": [b.accuracy for b in valid],
        "count": [float(b.count) for b in valid],
    }
