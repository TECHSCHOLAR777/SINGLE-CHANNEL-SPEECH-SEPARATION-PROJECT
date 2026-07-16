"""
Statistical analysis utilities for CALM-Sep evaluation (Dev C, P5-C1).

Provides:
  - Bootstrap confidence intervals (BCa method) for SI-SDRi, SDRi, DNSMOS.
  - Wilcoxon signed-rank test for pairwise system comparison.
  - Formatted summary table for reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap CIs (BCa — bias-corrected and accelerated)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap BCa confidence interval.

    Args:
        samples: (N,) array of observed values.
        statistic: Function mapping a sample array to a scalar.
        n_boot: Number of bootstrap resamples.
        alpha: Coverage = 1 - alpha (default 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (estimate, lower, upper) where [lower, upper] is the (1-alpha) CI.
    """
    rng = np.random.default_rng(seed)
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        v = float(statistic(samples))
        return v, v, v

    estimate = float(statistic(samples))

    boot_stats = np.array([
        statistic(rng.choice(samples, size=n, replace=True))
        for _ in range(n_boot)
    ])

    # Bias-correction factor z0.
    z0 = _norm_ppf(np.mean(boot_stats < estimate))

    # Acceleration factor a: jackknife estimate.
    jack = np.array([statistic(np.delete(samples, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    denom = 6.0 * np.sum((jack_mean - jack) ** 2) ** 1.5
    a = num / (denom + 1e-30)

    def _adjusted_quantile(p: float) -> float:
        z = _norm_ppf(p)
        adj_z = z0 + (z0 + z) / (1.0 - a * (z0 + z))
        return float(_norm_cdf(adj_z))

    lo = float(np.quantile(boot_stats, _adjusted_quantile(alpha / 2)))
    hi = float(np.quantile(boot_stats, _adjusted_quantile(1.0 - alpha / 2)))
    return estimate, lo, hi


def _norm_ppf(p: float) -> float:
    """Standard normal percent-point function (inverse CDF)."""
    from scipy.special import ndtri
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    return float(ndtri(p))


def _norm_cdf(z: float) -> float:
    from scipy.special import ndtr
    return float(ndtr(float(z)))


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Wilcoxon signed-rank test: is system A significantly different from B?

    Args:
        scores_a: (N,) per-utterance metric for system A.
        scores_b: (N,) per-utterance metric for system B.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        {'statistic': float, 'p_value': float, 'mean_diff': float}
    """
    from scipy.stats import wilcoxon

    diff = scores_a - scores_b
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff": 0.0}

    stat, pval = wilcoxon(diff, alternative=alternative)
    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "mean_diff": float(diff.mean()),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def format_summary_table(
    summary: dict[str, dict],
    metric: str = "si_sdri_mean",
    caption: str = "SI-SDRi (dB) by condition and N",
) -> str:
    """
    Format the summary dict from EvalMatrix.summary_by_condition() as a
    Markdown table with rows = conditions, columns = N values.

    Args:
        summary: nested dict from EvalMatrix.summary_by_condition().
        metric: Key within each stats dict to display.
        caption: Table caption.

    Returns:
        Markdown-formatted table string.
    """
    conds = list(summary.keys())
    n_vals = [2, 3, 4, 5]

    header = "| Condition | " + " | ".join(f"N={n}" for n in n_vals) + " |"
    sep = "|" + "|".join(["---"] * (len(n_vals) + 1)) + "|"
    rows = [f"**{caption}**", "", header, sep]

    for cond in conds:
        cells = []
        for n in n_vals:
            val = summary.get(cond, {}).get(n, {}).get(metric)
            if val is None:
                cells.append(" - ")
            else:
                cells.append(f"{val:.2f}")
        rows.append("| " + cond + " | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def compute_ci_table(
    records_by_cond_n: dict[tuple[str, int], list[float]],
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> dict[tuple[str, int], dict]:
    """
    Compute BCa CIs for each (condition, N) bucket.

    Args:
        records_by_cond_n: dict mapping (condition, N) → list of metric values.
        n_boot: Bootstrap resamples.
        alpha: CI coverage = 1 - alpha.

    Returns:
        dict mapping (condition, N) → {'mean', 'ci_low', 'ci_high'}.
    """
    result = {}
    for key, vals in records_by_cond_n.items():
        arr = np.array(vals, dtype=np.float64)
        mean, lo, hi = bootstrap_ci(arr, n_boot=n_boot, alpha=alpha)
        result[key] = {"mean": mean, "ci_low": lo, "ci_high": hi}
    return result
