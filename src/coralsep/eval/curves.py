"""Break-point curves and band-recovery contribution helpers (P5-B1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from coralsep.eval.stats import bootstrap_ci


def breakpoint_curve(
    metrics_by_n: dict[int, list[float]],
) -> dict[str, Any]:
    """metrics_by_n: {2: [...], 3: [...], ...} → mean + CI per N."""
    out: dict[str, Any] = {}
    for n, vals in sorted(metrics_by_n.items()):
        ci = bootstrap_ci(vals, n_resamples=1000)
        out[str(n)] = {"mean": ci.mean, "ci_low": ci.low, "ci_high": ci.high, "n": len(vals)}
    return out


def band_recovery_contribution(
    si_sdri_8k: list[float],
    si_sdri_16k: list[float],
    dnsmos_8k: list[float] | None = None,
    dnsmos_16k: list[float] | None = None,
) -> dict[str, Any]:
    """Matched-pair deltas for band recovery (BLUEPRINT §9.5 analysis 7)."""
    d_sisdr = np.asarray(si_sdri_16k, dtype=np.float64) - np.asarray(si_sdri_8k, dtype=np.float64)
    result: dict[str, Any] = {
        "delta_si_sdri_mean": float(d_sisdr.mean()) if d_sisdr.size else None,
        "delta_si_sdri_ci": None,
    }
    if d_sisdr.size:
        ci = bootstrap_ci(d_sisdr, n_resamples=1000)
        result["delta_si_sdri_ci"] = {"low": ci.low, "high": ci.high}
    if dnsmos_8k is not None and dnsmos_16k is not None:
        d_mos = np.asarray(dnsmos_16k, dtype=np.float64) - np.asarray(dnsmos_8k, dtype=np.float64)
        result["delta_dnsmos_mean"] = float(d_mos.mean()) if d_mos.size else None
    return result


def write_curves_report(payload: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
