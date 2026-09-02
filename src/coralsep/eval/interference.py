"""Cross-interference matrix for LoRA adapters (P1-C1, BLUEPRINT §9.5)."""

from __future__ import annotations

from typing import Any

import numpy as np

ADAPTERS = ("reverb", "noise", "codec")
CONDITIONS = ("clean", "reverb", "noise", "codec")
HARM_THRESHOLD_DB = 0.3


def build_interference_matrix(
    scores: dict[tuple[str, str], float],
) -> np.ndarray:
    """scores[(adapter, condition)] = mean SI-SDRi. Returns (A x C) matrix."""
    mat = np.full((len(ADAPTERS), len(CONDITIONS)), np.nan, dtype=np.float64)
    for i, a in enumerate(ADAPTERS):
        for j, c in enumerate(CONDITIONS):
            if (a, c) in scores:
                mat[i, j] = scores[(a, c)]
    return mat


def off_diagonal_harm(
    matrix: np.ndarray,
    clean_baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Measure off-diagonal harm vs matched diagonal / clean baseline.

    Harm > 0.3 dB triggers O-LoRA escalation (BLUEPRINT §5.3).
    """
    harms: list[dict[str, Any]] = []
    for i, a in enumerate(ADAPTERS):
        diag = matrix[i, CONDITIONS.index(a)] if a in CONDITIONS else np.nan
        for j, c in enumerate(CONDITIONS):
            if a == c or np.isnan(matrix[i, j]):
                continue
            # Harm relative to clean baseline for that adapter if provided.
            base = clean_baseline.get(a, 0.0) if clean_baseline else 0.0
            harm = float(base - matrix[i, j]) if c == "clean" else float(diag - matrix[i, j])
            if harm > HARM_THRESHOLD_DB:
                harms.append({"adapter": a, "condition": c, "harm_db": harm})
    return {
        "threshold_db": HARM_THRESHOLD_DB,
        "violations": harms,
        "needs_olora": len(harms) > 0,
        "matrix": matrix.tolist(),
        "adapters": list(ADAPTERS),
        "conditions": list(CONDITIONS),
    }
