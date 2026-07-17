"""Mandatory baselines runner scaffold (BLUEPRINT §9.6).

Modes (config off-switches → one-line baselines):
  - frozen_base
  - universal_adapter
  - uniform_blend
  - oracle_gating
  - frozen_base_plus_band_recovery

Scores are written under reports/baselines/. Real numbers require trained weights
and fixed_eval audio; this module defines the contract and a mock dry-run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipeline.infer import CalmSepEngine, MockCalmSepWrapper
from utils.logging import get_logger

log = get_logger("baselines")

BASELINE_NAMES = (
    "frozen_base",
    "universal_adapter",
    "uniform_blend",
    "oracle_gating",
    "frozen_base_plus_band_recovery",
)


def build_baseline_engine(name: str, device: str = "cpu") -> CalmSepEngine:
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name}")
    wrapper = MockCalmSepWrapper(2)
    if name == "frozen_base":
        return CalmSepEngine(
            device=device,
            wrapper=wrapper,
            base_only=True,
            use_band_recovery=False,
        )
    if name == "frozen_base_plus_band_recovery":
        return CalmSepEngine(
            device=device,
            wrapper=wrapper,
            base_only=True,
            use_band_recovery=True,
        )
    if name == "uniform_blend":
        eng = CalmSepEngine(device=device, wrapper=wrapper, use_gate=False, use_adapters=True)
        return eng
    if name in ("universal_adapter", "oracle_gating"):
        # Same engine path; oracle/universal weights loaded by user post-training.
        return CalmSepEngine(device=device, wrapper=wrapper, use_gate=(name == "oracle_gating"))
    raise ValueError(name)


def run_baseline_on_mixtures(
    name: str,
    mixtures: list[np.ndarray],
    sample_rate: int = 8000,
    device: str = "cpu",
) -> dict[str, Any]:
    engine = build_baseline_engine(name, device=device)
    counts = []
    for mix in mixtures:
        result = engine(mix, sample_rate)
        counts.append(result.speaker_count)
    return {
        "baseline": name,
        "n_mixtures": len(mixtures),
        "mean_count": float(np.mean(counts)) if counts else None,
        "status": "mock_dry_run",
        "note": "Replace MockCalmSepWrapper with real checkpoint + adapter weights for headline numbers",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="reports/baselines")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dry-run", action="store_true", default=True)
    args = p.parse_args()

    mixes = [(np.random.randn(8000).astype(np.float32) * 0.05) for _ in range(3)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in BASELINE_NAMES:
        stats = run_baseline_on_mixtures(name, mixes, device=args.device)
        summary[name] = stats
        (out / f"{name}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log.info("baseline_done", name=name)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
