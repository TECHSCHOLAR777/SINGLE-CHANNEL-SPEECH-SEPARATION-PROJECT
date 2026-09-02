"""P2-C1: per-layer gates vs per-adapter scalars ablation scaffold.

Run after Stage-3 training. Writes a JSON verdict under reports/.
If mean SI-SDRi gain of per-layer over per-adapter is < 0.1 dB, simpler wins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coralsep.eval.stats import bootstrap_ci
from coralsep.utils.logging import get_logger

log = get_logger("ablation_gate")


def compare_modes(
    si_sdri_per_layer: list[float],
    si_sdri_per_adapter: list[float],
    margin_db: float = 0.1,
) -> dict:
    a = np.asarray(si_sdri_per_layer, dtype=np.float64)
    b = np.asarray(si_sdri_per_adapter, dtype=np.float64)
    delta = a - b
    mean_delta = float(delta.mean()) if delta.size else 0.0
    if delta.size:
        ci_obj = bootstrap_ci(delta, n_resamples=1000)
        ci = {"mean": ci_obj.mean, "low": ci_obj.low, "high": ci_obj.high}
    else:
        ci = {"mean": 0.0, "low": 0.0, "high": 0.0}
    winner = "per_layer" if mean_delta >= margin_db else "per_adapter"
    return {
        "mean_delta_db": mean_delta,
        "ci95": ci,
        "margin_db": margin_db,
        "winner": winner,
        "n": int(delta.size),
        "note": "Simpler (per_adapter) wins unless per_layer gains >= margin_db",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-layer-scores", type=str, help="JSON list of SI-SDRi values")
    p.add_argument("--per-adapter-scores", type=str, help="JSON list of SI-SDRi values")
    p.add_argument("--out", type=str, default="reports/gate_ablation.json")
    p.add_argument("--demo", action="store_true", help="Write placeholder verdict with empty scores")
    args = p.parse_args()

    if args.demo or not args.per_layer_scores:
        verdict = {
            "status": "pending_training",
            "winner": None,
            "note": "Populate with real matched-pair scores after Stage-3 training",
        }
    else:
        pl = json.loads(Path(args.per_layer_scores).read_text(encoding="utf-8"))
        pa = json.loads(Path(args.per_adapter_scores).read_text(encoding="utf-8"))
        verdict = compare_modes(pl, pa)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    log.info("ablation_written", path=str(out), winner=verdict.get("winner"))


if __name__ == "__main__":
    main()
