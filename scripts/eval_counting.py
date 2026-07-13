"""
Unknown-N counting eval → confusion matrix + calibration curve (P3 / GATE M3).

Loads a trained stop-classifier, runs peel-off unknown-N inference over a cached
dev/test set (N is NOT given to the counter), writes a canonical RunLog, and
produces the M3 artifacts via `eval/counting_report.py`:
  * count_confusion_matrix.csv / .svg   (rows = true N, cols = estimated N)
  * count_calibration_curve.csv / .svg  (+ ECE)
  * counting_summary.json / counting_report.md

Example::

    python -m scripts.eval_counting \
        --cache-dir /kaggle/working/cache_mixed/dev \
        --checkpoint /kaggle/working/outputs/counting/stop_classifier.pt \
        --output-dir /kaggle/working/outputs/counting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.counting_infer import infer_counts_over_cache
from eval.counting_report import generate_counting_report
from eval.reporting import RunLog, RunRecord
from models.stop_classifier import StopClassifier


def evaluate_counting(args: argparse.Namespace) -> dict:
    clf = StopClassifier.load(args.checkpoint)
    rows = infer_counts_over_cache(clf, args.cache_dir, threshold=args.threshold)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log = RunLog(out / "counting_runlog.jsonl")
    records: list[RunRecord] = []
    for i, r in enumerate(rows):
        rec = RunRecord(
            run_id=f"count_{i:05d}",
            system="cascade-count",
            condition=args.condition,
            tier=args.tier,
            n_true=int(r["n_true"]),
            n_estimated=int(r["n_estimated"]),
            mean_si_sdri=float("nan"),  # counting eval — separation quality not the metric here
            count_confidence=float(r["confidence"]),
        )
        log.append(rec)
        records.append(rec)

    artifacts = generate_counting_report(records, out, count_range=tuple(args.count_range))
    summary = json.loads(Path(artifacts.summary_json).read_text(encoding="utf-8"))

    exact = summary["counting"]["accuracy"]
    ece = summary["calibration"]["ece"] if summary.get("calibration") else None
    verdict = {
        "n_samples": len(records),
        "count_accuracy": exact,
        "ece": ece,
        "confusion_csv": str(artifacts.confusion_csv),
        "confusion_svg": str(artifacts.confusion_svg),
        "calibration_csv": str(artifacts.calibration_csv) if artifacts.calibration_csv else None,
        "calibration_svg": str(artifacts.calibration_svg) if artifacts.calibration_svg else None,
        "summary_json": str(artifacts.summary_json),
    }
    print(json.dumps(verdict, indent=2))
    print("\nConfusion matrix (rows = true N, cols = estimated N):")
    print(json.dumps(summary["counting"]["confusion"], indent=2))
    return verdict


def main() -> None:
    p = argparse.ArgumentParser(description="Unknown-N counting eval → M3 artifacts")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="outputs/counting")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--condition", default="mixed")
    p.add_argument("--tier", default="L1")
    p.add_argument("--count-range", type=int, nargs=2, default=[2, 5])
    evaluate_counting(p.parse_args())


if __name__ == "__main__":
    main()
