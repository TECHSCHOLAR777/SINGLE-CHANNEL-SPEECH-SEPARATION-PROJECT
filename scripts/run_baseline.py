#!/usr/bin/env python3
"""Score the frozen backbone alone on a LibriMix split.

This is the corpus-transfer baseline: the number every adapter configuration
has to beat. It runs the frozen backbone with no adapters, no gate and no band
recovery, and reports PIT SI-SDRi against the clean stems.

    python scripts/run_baseline.py --data-root <librimix>/Libri2Mix --max-samples 30

Defaults match the recorded evaluations: 8 kHz, `min` mode, `test` subset. Those
are the settings behind every number in docs/restoration/RESULTS.md.

Relationship to `eval/run_eval.py`: that script scores the baseline and the full
system side by side and is what produced the published comparisons. This one
scores the baseline alone, which is useful when checking a fresh environment or
a new split before committing to a full run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is importable when invoked as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from coralsep.data.mixer_stub import discover_librimix_samples  # noqa: E402
from coralsep.models.baseline_runner import (  # noqa: E402
    run_corpus_transfer_baseline,
    write_baseline_log,
)
from coralsep.utils.logging import get_logger  # noqa: E402

log = get_logger("run_baseline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-root",
        required=True,
        help="Root of one LibriMix split, for example <librimix>/Libri2Mix",
    )
    p.add_argument("--subset", default="test", choices=("train", "dev", "test"))
    p.add_argument("--sample-rate", type=int, default=8000, choices=(8000, 16000))
    p.add_argument("--mode", default="min", choices=("min", "max"))
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of mixtures. The recorded runs used 30.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="Write the result JSON here")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    samples = discover_librimix_samples(
        args.data_root,
        subset=args.subset,
        max_samples=args.max_samples,
        sample_rate=args.sample_rate,
        mode=args.mode,
    )
    if not samples:
        raise SystemExit(
            f"no mixtures found under {args.data_root} "
            f"(subset={args.subset}, sample_rate={args.sample_rate}, mode={args.mode})"
        )
    log.info("samples_discovered", n=len(samples), data_root=args.data_root)

    stats = run_corpus_transfer_baseline(
        [s.mixture for s in samples],
        [s.references for s in samples],
        device=args.device,
    )
    stats.update(
        {
            "data_root": str(args.data_root),
            "subset": args.subset,
            "mode": args.mode,
            "requested_sample_rate": args.sample_rate,
        }
    )

    if args.out:
        write_baseline_log(stats, args.out)
        log.info("baseline_written", path=args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
