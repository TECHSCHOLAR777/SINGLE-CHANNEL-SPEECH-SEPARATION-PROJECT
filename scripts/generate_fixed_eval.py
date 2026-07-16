#!/usr/bin/env python3
"""
CLI entry for generating fixed evaluation manifests (BLUEPRINT §7.4).

Usage::

    python scripts/generate_fixed_eval.py
    python scripts/generate_fixed_eval.py --out-dir data/fixed_eval --seed 42
    python scripts/generate_fixed_eval.py --tier clean_2_3 --n-speakers 2 --n-items 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.synthesis.fixed_eval import (  # noqa: E402
    DEFAULT_EVAL_SEED,
    build_eval_manifest,
    generate_all_manifests,
)
from utils.hashing import hash_file  # noqa: E402
from utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seeded fixed evaluation manifests")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/fixed_eval"),
        help="Output directory for JSONL manifests and hash sidecars",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="Global evaluation seed",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Generate a single tier (default: full matrix)",
    )
    parser.add_argument(
        "--n-speakers",
        type=int,
        default=None,
        help="Speaker count N (required with --tier)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Override item count (required with --tier)",
    )
    args = parser.parse_args()

    log = get_logger("generate_fixed_eval")
    log.bind(seed=args.seed)

    if args.tier is not None:
        if args.n_speakers is None or args.n_items is None:
            parser.error("--tier requires --n-speakers and --n-items")
        path = build_eval_manifest(
            args.tier,
            args.n_speakers,
            args.n_items,
            args.seed,
            args.out_dir,
        )
        digest = hash_file(path)
        log.info("manifest_written", path=str(path), sha256=digest)
        print(f"Wrote {path} ({digest[:12]}...)")
        return

    paths = generate_all_manifests(args.out_dir, seed=args.seed)
    log.info("matrix_generated", n_manifests=len(paths), out_dir=str(args.out_dir))
    print(f"Generated {len(paths)} manifests under {args.out_dir}")


if __name__ == "__main__":
    main()
