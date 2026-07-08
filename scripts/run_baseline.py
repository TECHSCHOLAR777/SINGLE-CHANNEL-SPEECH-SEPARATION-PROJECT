#!/usr/bin/env python3
"""
CLI entry point for Phase 0 baseline evaluation.

Usage:
    python scripts/run_baseline.py --config configs/baseline.yaml
    python scripts/run_baseline.py --config configs/baseline.yaml --max-samples 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path when invoked as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.baseline_runner import BaselineConfig, run_baseline


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 CA-MoSE baseline on Libri3Mix")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline YAML config",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max_samples from config",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data_root from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda / cpu)",
    )
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    if args.max_samples is not None:
        cfg_dict["max_samples"] = args.max_samples
    if args.data_root is not None:
        cfg_dict["data_root"] = args.data_root
    if args.device is not None:
        cfg_dict["device"] = args.device

    config = BaselineConfig.from_dict(cfg_dict)
    run_baseline(config)


if __name__ == "__main__":
    main()
