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

# Ensure project root is on sys.path when invoked as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.baseline_runner import BaselineConfig, run_baseline  # noqa: E402
from utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 CA-MoSE baseline on Libri3Mix")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline YAML config",
    )
    parser.add_argument(
        "--defaults",
        type=str,
        default="configs/default.yaml",
        help="Optional shared defaults YAML layered under --config",
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

    defaults_path = Path(args.defaults)
    config_paths = [args.config]
    if defaults_path.exists():
        config_paths = [defaults_path, args.config]

    cfg_dict = load_config(*config_paths)
    overrides: dict = {}
    if args.max_samples is not None:
        overrides["max_samples"] = args.max_samples
    if args.data_root is not None:
        overrides["data_root"] = args.data_root
    if args.device is not None:
        overrides["device"] = args.device
    if overrides:
        cfg_dict = load_config(*config_paths, overrides=overrides)

    config = BaselineConfig.from_dict(cfg_dict)
    run_baseline(config)


if __name__ == "__main__":
    main()
