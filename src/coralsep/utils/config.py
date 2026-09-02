"""
Shared configuration loader (Dev C, Phase 0).

Loads YAML config files with layered deep-merging so every module reads
settings the same way: defaults first, task config over it, explicit
overrides last. Nothing in the codebase should hardcode a tunable that
belongs in configs/.

Usage:
    cfg = load_config("configs/default.yaml", "configs/devc.yaml")
    sr = cfg_get(cfg, "sample_rate", 16000)
    policy = cfg_get(cfg, "eval.scoring.missing_policy", "mixture_fallback")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where override wins, merging nested dicts recursively."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    *paths: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load and layer one or more YAML config files.

    Args:
        *paths: Config file paths, lowest precedence first. Missing files
            raise FileNotFoundError immediately rather than silently loading
            partial configuration.
        overrides: Optional final dict merged on top (e.g. CLI arguments).

    Returns:
        A plain dict with all layers deep-merged, later layers winning.
    """
    cfg: dict[str, Any] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            layer = yaml.safe_load(fh) or {}
        if not isinstance(layer, dict):
            raise ValueError(f"config root must be a mapping, got {type(layer)} in {p}")
        cfg = _deep_merge(cfg, layer)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def cfg_get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    Fetch a nested config value via dot-path, e.g. "eval.scoring.missing_policy".

    Args:
        cfg: Config dict from load_config.
        dotted_key: Dot-separated path into nested dicts.
        default: Returned when any segment of the path is missing.

    Returns:
        The value at the path, or default.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
