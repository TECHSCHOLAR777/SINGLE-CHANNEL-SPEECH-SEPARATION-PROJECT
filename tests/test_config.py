"""Tests for the shared config loader."""

from pathlib import Path

import pytest

from coralsep.utils.config import cfg_get, load_config


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_layered_deep_merge(tmp_path: Path) -> None:
    base = _write(tmp_path, "base.yaml", "a: 1\nnested: {x: 1, y: 2}\n")
    over = _write(tmp_path, "over.yaml", "nested: {y: 9}\nb: 2\n")
    cfg = load_config(base, over)
    assert cfg["a"] == 1 and cfg["b"] == 2
    assert cfg["nested"] == {"x": 1, "y": 9}


def test_overrides_win_last(tmp_path: Path) -> None:
    base = _write(tmp_path, "base.yaml", "sample_rate: 16000\n")
    cfg = load_config(base, overrides={"sample_rate": 8000})
    assert cfg["sample_rate"] == 8000


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_cfg_get_dot_path_and_default() -> None:
    cfg = {"eval": {"scoring": {"missing_policy": "mixture_fallback"}}}
    assert cfg_get(cfg, "eval.scoring.missing_policy") == "mixture_fallback"
    assert cfg_get(cfg, "eval.scoring.absent", "fallback") == "fallback"
    assert cfg_get(cfg, "totally.absent") is None


def test_repo_configs_load_together() -> None:
    cfg = load_config("configs/default.yaml", "configs/runtime.yaml")
    assert cfg["sample_rate"] == 16000
    assert cfg_get(cfg, "eval.scoring.missing_policy") == "mixture_fallback"
