"""Frozen-backbone baseline CLI and the LibriMix layout it discovers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from coralsep.data.mixer_stub import discover_librimix_samples
from scripts.run_baseline import parse_args


def _wav(path: Path, seconds: float = 0.1, sr: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(sr * seconds), dtype=np.float32), sr)


def _librimix(root: Path, sample_rate: int, mode: str, n_speakers: int, n_files: int) -> Path:
    subset = root / f"wav{sample_rate // 1000}k" / mode / "test"
    for i in range(n_files):
        uid = f"utt{i}"
        _wav(subset / "mix_both" / f"{uid}.wav", sr=sample_rate)
        for spk in range(1, n_speakers + 1):
            _wav(subset / f"s{spk}" / f"{uid}.wav", sr=sample_rate)
    return root


def test_cli_defaults_match_the_recorded_evaluation_settings():
    """Every published number used 8 kHz, min mode, test subset."""
    args = parse_args(["--data-root", "x"])
    assert args.sample_rate == 8000
    assert args.mode == "min"
    assert args.subset == "test"


def test_cli_requires_a_data_root():
    with pytest.raises(SystemExit):
        parse_args([])


def test_discovers_the_8k_min_tree_the_project_actually_uses(tmp_path):
    root = _librimix(tmp_path / "Libri3Mix", 8000, "min", n_speakers=3, n_files=4)
    samples = discover_librimix_samples(root, sample_rate=8000, mode="min")
    assert len(samples) == 4
    assert samples[0].references.shape[0] == 3
    assert samples[0].sample_rate == 8000


def test_discovers_the_16k_max_tree_the_phase_0_callers_used(tmp_path):
    root = _librimix(tmp_path / "Libri2Mix", 16000, "max", n_speakers=2, n_files=2)
    samples = discover_librimix_samples(root, sample_rate=16000, mode="max")
    assert len(samples) == 2
    assert samples[0].references.shape[0] == 2


def test_max_samples_caps_discovery(tmp_path):
    root = _librimix(tmp_path / "Libri2Mix", 8000, "min", n_speakers=2, n_files=10)
    assert len(discover_librimix_samples(root, sample_rate=8000, mode="min", max_samples=3)) == 3


def test_a_wrong_layout_names_the_path_it_looked_for(tmp_path):
    _librimix(tmp_path / "Libri2Mix", 8000, "min", n_speakers=2, n_files=1)
    with pytest.raises(FileNotFoundError, match="wav16k"):
        discover_librimix_samples(tmp_path / "Libri2Mix", sample_rate=16000, mode="max")


@pytest.mark.parametrize("bad", [22050, 44100, 0])
def test_unsupported_sample_rates_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="sample_rate"):
        discover_librimix_samples(tmp_path, sample_rate=bad)


def test_unsupported_modes_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        discover_librimix_samples(tmp_path, mode="medium")


def test_baseline_config_has_no_unresolved_placeholder():
    """The config used to ship a TODO and a v1 16 kHz sample rate."""
    cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))
    assert cfg["sample_rate"] == 8000
    assert cfg["mode"] == "min"
    assert cfg["data_root"] is None, "data_root must be explicit, not a guessed path"
    assert "TODO" not in Path("configs/baseline.yaml").read_text(encoding="utf-8")
