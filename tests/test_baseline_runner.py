"""Tests for SI-SDRi computation in baseline runner."""

from pathlib import Path

import numpy as np
import soundfile as sf

from models.baseline_runner import BaselineConfig, _samples_from_dynamic, compute_sisdri


def test_compute_sisdri_perfect_separation() -> None:
    """Identity separation should yield high positive SI-SDRi."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    ref1 = np.sin(2 * np.pi * 300 * t)
    ref2 = np.sin(2 * np.pi * 500 * t)
    refs = np.stack([ref1, ref2], axis=0)
    mixture = ref1 + ref2
    estimates = refs.copy()

    sisdri = compute_sisdri(estimates, refs, mixture)
    assert sisdri > 10.0


def test_compute_sisdri_worse_than_mixture() -> None:
    """Random estimates should not beat the mixture by much."""
    rng = np.random.default_rng(0)
    refs = rng.standard_normal((2, 4000)).astype(np.float32)
    mixture = refs.sum(axis=0)
    bad_est = rng.standard_normal((2, 4000)).astype(np.float32) * 0.01

    sisdri = compute_sisdri(bad_est, refs, mixture)
    assert sisdri < 5.0


# ── Dynamic mixing path ───────────────────────────────────────────────────────

_SR = 16_000


def _make_source_files(root: Path, n: int = 5) -> list[str]:
    """Write n clean single-speaker WAVs in LibriSpeech-style names."""
    paths: list[str] = []
    for i in range(n):
        spk = f"10{i}"
        t = np.linspace(0, 1.0, _SR, dtype=np.float32)
        wave = 0.5 * np.sin(2 * np.pi * (200 + i * 100) * t)
        p = root / f"{spk}-001-0001.wav"
        sf.write(str(p), wave, _SR)
        paths.append(str(p))
    return paths


def test_samples_from_dynamic_returns_correct_count(tmp_path: Path) -> None:
    files = _make_source_files(tmp_path)
    config = BaselineConfig(source_files=files, n_dynamic=6, allowed_n=[2])
    samples = _samples_from_dynamic(config)
    assert len(samples) == 6


def test_samples_from_dynamic_respects_max_samples(tmp_path: Path) -> None:
    files = _make_source_files(tmp_path)
    config = BaselineConfig(source_files=files, n_dynamic=10, max_samples=4, allowed_n=[2])
    samples = _samples_from_dynamic(config)
    assert len(samples) == 4


def test_baseline_config_from_dict_dynamic_fields() -> None:
    cfg = {
        "source_files": ["/a/1.wav", "/a/2.wav"],
        "n_dynamic": 20,
        "allowed_n": [2, 3],
    }
    config = BaselineConfig.from_dict(cfg)
    assert config.source_files == ["/a/1.wav", "/a/2.wav"]
    assert config.n_dynamic == 20
    assert config.allowed_n == [2, 3]


def test_baseline_config_defaults_empty_source_files() -> None:
    config = BaselineConfig.from_dict({"data_root": "/some/path"})
    assert config.source_files == []
    assert config.n_dynamic == 50
    assert config.allowed_n == [2, 3]
