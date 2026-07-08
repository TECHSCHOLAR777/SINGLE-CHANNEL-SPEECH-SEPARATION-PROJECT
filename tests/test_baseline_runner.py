"""Tests for SI-SDRi computation in baseline runner."""

import numpy as np

from models.baseline_runner import compute_sisdri


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
