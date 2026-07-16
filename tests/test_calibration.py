"""CPU smoke for calibration fit/load with hash verification."""

from __future__ import annotations

import numpy as np

from calibration.fit import fit_all
from calibration.load import load_calibrators
from calibration.temperature import TemperatureScaler


def test_temperature_fit_and_calibrate(tmp_path):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(100, 4))
    labels = rng.integers(0, 4, size=100)
    scaler = TemperatureScaler()
    t = scaler.fit(logits, labels)
    assert t > 0
    probs = scaler.calibrate_logits(logits)
    assert probs.shape == (100, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_fit_all_and_load(tmp_path):
    rng = np.random.default_rng(1)
    n = 64
    manifest = fit_all(
        count_logits=rng.normal(size=(n, 4)),
        count_labels=rng.integers(0, 4, size=n),
        conf_features=rng.normal(size=(n, 3)),
        conf_labels=(rng.random(n) > 0.5).astype(float),
        comp_features=rng.normal(size=(n, 3)),
        comp_labels=(rng.random(n) > 0.5).astype(float),
        out_dir=tmp_path,
    )
    assert "count_temperature" in manifest
    loaded = load_calibrators(tmp_path, verify_hash=True)
    assert loaded["temperature"].temperature > 0
    assert loaded["confidence"].weights is not None
