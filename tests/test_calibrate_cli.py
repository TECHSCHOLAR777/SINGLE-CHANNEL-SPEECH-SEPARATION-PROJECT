"""Calibration fitting and artifact hashing (BLUEPRINT §8.5)."""

from __future__ import annotations

import numpy as np
import pytest

from train.calibrate import (
    ARTIFACT_FILES,
    REQUIRED_ARRAYS,
    fit_all,
    load_held_out,
    synthetic_held_out,
)
from utils.hashing import verify_manifest


def test_synthetic_bundle_has_every_required_array():
    bundle = synthetic_held_out(n=32)
    assert set(bundle) == set(REQUIRED_ARRAYS)


def test_fit_all_writes_every_artifact_and_a_verifiable_manifest(tmp_path):
    manifest = fit_all(synthetic_held_out(n=64), tmp_path, synthetic=True)

    for name in ARTIFACT_FILES:
        assert (tmp_path / name).exists(), f"{name} was not written"

    assert manifest["n_files"] == len(ARTIFACT_FILES)
    assert manifest["synthetic"] is True
    assert manifest["n_held_out"] == 64
    assert len(manifest["set_hash"]) == 64
    assert verify_manifest(tmp_path / "manifest.json", root=tmp_path) == []


def test_manifest_detects_a_tampered_artifact(tmp_path):
    """The point of hashing: a changed artifact must stop verifying."""
    fit_all(synthetic_held_out(n=32), tmp_path, synthetic=True)
    assert verify_manifest(tmp_path / "manifest.json", root=tmp_path) == []

    (tmp_path / ARTIFACT_FILES[0]).write_bytes(b"tampered")
    assert verify_manifest(tmp_path / "manifest.json", root=tmp_path) != []


def test_fitting_is_deterministic_for_a_fixed_seed(tmp_path):
    a = fit_all(synthetic_held_out(n=32, seed=1), tmp_path / "a", synthetic=True)
    b = fit_all(synthetic_held_out(n=32, seed=1), tmp_path / "b", synthetic=True)
    assert a["count_temperature"] == pytest.approx(b["count_temperature"])


def test_load_held_out_reports_every_missing_array(tmp_path):
    path = tmp_path / "partial.npz"
    np.savez(path, count_logits=np.zeros((4, 4)))
    with pytest.raises(KeyError) as excinfo:
        load_held_out(path)
    message = str(excinfo.value)
    assert "count_labels" in message
    assert "ood_features" in message


def test_load_held_out_reports_a_missing_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_held_out(tmp_path / "absent.npz")
