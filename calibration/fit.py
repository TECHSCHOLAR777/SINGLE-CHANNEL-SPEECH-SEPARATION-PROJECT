"""Fit and hash calibration artifacts on held-out data (BLUEPRINT §8.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from calibration.logistic import LogisticCalibrator
from calibration.temperature import TemperatureScaler
from utils.hashing import hash_file


def fit_all(
    *,
    count_logits: np.ndarray,
    count_labels: np.ndarray,
    conf_features: np.ndarray,
    conf_labels: np.ndarray,
    comp_features: np.ndarray,
    comp_labels: np.ndarray,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Fit temperature + logistic models; write hashed JSON artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    temp = TemperatureScaler()
    temp.fit(count_logits, count_labels)
    temp_path = out / "count_temperature.json"
    temp.save(temp_path)

    conf = LogisticCalibrator()
    conf.fit(conf_features, conf_labels)
    conf_path = out / "stream_confidence.json"
    conf.save(conf_path)

    comp = LogisticCalibrator()
    comp.fit(comp_features, comp_labels)
    comp_path = out / "completeness.json"
    comp.save(comp_path)

    manifest = {
        "count_temperature": {
            "path": temp_path.name,
            "sha256": hash_file(temp_path),
            "temperature": temp.temperature,
        },
        "stream_confidence": {
            "path": conf_path.name,
            "sha256": hash_file(conf_path),
        },
        "completeness": {
            "path": comp_path.name,
            "sha256": hash_file(comp_path),
        },
    }
    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_sha256"] = hash_file(man_path)
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
