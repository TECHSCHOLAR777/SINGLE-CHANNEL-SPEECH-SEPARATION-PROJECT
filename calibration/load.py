"""Load hashed calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calibration.logistic import LogisticCalibrator
from calibration.temperature import TemperatureScaler
from utils.hashing import hash_file


def load_calibrators(artifact_dir: str | Path, verify_hash: bool = True) -> dict[str, Any]:
    root = Path(artifact_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if verify_hash:
        for key in ("count_temperature", "stream_confidence", "completeness"):
            entry = manifest[key]
            path = root / entry["path"]
            digest = hash_file(path)
            if digest != entry["sha256"]:
                raise ValueError(f"hash mismatch for {key}: {digest} != {entry['sha256']}")
    return {
        "temperature": TemperatureScaler.load(root / manifest["count_temperature"]["path"]),
        "confidence": LogisticCalibrator.load(root / manifest["stream_confidence"]["path"]),
        "completeness": LogisticCalibrator.load(root / manifest["completeness"]["path"]),
        "manifest": manifest,
    }
