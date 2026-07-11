"""P0-INT4: shared config -> baseline -> PIT SI-SDRi -> artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import models.baseline_runner as baseline_module
from data.mixer_stub import MixtureSample
from models.baseline_runner import BaselineConfig, run_baseline
from schemas.separation_result import SeparationResult
from utils.config import load_config


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_p0_shared_end_to_end_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = _write(
        tmp_path / "defaults.yaml",
        "sample_rate: 16000\ndevice: cpu\nsubset: test\nmax_samples: 1\n",
    )
    task = _write(
        tmp_path / "baseline.yaml",
        (
            "data_root: ignored-by-test\n"
            f"output_dir: {tmp_path / 'results'}\n"
            "experts: [sepformer]\n"
        ),
    )
    cfg = load_config(defaults, task)
    config = BaselineConfig.from_dict(cfg)

    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    references = np.stack(
        [
            0.25 * np.sin(2 * np.pi * 310 * time),
            0.25 * np.sin(2 * np.pi * 570 * time),
        ]
    ).astype(np.float32)
    mixture = references.sum(axis=0)
    sample = MixtureSample(
        mixture=mixture,
        references=references,
        sample_rate=sample_rate,
        utterance_id="p0-int4-known-answer",
    )

    class FakeExpert:
        def __init__(self, device: str = "cpu") -> None:
            self.device = device

        def separate(self, wave: np.ndarray, sample_rate: int) -> SeparationResult:
            # Deliberately reverse streams: a non-PIT metric would score this incorrectly.
            return SeparationResult(
                streams=references[::-1].copy(),
                sample_rate=sample_rate,
                speaker_count=2,
                mixture=np.asarray(wave, dtype=np.float32),
                expert_used="fake-sepformer",
            )

    monkeypatch.setattr(baseline_module, "discover_librimix_samples", lambda *a, **k: [sample])
    monkeypatch.setattr(baseline_module, "SepFormerExpert", FakeExpert)

    results = run_baseline(config)
    assert set(results) == {"sepformer"}
    assert results["sepformer"].num_samples == 1
    assert results["sepformer"].mean_sisdri_db > 30.0

    json_path = Path(config.output_dir) / "baseline_results.json"
    markdown_path = Path(config.output_dir) / "baseline_results.md"
    assert json_path.exists() and markdown_path.exists()
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["sepformer"]["num_samples"] == 1
    assert persisted["sepformer"]["mean_sisdri_db"] == pytest.approx(
        results["sepformer"].mean_sisdri_db
    )
    assert "Mean SI-SDRi" in markdown_path.read_text(encoding="utf-8")
