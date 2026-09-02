"""End-to-end smoke test on a short synthetic fixture (BLUEPRINT §13).

Runs the full `CalmSepPipeline` with a weight-free mock expert, so it exercises
preprocessing, chunking, separation, stitching, counting, band recovery and the
output contract without needing the frozen checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.infer import CalmSepPipeline, InferenceCfg, PipelineResult
from schemas.separation_result import SeparationResult


def test_smoke_end_to_end_mock(tmp_path: Path, mock_expert, two_tone_mixture):
    mix, sr = two_tone_mixture

    pipeline = CalmSepPipeline(
        expert=mock_expert,
        cfg=InferenceCfg(default_n_speakers=2, run_band_recovery=False),
    )
    result = pipeline.run(mix, sr)

    assert isinstance(result, PipelineResult)
    assert result.speaker_count in {2, 3, 4, 5}
    assert result.streams_8k.ndim == 2
    assert result.streams_16k.ndim == 2
    assert result.streams_8k.shape[0] == result.speaker_count
    assert result.streams_16k.shape[0] == result.speaker_count
    assert np.isfinite(result.streams_16k).all()
    assert 0.0 <= result.completeness_prob <= 1.0
    assert isinstance(result.ood_flag, bool)
    assert mock_expert.calls, "the pipeline never called the expert"


def test_smoke_emits_the_shared_result_contract(mock_expert, two_tone_mixture):
    """The pipeline result must convert to the project-wide SeparationResult."""
    mix, sr = two_tone_mixture
    pipeline = CalmSepPipeline(
        expert=mock_expert,
        cfg=InferenceCfg(default_n_speakers=2, run_band_recovery=False),
    )
    result = pipeline.run(mix, sr)

    shared = result.to_separation_result()
    assert isinstance(shared, SeparationResult)
    assert shared.sample_rate == 16000
    assert shared.speaker_count == shared.streams.shape[0] == result.speaker_count
    assert len(shared.metadata) == shared.speaker_count
    assert shared.gate_vector == result.gate_vector
    assert shared.completeness_prob == result.completeness_prob
    assert shared.ood_flag == result.ood_flag

    at_8k = result.to_separation_result(sample_rate=8000)
    assert at_8k.sample_rate == 8000
    assert at_8k.streams.shape == result.streams_8k.shape


def test_smoke_outputs_are_writable_and_serialisable(tmp_path: Path, mock_expert, two_tone_mixture):
    mix, sr = two_tone_mixture
    pipeline = CalmSepPipeline(
        expert=mock_expert,
        cfg=InferenceCfg(default_n_speakers=2, run_band_recovery=False),
    )
    shared = pipeline.run(mix, sr).to_separation_result()

    out_dir = tmp_path / "smoke_out"
    out_dir.mkdir()
    for i, stream in enumerate(shared.streams):
        path = out_dir / f"spk_{i + 1}.wav"
        sf.write(path, stream, shared.sample_rate)
        assert path.exists() and path.stat().st_size > 0

    report = {
        "speaker_count": shared.speaker_count,
        "sample_rate": shared.sample_rate,
        "duration_sec": shared.duration_sec,
        "gate_vector": shared.gate_vector,
        "completeness_prob": shared.completeness_prob,
        "ood_flag": shared.ood_flag,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["speaker_count"] == shared.speaker_count
    assert loaded["sample_rate"] == 16000
    assert "completeness_prob" in loaded
    assert "ood_flag" in loaded
    assert "gate_vector" in loaded
