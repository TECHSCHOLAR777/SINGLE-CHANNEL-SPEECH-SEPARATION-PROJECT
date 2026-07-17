"""End-to-end smoke test on a short synthetic fixture (BLUEPRINT §13)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.infer import CalmSepEngine, MockCalmSepWrapper
from schemas.separation_result import SeparationResult


def test_smoke_end_to_end_mock(tmp_path: Path):
    sr = 8000
    duration = 3.0
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    # Two overlapping tones as a crude multi-speaker mixture.
    s1 = 0.3 * np.sin(2 * np.pi * 220 * t)
    s2 = 0.3 * np.sin(2 * np.pi * 440 * t)
    mix = (s1 + s2).astype(np.float32)

    engine = CalmSepEngine(wrapper=MockCalmSepWrapper(n_speakers=2), base_only=True)
    result = engine(mix, sr)

    assert isinstance(result, SeparationResult)
    assert result.speaker_count in {2, 3, 4, 5}
    assert result.streams.ndim == 2
    assert result.streams.shape[0] == result.speaker_count
    assert result.sample_rate == 16000
    assert result.p_k is not None
    assert result.completeness is not None
    assert result.gate_vector is not None
    assert result.condition_estimates is not None

    # Write outputs and validate files.
    out_dir = tmp_path / "smoke_out"
    out_dir.mkdir()
    for i, stream in enumerate(result.streams):
        path = out_dir / f"spk_{i+1}.wav"
        sf.write(path, stream, result.sample_rate)
        assert path.exists() and path.stat().st_size > 0

    report = result.to_report_dict()
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert "speaker_count" in loaded
    assert "completeness" in loaded
    assert "ood_flag" in loaded
    assert "gate_vector" in loaded
