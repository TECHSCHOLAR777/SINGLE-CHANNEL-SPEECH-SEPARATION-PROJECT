"""Baseline specification and construction tests (BLUEPRINT §9.6)."""

from __future__ import annotations

import numpy as np
import pytest

from eval.baselines import (
    BASELINE_NAMES,
    BASELINE_SPECS,
    build_pipeline,
    describe_baselines,
    get_spec,
    run_baseline_on_mixtures,
)


def test_all_five_mandatory_baselines_are_defined():
    assert set(BASELINE_NAMES) == {
        "frozen_base",
        "frozen_base_plus_band_recovery",
        "universal_adapter",
        "uniform_blend",
        "oracle_gating",
    }
    assert len(BASELINE_SPECS) == 5


def test_frozen_base_uses_neither_adapters_nor_band_recovery():
    spec = get_spec("frozen_base")
    assert not spec.use_adapters
    assert not spec.use_gate
    assert not spec.band_recovery


def test_only_oracle_gating_predicts_gates():
    gated = [s.name for s in BASELINE_SPECS if s.use_gate]
    assert gated == ["oracle_gating"]


def test_get_spec_rejects_an_unknown_baseline():
    with pytest.raises(ValueError, match="unknown baseline"):
        get_spec("not_a_baseline")


def test_describe_reports_structure_without_running_a_model():
    described = describe_baselines()
    assert set(described) == set(BASELINE_NAMES)
    assert all(r["status"] == "structural_only" for r in described.values())
    assert all(r["note"] for r in described.values())


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_every_baseline_builds_a_pipeline(name, mock_expert):
    pipeline = build_pipeline(name, mock_expert)
    spec = get_spec(name)
    assert pipeline.cfg.run_band_recovery is spec.band_recovery
    assert pipeline.lora is None  # no library supplied in this test
    assert pipeline.gate is None


def test_frozen_base_runs_end_to_end_on_a_mock_expert(mock_expert, two_tone_mixture):
    mix, sr = two_tone_mixture
    stats = run_baseline_on_mixtures("frozen_base", [mix, mix], mock_expert, sample_rate=sr)
    assert stats["baseline"] == "frozen_base"
    assert stats["n_mixtures"] == 2
    assert stats["status"] == "ran"
    assert len(stats["counts"]) == 2
    assert np.isfinite(stats["mean_count"])
