"""Tests for eval/stats.py (Dev C)."""

import numpy as np
import pytest

from coralsep.eval.stats import bootstrap_ci, compute_ci_table, format_summary_table


class TestBootstrapCI:
    def test_mean_estimate_accurate(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=5.0, scale=1.0, size=500)
        est, lo, hi = bootstrap_ci(samples, n_boot=500, seed=0)
        assert abs(est - 5.0) < 0.2
        assert lo < est < hi

    def test_ci_contains_true_mean(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(loc=10.0, scale=2.0, size=200)
        _, lo, hi = bootstrap_ci(samples, n_boot=1000, alpha=0.05, seed=1)
        assert lo < 10.0 < hi

    def test_empty_returns_nan(self):
        est, lo, hi = bootstrap_ci(np.array([]))
        assert np.isnan(est) and np.isnan(lo) and np.isnan(hi)

    def test_single_sample(self):
        est, lo, hi = bootstrap_ci(np.array([3.0]), n_boot=100)
        assert est == pytest.approx(3.0)

    def test_ci_order(self):
        samples = np.random.default_rng(5).normal(0, 1, 100)
        est, lo, hi = bootstrap_ci(samples, n_boot=500)
        assert lo <= est <= hi


class TestComputeCiTable:
    def test_output_keys(self):
        data = {
            ("clean", 2): [5.0, 6.0, 7.0, 5.5],
            ("reverb", 3): [3.0, 2.5, 4.0],
        }
        result = compute_ci_table(data, n_boot=200)
        assert ("clean", 2) in result
        assert ("reverb", 3) in result
        for key in result:
            assert "mean" in result[key]
            assert "ci_low" in result[key]
            assert "ci_high" in result[key]

    def test_ci_bounds_order(self):
        data = {("clean", 2): list(np.random.default_rng(0).normal(5, 1, 50))}
        result = compute_ci_table(data, n_boot=300)
        entry = result[("clean", 2)]
        assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"]


class TestFormatSummaryTable:
    def test_contains_condition_names(self):
        summary = {
            "clean": {2: {"si_sdri_mean": 5.1}, 3: {"si_sdri_mean": 4.8}},
            "reverb": {2: {"si_sdri_mean": 3.2}, 3: {"si_sdri_mean": 2.9}},
        }
        table = format_summary_table(summary)
        assert "clean" in table
        assert "reverb" in table
        assert "N=2" in table
        assert "N=3" in table

    def test_missing_cell_shown_as_dash(self):
        summary = {
            "clean": {2: {"si_sdri_mean": 5.0}},
        }
        table = format_summary_table(summary)
        assert " - " in table  # N=3,4,5 are missing

    def test_markdown_table_header(self):
        summary = {"clean": {2: {"si_sdri_mean": 5.0}}}
        table = format_summary_table(summary)
        assert "| Condition |" in table
