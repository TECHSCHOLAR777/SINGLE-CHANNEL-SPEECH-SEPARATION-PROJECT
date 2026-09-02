"""
Tests for data/sparselibrimix.py (SparseLibriMix loader).

Builds a tiny on-disk SparseLibriMix tree with soundfile and verifies discovery,
overlap-ratio tagging, speaker-count filtering, and reference shapes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from coralsep.data.sparselibrimix import (
    SparseSample,
    _parse_config_dir,
    discover_sparse_samples,
)

SR = 16_000


def _write_config(
    data_root: Path,
    n_src: int,
    ratio: str,
    uids: list[str],
    freq: int = SR,
    mixture: str = "mix_clean",
) -> None:
    """Write one sparse_{n}_{ratio}/wav{freq} set with the given utterance ids."""
    base = data_root / f"sparse_{n_src}_{ratio}" / f"wav{freq}"
    rng = np.random.default_rng(abs(hash((n_src, ratio))) % (2**32))
    for uid in uids:
        refs = [rng.standard_normal(SR).astype(np.float32) for _ in range(n_src)]
        mix = np.sum(refs, axis=0)
        (base / mixture).mkdir(parents=True, exist_ok=True)
        sf.write(str(base / mixture / f"{uid}.wav"), mix, freq, subtype="FLOAT")
        for i, ref in enumerate(refs, start=1):
            (base / f"s{i}").mkdir(parents=True, exist_ok=True)
            sf.write(str(base / f"s{i}" / f"{uid}.wav"), ref, freq, subtype="FLOAT")


# ── _parse_config_dir ─────────────────────────────────────────────────────────


def test_parse_config_dir_decimal() -> None:
    assert _parse_config_dir("sparse_3_0.2") == (3, 0.2)


def test_parse_config_dir_zero_and_one() -> None:
    assert _parse_config_dir("sparse_2_0") == (2, 0.0)
    assert _parse_config_dir("sparse_3_1") == (3, 1.0)


def test_parse_config_dir_invalid_raises() -> None:
    with pytest.raises(ValueError):
        _parse_config_dir("not_a_config")


# ── discovery ─────────────────────────────────────────────────────────────────


def test_discover_returns_sparse_samples(tmp_path: Path) -> None:
    _write_config(tmp_path, 2, "0.2", ["utt1", "utt2"])
    samples = discover_sparse_samples(tmp_path, n_src=2)
    assert len(samples) == 2
    assert all(isinstance(s, SparseSample) for s in samples)


def test_discover_tags_overlap_ratio(tmp_path: Path) -> None:
    _write_config(tmp_path, 3, "0.4", ["a"])
    _write_config(tmp_path, 3, "0.8", ["b"])
    samples = discover_sparse_samples(tmp_path, n_src=3)
    ratios = sorted(s.overlap_ratio for s in samples)
    assert ratios == [0.4, 0.8]


def test_discover_filters_by_overlap_ratio(tmp_path: Path) -> None:
    _write_config(tmp_path, 3, "0", ["a"])
    _write_config(tmp_path, 3, "1", ["b"])
    only_full = discover_sparse_samples(tmp_path, n_src=3, overlap_ratio=1.0)
    assert len(only_full) == 1
    assert only_full[0].overlap_ratio == 1.0


def test_discover_reference_shape_matches_n_src(tmp_path: Path) -> None:
    _write_config(tmp_path, 3, "0.6", ["x"])
    sample = discover_sparse_samples(tmp_path, n_src=3)[0]
    assert sample.references.shape[0] == 3
    assert sample.n_src == 3
    assert sample.mixture.shape == sample.references[0].shape


def test_discover_only_returns_requested_n_src(tmp_path: Path) -> None:
    _write_config(tmp_path, 2, "0.2", ["two"])
    _write_config(tmp_path, 3, "0.2", ["three"])
    samples = discover_sparse_samples(tmp_path, n_src=2)
    assert all(s.n_src == 2 for s in samples)
    assert len(samples) == 1


def test_discover_mixture_equals_sum_of_references(tmp_path: Path) -> None:
    _write_config(tmp_path, 2, "0.4", ["s"])
    sample = discover_sparse_samples(tmp_path, n_src=2)[0]
    np.testing.assert_allclose(sample.mixture, sample.references.sum(axis=0), rtol=0, atol=1e-4)


def test_discover_max_samples_caps_per_ratio(tmp_path: Path) -> None:
    _write_config(tmp_path, 2, "0.2", ["a", "b", "c"])
    samples = discover_sparse_samples(tmp_path, n_src=2, max_samples=2)
    assert len(samples) == 2


def test_discover_sample_rate(tmp_path: Path) -> None:
    _write_config(tmp_path, 2, "0.2", ["a"])
    sample = discover_sparse_samples(tmp_path, n_src=2)[0]
    assert sample.sample_rate == SR


def test_discover_empty_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No SparseLibriMix"):
        discover_sparse_samples(tmp_path, n_src=2)


def test_discover_skips_incomplete_mixture_missing_stem(tmp_path: Path) -> None:
    """A mixture whose s2 stem is missing is skipped, not crashed on."""
    _write_config(tmp_path, 2, "0.2", ["good"])
    # Add a mixture with no matching stems
    base = tmp_path / "sparse_2_0.2" / "wav16000"
    sf.write(str(base / "mix_clean" / "orphan.wav"), np.zeros(SR, np.float32), SR, subtype="FLOAT")

    samples = discover_sparse_samples(tmp_path, n_src=2)
    uids = {s.utterance_id for s in samples}
    assert "good" in uids
    assert "orphan" not in uids
