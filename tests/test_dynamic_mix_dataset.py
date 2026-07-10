"""Tests for data/dynamic_mix_dataset.py — DynamicMixDataset + collate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from data.dynamic_mix_dataset import DynamicMixDataset, collate_mixture_samples
from data.mixer_stub import MixtureSample
from data.overlap_scheduler import OverlapScheduler

SR = 16_000


def _make_files(root: Path, n_speakers: int = 5, duration_s: float = 1.0) -> list[Path]:
    """Write n_speakers single-speaker WAV files in LibriSpeech-style layout."""
    files: list[Path] = []
    for spk_idx in range(n_speakers):
        spk_id = f"10{spk_idx}"
        t = np.linspace(0, duration_s, int(SR * duration_s), dtype=np.float32)
        wave = 0.5 * np.sin(2 * np.pi * (200 + spk_idx * 100) * t)
        path = root / f"{spk_id}-001-0001.wav"
        sf.write(str(path), wave, SR)
        files.append(path)
    return files


# ── Construction ──────────────────────────────────────────────────────────────

def test_dataset_length(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ds = DynamicMixDataset(files, n_samples=7, allowed_n=[2])
    assert len(ds) == 7


def test_n_samples_one(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ds = DynamicMixDataset(files, n_samples=1, allowed_n=[2])
    assert len(ds) == 1
    sample = ds[0]
    assert isinstance(sample, MixtureSample)


def test_invalid_n_samples_raises(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    with pytest.raises(ValueError, match="n_samples"):
        DynamicMixDataset(files, n_samples=0, allowed_n=[2])


# ── Item shape and type ───────────────────────────────────────────────────────

def test_item_is_mixture_sample(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ds = DynamicMixDataset(files, n_samples=4, allowed_n=[2])
    sample = ds[0]
    assert isinstance(sample, MixtureSample)
    assert sample.sample_rate == SR
    assert sample.mixture.ndim == 1
    assert sample.references.ndim == 2
    assert sample.utterance_id.startswith("dyn_")


def test_item_n_speakers_matches_allowed_n(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ds = DynamicMixDataset(files, n_samples=4, allowed_n=[3])
    for i in range(len(ds)):
        assert ds[i].references.shape[0] == 3


def test_mixture_equals_sum_of_refs_full_overlap(tmp_path: Path) -> None:
    """Without overlap_scheduler, DynamicMixer uses full overlap → mixture == sum(refs)."""
    files = _make_files(tmp_path)
    ds = DynamicMixDataset(files, n_samples=3, allowed_n=[2], seed=7)
    for i in range(len(ds)):
        s = ds[i]
        np.testing.assert_allclose(s.mixture, s.references.sum(axis=0), atol=1e-5)


# ── Progress / overlap curriculum ─────────────────────────────────────────────

def test_progress_reaches_first_and_last_phase(tmp_path: Path) -> None:
    """idx=0 → progress=0 (full overlap), idx=n-1 → progress=1 (sparser)."""
    files = _make_files(tmp_path, n_speakers=5)
    scheduler = OverlapScheduler(phases=[(0.0, 1.0), (0.5, 0.5), (1.0, 0.1)])
    ds = DynamicMixDataset(files, n_samples=5, allowed_n=[2], overlap_scheduler=scheduler, seed=0)

    first = ds[0]
    last = ds[len(ds) - 1]

    # With ratio=1.0, mixture == sum of refs (full overlap).
    np.testing.assert_allclose(first.mixture, first.references.sum(axis=0), atol=1e-5)
    # With ratio=0.1 (sparse), total length > any single ref.
    assert last.mixture.shape[0] > last.references.shape[1] * 0.9


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_same_seed_gives_same_first_item(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ds_a = DynamicMixDataset(files, n_samples=4, allowed_n=[2], seed=42)
    ds_b = DynamicMixDataset(files, n_samples=4, allowed_n=[2], seed=42)
    np.testing.assert_array_equal(ds_a[0].mixture, ds_b[0].mixture)
    np.testing.assert_array_equal(ds_a[0].references, ds_b[0].references)


def test_different_seeds_give_different_items(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ds_a = DynamicMixDataset(files, n_samples=4, allowed_n=[2], seed=0)
    ds_b = DynamicMixDataset(files, n_samples=4, allowed_n=[2], seed=99)
    # Highly unlikely to be identical with different seeds.
    assert not np.array_equal(ds_a[0].mixture, ds_b[0].mixture)


# ── Collation ─────────────────────────────────────────────────────────────────

def test_collate_uniform_batch(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ds = DynamicMixDataset(files, n_samples=4, allowed_n=[2], seed=1)
    batch = [ds[i] for i in range(4)]
    out = collate_mixture_samples(batch)

    assert set(out.keys()) == {"mixture", "references", "n_speakers", "sample_rate"}
    assert out["mixture"].shape == (4, out["mixture"].shape[1])
    assert out["references"].shape == (4, 2, out["mixture"].shape[1])
    assert out["n_speakers"].tolist() == [2, 2, 2, 2]
    assert (out["sample_rate"] == SR).all()


def test_collate_pads_variable_length(tmp_path: Path) -> None:
    """Manually construct MixtureSamples of different lengths; collate must pad."""
    short = MixtureSample(
        mixture=np.zeros(100, dtype=np.float32),
        references=np.zeros((2, 100), dtype=np.float32),
        sample_rate=SR,
        utterance_id="a",
    )
    long_ = MixtureSample(
        mixture=np.zeros(200, dtype=np.float32),
        references=np.zeros((2, 200), dtype=np.float32),
        sample_rate=SR,
        utterance_id="b",
    )
    out = collate_mixture_samples([short, long_])
    assert out["mixture"].shape == (2, 200)
    assert out["references"].shape == (2, 2, 200)
    # Short item's padding region must be zero.
    assert out["mixture"][0, 100:].sum() == 0.0


def test_collate_variable_n_speakers(tmp_path: Path) -> None:
    """Batch with 2-speaker and 3-speaker items — N dim padded to max."""
    s2 = MixtureSample(
        mixture=np.zeros(100, dtype=np.float32),
        references=np.zeros((2, 100), dtype=np.float32),
        sample_rate=SR,
        utterance_id="a",
    )
    s3 = MixtureSample(
        mixture=np.zeros(100, dtype=np.float32),
        references=np.zeros((3, 100), dtype=np.float32),
        sample_rate=SR,
        utterance_id="b",
    )
    out = collate_mixture_samples([s2, s3])
    assert out["references"].shape == (2, 3, 100)
    assert out["n_speakers"].tolist() == [2, 3]
    # Padded speaker row for s2 must be zero.
    assert out["references"][0, 2, :].sum() == 0.0


# ── DataLoader integration ────────────────────────────────────────────────────

def test_dataloader_round_trip(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ds = DynamicMixDataset(files, n_samples=6, allowed_n=[2], seed=3)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_mixture_samples)
    batches = list(loader)
    assert len(batches) == 2
    for b in batches:
        assert isinstance(b["mixture"], torch.Tensor)
        assert b["mixture"].shape[0] == 3
        assert b["references"].ndim == 3
