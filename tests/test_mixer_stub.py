"""Tests for LibriNMix mixer stub (N=2..5)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data.mixer_stub import discover_librimix_samples

SR = 16_000


def _write_librimix_fixture(
    root: Path,
    n_speakers: int = 3,
    uid: str = "sample_000",
    subset: str = "test",
) -> None:
    """Create a minimal LibriNMix directory tree with one utterance."""
    t = np.linspace(0, 1, SR, dtype=np.float32)
    stems = [0.5 * np.sin(2 * np.pi * (200 + i * 100) * t) for i in range(n_speakers)]
    mixture = sum(stems)  # type: ignore[arg-type]

    entries = [("mix_both", mixture)] + [(f"s{i + 1}", s) for i, s in enumerate(stems)]
    for sub, audio in entries:
        d = root / "wav16k" / "max" / subset / sub
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / f"{uid}.wav"), audio, SR)


# ── Regression: original N=3 behaviour ───────────────────────────────────────


def test_discover_librimix_samples_n3(tmp_path: Path) -> None:
    _write_librimix_fixture(tmp_path, n_speakers=3)
    samples = discover_librimix_samples(tmp_path, subset="test")
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_rate == SR
    assert s.mixture.shape == s.references[0].shape
    assert s.references.shape[0] == 3


# ── N=2 (Libri2Mix) ───────────────────────────────────────────────────────────


def test_discover_librimix_samples_n2(tmp_path: Path) -> None:
    _write_librimix_fixture(tmp_path, n_speakers=2)
    samples = discover_librimix_samples(tmp_path, subset="test")
    assert len(samples) == 1
    assert samples[0].references.shape[0] == 2


# ── N=4 (Libri4Mix) ───────────────────────────────────────────────────────────


def test_discover_librimix_samples_n4(tmp_path: Path) -> None:
    _write_librimix_fixture(tmp_path, n_speakers=4)
    samples = discover_librimix_samples(tmp_path, subset="test")
    assert len(samples) == 1
    assert samples[0].references.shape[0] == 4


# ── N=5 (Libri5Mix) ───────────────────────────────────────────────────────────


def test_discover_librimix_samples_n5(tmp_path: Path) -> None:
    _write_librimix_fixture(tmp_path, n_speakers=5)
    samples = discover_librimix_samples(tmp_path, subset="test")
    assert len(samples) == 1
    assert samples[0].references.shape[0] == 5


# ── max_samples cap ───────────────────────────────────────────────────────────


def test_max_samples_cap(tmp_path: Path) -> None:
    for uid in ("aaa_000", "bbb_001", "ccc_002"):
        _write_librimix_fixture(tmp_path, n_speakers=3, uid=uid)
    samples = discover_librimix_samples(tmp_path, subset="test", max_samples=2)
    assert len(samples) == 2


# ── Error cases ───────────────────────────────────────────────────────────────


def test_discover_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_librimix_samples(tmp_path / "nonexistent")


