"""Tests for Libri3Mix mixer stub."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data.mixer_stub import discover_librimix_samples


def _write_librimix_fixture(root: Path, uid: str = "sample_000") -> None:
    """Create a minimal Libri3Mix directory tree with one utterance."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    s1 = 0.5 * np.sin(2 * np.pi * 200 * t)
    s2 = 0.4 * np.sin(2 * np.pi * 350 * t)
    s3 = 0.3 * np.sin(2 * np.pi * 500 * t)
    mixture = s1 + s2 + s3

    for sub, audio in [
        ("mix_both", mixture),
        ("s1", s1),
        ("s2", s2),
        ("s3", s3),
    ]:
        d = root / "wav16k" / "max" / "test" / sub
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / f"{uid}.wav"), audio, sr)


def test_discover_librimix_samples(tmp_path: Path) -> None:
    _write_librimix_fixture(tmp_path)
    samples = discover_librimix_samples(tmp_path, subset="test")
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_rate == 16000
    assert s.mixture.shape == s.references[0].shape
    assert s.references.shape[0] == 3


def test_discover_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_librimix_samples(tmp_path / "nonexistent")
