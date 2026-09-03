"""
Unit tests for data/rir_bank.py (Dev A, P0-A3).

No pyroomacoustics required for most tests, we synthesize minimal RirRecord
fixtures directly. Tests that call generate_rir / build_rir_bank are skipped
when pyroomacoustics is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from coralsep.data.rir_bank import (
    SEVERE_FRACTION,
    SEVERE_T60_S,
    T60_MAX_S,
    T60_MIN_S,
    RirBank,
    RirRecord,
    find_direct_path_peak,
    measure_t60,
    sample_t60,
)

# ---------------------------------------------------------------------------
# measure_t60
# ---------------------------------------------------------------------------


def _synthetic_rir(t60_s: float, sr: int = 8_000, duration_s: float = 1.0) -> np.ndarray:
    """Exponentially decaying noise RIR with known T60."""
    rng = np.random.default_rng(0)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    decay = np.exp(-6.908 * t / t60_s)  # -60 dB at t60_s
    return (rng.standard_normal(n) * decay).astype(np.float32)


def test_measure_t60_approximate():
    rir = _synthetic_rir(t60_s=0.5)
    est = measure_t60(rir)
    assert 0.2 < est < 1.2, f"T60 estimate {est:.3f} s outside plausible range"


def test_measure_t60_shorter_longer():
    short = _synthetic_rir(t60_s=0.3)
    long_ = _synthetic_rir(t60_s=0.8)
    assert measure_t60(short) < measure_t60(long_)


# ---------------------------------------------------------------------------
# find_direct_path_peak
# ---------------------------------------------------------------------------


def test_find_direct_path_peak_impulse():
    rir = np.zeros(100, dtype=np.float32)
    rir[17] = 1.0
    assert find_direct_path_peak(rir) == 17


def test_find_direct_path_peak_negative_impulse():
    rir = np.zeros(50, dtype=np.float32)
    rir[5] = -0.9
    assert find_direct_path_peak(rir) == 5


# ---------------------------------------------------------------------------
# sample_t60
# ---------------------------------------------------------------------------


def test_sample_t60_in_range():
    rng = np.random.default_rng(42)
    for _ in range(200):
        t = sample_t60(rng, allow_severe=True)
        assert T60_MIN_S <= t <= T60_MAX_S


def test_sample_t60_no_severe():
    rng = np.random.default_rng(99)
    for _ in range(200):
        t = sample_t60(rng, allow_severe=False)
        assert t < SEVERE_T60_S, f"severe T60 {t:.3f} drawn when allow_severe=False"


def test_sample_t60_severe_fraction():
    rng = np.random.default_rng(7)
    samples = [
        sample_t60(rng, allow_severe=True, severe_fraction=SEVERE_FRACTION) for _ in range(2000)
    ]
    severe_count = sum(1 for t in samples if t >= SEVERE_T60_S)
    severe_frac = severe_count / len(samples)
    # Allow generous tolerance: binomial std for 2000 trials ≈ 0.006
    assert (
        abs(severe_frac - SEVERE_FRACTION) < 0.05
    ), f"severe fraction {severe_frac:.3f} differs from {SEVERE_FRACTION}"


# ---------------------------------------------------------------------------
# RirRecord serialisation
# ---------------------------------------------------------------------------


def test_rir_record_to_dict_round_trip():
    rec = RirRecord(
        rir_id="test_000",
        path="datasets/rirs/test_000.npy",
        t60_requested_s=0.5,
        t60_achieved_s=0.48,
        room_dim_m=[5.0, 4.0, 3.0],
        source_pos_m=[2.5, 2.0, 1.5],
        mic_pos_m=[1.0, 2.0, 1.5],
        absorption=0.3,
        max_order=17,
        n_peak=32,
        sample_rate=8_000,
    )
    d = rec.to_dict()
    assert d["rir_id"] == "test_000"
    assert d["t60_achieved_s"] == 0.48
    assert d["n_peak"] == 32


# ---------------------------------------------------------------------------
# RirBank (file-based)
# ---------------------------------------------------------------------------


def _write_bank(tmp_path: Path, n: int = 4) -> Path:
    """Write a minimal bank.json with synthetic .npy RIRs."""
    records = []
    for i in range(n):
        rir = _synthetic_rir(t60_s=0.3 + 0.1 * i)
        rir_path = tmp_path / f"rir_{i:04d}.npy"
        np.save(str(rir_path), rir)
        rec = RirRecord(
            rir_id=f"rir_{i:04d}",
            path=str(rir_path),
            t60_requested_s=0.3 + 0.1 * i,
            t60_achieved_s=0.3 + 0.1 * i,
            room_dim_m=[5.0, 4.0, 3.0],
            source_pos_m=[1.0, 1.0, 1.0],
            mic_pos_m=[2.0, 2.0, 1.0],
            absorption=0.25,
            max_order=10,
            n_peak=find_direct_path_peak(rir),
            sample_rate=8_000,
        )
        records.append(rec.to_dict())
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({"records": records}))
    return tmp_path


def test_rir_bank_load_and_sample(tmp_path):
    bank_dir = _write_bank(tmp_path, n=4)
    bank = RirBank(bank_dir / "bank.json")
    rec = bank.sample(t60_s=0.4, tolerance_s=0.2)
    assert rec is not None
    rir = bank.load(rec)
    assert isinstance(rir, np.ndarray)
    assert rir.ndim == 1
    assert rir.dtype == np.float32


def test_rir_bank_load_shape(tmp_path):
    bank_dir = _write_bank(tmp_path, n=4)
    bank = RirBank(bank_dir / "bank.json")
    rec = bank.sample(t60_s=0.5, tolerance_s=0.5)
    rir = bank.load(rec)
    assert rir.shape[0] > 0


def test_rir_bank_sample_out_of_range_raises(tmp_path):
    bank_dir = _write_bank(tmp_path, n=4)
    bank = RirBank(bank_dir / "bank.json")
    with pytest.raises(ValueError):
        bank.sample(t60_s=5.0, tolerance_s=0.05)


# ---------------------------------------------------------------------------
# generate_rir (optional, skipped when pyroomacoustics unavailable)
# ---------------------------------------------------------------------------

pyroomacoustics = pytest.importorskip("pyroomacoustics", reason="pyroomacoustics not installed")


def test_generate_rir_returns_record():
    from coralsep.data.rir_bank import generate_rir

    rng = np.random.default_rng(123)
    rir, meta = generate_rir(t60_s=0.5, rng=rng)
    assert isinstance(rir, np.ndarray)
    assert rir.ndim == 1
    assert meta["sample_rate"] == 8_000
    assert 0 <= meta["n_peak"] < len(rir)


def test_generate_rir_t60_in_spec():
    from coralsep.data.rir_bank import generate_rir

    rng = np.random.default_rng(0)
    _, meta = generate_rir(t60_s=0.6, rng=rng)
    assert 0.2 <= meta["t60_achieved_s"] <= 2.0
