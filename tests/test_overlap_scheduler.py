"""
Unit tests for data/overlap_scheduler.py (P0-A6).

Pure numpy logic: no I/O, no external deps.
"""

from __future__ import annotations

import numpy as np
import pytest

from coralsep.data.overlap_scheduler import OverlapScheduler, apply_overlap

# ---------------------------------------------------------------------------
# OverlapScheduler, validation
# ---------------------------------------------------------------------------


def test_empty_phases_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        OverlapScheduler(phases=[])


def test_first_phase_must_be_zero() -> None:
    with pytest.raises(ValueError, match="first phase progress must be 0.0"):
        OverlapScheduler(phases=[(0.1, 1.0), (0.5, 0.4)])


def test_non_monotonic_phases_raise() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        OverlapScheduler(phases=[(0.0, 1.0), (0.5, 0.4), (0.3, 0.2)])


def test_ratio_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="overlap ratio"):
        OverlapScheduler(phases=[(0.0, 1.5)])


# ---------------------------------------------------------------------------
# OverlapScheduler, step schedule (default)
# ---------------------------------------------------------------------------


def test_default_schedule_stages() -> None:
    sched = OverlapScheduler()  # 100% -> 40% -> 20% at 0.0 / 0.34 / 0.67
    assert sched.ratio_at(0.0) == 1.0
    assert sched.ratio_at(0.2) == 1.0  # still in first stage
    assert sched.ratio_at(0.34) == 0.4  # boundary enters second stage
    assert sched.ratio_at(0.5) == 0.4
    assert sched.ratio_at(0.67) == 0.2
    assert sched.ratio_at(1.0) == 0.2


def test_progress_is_clamped() -> None:
    sched = OverlapScheduler()
    assert sched.ratio_at(-5.0) == 1.0
    assert sched.ratio_at(9.0) == 0.2


def test_ratio_at_step() -> None:
    sched = OverlapScheduler()
    assert sched.ratio_at_step(0, 100) == 1.0
    assert sched.ratio_at_step(50, 100) == 0.4
    assert sched.ratio_at_step(100, 100) == 0.2


def test_ratio_at_step_bad_total_raises() -> None:
    with pytest.raises(ValueError, match="total_steps must be positive"):
        OverlapScheduler().ratio_at_step(1, 0)


# ---------------------------------------------------------------------------
# OverlapScheduler, interpolation
# ---------------------------------------------------------------------------


def test_interpolation_between_breakpoints() -> None:
    sched = OverlapScheduler(phases=[(0.0, 1.0), (1.0, 0.0)], interpolate=True)
    assert sched.ratio_at(0.0) == pytest.approx(1.0)
    assert sched.ratio_at(0.5) == pytest.approx(0.5)
    assert sched.ratio_at(0.25) == pytest.approx(0.75)
    assert sched.ratio_at(1.0) == pytest.approx(0.0)


def test_interpolation_past_last_breakpoint_holds() -> None:
    sched = OverlapScheduler(phases=[(0.0, 1.0), (0.5, 0.2)], interpolate=True)
    assert sched.ratio_at(0.8) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# apply_overlap, validation
# ---------------------------------------------------------------------------


def test_apply_overlap_requires_2d() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        apply_overlap(np.zeros(10), 0.5)


def test_apply_overlap_bad_ratio_raises() -> None:
    with pytest.raises(ValueError, match="overlap_ratio must be in"):
        apply_overlap(np.zeros((2, 10)), 1.5)


# ---------------------------------------------------------------------------
# apply_overlap, behaviour
# ---------------------------------------------------------------------------


def _stems(n: int = 2, length: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Non-zero everywhere so trimming keeps full length.
    return (rng.standard_normal((n, length)) + 5.0).astype(np.float32)


def test_full_overlap_matches_direct_sum() -> None:
    refs = _stems(3, 100)
    mixture, out = apply_overlap(refs, 1.0, shuffle=False)
    assert out.shape == (3, 100)
    np.testing.assert_allclose(mixture, refs.sum(axis=0), atol=1e-5)


def test_zero_overlap_is_sequential() -> None:
    refs = _stems(3, 100)
    mixture, out = apply_overlap(refs, 0.0, shuffle=False)
    # End to end: total length == sum of stem lengths.
    assert out.shape[1] == 300
    # At any sample at most one stem is active (no temporal overlap).
    active = (np.abs(out) > 1e-6).sum(axis=0)
    assert active.max() <= 1


def test_partial_overlap_length_between_extremes() -> None:
    refs = _stems(2, 100)
    _, out = apply_overlap(refs, 0.5, shuffle=False)
    # Second stem starts at (1 - 0.5) * 100 = 50 -> total length 150.
    assert out.shape[1] == 150


def test_references_preserve_order_and_energy() -> None:
    refs = _stems(3, 80, seed=3)
    _, out = apply_overlap(refs, 0.4, shuffle=False)
    # Each output stem contains its source stem's samples somewhere (energy kept).
    for i in range(3):
        assert np.abs(out[i]).sum() == pytest.approx(np.abs(refs[i]).sum(), rel=1e-5)


def test_reproducible_with_seeded_rng() -> None:
    refs = _stems(4, 60, seed=7)
    m_a, o_a = apply_overlap(refs, 0.5, rng=np.random.default_rng(1))
    m_b, o_b = apply_overlap(refs, 0.5, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(m_a, m_b)
    np.testing.assert_array_equal(o_a, o_b)


def test_output_dtype_float32() -> None:
    mixture, out = apply_overlap(_stems(2, 50), 0.5)
    assert mixture.dtype == np.float32
    assert out.dtype == np.float32
