"""
Overlap curriculum scheduler + overlap-mixing helper for CA-MoSE — Phase 0 (P0-A6).

The dynamic mixer (data/mixer.py) currently sums all stems from t=0, i.e. full
(100%) temporal overlap.  The evaluation tiers and the training curriculum,
however, call for *sparse* overlap — speakers that only partly talk over each
other (L2 ~40-60%, L3 ~20-40%).  This module supplies the two pieces needed to
get there, both standalone so the tested DynamicMixer stays untouched:

  * ``OverlapScheduler`` — maps training progress in [0, 1] to a target overlap
    ratio following a phase schedule (default 100% -> 40% -> 20%).  The training
    loop (P2) calls ``ratio_at(progress)`` each step and feeds the result to...

  * ``apply_overlap`` — re-places clean stems with temporal start offsets so
    their pairwise overlap matches a target ratio, returning (mixture, refs).

Overlap-ratio definition (placeholder for P0)
---------------------------------------------
For a set of equal-treatment stems, ratio r in [0, 1] scales how much each
successive stem overlaps the running mixture:

  * r = 1.0 -> every stem starts at 0 (full overlap; matches the current mixer).
  * r = 0.0 -> stems are laid end to end (no temporal overlap at all).
  * 0 < r < 1 -> stem k starts at ``(1 - r)`` times the current mixture length,
    so a fraction ~r of it overlaps what came before.

This is a deliberately simple, well-defined stub; a perceptually-calibrated
definition can replace ``apply_overlap`` later without touching the scheduler.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

# Default curriculum: full overlap early, sparser as training progresses.
# Each entry is (progress_fraction, overlap_ratio); progress is in [0, 1].
_DEFAULT_PHASES: list[tuple[float, float]] = [(0.0, 1.0), (0.34, 0.4), (0.67, 0.2)]


@dataclass
class OverlapScheduler:
    """
    Maps training progress in [0, 1] to a target overlap ratio.

    Parameters
    ----------
    phases:
        Sorted list of ``(progress_fraction, overlap_ratio)`` breakpoints.
        The first fraction must be 0.0.  Defaults to 100% -> 40% -> 20%.
    interpolate:
        If True, linearly interpolate the ratio between breakpoints.  If False
        (default), hold each phase's ratio until the next breakpoint (a step
        schedule), which matches the "curriculum stage" reading of the plan.
    """

    phases: list[tuple[float, float]] = field(default_factory=lambda: list(_DEFAULT_PHASES))
    interpolate: bool = False

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("phases must contain at least one (progress, ratio) entry.")
        fracs = [p for p, _ in self.phases]
        if fracs[0] != 0.0:
            raise ValueError(f"first phase progress must be 0.0, got {fracs[0]}.")
        if fracs != sorted(fracs):
            raise ValueError("phase progress fractions must be non-decreasing.")
        for p, r in self.phases:
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"phase progress {p} out of range [0, 1].")
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"overlap ratio {r} out of range [0, 1].")
        self._fracs = fracs
        self._ratios = [r for _, r in self.phases]

    def ratio_at(self, progress: float) -> float:
        """Return the target overlap ratio at a training progress in [0, 1]."""
        progress = float(np.clip(progress, 0.0, 1.0))
        idx = bisect_right(self._fracs, progress) - 1  # last breakpoint <= progress

        if not self.interpolate or idx >= len(self._fracs) - 1:
            return self._ratios[idx]

        p0, r0 = self._fracs[idx], self._ratios[idx]
        p1, r1 = self._fracs[idx + 1], self._ratios[idx + 1]
        if p1 == p0:
            return r1
        w = (progress - p0) / (p1 - p0)
        return float(r0 + w * (r1 - r0))

    def ratio_at_step(self, step: int, total_steps: int) -> float:
        """Convenience wrapper: ratio at ``step / total_steps``."""
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        return self.ratio_at(step / total_steps)


def apply_overlap(
    references: np.ndarray,
    overlap_ratio: float,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-place clean stems at temporal offsets realising a target overlap ratio.

    Parameters
    ----------
    references:
        Clean stems, shape ``[N, T]`` (equal length, as produced by the mixer).
    overlap_ratio:
        Target overlap in [0, 1]; see the module docstring for the definition.
    rng:
        Optional generator used only when ``shuffle`` is True, to randomise the
        order in which stems are laid down (so the same speaker is not always
        first).  Defaults to a fresh default_rng().
    shuffle:
        Randomise stem placement order.  Set False for deterministic layout.

    Returns
    -------
    (mixture, refs_out)
        ``refs_out`` has shape ``[N, T_out]`` with each stem zero-padded into
        its offset position (reference order preserved).  ``mixture`` is their
        sum, shape ``[T_out]``.  Both float32.
    """
    refs = np.asarray(references, dtype=np.float32)
    if refs.ndim != 2:
        raise ValueError(f"references must be 2-D [N, T], got shape {refs.shape}.")
    if not 0.0 <= overlap_ratio <= 1.0:
        raise ValueError(f"overlap_ratio must be in [0, 1], got {overlap_ratio}.")

    n, length = refs.shape
    if n == 0:
        raise ValueError("references must contain at least one stem.")
    rng = rng if rng is not None else np.random.default_rng()

    order = list(range(n))
    if shuffle:
        rng.shuffle(order)

    # Trim leading/trailing all-zero padding so overlap is measured on real audio.
    trimmed = {i: _trim_silence(refs[i]) for i in range(n)}

    # Assign a start offset to each stem in placement order.
    offsets = [0] * n
    cursor = 0  # current end of the laid-down mixture
    for placed_idx, ref_idx in enumerate(order):
        seg_len = len(trimmed[ref_idx])
        if placed_idx == 0:
            start = 0
        else:
            # The new stem overlaps the running mixture by a fraction ~ratio:
            # it starts (1 - ratio) of the way through the current content.
            start = int(round((1.0 - overlap_ratio) * cursor))
        offsets[ref_idx] = start
        cursor = max(cursor, start + seg_len)

    total_len = max(cursor, length, 1)
    refs_out = np.zeros((n, total_len), dtype=np.float32)
    for i in range(n):
        seg = trimmed[i]
        start = offsets[i]
        refs_out[i, start : start + len(seg)] = seg

    mixture = refs_out.sum(axis=0).astype(np.float32)
    return mixture, refs_out


def _trim_silence(wave: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Strip leading/trailing (near) zero samples; return at least one sample."""
    nz = np.flatnonzero(np.abs(wave) > eps)
    if nz.size == 0:
        return wave[:1].astype(np.float32)
    return wave[nz[0] : nz[-1] + 1].astype(np.float32)
