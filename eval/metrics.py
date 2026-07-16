"""
Evaluation metrics: SI-SDR, SI-SDRi, permutation-invariant scoring, counting (Dev C, Phase 0).

Canonical implementation for the whole project. The baseline runner's internal
compute_sisdri is a Phase 0 stopgap; new code should import from here. All
functions are framework-free numpy so the same code scores training runs,
Kaggle jobs, CI, and the demo backend. Torch callers pass
tensor.detach().cpu().numpy().

Mismatched-count handling (the unknown-N cases) follows the protocol in
MASTER_PROJECT.md section 10 and is configurable, never silent: see
pit_si_sdr's missing_policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from schemas.separation_result import SeparationResult

EPS: float = 1e-8
"""Numerical guard for log/divide. Not a tunable; tests reference this exact value."""

HALLUCINATION_PENALTY_DB: float = 1.0
"""Per-hallucinated-stream penalty for cardinality-aware scoring (BLUEPRINT §9.2)."""

MissingPolicy = Literal["mixture_fallback", "silence_floor"]


@dataclass
class PitResult:
    """Permutation-invariant scoring result for one mixture.

    Attributes:
        si_sdr_per_stream: SI-SDR (dB) per reference stream, in reference order.
        si_sdri_per_stream: Improvement over the mixture per reference stream.
        mean_si_sdr: Mean SI-SDR across reference streams.
        mean_si_sdri: Mean SI-SDRi across reference streams.
        penalized_si_sdri: Cardinality-aware score: mean SI-SDRi minus 1 dB per
            hallucinated (unassigned) estimate stream (BLUEPRINT §9.2).
        assignment: Matched (estimate_index, reference_index) pairs.
        unassigned_estimates: Estimate indices left unmatched (over-separation).
        missing_references: Reference indices with no estimate (under-separation).
        n_estimated: Number of estimated streams scored.
        n_reference: Number of reference streams.
    """

    si_sdr_per_stream: list[float]
    si_sdri_per_stream: list[float]
    mean_si_sdr: float
    mean_si_sdri: float
    assignment: list[tuple[int, int]]
    unassigned_estimates: list[int] = field(default_factory=list)
    missing_references: list[int] = field(default_factory=list)
    n_estimated: int = 0
    n_reference: int = 0
    penalized_si_sdri: float = 0.0


def _validate_1d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (samples,), got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio in dB (Le Roux et al., 2019).

    Both signals are zero-meaned, the estimate is projected onto the reference
    (the step that makes the metric scale-invariant), and the ratio of target
    to residual energy is returned in dB.

    Args:
        estimate: Separated waveform, shape [T].
        reference: Clean ground-truth waveform, shape [T], same length.

    Returns:
        SI-SDR in dB. Larger is better. EPS-guarded, so a perfect match
        returns a large finite value rather than infinity.
    """
    est = _validate_1d(estimate, "estimate")
    ref = _validate_1d(reference, "reference")
    if est.shape != ref.shape:
        raise ValueError(f"length mismatch: estimate {est.shape} vs reference {ref.shape}")

    est = est - est.mean()
    ref = ref - ref.mean()

    ref_energy = float(np.dot(ref, ref))
    if ref_energy < EPS:
        raise ValueError("reference is (near) silent; SI-SDR is undefined")

    scale = float(np.dot(est, ref)) / (ref_energy + EPS)
    target = scale * ref
    residual = est - target

    num = float(np.dot(target, target))
    den = float(np.dot(residual, residual))
    return 10.0 * float(np.log10((num + EPS) / (den + EPS)))


def si_sdr_improvement(estimate: np.ndarray, reference: np.ndarray, mixture: np.ndarray) -> float:
    """SI-SDRi in dB: improvement of the estimate over just using the mixture."""
    return si_sdr(estimate, reference) - si_sdr(mixture, reference)


def pairwise_si_sdr(estimates: np.ndarray, references: np.ndarray) -> np.ndarray:
    """
    SI-SDR for every (estimate, reference) pair.

    Args:
        estimates: Array of shape [n_est, T].
        references: Array of shape [n_ref, T].

    Returns:
        Array of shape [n_est, n_ref]; cell (i, j) = si_sdr(estimates[i], references[j]).
    """
    est = np.atleast_2d(np.asarray(estimates, dtype=np.float64))
    ref = np.atleast_2d(np.asarray(references, dtype=np.float64))
    out = np.empty((est.shape[0], ref.shape[0]), dtype=np.float64)
    for i in range(est.shape[0]):
        for j in range(ref.shape[0]):
            out[i, j] = si_sdr(est[i], ref[j])
    return out


def pit_si_sdr(
    estimates: np.ndarray,
    references: np.ndarray,
    mixture: np.ndarray,
    missing_policy: MissingPolicy = "mixture_fallback",
    silence_floor_db: float = -30.0,
) -> PitResult:
    """
    Permutation-invariant SI-SDR / SI-SDRi with mismatched-count handling.

    Hungarian matching on the negated pairwise SI-SDR matrix covers the
    rectangular (unknown-N) cases:

    Over-separation (n_est > n_ref): the best n_ref estimates are matched;
    leftover estimates are reported in unassigned_estimates and do not affect
    the score (Stem Hygiene upstream is expected to prune them).

    Under-separation (n_est < n_ref): unmatched references are scored by
    missing_policy. "mixture_fallback" scores the raw mixture as that
    speaker's estimate, giving SI-SDRi of exactly 0 (the listener is left with
    the mixture). "silence_floor" assigns silence_floor_db as that speaker's
    SI-SDR. Missing speakers are always reported in missing_references, never
    silently absorbed.

    Args:
        estimates: [n_est, T] separated stems.
        references: [n_ref, T] ground-truth stems.
        mixture: [T] original mixed signal.
        missing_policy: How to score references with no matching estimate.
        silence_floor_db: Floor used by the "silence_floor" policy.

    Returns:
        PitResult with per-stream and mean scores in reference order.
    """
    est = np.atleast_2d(np.asarray(estimates, dtype=np.float64))
    ref = np.atleast_2d(np.asarray(references, dtype=np.float64))
    mix = _validate_1d(mixture, "mixture")

    n_est, n_ref = est.shape[0], ref.shape[0]
    pair = pairwise_si_sdr(est, ref)

    est_idx, ref_idx = linear_sum_assignment(-pair)
    assignment = list(zip(est_idx.tolist(), ref_idx.tolist(), strict=True))

    matched_refs = {int(r) for r in ref_idx}
    matched_ests = {int(e) for e in est_idx}
    missing = [j for j in range(n_ref) if j not in matched_refs]
    unassigned = [i for i in range(n_est) if i not in matched_ests]

    mixture_sdr_per_ref = [si_sdr(mix, ref[j]) for j in range(n_ref)]

    sdr_by_ref: dict[int, float] = {int(j): float(pair[i, j]) for i, j in assignment}
    for j in missing:
        if missing_policy == "mixture_fallback":
            sdr_by_ref[j] = mixture_sdr_per_ref[j]
        elif missing_policy == "silence_floor":
            sdr_by_ref[j] = float(silence_floor_db)
        else:  # pragma: no cover - guarded by Literal, kept for safety
            raise ValueError(f"unknown missing_policy: {missing_policy}")

    sdr = [sdr_by_ref[j] for j in range(n_ref)]
    sdri = [sdr[j] - mixture_sdr_per_ref[j] for j in range(n_ref)]

    mean_sdri = float(np.mean(sdri))
    penalized = cardinality_aware_score(mean_sdri, n_hallucinated=len(unassigned))

    return PitResult(
        si_sdr_per_stream=[float(v) for v in sdr],
        si_sdri_per_stream=[float(v) for v in sdri],
        mean_si_sdr=float(np.mean(sdr)),
        mean_si_sdri=mean_sdri,
        penalized_si_sdri=penalized,
        assignment=assignment,
        unassigned_estimates=unassigned,
        missing_references=missing,
        n_estimated=n_est,
        n_reference=n_ref,
    )


def cardinality_aware_score(
    mean_si_sdri: float,
    n_hallucinated: int,
    penalty_db: float = HALLUCINATION_PENALTY_DB,
) -> float:
    """
    Cardinality-aware penalized SI-SDRi (BLUEPRINT §9.2).

    Each matched pair contributes via mean_si_sdri (missed speakers already
    scored at 0 dB improvement under mixture_fallback). Each hallucinated
    (unassigned) estimate stream subtracts ``penalty_db`` from the mean.

    Args:
        mean_si_sdri: Mean SI-SDRi over reference streams after Hungarian match.
        n_hallucinated: Number of unassigned estimate streams.
        penalty_db: Penalty per hallucinated stream (default 1.0 dB).

    Returns:
        Penalized score in dB.
    """
    if n_hallucinated < 0:
        raise ValueError(f"n_hallucinated must be non-negative, got {n_hallucinated}")
    if penalty_db < 0:
        raise ValueError(f"penalty_db must be non-negative, got {penalty_db}")
    return float(mean_si_sdri - penalty_db * n_hallucinated)


def score_result(
    result: SeparationResult,
    references: np.ndarray,
    missing_policy: MissingPolicy = "mixture_fallback",
    silence_floor_db: float = -30.0,
) -> PitResult:
    """
    Score a SeparationResult against ground-truth stems.

    Convenience wrapper so pipeline code scores the shared schema object
    directly. Requires result.mixture to be present (every expert wrapper
    attaches it).

    Args:
        result: Output of any expert wrapper or the fused pipeline.
        references: [n_ref, T] ground-truth stems at result.sample_rate.
        missing_policy: See pit_si_sdr.
        silence_floor_db: See pit_si_sdr.

    Returns:
        PitResult for this mixture.
    """
    if result.mixture is None:
        raise ValueError("SeparationResult.mixture is required for SI-SDRi scoring")
    return pit_si_sdr(
        estimates=result.streams,
        references=references,
        mixture=result.mixture,
        missing_policy=missing_policy,
        silence_floor_db=silence_floor_db,
    )


def count_accuracy(true_counts: Sequence[int], estimated_counts: Sequence[int]) -> float:
    """Fraction of mixtures where the estimated speaker count is exact."""
    t = np.asarray(true_counts)
    e = np.asarray(estimated_counts)
    if t.shape != e.shape:
        raise ValueError("true and estimated count lists differ in length")
    if t.size == 0:
        raise ValueError("empty count lists")
    return float(np.mean(t == e))


def count_confusion_matrix(
    true_counts: Sequence[int],
    estimated_counts: Sequence[int],
    count_range: tuple[int, int],
) -> np.ndarray:
    """
    Confusion matrix over speaker counts (rows: true N, columns: estimated N).

    Estimates outside count_range are clipped into the edge bins so a wildly
    wrong estimate stays visible at the boundary rather than being dropped.

    Args:
        true_counts: True N per mixture.
        estimated_counts: Estimated N per mixture.
        count_range: Inclusive (lowest N, highest N) span, e.g. (2, 5).

    Returns:
        Integer matrix of shape [span, span].
    """
    lo, hi = count_range
    if hi < lo:
        raise ValueError(f"count_range upper bound below lower bound: {count_range}")
    size = hi - lo + 1
    mat = np.zeros((size, size), dtype=np.int64)
    for t, e in zip(true_counts, estimated_counts, strict=True):
        ti = int(np.clip(t, lo, hi)) - lo
        ei = int(np.clip(e, lo, hi)) - lo
        mat[ti, ei] += 1
    return mat
