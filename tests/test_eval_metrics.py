"""Known-answer tests for eval/metrics.py: the M0 ruler-is-straight contract."""

import numpy as np
import pytest

from eval.metrics import (
    HALLUCINATION_PENALTY_DB,
    cardinality_aware_score,
    count_accuracy,
    count_confusion_matrix,
    pairwise_si_sdr,
    pit_si_sdr,
    score_result,
    si_sdr,
    si_sdr_improvement,
)
from schemas.separation_result import SeparationResult

RNG = np.random.default_rng(seed=1337)
SAMPLES = 16000


def make_sources(n: int, samples: int = SAMPLES) -> np.ndarray:
    src = RNG.standard_normal((n, samples))
    return src / src.std(axis=1, keepdims=True)


def add_noise_at_snr(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Signal plus noise orthogonalized to the signal: SI-SDR equals snr_db exactly."""
    noise = RNG.standard_normal(signal.shape)
    sig0 = signal - signal.mean()
    noise = noise - (np.dot(noise, sig0) / np.dot(sig0, sig0)) * sig0
    noise = noise / np.linalg.norm(noise) * np.linalg.norm(sig0)
    return signal + 10 ** (-snr_db / 20.0) * noise


def test_oracle_scores_very_high() -> None:
    ref = make_sources(1)[0]
    assert si_sdr(ref.copy(), ref) > 60.0


def test_scale_invariance_nondegenerate() -> None:
    # At exact perfect match the residual is zero and the EPS guard dominates,
    # leaking amplitude by construction; invariance is guaranteed (and tested)
    # in the finite-residual regime.
    ref = make_sources(1)[0]
    est = add_noise_at_snr(ref, snr_db=10.0)
    assert si_sdr(0.31 * est, ref) == pytest.approx(si_sdr(3.7 * est, ref), abs=1e-6)


def test_known_snr_recovered() -> None:
    ref = make_sources(1)[0]
    for snr in (0.0, 10.0, 20.0):
        assert si_sdr(add_noise_at_snr(ref, snr), ref) == pytest.approx(snr, abs=0.05)


def test_mixture_as_estimate_gives_zero_improvement() -> None:
    a, b = make_sources(2)
    mix = a + b
    assert si_sdr_improvement(mix, a, mix) == pytest.approx(0.0, abs=1e-9)


def test_silent_reference_rejected() -> None:
    with pytest.raises(ValueError):
        si_sdr(make_sources(1)[0], np.zeros(SAMPLES))


def test_pit_permuted_oracle_recovers_assignment() -> None:
    refs = make_sources(3)
    mix = refs.sum(axis=0)
    perm = [2, 0, 1]
    res = pit_si_sdr(refs[perm], refs, mix)
    assert sorted(res.assignment) == sorted((i, perm[i]) for i in range(3))
    assert res.mean_si_sdr > 60.0
    assert res.missing_references == [] and res.unassigned_estimates == []


def test_pit_over_separation_reported_not_scored() -> None:
    refs = make_sources(2)
    mix = refs.sum(axis=0)
    res = pit_si_sdr(np.vstack([refs, make_sources(1)]), refs, mix)
    assert res.n_estimated == 3 and res.n_reference == 2
    assert len(res.unassigned_estimates) == 1
    assert res.mean_si_sdr > 60.0
    assert res.penalized_si_sdri == pytest.approx(res.mean_si_sdri - HALLUCINATION_PENALTY_DB)


def test_cardinality_aware_score_penalty() -> None:
    assert cardinality_aware_score(10.0, n_hallucinated=0) == pytest.approx(10.0)
    assert cardinality_aware_score(10.0, n_hallucinated=2) == pytest.approx(8.0)
    assert cardinality_aware_score(10.0, n_hallucinated=2, penalty_db=0.5) == pytest.approx(9.0)


def test_pit_under_separation_mixture_fallback() -> None:
    refs = make_sources(3)
    mix = refs.sum(axis=0)
    res = pit_si_sdr(refs[:2], refs, mix, missing_policy="mixture_fallback")
    missing = res.missing_references[0]
    assert res.si_sdri_per_stream[missing] == pytest.approx(0.0, abs=1e-9)
    for j in set(range(3)) - {missing}:
        assert res.si_sdri_per_stream[j] > 30.0


def test_pit_under_separation_silence_floor() -> None:
    refs = make_sources(3)
    mix = refs.sum(axis=0)
    res = pit_si_sdr(refs[:2], refs, mix, missing_policy="silence_floor", silence_floor_db=-25.0)
    assert res.si_sdr_per_stream[res.missing_references[0]] == pytest.approx(-25.0)


def test_pairwise_shape() -> None:
    assert pairwise_si_sdr(make_sources(3), make_sources(2)).shape == (3, 2)


def test_score_result_uses_schema() -> None:
    refs = make_sources(3)
    mix = refs.sum(axis=0)
    result = SeparationResult(
        streams=refs.astype(np.float32),
        sample_rate=16000,
        speaker_count=3,
        mixture=mix.astype(np.float32),
        expert_used="oracle",
    )
    res = score_result(result, refs)
    assert res.mean_si_sdri > 30.0


def test_score_result_requires_mixture() -> None:
    refs = make_sources(2)
    result = SeparationResult(streams=refs.astype(np.float32), sample_rate=16000, speaker_count=2)
    with pytest.raises(ValueError, match="mixture"):
        score_result(result, refs)


def test_count_accuracy_and_confusion() -> None:
    assert count_accuracy([2, 3, 4, 5], [2, 3, 3, 5]) == pytest.approx(0.75)
    mat = count_confusion_matrix([2, 3, 4, 4], [2, 3, 3, 4], count_range=(2, 5))
    assert mat[0, 0] == 1 and mat[1, 1] == 1 and mat[2, 2] == 1 and mat[2, 1] == 1
    assert mat.sum() == 4


def test_confusion_out_of_range_clipped() -> None:
    mat = count_confusion_matrix([2], [9], count_range=(2, 5))
    assert mat[0, 3] == 1
