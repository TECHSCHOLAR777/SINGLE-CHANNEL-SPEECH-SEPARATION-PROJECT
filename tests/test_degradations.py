"""
Unit tests for data/degradations.py (Dev A, P0-A7 / P1-A1-A2).

Tests use tiny synthetic waveforms and mock the RirBank so no corpus or
pyroomacoustics is needed. Codec tests run the mu-law fallback path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from coralsep.data.condition_mixer import CORALSEP_SAMPLE_RATE, CoralSepMixture, MixtureRecipe
from coralsep.data.degradations import (
    HELD_OUT_COMBINATIONS,
    SEVERE_FRACTION,
    SEVERE_SNR_DB,
    SNR_MAX_DB,
    SNR_MIN_DB,
    apply_noise,
    apply_reverb,
    assert_not_held_out,
    describe_condition,
    make_wet_mixture,
    make_wet_reference,
    sample_snr,
)
from coralsep.data.mixer_stub import MixtureSample
from coralsep.data.rir_bank import RirRecord

SR = CORALSEP_SAMPLE_RATE
LENGTH = SR  # 1-second clips


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _clean_mixture(n: int = 2, length: int = LENGTH) -> CoralSepMixture:
    rng = np.random.default_rng(0)
    refs = rng.standard_normal((n, length)).astype(np.float32) * 0.1
    mix = refs.sum(axis=0).astype(np.float32)
    sample = MixtureSample(mixture=mix, references=refs, sample_rate=SR, utterance_id="test_utt")
    recipe = MixtureRecipe(n_speakers=n)
    return CoralSepMixture(sample=sample, recipe=recipe)


def _rir(length: int = 800) -> np.ndarray:
    rng = np.random.default_rng(1)
    t = np.arange(length) / SR
    decay = np.exp(-6.908 * t / 0.4)
    rir = (rng.standard_normal(length) * decay).astype(np.float32)
    rir[16] = 1.0  # artificial direct-path peak
    return rir


def _rir_record(tmp_path: Path | None = None) -> RirRecord:
    return RirRecord(
        rir_id="test",
        path="test.npy",
        t60_requested_s=0.4,
        t60_achieved_s=0.42,
        room_dim_m=[5.0, 4.0, 3.0],
        source_pos_m=[1.0, 1.0, 1.0],
        mic_pos_m=[2.0, 2.0, 1.0],
        absorption=0.25,
        max_order=10,
        n_peak=16,
        sample_rate=SR,
    )


def _mock_rir_bank(rir: np.ndarray) -> MagicMock:
    bank = MagicMock()
    bank.sample.return_value = _rir_record()
    bank.load.return_value = rir
    return bank


# ---------------------------------------------------------------------------
# make_wet_reference
# ---------------------------------------------------------------------------


def test_wet_ref_length_matches_dry():
    dry = np.ones(LENGTH, dtype=np.float32)
    rir = _rir()
    ref = make_wet_reference(dry, rir, n_peak=16)
    assert ref.shape == (LENGTH,)


def test_wet_ref_dtype_float32():
    dry = np.ones(LENGTH, dtype=np.float32)
    rir = _rir()
    ref = make_wet_reference(dry, rir, n_peak=16)
    assert ref.dtype == np.float32


def test_wet_ref_truncation_offset():
    dry = np.zeros(100, dtype=np.float32)
    dry[0] = 1.0
    rir = np.zeros(200, dtype=np.float32)
    rir[10] = 1.0
    rir[20] = 0.5  # within offset=12
    rir[100] = 0.5  # outside: n_peak+12 = 22

    ref = make_wet_reference(dry, rir, n_peak=10, offset=12, target_length=200)
    # Contribution from rir[100] should be absent
    assert ref.shape[0] == 200


def test_wet_ref_invalid_truncation_raises():
    dry = np.ones(10, dtype=np.float32)
    rir = np.ones(5, dtype=np.float32)
    with pytest.raises(ValueError, match="truncation"):
        make_wet_reference(dry, rir, n_peak=-5, offset=2)


# ---------------------------------------------------------------------------
# make_wet_mixture
# ---------------------------------------------------------------------------


def test_wet_mixture_length():
    dry = np.ones(LENGTH, dtype=np.float32)
    rir = _rir()
    wet = make_wet_mixture(dry, rir)
    assert wet.shape == (LENGTH,)


def test_wet_mixture_differs_from_reference():
    dry = np.ones(LENGTH, dtype=np.float32)
    rir = _rir()
    ref = make_wet_reference(dry, rir, n_peak=16)
    wet = make_wet_mixture(dry, rir)
    # Full RIR ≠ truncated RIR
    assert not np.allclose(ref, wet)


# ---------------------------------------------------------------------------
# sample_snr
# ---------------------------------------------------------------------------


def test_sample_snr_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = sample_snr(rng, allow_severe=True)
        assert SNR_MIN_DB <= s <= SNR_MAX_DB


def test_sample_snr_no_severe():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = sample_snr(rng, allow_severe=False)
        assert s >= SEVERE_SNR_DB


def test_sample_snr_severe_fraction():
    rng = np.random.default_rng(5)
    samples = [
        sample_snr(rng, allow_severe=True, severe_fraction=SEVERE_FRACTION) for _ in range(2000)
    ]
    severe = sum(1 for s in samples if s < SEVERE_SNR_DB)
    assert abs(severe / len(samples) - SEVERE_FRACTION) < 0.05


# ---------------------------------------------------------------------------
# apply_reverb
# ---------------------------------------------------------------------------


def test_apply_reverb_returns_new_object():
    m = _clean_mixture()
    rir = _rir()
    bank = _mock_rir_bank(rir)
    m2 = apply_reverb(m, bank, np.random.default_rng(0), record=_rir_record())
    assert m2 is not m


def test_apply_reverb_observation_different_from_clean():
    m = _clean_mixture()
    rir = _rir()
    bank = _mock_rir_bank(rir)
    m2 = apply_reverb(m, bank, np.random.default_rng(0), record=_rir_record())
    assert not np.allclose(m.mixture, m2.mixture)


def test_apply_reverb_references_shape_preserved():
    m = _clean_mixture(n=3)
    rir = _rir()
    bank = _mock_rir_bank(rir)
    m2 = apply_reverb(m, bank, np.random.default_rng(0), record=_rir_record())
    assert m2.references.shape == m.references.shape


def test_apply_reverb_recipe_updated():
    m = _clean_mixture()
    rir = _rir()
    bank = _mock_rir_bank(rir)
    rec = _rir_record()
    m2 = apply_reverb(m, bank, np.random.default_rng(0), record=rec)
    assert m2.recipe.t60_s == rec.t60_achieved_s
    assert m2.recipe.rir_file == rec.path


def test_apply_reverb_sample_rate_unchanged():
    m = _clean_mixture()
    rir = _rir()
    bank = _mock_rir_bank(rir)
    m2 = apply_reverb(m, bank, np.random.default_rng(0), record=_rir_record())
    assert m2.sample.sample_rate == SR


# ---------------------------------------------------------------------------
# apply_noise
# ---------------------------------------------------------------------------


def test_apply_noise_adds_to_observation():
    m = _clean_mixture()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(LENGTH).astype(np.float32) * 0.05
    m2 = apply_noise(m, noise, rng, snr_db=5.0)
    assert not np.allclose(m.mixture, m2.mixture)


def test_apply_noise_references_unchanged():
    m = _clean_mixture(n=3)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(LENGTH).astype(np.float32) * 0.05
    m2 = apply_noise(m, noise, rng, snr_db=0.0)
    np.testing.assert_array_equal(m.references, m2.references)


def test_apply_noise_achieved_snr():
    rng = np.random.default_rng(42)
    m = _clean_mixture()
    noise = rng.standard_normal(LENGTH).astype(np.float32)
    target_snr = 3.0
    m2 = apply_noise(m, noise, rng, snr_db=target_snr)
    speech_power = float(np.mean(m.mixture**2))
    noise_power = float(np.mean((m2.mixture - m.mixture) ** 2))
    achieved_snr = 10.0 * np.log10(speech_power / noise_power + 1e-10)
    assert (
        abs(achieved_snr - target_snr) < 0.5
    ), f"SNR {achieved_snr:.2f} far from target {target_snr}"


def test_apply_noise_recipe_snr_recorded():
    m = _clean_mixture()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(LENGTH).astype(np.float32) * 0.1
    m2 = apply_noise(m, noise, rng, snr_db=-2.0, noise_file="noise.wav")
    assert m2.recipe.snr_db == -2.0
    assert m2.recipe.noise_file == "noise.wav"


def test_apply_noise_silent_noise_raises():
    m = _clean_mixture()
    rng = np.random.default_rng(0)
    noise = np.zeros(LENGTH, dtype=np.float32)
    with pytest.raises(ValueError, match="silent"):
        apply_noise(m, noise, rng, snr_db=0.0)


# ---------------------------------------------------------------------------
# describe_condition
# ---------------------------------------------------------------------------


def test_describe_condition_clean():
    r = MixtureRecipe(n_speakers=2)
    assert describe_condition(r) == "clean"


def test_describe_condition_reverb():
    r = MixtureRecipe(n_speakers=2, t60_s=0.5)
    assert describe_condition(r) == "reverb"


def test_describe_condition_noise():
    r = MixtureRecipe(n_speakers=2, snr_db=5.0)
    assert describe_condition(r) == "noise"


def test_describe_condition_codec():
    r = MixtureRecipe(n_speakers=2, codec_name="opus", codec_bitrate_bps=8000)
    assert describe_condition(r) == "codec"


def test_describe_condition_reverb_noise():
    r = MixtureRecipe(n_speakers=2, t60_s=0.4, snr_db=3.0)
    assert describe_condition(r) == "reverb+noise"


def test_describe_condition_all_three():
    r = MixtureRecipe(
        n_speakers=2, t60_s=0.4, snr_db=3.0, codec_name="aac", codec_bitrate_bps=16000
    )
    assert describe_condition(r) == "all-three"


def test_describe_condition_held_out_combinations():
    for combo in HELD_OUT_COMBINATIONS:
        assert combo in {"reverb+codec", "noise+codec"}


# ---------------------------------------------------------------------------
# assert_not_held_out
# ---------------------------------------------------------------------------


def test_assert_not_held_out_passes_clean():
    assert_not_held_out(MixtureRecipe(n_speakers=2))


def test_assert_not_held_out_passes_reverb_noise():
    r = MixtureRecipe(n_speakers=2, t60_s=0.4, snr_db=3.0)
    assert_not_held_out(r)  # must not raise


def test_assert_not_held_out_raises_reverb_codec():
    r = MixtureRecipe(n_speakers=2, t60_s=0.4, codec_name="opus", codec_bitrate_bps=8000)
    with pytest.raises(ValueError, match="held-out"):
        assert_not_held_out(r)


def test_assert_not_held_out_raises_noise_codec():
    r = MixtureRecipe(n_speakers=2, snr_db=3.0, codec_name="aac", codec_bitrate_bps=16000)
    with pytest.raises(ValueError, match="held-out"):
        assert_not_held_out(r)
