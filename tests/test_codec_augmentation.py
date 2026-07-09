"""
Unit tests for data/codec_augmentation.py.

All tests use the mu-law fallback (use_ffmpeg=False) or mock subprocess.run,
so no ffmpeg binary is required to run the suite.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import soundfile as sf

from data.codec_augmentation import (
    CodecAugmentor,
    CodecConfig,
    _fit_length,
    is_ffmpeg_available,
)
from data.mixer_stub import MixtureSample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 16_000
LENGTH = SR  # 1 second


def _make_sample(seed: int = 42) -> MixtureSample:
    rng = np.random.default_rng(seed)
    refs = rng.standard_normal((3, LENGTH)).astype(np.float32)
    mixture = refs.sum(axis=0)
    return MixtureSample(
        mixture=mixture,
        references=refs,
        sample_rate=SR,
        utterance_id="codec_test_000",
    )


def _mulaw_config(**kwargs) -> CodecConfig:
    """CodecConfig with ffmpeg disabled (forces mu-law path)."""
    defaults = dict(codec="opus", codec_prob=1.0, use_ffmpeg=False)
    defaults.update(kwargs)
    return CodecConfig(**defaults)


# ---------------------------------------------------------------------------
# CodecConfig validation
# ---------------------------------------------------------------------------


def test_config_invalid_codec_raises() -> None:
    with pytest.raises(ValueError, match="codec must be one of"):
        CodecConfig(codec="mp3")


def test_config_negative_bitrate_raises() -> None:
    with pytest.raises(ValueError, match="bitrate values must be positive"):
        CodecConfig(bitrate_min_kbps=-1.0)


def test_config_min_exceeds_max_raises() -> None:
    with pytest.raises(ValueError, match="bitrate_min_kbps must be <="):
        CodecConfig(bitrate_min_kbps=32.0, bitrate_max_kbps=6.0)


def test_config_invalid_prob_raises() -> None:
    with pytest.raises(ValueError, match="codec_prob must be in"):
        CodecConfig(codec_prob=1.5)


# ---------------------------------------------------------------------------
# Return type and metadata preservation
# ---------------------------------------------------------------------------


def test_returns_mixture_sample_type() -> None:
    out = CodecAugmentor(_mulaw_config())(_make_sample())
    assert isinstance(out, MixtureSample)


def test_utterance_id_preserved() -> None:
    sample = _make_sample()
    out = CodecAugmentor(_mulaw_config())(sample)
    assert out.utterance_id == sample.utterance_id


def test_sample_rate_preserved() -> None:
    sample = _make_sample()
    out = CodecAugmentor(_mulaw_config())(sample)
    assert out.sample_rate == SR


# ---------------------------------------------------------------------------
# Probability gate
# ---------------------------------------------------------------------------


def test_prob_zero_skips_augmentation() -> None:
    sample = _make_sample()
    cfg = _mulaw_config(codec_prob=0.0)
    out = CodecAugmentor(cfg, rng=np.random.default_rng(0))(sample)
    np.testing.assert_array_equal(out.mixture, sample.mixture)


def test_prob_one_always_applies() -> None:
    sample = _make_sample()
    cfg = _mulaw_config(codec_prob=1.0)
    out = CodecAugmentor(cfg, rng=np.random.default_rng(0))(sample)
    assert not np.allclose(out.mixture, sample.mixture)


# ---------------------------------------------------------------------------
# References are never modified
# ---------------------------------------------------------------------------


def test_references_unchanged_after_codec() -> None:
    sample = _make_sample()
    refs_copy = sample.references.copy()
    out = CodecAugmentor(_mulaw_config())(_make_sample())
    np.testing.assert_array_equal(out.references, refs_copy)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_output_length_preserved() -> None:
    sample = _make_sample()
    out = CodecAugmentor(_mulaw_config())(sample)
    assert len(out.mixture) == LENGTH


def test_output_dtype_is_float32() -> None:
    sample = _make_sample()
    out = CodecAugmentor(_mulaw_config())(sample)
    assert out.mixture.dtype == np.float32


# ---------------------------------------------------------------------------
# Mu-law fallback behaviour
# ---------------------------------------------------------------------------


def test_mulaw_changes_audio() -> None:
    rng = np.random.default_rng(7)
    audio = rng.standard_normal(SR).astype(np.float32)
    aug = CodecAugmentor(_mulaw_config())
    result = aug._mulaw_roundtrip(audio)
    assert not np.allclose(result, audio)


def test_mulaw_output_is_float32() -> None:
    audio = np.ones(1000, dtype=np.float32) * 0.5
    result = CodecAugmentor(_mulaw_config())._mulaw_roundtrip(audio)
    assert result.dtype == np.float32


def test_mulaw_output_stays_within_minus1_to_1() -> None:
    rng = np.random.default_rng(13)
    audio = rng.uniform(-2.0, 2.0, SR).astype(np.float32)  # intentionally out-of-range
    result = CodecAugmentor(_mulaw_config())._mulaw_roundtrip(audio)
    assert result.min() >= -1.0 - 1e-6
    assert result.max() <= 1.0 + 1e-6


def test_mulaw_output_length_matches_input() -> None:
    audio = np.zeros(12345, dtype=np.float32)
    result = CodecAugmentor(_mulaw_config())._mulaw_roundtrip(audio)
    assert len(result) == 12345


# ---------------------------------------------------------------------------
# ffmpeg path — subprocess mocked
# ---------------------------------------------------------------------------


def _make_ffmpeg_mock(tmp_path, audio: np.ndarray) -> MagicMock:
    """
    Return a mock for subprocess.run that writes a decoded WAV on the
    second call (the decode step), simulating a successful ffmpeg roundtrip.
    """
    decoded_wav = tmp_path / "decoded_ref.wav"
    sf.write(str(decoded_wav), audio * 0.9, SR, subtype="FLOAT")

    call_count = [0]

    def _side_effect(cmd, **kwargs):
        call_count[0] += 1
        result = MagicMock()
        result.returncode = 0
        if call_count[0] == 2:
            # Decode call: write decoded wav where ffmpeg would put it
            import shutil as _shutil
            dst = Path(cmd[-1])  # last arg is the output path
            _shutil.copy(str(decoded_wav), str(dst))
        return result

    return MagicMock(side_effect=_side_effect)


def test_ffmpeg_called_with_opus_args(tmp_path) -> None:
    sample = _make_sample()
    cfg = CodecConfig(codec="opus", codec_prob=1.0, use_ffmpeg=True)
    aug = CodecAugmentor(cfg, rng=np.random.default_rng(0))
    aug._ffmpeg_ok = True  # skip shutil.which check

    with patch("data.codec_augmentation.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        # We don't need a real decoded file here — just check args
        with patch("data.codec_augmentation.sf.read", return_value=(
            np.zeros((LENGTH, 1), dtype=np.float32), SR
        )):
            aug(sample)

    all_calls = mock_run.call_args_list
    assert len(all_calls) >= 1
    encode_cmd = all_calls[0][0][0]
    assert "libopus" in encode_cmd


def test_ffmpeg_called_with_aac_args(tmp_path) -> None:
    sample = _make_sample()
    cfg = CodecConfig(codec="aac", codec_prob=1.0, use_ffmpeg=True)
    aug = CodecAugmentor(cfg, rng=np.random.default_rng(0))
    aug._ffmpeg_ok = True

    with patch("data.codec_augmentation.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with patch("data.codec_augmentation.sf.read", return_value=(
            np.zeros((LENGTH, 1), dtype=np.float32), SR
        )):
            aug(sample)

    encode_cmd = mock_run.call_args_list[0][0][0]
    assert "aac" in encode_cmd


def test_ffmpeg_failure_falls_back_to_mulaw() -> None:
    sample = _make_sample()
    cfg = CodecConfig(codec="opus", codec_prob=1.0, use_ffmpeg=True)
    aug = CodecAugmentor(cfg, rng=np.random.default_rng(0))
    aug._ffmpeg_ok = True

    failing = MagicMock(returncode=1)
    with patch("data.codec_augmentation.subprocess.run", return_value=failing):
        import warnings as _warnings
        with _warnings.catch_warnings(record=True):
            out = aug(sample)

    # Mu-law still changes the mixture
    assert not np.allclose(out.mixture, sample.mixture)
    assert len(out.mixture) == LENGTH


def test_ffmpeg_unavailable_uses_mulaw() -> None:
    sample = _make_sample()
    cfg = CodecConfig(codec="opus", codec_prob=1.0, use_ffmpeg=True)
    aug = CodecAugmentor(cfg, rng=np.random.default_rng(0))
    aug._ffmpeg_ok = False  # simulate ffmpeg not on PATH

    out = aug(sample)
    assert not np.allclose(out.mixture, sample.mixture)
    assert len(out.mixture) == LENGTH


# ---------------------------------------------------------------------------
# Random codec selection
# ---------------------------------------------------------------------------


def test_random_codec_resolves_to_opus_or_aac() -> None:
    cfg = CodecConfig(codec="random", codec_prob=1.0, use_ffmpeg=False)
    aug = CodecAugmentor(cfg, rng=np.random.default_rng(0))
    seen = set()
    for _ in range(30):
        seen.add(aug._resolve_codec())
    assert seen == {"opus", "aac"}


# ---------------------------------------------------------------------------
# Bitrate is sampled within the configured range
# ---------------------------------------------------------------------------


def test_bitrate_sampled_within_range() -> None:
    cfg = CodecConfig(codec="opus", codec_prob=1.0, use_ffmpeg=True, bitrate_min_kbps=8.0, bitrate_max_kbps=16.0)
    aug = CodecAugmentor(cfg, rng=np.random.default_rng(5))
    aug._ffmpeg_ok = True

    captured = []

    original = aug._ffmpeg_roundtrip

    def _capture(audio, sr, codec, bitrate_kbps):
        captured.append(bitrate_kbps)
        return None  # force mulaw fallback

    with patch.object(aug, "_ffmpeg_roundtrip", side_effect=_capture):
        import warnings as _warnings
        for _ in range(20):
            with _warnings.catch_warnings(record=True):
                aug(_make_sample())

    assert all(8.0 <= b <= 16.0 for b in captured), f"Out-of-range bitrate found: {captured}"


# ---------------------------------------------------------------------------
# RNG reproducibility
# ---------------------------------------------------------------------------


def test_rng_reproducibility() -> None:
    sample = _make_sample()
    cfg = _mulaw_config(codec_prob=1.0)
    out_a = CodecAugmentor(cfg, rng=np.random.default_rng(77))(sample)
    out_b = CodecAugmentor(cfg, rng=np.random.default_rng(77))(sample)
    np.testing.assert_array_equal(out_a.mixture, out_b.mixture)


# ---------------------------------------------------------------------------
# is_ffmpeg_available helper
# ---------------------------------------------------------------------------


def test_is_ffmpeg_available_returns_bool() -> None:
    result = is_ffmpeg_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _fit_length helper
# ---------------------------------------------------------------------------


def test_fit_length_trims_long_audio() -> None:
    audio = np.ones(2000, dtype=np.float32)
    assert len(_fit_length(audio, 1000)) == 1000


def test_fit_length_pads_short_audio() -> None:
    audio = np.ones(500, dtype=np.float32)
    result = _fit_length(audio, 1000)
    assert len(result) == 1000
    assert np.all(result[500:] == 0.0)


def test_fit_length_exact_unchanged() -> None:
    audio = np.ones(SR, dtype=np.float32)
    assert len(_fit_length(audio, SR)) == SR
