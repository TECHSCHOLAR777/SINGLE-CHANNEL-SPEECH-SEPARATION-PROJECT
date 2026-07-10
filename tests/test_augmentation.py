"""
Unit tests for data/augmentation.py.

All tests use synthetic numpy arrays and tmp_path fixtures.
No GPU, no real WHAM! dataset, and no network access required.
pyroomacoustics is mocked for RIR tests so the full test suite
runs even without the optional dependency installed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from data.augmentation import AugmentationConfig, AugmentationPipeline, _fit_to_length
from data.mixer_stub import MixtureSample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 16_000
N_SPEAKERS = 3
LENGTH = SR  # 1 second


def _make_sample(rng: np.random.Generator | None = None) -> MixtureSample:
    """Synthetic 1-second MixtureSample with 3 clean stems."""
    r = rng or np.random.default_rng(42)
    refs = r.standard_normal((N_SPEAKERS, LENGTH)).astype(np.float32)
    mixture = refs.sum(axis=0)
    return MixtureSample(
        mixture=mixture,
        references=refs,
        sample_rate=SR,
        utterance_id="test_utt_000",
    )


def _write_noise_wav(path: Path, length: int = SR, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(length).astype(np.float32) * 0.1
    sf.write(str(path), audio, SR, subtype="FLOAT")


def _make_rir(length: int = 400) -> np.ndarray:
    """Simple synthetic RIR: a decaying impulse."""
    rir = np.zeros(length, dtype=np.float32)
    rir[0] = 1.0
    for i in range(1, length):
        rir[i] = rir[i - 1] * 0.98
    return rir


def _mock_pra(rir: np.ndarray) -> MagicMock:
    """Build a pyroomacoustics mock that returns the given RIR."""
    mic_mock = MagicMock()
    mic_mock.__getitem__ = lambda self, idx: [rir]  # room.rir[0][0]

    room_mock = MagicMock()
    room_mock.rir = [mic_mock]

    pra_mock = MagicMock()
    pra_mock.Material.return_value = MagicMock()
    pra_mock.ShoeBox.return_value = room_mock

    return pra_mock


# ---------------------------------------------------------------------------
# Type and structure
# ---------------------------------------------------------------------------


def test_returns_mixture_sample_type(tmp_path: Path) -> None:
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=0.0)
    pipeline = AugmentationPipeline(cfg)
    out = pipeline(sample)
    assert isinstance(out, MixtureSample)


def test_utterance_id_preserved(tmp_path: Path) -> None:
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=0.0)
    out = AugmentationPipeline(cfg)(sample)
    assert out.utterance_id == sample.utterance_id


def test_sample_rate_preserved() -> None:
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=0.0)
    out = AugmentationPipeline(cfg)(sample)
    assert out.sample_rate == SR


# ---------------------------------------------------------------------------
# Stage 1 — prob=0 skips, prob=1 fires
# ---------------------------------------------------------------------------


def test_rir_prob_zero_skips() -> None:
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=0.0)
    out = AugmentationPipeline(cfg, rng=np.random.default_rng(0))(sample)
    np.testing.assert_array_equal(out.mixture, sample.mixture)


def test_rir_prob_one_changes_mixture() -> None:
    """_apply_rir with a non-trivial RIR should change the mixture."""
    import sys

    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=1.0, noise_prob=0.0)
    pipeline = AugmentationPipeline(cfg, rng=np.random.default_rng(1))
    rir = _make_rir()
    pra_mock = _mock_pra(rir)

    # Inject the mock into sys.modules so the `import pyroomacoustics as pra` inside
    # _apply_rir resolves without the package being installed.
    with patch.dict(sys.modules, {"pyroomacoustics": pra_mock}):
        with patch.object(pipeline, "_generate_rir", return_value=rir):
            result = pipeline._apply_rir(sample.mixture, SR)

    assert not np.allclose(result, sample.mixture)


def test_rir_changes_mixture_via_full_pipeline(tmp_path: Path) -> None:
    """Full pipeline call: patch _apply_rir so we don't need pra installed."""
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=1.0, noise_prob=0.0)
    pipeline = AugmentationPipeline(cfg, rng=np.random.default_rng(2))

    fake_aug = np.zeros(LENGTH, dtype=np.float32)

    with patch.object(pipeline, "_apply_rir", return_value=fake_aug) as mock_rir:
        out = pipeline(sample)

    mock_rir.assert_called_once()
    np.testing.assert_array_equal(out.mixture, fake_aug)


# ---------------------------------------------------------------------------
# Stage 1 — references unchanged
# ---------------------------------------------------------------------------


def test_rir_does_not_change_references() -> None:
    sample = _make_sample()
    refs_copy = sample.references.copy()
    cfg = AugmentationConfig(rir_prob=1.0, noise_prob=0.0)
    pipeline = AugmentationPipeline(cfg, rng=np.random.default_rng(3))
    fake_aug = np.zeros(LENGTH, dtype=np.float32)

    with patch.object(pipeline, "_apply_rir", return_value=fake_aug):
        out = pipeline(sample)

    np.testing.assert_array_equal(out.references, refs_copy)


# ---------------------------------------------------------------------------
# Stage 1 — output length preserved
# ---------------------------------------------------------------------------


def test_output_length_preserved_after_rir() -> None:
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=1.0, noise_prob=0.0)
    pipeline = AugmentationPipeline(cfg, rng=np.random.default_rng(4))
    fake_aug = np.zeros(LENGTH, dtype=np.float32)

    with patch.object(pipeline, "_apply_rir", return_value=fake_aug):
        out = pipeline(sample)

    assert len(out.mixture) == LENGTH


# ---------------------------------------------------------------------------
# Stage 2 — noise prob=0 skips
# ---------------------------------------------------------------------------


def test_noise_prob_zero_skips(tmp_path: Path) -> None:
    noise_wav = tmp_path / "noise.wav"
    _write_noise_wav(noise_wav)
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=0.0, wham_dir=tmp_path)
    out = AugmentationPipeline(cfg, rng=np.random.default_rng(5))(sample)
    np.testing.assert_array_equal(out.mixture, sample.mixture)


def test_wham_dir_none_skips_noise_regardless_of_prob() -> None:
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=1.0, wham_dir=None)
    out = AugmentationPipeline(cfg, rng=np.random.default_rng(6))(sample)
    np.testing.assert_array_equal(out.mixture, sample.mixture)


# ---------------------------------------------------------------------------
# Stage 2 — noise changes mixture
# ---------------------------------------------------------------------------


def test_noise_prob_one_changes_mixture(tmp_path: Path) -> None:
    noise_wav = tmp_path / "noise.wav"
    _write_noise_wav(noise_wav, seed=99)
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=1.0, wham_dir=tmp_path)
    out = AugmentationPipeline(cfg, rng=np.random.default_rng(7))(sample)
    assert not np.allclose(out.mixture, sample.mixture)


def test_noise_does_not_change_references(tmp_path: Path) -> None:
    noise_wav = tmp_path / "noise.wav"
    _write_noise_wav(noise_wav)
    sample = _make_sample()
    refs_copy = sample.references.copy()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=1.0, wham_dir=tmp_path)
    out = AugmentationPipeline(cfg, rng=np.random.default_rng(8))(sample)
    np.testing.assert_array_equal(out.references, refs_copy)


def test_output_length_preserved_after_noise(tmp_path: Path) -> None:
    noise_wav = tmp_path / "noise.wav"
    _write_noise_wav(noise_wav)
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=1.0, wham_dir=tmp_path)
    out = AugmentationPipeline(cfg, rng=np.random.default_rng(9))(sample)
    assert len(out.mixture) == LENGTH


# ---------------------------------------------------------------------------
# Stage 2 — SNR is within configured range
# ---------------------------------------------------------------------------


def test_snr_within_configured_range(tmp_path: Path) -> None:
    noise_wav = tmp_path / "noise.wav"
    _write_noise_wav(noise_wav, length=SR * 2)

    snr_min, snr_max = 8.0, 12.0
    sample = _make_sample()
    cfg = AugmentationConfig(
        rir_prob=0.0,
        noise_prob=1.0,
        snr_min_db=snr_min,
        snr_max_db=snr_max,
        wham_dir=tmp_path,
    )
    rng = np.random.default_rng(10)

    for _ in range(10):
        out = AugmentationPipeline(cfg, rng=rng)(sample)
        noise_component = out.mixture - sample.mixture
        sig_rms = float(np.sqrt(np.mean(sample.mixture ** 2)))
        noise_rms = float(np.sqrt(np.mean(noise_component ** 2)))
        if noise_rms < 1e-8:
            continue
        measured_snr = 20.0 * np.log10(sig_rms / noise_rms)
        assert snr_min - 0.5 <= measured_snr <= snr_max + 0.5, (
            f"SNR {measured_snr:.2f} dB outside [{snr_min}, {snr_max}] dB"
        )


# ---------------------------------------------------------------------------
# Stage 2 — short noise is tiled to match mixture length
# ---------------------------------------------------------------------------


def test_short_noise_is_tiled_to_match_length(tmp_path: Path) -> None:
    short_wav = tmp_path / "short_noise.wav"
    _write_noise_wav(short_wav, length=SR // 4)  # 0.25 s noise for 1 s mixture
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=1.0, wham_dir=tmp_path)
    out = AugmentationPipeline(cfg, rng=np.random.default_rng(11))(sample)
    assert len(out.mixture) == LENGTH


# ---------------------------------------------------------------------------
# Both stages together
# ---------------------------------------------------------------------------


def test_both_stages_apply_together(tmp_path: Path) -> None:
    noise_wav = tmp_path / "noise.wav"
    _write_noise_wav(noise_wav)
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=1.0, noise_prob=1.0, wham_dir=tmp_path)
    pipeline = AugmentationPipeline(cfg, rng=np.random.default_rng(12))
    fake_rir_out = sample.mixture * 0.8  # slightly different from original

    rir_calls = []
    noise_calls = []

    original_rir = pipeline._apply_rir
    original_noise = pipeline._apply_noise

    def _track_rir(audio, sr):
        rir_calls.append(True)
        return fake_rir_out

    def _track_noise(audio, sr):
        noise_calls.append(True)
        return original_noise(audio, sr)

    with patch.object(pipeline, "_apply_rir", side_effect=_track_rir):
        with patch.object(pipeline, "_apply_noise", side_effect=_track_noise):
            out = pipeline(sample)

    assert len(rir_calls) == 1, "RIR stage should have been called"
    assert len(noise_calls) == 1, "Noise stage should have been called"
    assert len(out.mixture) == LENGTH


# ---------------------------------------------------------------------------
# RNG reproducibility
# ---------------------------------------------------------------------------


def test_rng_reproducibility(tmp_path: Path) -> None:
    noise_wav = tmp_path / "noise.wav"
    _write_noise_wav(noise_wav)
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=1.0, wham_dir=tmp_path)

    out_a = AugmentationPipeline(cfg, rng=np.random.default_rng(99))(sample)
    out_b = AugmentationPipeline(cfg, rng=np.random.default_rng(99))(sample)
    np.testing.assert_array_equal(out_a.mixture, out_b.mixture)


# ---------------------------------------------------------------------------
# Helper: _fit_to_length
# ---------------------------------------------------------------------------


def test_fit_to_length_short_input_is_tiled() -> None:
    rng = np.random.default_rng(0)
    short = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _fit_to_length(short, 7, rng)
    assert len(out) == 7


def test_fit_to_length_long_input_is_cropped() -> None:
    rng = np.random.default_rng(0)
    long_audio = np.ones(1000, dtype=np.float32)
    out = _fit_to_length(long_audio, 100, rng)
    assert len(out) == 100


def test_fit_to_length_exact_length_unchanged() -> None:
    rng = np.random.default_rng(0)
    audio = np.ones(SR, dtype=np.float32)
    out = _fit_to_length(audio, SR, rng)
    assert len(out) == SR


# ---------------------------------------------------------------------------
# Missing WHAM! files raises clear error
# ---------------------------------------------------------------------------


def test_empty_wham_dir_raises_file_not_found(tmp_path: Path) -> None:
    sample = _make_sample()
    cfg = AugmentationConfig(rir_prob=0.0, noise_prob=1.0, wham_dir=tmp_path)
    pipeline = AugmentationPipeline(cfg, rng=np.random.default_rng(0))
    with pytest.raises(FileNotFoundError, match="No WAV files found"):
        pipeline(sample)
