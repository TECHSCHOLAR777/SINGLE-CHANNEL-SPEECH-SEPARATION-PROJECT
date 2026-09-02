"""Tests for models/preprocess.py."""

import numpy as np

from coralsep.models.preprocess import (
    PROJECT_SAMPLE_RATE,
    STFT_HOP_LENGTH,
    STFT_N_FFT,
    peak_normalize_dbfs,
    preprocess,
    resample_audio,
)


def test_resample_identity_at_target_rate() -> None:
    t = np.random.randn(8000).astype(np.float32)
    out = resample_audio(t, PROJECT_SAMPLE_RATE, PROJECT_SAMPLE_RATE)
    assert np.allclose(out, t)


def test_peak_normalize_target_dbfs() -> None:
    wav = np.array([0.5, -1.0, 0.25], dtype=np.float32)
    out = peak_normalize_dbfs(wav, target_dbfs=-26.0)
    target_linear = 10.0 ** (-26.0 / 20.0)
    assert np.isclose(float(np.max(np.abs(out))), target_linear, rtol=1e-5)


def test_preprocess_output_shapes() -> None:
    sr = 8000
    t = np.random.randn(sr * 2).astype(np.float32)
    pre = preprocess(t, sample_rate=sr)
    assert pre.sample_rate == PROJECT_SAMPLE_RATE
    assert pre.waveform.ndim == 1
    # 2 s at 8 kHz → 4 s at 16 kHz after resampling
    assert pre.waveform.shape[0] == sr * 4
    assert pre.stft.ndim == 2
    assert pre.stft.shape[0] == STFT_N_FFT // 2 + 1
    assert pre.stft.shape[1] >= 1
    assert pre.stft.shape[1] * STFT_HOP_LENGTH >= pre.waveform.shape[0] - STFT_N_FFT
