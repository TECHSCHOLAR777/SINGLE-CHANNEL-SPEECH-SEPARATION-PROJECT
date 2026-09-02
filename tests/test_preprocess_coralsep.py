"""Tests for the CoRAL-Sep dual-branch preprocessor (Dev C)."""

import numpy as np
import pytest

from coralsep.models.preprocess import (
    CORALSEP_SAMPLE_RATE,
    CORALSEP_STFT_BINS,
    CORALSEP_STFT_HOP,
    CORALSEP_STFT_WIN,
    CoralSepPreprocessedAudio,
    coralsep_preprocess,
)


def _sine(duration_s: float = 2.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(duration_s * sr), dtype=np.float32) / sr
    return np.sin(2 * np.pi * 440.0 * t)


class TestCoralSepPreprocess:
    def test_output_types(self):
        wav = _sine()
        result = coralsep_preprocess(wav, 16000)
        assert isinstance(result, CoralSepPreprocessedAudio)
        assert result.waveform_8k.dtype == np.float32
        assert result.waveform_16k.dtype == np.float32

    def test_8k_sample_rate_locked(self):
        wav = _sine()
        result = coralsep_preprocess(wav, 16000)
        assert result.sample_rate_8k == CORALSEP_SAMPLE_RATE
        # Length should correspond to 8 kHz.
        expected_len = int(2.0 * CORALSEP_SAMPLE_RATE)
        assert abs(len(result.waveform_8k) - expected_len) <= 2

    def test_8k_stft_bins(self):
        wav = _sine()
        result = coralsep_preprocess(wav, 16000)
        # n_fft=128 → 65 bins
        assert result.stft_8k.shape[0] == CORALSEP_STFT_BINS
        assert result.stft_8k.dtype == np.complex64

    def test_16k_waveform_length(self):
        wav = _sine(duration_s=3.0, sr=16000)
        result = coralsep_preprocess(wav, 16000)
        expected = int(3.0 * 16000)
        assert abs(len(result.waveform_16k) - expected) <= 2

    def test_16k_stft_bins(self):
        wav = _sine()
        result = coralsep_preprocess(wav, 16000)
        # n_fft=512 → 257 bins
        assert result.stft_16k.shape[0] == 257
        assert result.stft_16k.dtype == np.complex64

    def test_input_at_native_8k(self):
        wav = _sine(sr=8000)
        result = coralsep_preprocess(wav, 8000)
        assert result.sample_rate_8k == 8000
        assert len(result.waveform_8k) > 0

    def test_waveform_8k_torch(self):
        import torch
        wav = _sine()
        result = coralsep_preprocess(wav, 16000)
        t = result.waveform_8k_torch("cpu")
        assert isinstance(t, torch.Tensor)
        assert t.ndim == 1

    def test_original_length_recorded(self):
        wav = _sine(duration_s=2.0, sr=8000)
        result = coralsep_preprocess(wav, 8000)
        expected = int(2.0 * 8000)
        assert abs(result.original_length_8k - expected) <= 2

    def test_peak_normalized(self):
        wav = _sine() * 0.001  # very quiet input
        result = coralsep_preprocess(wav, 16000)
        # Peak normalization should bring it to ~-26 dBFS.
        peak = float(np.max(np.abs(result.waveform_8k)))
        target = 10 ** (-26.0 / 20.0)
        assert abs(peak - target) < 0.01
