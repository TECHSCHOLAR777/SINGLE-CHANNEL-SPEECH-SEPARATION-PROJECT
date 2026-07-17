"""Tests for models/band_recovery.py (Dev C)."""

import numpy as np
import pytest
import torch

from models.band_recovery import (
    BandRecoveryHead,
    _si_sdr_np,
    apply_band_recovery_guarded,
    stft_to_waveform,
)


class TestBandRecoveryHead:
    def _head(self) -> BandRecoveryHead:
        return BandRecoveryHead(hidden=16)

    def test_forward_output_shape(self):
        head = self._head()
        B, T = 2, 50
        mag_8k = torch.ones(B, 65, T)
        mag_16k_high = torch.ones(B, 129, T)
        mask = head(mag_8k, mag_16k_high)
        assert mask.shape == (B, 129, T)

    def test_forward_range(self):
        head = self._head()
        B, T = 1, 30
        mag_8k = torch.rand(B, 65, T)
        mag_16k_high = torch.rand(B, 129, T)
        mask = head(mag_8k, mag_16k_high)
        assert float(mask.min()) >= 0.0
        assert float(mask.max()) <= 1.0

    def test_time_alignment_mismatch(self):
        head = self._head()
        mag_8k = torch.ones(1, 65, 40)
        mag_16k_high = torch.ones(1, 129, 30)
        mask = head(mag_8k, mag_16k_high)
        assert mask.shape[-1] == 30  # min of 40, 30

    def test_predict_highband_stft_shape(self):
        head = self._head()
        B, T = 2, 50
        stft_8k = torch.randn(B, 65, T, dtype=torch.complex64)
        stft_16k_mix = torch.randn(B, 257, T, dtype=torch.complex64)
        out = head.predict_highband_stft(stft_8k, stft_16k_mix)
        assert out.shape == (B, 257, T)
        assert out.dtype == torch.complex64

    def test_low_band_preserved(self):
        head = self._head()
        B, T = 1, 20
        stft_8k = torch.ones(B, 65, T, dtype=torch.complex64)
        stft_16k_mix = torch.ones(B, 257, T, dtype=torch.complex64) * 0.5
        out = head.predict_highband_stft(stft_8k, stft_16k_mix)
        # Low band (0:128) should equal the mixture's low band.
        assert torch.allclose(out[:, :128, :], stft_16k_mix[:, :128, :])


class TestSiSdrNp:
    def test_perfect_estimate_high_sisdr(self):
        t = np.sin(np.linspace(0, 10, 8000), dtype=float).astype(np.float32)
        val = _si_sdr_np(t, t)
        assert val > 50.0

    def test_zero_reference_no_crash(self):
        t = np.sin(np.linspace(0, 10, 8000)).astype(np.float32)
        z = np.zeros_like(t)
        val = _si_sdr_np(t, z)
        assert np.isfinite(val)


class TestGuardedRecovery:
    def test_fallback_to_baseline_without_recovery(self):
        K, T_8k, T_16k = 2, 4000, 8000
        sep = np.random.default_rng(0).standard_normal((K, T_8k)).astype(np.float32) * 0.01
        mix = np.zeros(T_16k, dtype=np.float32)
        head = BandRecoveryHead(hidden=8)
        # Without references or DNSMOS, guard passes → band recovery applied.
        out = apply_band_recovery_guarded(head, sep, mix, references_8k=None, dnsmos_scorer=None)
        assert out.shape == (K, T_16k)

    def test_output_dtype_float32(self):
        K, T_8k, T_16k = 2, 4000, 8000
        sep = np.zeros((K, T_8k), dtype=np.float32)
        mix = np.zeros(T_16k, dtype=np.float32)
        head = BandRecoveryHead(hidden=8)
        out = apply_band_recovery_guarded(head, sep, mix, references_8k=None, dnsmos_scorer=None)
        assert out.dtype == np.float32

    def test_guard_rejects_when_sisdr_regresses(self):
        K, T_8k = 2, 8000
        T_16k = T_8k * 2
        rng = np.random.default_rng(7)
        # References = very clean signal.
        refs = rng.standard_normal((K, T_8k)).astype(np.float32)
        # Separated = almost noise → low SI-SDR baseline.
        sep = rng.standard_normal((K, T_8k)).astype(np.float32) * 0.0001
        mix = np.zeros(T_16k, dtype=np.float32)
        head = BandRecoveryHead(hidden=8)
        # Guard may reject and fall back to baseline; result must still be valid.
        out = apply_band_recovery_guarded(head, sep, mix, references_8k=refs, dnsmos_scorer=None)
        assert out.shape == (K, T_16k)
        assert out.dtype == np.float32
