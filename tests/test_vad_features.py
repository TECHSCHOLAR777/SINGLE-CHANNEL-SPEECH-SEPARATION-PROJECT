"""Tests for data/vad_features.py, STFT fallback without Silero."""

from __future__ import annotations

import numpy as np
import pytest

from coralsep.data import vad_features


def test_voiced_density_silence_near_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad_features, "_silero_voiced_density", lambda w, sr: None)
    wav = np.zeros(8000, dtype=np.float32)
    density = vad_features.voiced_frame_density(wav, sr=8000)
    assert 0.0 <= density <= 0.5


def test_voiced_density_active_speech_higher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad_features, "_silero_voiced_density", lambda w, sr: None)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(16000).astype(np.float32) * 0.01
    speech = np.sin(2 * np.pi * 200 * np.linspace(0, 1, 16000, endpoint=False)).astype(np.float32)
    speech = speech / (np.max(np.abs(speech)) + 1e-8)

    d_noise = vad_features.voiced_frame_density(noise, sr=8000)
    d_speech = vad_features.voiced_frame_density(speech, sr=8000)
    assert d_speech > d_noise


def test_stft_fallback_used_when_silero_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad_features, "_silero_voiced_density", lambda w, sr: None)
    wav = np.random.default_rng(1).standard_normal(8000).astype(np.float32)
    val = vad_features.voiced_frame_density(wav, sr=8000)
    assert 0.0 <= val <= 1.0


def test_validate_vad_proxy_spread() -> None:
    rng = np.random.default_rng(2)
    low = [rng.standard_normal(4000).astype(np.float32) * 0.01 for _ in range(5)]
    high = [
        np.sin(2 * np.pi * 180 * np.linspace(0, 0.5, 4000, endpoint=False)).astype(np.float32)
        for _ in range(5)
    ]
    waveforms = low + high
    overlap = [0.1] * 5 + [0.9] * 5
    report = vad_features.validate_vad_proxy(waveforms, overlap_labels=overlap, min_spread=0.01)
    assert report.n_items == 10
    assert report.max_density >= report.min_density
