"""
Audio preprocessing for CA-MoSE inference (Dev B, Phase 1).

Resamples to 16 kHz, peak-normalizes to -26 dBFS, and produces dual branches:
waveform for time-domain experts (MossFormer2) and STFT for time-frequency
experts (SR-CorrNet). All downstream modules should consume PreprocessedAudio
rather than reimplementing normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

PROJECT_SAMPLE_RATE = 16000
TARGET_PEAK_DBFS = -26.0
STFT_N_FFT = 512
STFT_HOP_LENGTH = 128


@dataclass
class PreprocessedAudio:
    """Dual-branch preprocessed mixture ready for expert inference.

    Attributes:
        waveform: Mono float32 waveform [T] at PROJECT_SAMPLE_RATE.
        stft: Complex STFT [F, frames] with n_fft=512, hop=128.
        sample_rate: Always PROJECT_SAMPLE_RATE after preprocessing.
        original_length: Sample count before any padding applied for STFT.
    """

    waveform: np.ndarray
    stft: np.ndarray
    sample_rate: int = PROJECT_SAMPLE_RATE
    original_length: int = 0

    def __post_init__(self) -> None:
        self.waveform = np.asarray(self.waveform, dtype=np.float32).squeeze()
        if self.waveform.ndim != 1:
            raise ValueError(f"waveform must be 1-D, got shape {self.waveform.shape}")
        self.stft = np.asarray(self.stft, dtype=np.complex64)
        if self.stft.ndim != 2:
            raise ValueError(f"stft must be 2-D [F, frames], got shape {self.stft.shape}")
        if self.original_length <= 0:
            self.original_length = int(self.waveform.shape[0])

    @property
    def duration_sec(self) -> float:
        return float(self.waveform.shape[0]) / float(self.sample_rate)

    def waveform_torch(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return waveform as [T] float tensor on device."""
        return torch.from_numpy(self.waveform).to(device)

    def stft_torch(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return complex STFT as [F, frames] tensor on device."""
        return torch.from_numpy(self.stft).to(device)


def resample_audio(
    audio: np.ndarray | torch.Tensor,
    orig_sr: int,
    target_sr: int = PROJECT_SAMPLE_RATE,
) -> np.ndarray:
    """Resample mono audio to target_sr, returning float32 numpy [T]."""
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32).squeeze()

    if isinstance(audio, np.ndarray):
        wav = audio.astype(np.float32).squeeze()
    else:
        wav = audio.float().squeeze().detach().cpu().numpy()
    if wav.ndim != 1:
        raise ValueError(f"Expected mono audio [T], got shape {wav.shape}")

    from scipy import signal

    n_out = int(round(wav.shape[0] * target_sr / orig_sr))
    return signal.resample(wav, n_out).astype(np.float32)


def peak_normalize_dbfs(waveform: np.ndarray, target_dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    """
    Peak-normalize waveform so the absolute peak equals target_dbfs dBFS.

    dBFS = 20 * log10(peak / full_scale). Full scale is 1.0 for float audio.
    """
    wav = np.asarray(waveform, dtype=np.float32).squeeze()
    peak = float(np.max(np.abs(wav)))
    if peak < 1e-8:
        return wav
    target_linear = 10.0 ** (target_dbfs / 20.0)
    return (wav * (target_linear / peak)).astype(np.float32)


def compute_stft(
    waveform: np.ndarray,
    n_fft: int = STFT_N_FFT,
    hop_length: int = STFT_HOP_LENGTH,
) -> np.ndarray:
    """Compute complex STFT [F, frames] from mono waveform [T]."""
    wav_t = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    spec = torch.stft(
        wav_t,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        return_complex=True,
        center=True,
    )
    return spec.numpy().astype(np.complex64)


def preprocess(
    mixture: np.ndarray | torch.Tensor,
    sample_rate: int,
    target_dbfs: float = TARGET_PEAK_DBFS,
) -> PreprocessedAudio:
    """
    Full preprocessing pipeline: resample → peak normalize → STFT branch.

    Args:
        mixture: Mono mixture [T] or [1, T].
        sample_rate: Input sample rate in Hz.
        target_dbfs: Peak normalization target (default -26 dBFS per MASTER §4.2).

    Returns:
        PreprocessedAudio with waveform and STFT branches at 16 kHz.
    """
    wav = resample_audio(mixture, sample_rate, PROJECT_SAMPLE_RATE)
    orig_len = int(wav.shape[0])
    wav = peak_normalize_dbfs(wav, target_dbfs)
    stft = compute_stft(wav)
    return PreprocessedAudio(
        waveform=wav,
        stft=stft,
        sample_rate=PROJECT_SAMPLE_RATE,
        original_length=orig_len,
    )
