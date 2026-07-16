"""Audio preprocessing for CALM-Sep (8 kHz) and legacy 16 kHz paths.

CALM-Sep operates at 8 kHz internally (STFT window 128, hop 64) with a parallel
16 kHz mixture STFT for band recovery. The legacy ``preprocess`` API remains for
older 16 kHz expert code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Legacy CA-MoSE constants (kept for backward compatibility).
PROJECT_SAMPLE_RATE = 16000
TARGET_PEAK_DBFS = -26.0
STFT_N_FFT = 512
STFT_HOP_LENGTH = 128

# CALM-Sep constants (BLUEPRINT fixed constraints).
CALMSEP_SR = 8000
OUTPUT_SR = 16000
CALMSEP_N_FFT = 128
CALMSEP_HOP = 64
BAND_N_FFT = 256
BAND_HOP = 128


@dataclass
class PreprocessedAudio:
    """Dual-branch preprocessed mixture ready for legacy 16 kHz expert inference."""

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
        return torch.from_numpy(self.waveform).to(device)

    def stft_torch(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.from_numpy(self.stft).to(device)


@dataclass
class CalmSepPreprocessed:
    """CALM-Sep dual-rate preprocessed mixture (BLUEPRINT §5.2)."""

    wav_8k: np.ndarray
    stft_8k: np.ndarray
    wav_16k: np.ndarray
    stft_16k: np.ndarray
    sample_rate_internal: int = CALMSEP_SR
    sample_rate_output: int = OUTPUT_SR
    original_length_8k: int = 0

    def __post_init__(self) -> None:
        self.wav_8k = np.asarray(self.wav_8k, dtype=np.float32).squeeze()
        self.wav_16k = np.asarray(self.wav_16k, dtype=np.float32).squeeze()
        self.stft_8k = np.asarray(self.stft_8k, dtype=np.complex64)
        self.stft_16k = np.asarray(self.stft_16k, dtype=np.complex64)
        if self.original_length_8k <= 0:
            self.original_length_8k = int(self.wav_8k.shape[0])


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
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    return signal.resample(wav, n_out).astype(np.float32)


def peak_normalize_dbfs(waveform: np.ndarray, target_dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    """Peak-normalize waveform so the absolute peak equals target_dbfs dBFS."""
    wav = np.asarray(waveform, dtype=np.float32).squeeze()
    peak = float(np.max(np.abs(wav)))
    if peak < 1e-8:
        return wav
    target_linear = 10.0 ** (target_dbfs / 20.0)
    return (wav * (target_linear / peak)).astype(np.float32)


def rms_normalize(waveform: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """RMS-normalize to a fixed target (BLUEPRINT §5.2)."""
    wav = np.asarray(waveform, dtype=np.float32).squeeze()
    rms = float(np.sqrt(np.mean(wav**2) + 1e-12))
    if rms < 1e-8:
        return wav
    return (wav * (target_rms / rms)).astype(np.float32)


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
    """Legacy 16 kHz preprocessing pipeline."""
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


def preprocess_calmsep(
    mixture: np.ndarray | torch.Tensor,
    sample_rate: int,
    target_rms: float = 0.1,
) -> CalmSepPreprocessed:
    """CALM-Sep dual-rate preprocess: 8 kHz ops + parallel 16 kHz mixture STFT."""
    wav_8k = resample_audio(mixture, sample_rate, CALMSEP_SR)
    wav_8k = rms_normalize(wav_8k, target_rms)
    wav_16k = resample_audio(mixture, sample_rate, OUTPUT_SR)
    wav_16k = rms_normalize(wav_16k, target_rms)
    stft_8k = compute_stft(wav_8k, n_fft=CALMSEP_N_FFT, hop_length=CALMSEP_HOP)
    stft_16k = compute_stft(wav_16k, n_fft=BAND_N_FFT, hop_length=BAND_HOP)
    return CalmSepPreprocessed(
        wav_8k=wav_8k,
        stft_8k=stft_8k,
        wav_16k=wav_16k,
        stft_16k=stft_16k,
        original_length_8k=int(wav_8k.shape[0]),
    )
