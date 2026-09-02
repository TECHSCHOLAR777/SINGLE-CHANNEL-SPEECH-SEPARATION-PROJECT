"""
Audio preprocessing for CoRAL-Sep inference (Dev C, P0-C2).

Produces a dual-branch preprocessed audio object:
  Branch 1 (8 kHz): waveform + STFT(win=128, hop=64, F=65) for SR-CorrNet.
  Branch 2 (16 kHz): mixture STFT only, for the band-recovery module.

The 8 kHz branch is the primary inference branch; 16 kHz is used only by
band_recovery.py to predict 4-8 kHz content. This matches BLUEPRINT §5.2.

Backward compatible: `preprocess()` still returns `PreprocessedAudio` at 16 kHz
for legacy callers. `coralsep_preprocess()` returns `CoralSepPreprocessedAudio`
with both branches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

PROJECT_SAMPLE_RATE = 16000
TARGET_PEAK_DBFS = -26.0
STFT_N_FFT = 512
STFT_HOP_LENGTH = 128

# CoRAL-Sep SR-CorrNet 8 kHz branch (BLUEPRINT fixed constraints)
CORALSEP_SAMPLE_RATE = 8_000
CORALSEP_STFT_WIN = 128
CORALSEP_STFT_HOP = 64
CORALSEP_STFT_BINS = 65  # n_fft//2 + 1 = 65


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


# ---------------------------------------------------------------------------
# CoRAL-Sep dual-branch preprocessor (Dev C, P0-C2)
# ---------------------------------------------------------------------------


@dataclass
class CoralSepPreprocessedAudio:
    """
    Dual-branch preprocessed mixture for the CoRAL-Sep inference pipeline.

    Attributes:
        waveform_8k: [T_8k] float32 at 8 kHz — fed directly to SR-CorrNet.
        stft_8k: [65, frames] complex64 STFT at 8 kHz (win=128, hop=64).
        waveform_16k: [T_16k] float32 at 16 kHz — for band recovery & DNSMOS.
        stft_16k: [257, frames] complex64 STFT at 16 kHz (win=512, hop=128).
        sample_rate_8k: Always CORALSEP_SAMPLE_RATE (8000).
        sample_rate_16k: Always PROJECT_SAMPLE_RATE (16000).
        original_length_8k: Sample count before padding (at 8 kHz).
    """

    waveform_8k: np.ndarray
    stft_8k: np.ndarray
    waveform_16k: np.ndarray
    stft_16k: np.ndarray
    sample_rate_8k: int = CORALSEP_SAMPLE_RATE
    sample_rate_16k: int = PROJECT_SAMPLE_RATE
    original_length_8k: int = 0

    def __post_init__(self) -> None:
        self.waveform_8k = np.asarray(self.waveform_8k, dtype=np.float32).squeeze()
        self.waveform_16k = np.asarray(self.waveform_16k, dtype=np.float32).squeeze()
        if self.original_length_8k <= 0:
            self.original_length_8k = int(self.waveform_8k.shape[0])

    def waveform_8k_torch(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """[T_8k] float tensor for SR-CorrNet input."""
        return torch.from_numpy(self.waveform_8k).to(device)

    def stft_8k_torch(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """[65, frames] complex tensor for condition analysis."""
        return torch.from_numpy(self.stft_8k).to(device)


def coralsep_preprocess(
    mixture: np.ndarray | torch.Tensor,
    sample_rate: int,
    target_dbfs: float = TARGET_PEAK_DBFS,
) -> CoralSepPreprocessedAudio:
    """
    CoRAL-Sep dual-branch preprocessing.

    Produces both the 8 kHz branch (for SR-CorrNet) and the 16 kHz branch
    (for band recovery and DNSMOS) from a single input mixture.

    Args:
        mixture: Mono mixture [T] or [1, T].
        sample_rate: Input sample rate in Hz.
        target_dbfs: Peak normalization target.

    Returns:
        CoralSepPreprocessedAudio with both branches.
    """
    wav8 = resample_audio(mixture, sample_rate, CORALSEP_SAMPLE_RATE)
    orig_len_8k = int(wav8.shape[0])
    wav8 = peak_normalize_dbfs(wav8, target_dbfs)
    stft8 = compute_stft(wav8, n_fft=CORALSEP_STFT_WIN, hop_length=CORALSEP_STFT_HOP)

    wav16 = resample_audio(mixture, sample_rate, PROJECT_SAMPLE_RATE)
    wav16 = peak_normalize_dbfs(wav16, target_dbfs)
    stft16 = compute_stft(wav16, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH)

    return CoralSepPreprocessedAudio(
        waveform_8k=wav8,
        stft_8k=stft8,
        waveform_16k=wav16,
        stft_16k=stft16,
        original_length_8k=orig_len_8k,
    )
