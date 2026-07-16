"""Band-recovery head: 8 kHz → 16 kHz with dual-metric guard (BLUEPRINT §5.9)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.preprocess import compute_stft, resample_audio

SR_8K = 8000
SR_16K = 16000


class BandRecoveryHead(nn.Module):
    """Two-conv head predicting high-band (4–8 kHz) mask from low-band + mix HB."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        # Input: concat |S_low|, |Y_high| along channel → 2 channels over F_high x T
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=3, padding=1),
        )
        self.enabled = True

    def forward(
        self,
        low_mag: torch.Tensor,
        high_mix_mag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            low_mag: (B, F_low, T) magnitude of separated 8 kHz STFT
            high_mix_mag: (B, F_high, T) magnitude of mixture 16 kHz high band
        Returns:
            mask: (B, F_high, T) in (0, 1)
        """
        # Align time.
        t = min(low_mag.shape[-1], high_mix_mag.shape[-1])
        low = low_mag[..., :t]
        high = high_mix_mag[..., :t]
        # Upsample frequency axis of low to high-band length by interpolation.
        f_high = high.shape[-2]
        low_up = F.interpolate(low.unsqueeze(1), size=(f_high, t), mode="bilinear", align_corners=False)
        high_u = high.unsqueeze(1)
        x = torch.cat([low_up, high_u], dim=1)
        mask = torch.sigmoid(self.net(x)).mean(dim=1)  # (B, F_high, T)
        return mask


@dataclass
class BandRecoveryResult:
    waveforms_16k: np.ndarray  # (K, T16)
    applied: list[bool]
    bypass_reason: list[str]


def _istft_np(spec: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    wav = torch.istft(
        torch.from_numpy(spec.astype(np.complex64)),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        center=True,
    )
    return wav.numpy().astype(np.float32)


def zero_pad_8k_to_16k(wav_8k: np.ndarray) -> np.ndarray:
    """Pass-through: resample 8 kHz → 16 kHz (no true high-band content)."""
    return resample_audio(wav_8k, SR_8K, SR_16K)


def apply_band_recovery(
    streams_8k: np.ndarray,
    mixture_16k: np.ndarray,
    head: BandRecoveryHead,
    *,
    guard_sisdri: float | None = None,
    guard_dnsmos: float | None = None,
    references_16k: np.ndarray | None = None,
    dnsmos_fn: Callable[[np.ndarray, int], dict[str, float]] | None = None,
) -> BandRecoveryResult:
    """Recover 16 kHz streams with dual-metric per-chunk guard.

    If either SI-SDRi or DNSMOS would decrease vs pass-through, bypass that stream.
    When references/dnsmos unavailable, apply head unconditionally (training/demo).
    """
    if not head.enabled:
        out = np.stack([zero_pad_8k_to_16k(s) for s in streams_8k], axis=0)
        return BandRecoveryResult(out, [False] * len(streams_8k), ["disabled"] * len(streams_8k))

    mix_stft = compute_stft(mixture_16k, n_fft=256, hop_length=128)
    # High band bins: freqs > 4 kHz → bin > 4k / (sr/n_fft) = 4000/(16000/256)=64
    # Full bins = 129; low mirrors 65 bins at 8k. High-band ≈ bins 65..128
    hb = mix_stft[65:, :]
    hb_mag = np.abs(hb)
    hb_phase = np.angle(hb)

    recovered: list[np.ndarray] = []
    applied: list[bool] = []
    reasons: list[str] = []

    head.eval()
    with torch.no_grad():
        for i, stream in enumerate(streams_8k):
            low_stft = compute_stft(stream, n_fft=128, hop_length=64)
            low_mag = torch.from_numpy(np.abs(low_stft)).unsqueeze(0)
            high_mag = torch.from_numpy(hb_mag).unsqueeze(0)
            mask = head(low_mag, high_mag)[0].cpu().numpy()
            t = min(mask.shape[-1], hb.shape[-1])
            masked = mask[:, :t] * hb_mag[:, :t] * np.exp(1j * hb_phase[:, :t])
            # Build full 16k STFT: upsample low to 16k grid low bins + predicted high.
            low_16 = compute_stft(zero_pad_8k_to_16k(stream), n_fft=256, hop_length=128)
            full = low_16.copy()
            t2 = min(t, full.shape[-1])
            full[65 : 65 + masked.shape[0], :t2] = masked[: full.shape[0] - 65, :t2]
            cand = _istft_np(full, n_fft=256, hop=128)
            passthrough = zero_pad_8k_to_16k(stream)

            use = True
            reason = "applied"
            if references_16k is not None and guard_sisdri is not None:
                from eval.metrics import si_sdr

                ref = references_16k[i]
                n = min(len(cand), len(ref), len(passthrough))
                s_cand = si_sdr(cand[:n], ref[:n])
                s_pass = si_sdr(passthrough[:n], ref[:n])
                if s_cand + 1e-6 < s_pass + guard_sisdri:
                    use = False
                    reason = "sisdri_guard"
            if use and dnsmos_fn is not None and guard_dnsmos is not None:
                try:
                    d_cand = float(dnsmos_fn(cand, SR_16K).get("ovrl", 0.0))
                    d_pass = float(dnsmos_fn(passthrough, SR_16K).get("ovrl", 0.0))
                    if d_cand + 1e-6 < d_pass + guard_dnsmos:
                        use = False
                        reason = "dnsmos_guard"
                except Exception:
                    pass

            recovered.append(cand if use else passthrough)
            applied.append(use)
            reasons.append(reason)

    # Align lengths.
    max_len = max(len(w) for w in recovered)
    arr = np.stack(
        [np.pad(w, (0, max_len - len(w))) if len(w) < max_len else w[:max_len] for w in recovered],
        axis=0,
    ).astype(np.float32)
    return BandRecoveryResult(arr, applied, reasons)
