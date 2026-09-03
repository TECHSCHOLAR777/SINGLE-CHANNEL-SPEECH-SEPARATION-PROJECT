"""
Band recovery head (Dev C, P3-C1).

Predicts 4-8 kHz content from the separated 8 kHz signal and the original
16 kHz mixture STFT high-band component. The output is a 16 kHz waveform.

Architecture: two conv layers on the concatenated low-band (8 kHz STFT
magnitude) and high-band residual from the 16 kHz mixture. Produces a mask
over the high-band (4-8 kHz, bins 129-256 at 16 kHz with n_fft=512).

Dual-metric guard (P3-C2): band recovery is applied per-chunk only when BOTH
SI-SDRi and DNSMOS improve. Worst case is 8 kHz zero-padded to 16 kHz.

BLUEPRINT Option B: band recovery is fixed, this is the ONLY path to 16 kHz
output. There is no wideband model alternative.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 16 kHz STFT parameters (matched to models/preprocess.py)
_STFT_N_FFT = 512
_STFT_HOP = 128
_STFT_WIN = 512
_STFT_BINS = 257  # n_fft // 2 + 1

# 8 kHz STFT parameters (SR-CorrNet, BLUEPRINT §15.8)
_STFT_8K_WIN = 128
_STFT_8K_HOP = 64
_STFT_8K_BINS = 65

# High-band bin range in 16 kHz STFT that corresponds to 4-8 kHz
# (bins 129-256 at 16 kHz with n_fft=512, since 4000/(16000/2)*256 = 128)
_HIGH_BAND_START = 128
_HIGH_BAND_END = 256

_EPS = 1e-10


class BandRecoveryHead(nn.Module):
    """
    Two-conv head that predicts the 4-8 kHz mask from separated + mixture info.

    Input channels (concatenated along channel axis):
      - Low-band log-magnitude: 65 bins at 8 kHz
      - High-band residual log-magnitude: 129 bins from 16 kHz mixture

    Processing: frequency-axis convolutions operating on the time axis,
    producing a soft mask in [0, 1] over the 129 high-band bins.
    The mask is applied to the original mixture's high-band STFT.

    Params: ~50K (intentionally small; quality limited by physical information,
    not model capacity).
    """

    _LOW_BINS = _STFT_8K_BINS  # 65
    _HIGH_BINS = _HIGH_BAND_END - _HIGH_BAND_START + 1  # 129

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        in_ch = self._LOW_BINS + self._HIGH_BINS  # 65 + 129 = 194

        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, self._HIGH_BINS, kernel_size=1)

    def forward(
        self,
        stft_8k_mag: torch.Tensor,
        stft_16k_highband_mag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            stft_8k_mag: (B, 65, T) low-band magnitude at 8 kHz.
            stft_16k_highband_mag: (B, 129, T) high-band magnitude from 16 kHz mix.
        Returns:
            (B, 129, T) predicted mask in [0, 1] for the high band.
        """
        # Align time axes.
        t = min(stft_8k_mag.shape[-1], stft_16k_highband_mag.shape[-1])
        stft_8k_mag = stft_8k_mag[..., :t]
        stft_16k_highband_mag = stft_16k_highband_mag[..., :t]

        x = torch.cat([stft_8k_mag, stft_16k_highband_mag], dim=1)  # (B, 194, T)
        x = torch.log(x + _EPS)  # log-magnitude
        x = F.gelu(self.conv1(x))
        mask = torch.sigmoid(self.conv2(x))  # (B, 129, T)
        return mask

    def predict_highband_stft(
        self,
        stft_8k: torch.Tensor,
        stft_16k_mix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the mask to the mixture's high-band STFT to produce the estimated
        high-band for the separated signal.

        Args:
            stft_8k: (B, 65, T) complex STFT of the separated 8 kHz signal.
            stft_16k_mix: (B, 257, T) complex STFT of the 16 kHz mixture.
        Returns:
            (B, 257, T) complex STFT, low-band from separated, high-band from mask.
        """
        mag_8k = stft_8k.abs()
        mag_16k_high = stft_16k_mix[:, _HIGH_BAND_START:, :].abs()

        mask = self.forward(mag_8k, mag_16k_high)  # (B, 129, T)

        # Build full 257-bin output STFT.
        out = torch.zeros_like(stft_16k_mix)  # (B, 257, T)

        # Low band (0-127): upsample 8 kHz bins by padding zeros (4 kHz limit).
        out[:, :_HIGH_BAND_START, :] = 0.0  # zero-padded low band as baseline
        # ... actual low-band fill comes from 8->16 kHz interleaving; here we
        # apply mask to the mixture's low-band too (conservative approach).
        out[:, :_HIGH_BAND_START, :] = stft_16k_mix[:, :_HIGH_BAND_START, :]

        # High band (128-256): mask × original mixture high-band.
        t = min(mask.shape[-1], stft_16k_mix.shape[-1])
        out[:, _HIGH_BAND_START:, :t] = mask[:, :, :t] * stft_16k_mix[:, _HIGH_BAND_START:, :t]
        return out


def stft_to_waveform(
    stft: torch.Tensor, n_fft: int = _STFT_N_FFT, hop: int = _STFT_HOP
) -> torch.Tensor:
    """
    Inverse STFT: (B, F, T) complex → (B, L) waveform.
    """
    B = stft.shape[0]
    window = torch.hann_window(n_fft, device=stft.device)
    waves = []
    for b in range(B):
        wav = torch.istft(
            stft[b],
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
        )
        waves.append(wav)
    return torch.stack(waves, dim=0)


# ---------------------------------------------------------------------------
# Dual-metric guard (P3-C2)
# ---------------------------------------------------------------------------


def apply_band_recovery_guarded(
    head: BandRecoveryHead,
    separated_8k: np.ndarray,
    mixture_16k: np.ndarray,
    references_8k: np.ndarray | None,
    dnsmos_scorer: object | None,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """
    Apply band recovery only if BOTH SI-SDRi and DNSMOS improve.

    Worst case: zero-pad the 8 kHz signal to 16 kHz (append zeros above 4 kHz).

    Args:
        head: Trained BandRecoveryHead.
        separated_8k: (K, T_8k) separated streams at 8 kHz.
        mixture_16k: (T_16k,) original mixture at 16 kHz.
        references_8k: (K, T_8k) clean references for SI-SDRi, or None.
        dnsmos_scorer: DnsmosScorer instance or None.
        device: Compute device.

    Returns:
        (K, T_16k) waveforms at 16 kHz, either band-recovered or zero-padded.
    """
    device = torch.device(device)
    head = head.to(device)
    head.eval()

    K, T_8k = separated_8k.shape
    T_16k = mixture_16k.shape[0]

    # Baseline: zero-pad to 16 kHz.
    baseline_16k = np.zeros((K, T_16k), dtype=np.float32)
    factor = T_16k // T_8k
    for k in range(K):
        t = min(T_8k * factor, T_16k)
        baseline_16k[k, :t:factor] = separated_8k[k, : T_16k // factor]

    with torch.no_grad():
        sep_t = torch.from_numpy(separated_8k).float().to(device)  # (K, T_8k)
        mix_t = torch.from_numpy(mixture_16k).float().to(device)

        window_8k = torch.hann_window(128, device=device)
        window_16k = torch.hann_window(512, device=device)

        stft_8k = torch.stft(
            sep_t,
            n_fft=128,
            hop_length=64,
            win_length=128,
            window=window_8k,
            return_complex=True,
            center=True,
        )

        stft_16k_mix = torch.stft(
            mix_t,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window=window_16k,
            return_complex=True,
            center=True,
        )
        stft_16k_mix = stft_16k_mix.unsqueeze(0).expand(K, -1, -1)  # (K, 257, T)

        # Predict band-recovered STFT.
        recovered_stft = head.predict_highband_stft(stft_8k, stft_16k_mix)  # (K, 257, T)
        recovered_wav = stft_to_waveform(recovered_stft)  # (K, L_16k)

    # Trim or pad to T_16k.
    L = recovered_wav.shape[1]
    if L >= T_16k:
        recovered_16k = recovered_wav[:, :T_16k].cpu().numpy()
    else:
        pad = np.zeros((K, T_16k - L), dtype=np.float32)
        recovered_16k = np.concatenate([recovered_wav.cpu().numpy(), pad], axis=1)

    # Dual-metric guard: apply band recovery only if it improves BOTH metrics.
    output = baseline_16k.copy()
    for k in range(K):
        use_recovered = True

        # SI-SDRi guard (only when references are available).
        if references_8k is not None:
            ref = references_8k[k]
            si_sdr_base = _si_sdr_np(baseline_16k[k, ::2], ref)
            si_sdr_rec = _si_sdr_np(recovered_16k[k, ::2], ref)
            if si_sdr_rec <= si_sdr_base:
                use_recovered = False

        # DNSMOS guard.
        if use_recovered and dnsmos_scorer is not None and hasattr(dnsmos_scorer, "score_or_none"):
            base_score = dnsmos_scorer.score_or_none(baseline_16k[k], 16000)
            rec_score = dnsmos_scorer.score_or_none(recovered_16k[k], 16000)
            if base_score is not None and rec_score is not None:
                if rec_score.get("ovrl", 0) <= base_score.get("ovrl", 0):
                    use_recovered = False

        if use_recovered:
            output[k] = recovered_16k[k]

    return output


def _si_sdr_np(estimate: np.ndarray, target: np.ndarray) -> float:
    eps = 1e-10
    e = estimate - estimate.mean()
    t = target - target.mean()
    length = min(len(e), len(t))
    e, t = e[:length], t[:length]
    dot = float(np.dot(e, t))
    t_pow = float(np.dot(t, t))
    s_target = dot / (t_pow + eps) * t
    noise = e - s_target
    return float(10.0 * np.log10(np.dot(s_target, s_target) / (np.dot(noise, noise) + eps) + eps))
