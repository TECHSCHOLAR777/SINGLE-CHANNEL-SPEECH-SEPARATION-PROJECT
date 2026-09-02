"""
Shared loss functions for CoRAL-Sep adapter training (Dev B, P1-B3).

Reuses the backbone engine losses where possible:
  primary: PIT SI-SNR on waveforms (time domain)
  secondary: 0.5 × PIT SI-SNR on magnitude STFT (frequency domain)
  attractor: BCE on pres["logits"] (shape 1,7)

Cardinality-aware extension (BLUEPRINT §8.2):
  Missed speakers score 0 dB. Hallucinated streams incur -1 dB per stream.
  Applied at PIT selection time: the PIT matrix is extended with zero-columns
  for missing references so the Hungarian assignment can map a stream to silence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-10
_HALL_PENALTY_DB = -1.0
_STFT_WIN = 128
_STFT_HOP = 64


# ---------------------------------------------------------------------------
# SI-SNR
# ---------------------------------------------------------------------------


def si_snr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Scale-invariant SNR, shape-agnostic.

    Args:
        estimate: [..., T_e]
        target:   [..., T_r]  (may differ from T_e due to STFT/iSTFT boundary)
    Returns:
        [...] SI-SNR in dB.
    """
    min_t = min(estimate.shape[-1], target.shape[-1])
    estimate = estimate[..., :min_t]
    target = target[..., :min_t]
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    s_target = dot / (target.pow(2).sum(dim=-1, keepdim=True) + _EPS) * target
    noise = estimate - s_target
    return 10.0 * torch.log10(s_target.pow(2).sum(-1) / (noise.pow(2).sum(-1) + _EPS) + _EPS)


def pit_si_snr(
    estimates: torch.Tensor, references: torch.Tensor
) -> tuple[torch.Tensor, list[list[int]]]:
    """
    Permutation-Invariant Training SI-SNR.

    Args:
        estimates:  (B, K_hat, T) separated streams.
        references: (B, K_ref, T) clean references.
    Returns:
        mean_sisnr: (B,) mean SI-SNR after optimal permutation.
        perms: List of B permutation lists mapping K_hat → K_ref.
    """
    B, K_hat, T_est = estimates.shape
    K_ref, T_ref = references.shape[1], references.shape[2]
    T = min(T_est, T_ref)  # align lengths: iSTFT can differ by a few samples
    estimates = estimates[..., :T]
    references = references[..., :T]
    K = max(K_hat, K_ref)

    # Pad smaller tensor to K.
    if K_hat < K:
        pad = torch.zeros(B, K - K_hat, T, device=estimates.device)
        estimates_padded = torch.cat([estimates, pad], dim=1)
    else:
        estimates_padded = estimates

    if K_ref < K:
        pad = torch.zeros(B, K - K_ref, T, device=references.device)
        refs_padded = torch.cat([references, pad], dim=1)
    else:
        refs_padded = references

    # Build cost matrix [B, K, K].
    cost = torch.zeros(B, K, K, device=estimates.device)
    for i in range(K):
        for j in range(K):
            cost[:, i, j] = -si_snr(estimates_padded[:, i], refs_padded[:, j])

    # Hungarian via scipy (CPU only for now — small K).
    # IMPORTANT: use -cost values from the pre-computed PyTorch cost matrix
    # rather than calling si_snr() again, so gradient flows through cost → estimates.
    from scipy.optimize import linear_sum_assignment

    perms: list[list[int]] = []
    sisnr_per_sample: list[torch.Tensor] = []

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b].detach().cpu().numpy())
        perm = list(col_ind[:K_hat])
        perms.append(perm)
        n_pairs = min(K_hat, K_ref)
        # Keep values as tensors (NOT .item()) so backward can flow through them.
        matched = torch.stack([-cost[b, row_ind[i], col_ind[i]] for i in range(n_pairs)])
        n_hall = max(0, K_hat - K_ref)
        sisnr_per_sample.append(matched.mean() + n_hall * _HALL_PENALTY_DB)

    sisnr_vals = torch.stack(sisnr_per_sample)  # (B,) with grad_fn
    return sisnr_vals, perms


# ---------------------------------------------------------------------------
# STFT magnitude loss
# ---------------------------------------------------------------------------


def _mag_stft(waveform: torch.Tensor) -> torch.Tensor:
    """[B, T] → [B, F, frames] magnitude STFT at 8 kHz SR-CorrNet params."""
    B, T = waveform.shape
    window = torch.hann_window(_STFT_WIN, device=waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=_STFT_WIN,
        hop_length=_STFT_HOP,
        win_length=_STFT_WIN,
        window=window,
        return_complex=True,
        center=True,
    )
    return spec.abs()  # [B, F, frames]


def pit_si_snr_mag(estimates: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """PIT SI-SNR on magnitude STFT, averaged over B. Returns scalar."""
    B, K_hat, T = estimates.shape
    K_ref = references.shape[1]

    # Flatten to [B*K, T] for batch STFT.
    est_mag = _mag_stft(estimates.reshape(B * K_hat, T)).reshape(B, K_hat, -1)
    ref_mag = _mag_stft(references.reshape(B * K_ref, T)).reshape(B, K_ref, -1)

    sisnr_vals, _ = pit_si_snr(est_mag, ref_mag)
    return sisnr_vals.mean()


# ---------------------------------------------------------------------------
# Attractor BCE
# ---------------------------------------------------------------------------


def attractor_bce(logits: torch.Tensor, n_speakers: list[int]) -> torch.Tensor:
    """
    BCE on attractor presence logits (shape B, 7 or 1, 7).

    Args:
        logits: Raw logits from pres["logits"], shape (B, 7) or (B, 1, 7).
        n_speakers: True speaker count per sample.
    Returns:
        Scalar BCE loss.
    """
    logits = logits.squeeze(1) if logits.ndim == 3 else logits  # (B, 7)
    targets = torch.zeros_like(logits)
    for b, n in enumerate(n_speakers):
        # Slots 1..n are active.
        targets[b, 1 : n + 1] = 1.0
    return F.binary_cross_entropy_with_logits(logits, targets)


# ---------------------------------------------------------------------------
# Combined training loss
# ---------------------------------------------------------------------------


def coralsep_loss(
    estimates: torch.Tensor,
    references: torch.Tensor,
    logits: torch.Tensor | None,
    n_speakers: list[int],
    mag_weight: float = 0.5,
    att_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """
    Combined CoRAL-Sep separation loss.

    = PIT_SISNR_time + mag_weight × PIT_SISNR_mag + att_weight × BCE(attractor)

    Args:
        estimates:  (B, K_hat, T) separated waveforms.
        references: (B, K_ref, T) clean reference waveforms.
        logits:     (B, 7) or None. If None, attractor loss is skipped.
        n_speakers: True count per sample.
        mag_weight: Weight on STFT magnitude loss.
        att_weight: Weight on attractor BCE.

    Returns:
        Dict with 'total', 'time', 'mag', 'att' scalar tensors.
    """
    # Align time axis: iSTFT may produce ±few samples vs the reference
    min_t = min(estimates.shape[-1], references.shape[-1])
    estimates = estimates[..., :min_t]
    references = references[..., :min_t]

    sisnr_vals, _ = pit_si_snr(estimates, references)
    loss_time = -sisnr_vals.mean()
    loss_mag = -pit_si_snr_mag(estimates, references)

    total = loss_time + mag_weight * loss_mag

    loss_att = torch.tensor(0.0, device=estimates.device)
    if logits is not None:
        loss_att = attractor_bce(logits, n_speakers)
        total = total + att_weight * loss_att

    return {"total": total, "time": loss_time, "mag": loss_mag, "att": loss_att}
