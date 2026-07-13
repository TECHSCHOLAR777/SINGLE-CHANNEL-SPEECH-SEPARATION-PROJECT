"""
Composite training losses for CA-MoSE (Dev B, Phase 2).

Assembles all seven loss terms from MASTER_PROJECT section 7.2:
SI-SDR-uPIT, multi-resolution STFT, count BCE, router load-balance,
null-sparsity, residual regularization, and speaker-consistency (ArcFace).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.router import load_balance_loss, null_sparsity_loss

_EPS = 1e-8


@dataclass
class LossWeights:
    """MASTER §7.2 starting lambda values."""

    si_sdr_upit: float = 1.0
    multi_res_stft: float = 0.5
    count_bce: float = 0.3
    load_balance: float = 0.1
    null_sparsity: float = 0.1
    residual_reg: float = 0.1
    speaker_consistency: float = 0.1


@dataclass
class LossBreakdown:
    """Per-term unweighted and weighted losses for logging."""

    total: torch.Tensor
    si_sdr_upit: torch.Tensor
    multi_res_stft: torch.Tensor
    count_bce: torch.Tensor
    load_balance: torch.Tensor
    null_sparsity: torch.Tensor
    residual_reg: torch.Tensor
    speaker_consistency: torch.Tensor
    weighted: dict[str, float] = field(default_factory=dict)


def neg_si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Negative scale-invariant SDR (differentiable) for one stream pair.

    Args:
        estimate: [T] or [B, T].
        reference: Same shape as estimate.

    Returns:
        Scalar or [B] negative SI-SDR in dB (minimize to maximize SI-SDR).
    """
    est = estimate - estimate.mean(dim=-1, keepdim=True)
    ref = reference - reference.mean(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True).clamp_min(_EPS)
    scale = (est * ref).sum(dim=-1, keepdim=True) / ref_energy
    target = scale * ref
    residual = est - target
    num = (target * target).sum(dim=-1)
    den = (residual * residual).sum(dim=-1).clamp_min(_EPS)
    si_sdr_db = 10.0 * torch.log10((num + _EPS) / den)
    return -si_sdr_db


def pit_si_sdr_loss(
    estimates: torch.Tensor,
    references: torch.Tensor,
    true_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Utterance-level permutation-invariant negative SI-SDR loss.

    Enumerates permutations (feasible for K <= 5) and picks the assignment
    with lowest loss per batch item. Permutation selection is detached so
    gradients flow only through the chosen assignment.

    Args:
        estimates: [B, K_est, T].
        references: [B, K_ref, T].
        true_counts: Optional [B] real speaker count per item. In a mixed-N
            batch, references beyond a sample's true count are zero pads; scoring
            them explodes SI-SDR (energy ~0 -> huge negative dB). When provided,
            each item is scored only over its first true_count speakers.

    Returns:
        Scalar mean loss across the batch.
    """
    if estimates.ndim != 3 or references.ndim != 3:
        raise ValueError("estimates and references must be [B, K, T]")
    b, k_est, t = estimates.shape
    k_ref = references.shape[1]
    if references.shape[0] != b or references.shape[2] != t:
        raise ValueError("batch/time dimensions must match between estimates and references")

    k_full = min(k_est, k_ref)
    losses: list[torch.Tensor] = []
    for batch_idx in range(b):
        k = k_full
        if true_counts is not None:
            k = int(min(int(true_counts[batch_idx].item()), k_full))
            k = max(k, 1)
        est = estimates[batch_idx, :k]
        ref = references[batch_idx, :k]
        best_perm: tuple[int, ...] | None = None
        best_val = float("inf")
        for perm in itertools.permutations(range(k)):
            perm_loss = torch.stack([neg_si_sdr(est[perm[i]], ref[i]) for i in range(k)]).mean()
            val = float(perm_loss.detach().item())
            if val < best_val:
                best_val = val
                best_perm = perm
        assert best_perm is not None
        chosen = torch.stack([neg_si_sdr(est[best_perm[i]], ref[i]) for i in range(k)]).mean()
        losses.append(chosen)
    return torch.stack(losses).mean()


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT magnitude L1 loss (P2-B7).

    Penalizes spectral leakage in quiet regions that SI-SDR underweights.
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: tuple[int, ...] = (128, 256, 512),
        win_lengths: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes)):
            raise ValueError("fft_sizes and hop_sizes must have equal length")
        if win_lengths is None:
            win_lengths = fft_sizes
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        window = torch.hann_window(win, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=True,
            return_complex=True,
        )
        return spec.abs()

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate: [B, K, T] or [B, T].
            reference: Same shape as estimate.

        Returns:
            Scalar mean L1 magnitude loss across resolutions.
        """
        if estimate.shape != reference.shape:
            raise ValueError("estimate and reference shapes must match")
        if estimate.ndim == 3:
            b, k, t = estimate.shape
            estimate = estimate.reshape(b * k, t)
            reference = reference.reshape(b * k, t)

        losses: list[torch.Tensor] = []
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths, strict=True):
            if estimate.shape[-1] < n_fft:
                continue
            est_mag = self._stft_mag(estimate, n_fft, hop, win)
            ref_mag = self._stft_mag(reference, n_fft, hop, win)
            min_frames = min(est_mag.shape[-1], ref_mag.shape[-1])
            losses.append(F.l1_loss(est_mag[..., :min_frames], ref_mag[..., :min_frames]))
        if not losses:
            return estimate.sum() * 0.0
        return torch.stack(losses).mean()


def count_bce_loss(count_logits: torch.Tensor, true_counts: torch.Tensor) -> torch.Tensor:
    """
    Coarse speaker-count BCE loss (P2-B5 / MASTER count term).

    Treats count prediction as K-class classification over {1..max_speakers}.
    true_counts are 1-indexed speaker counts (2 -> index 1 for 2 speakers).
    """
    if count_logits.ndim != 2:
        raise ValueError("count_logits must be [B, max_speakers]")
    max_speakers = count_logits.shape[1]
    targets = (true_counts.long().clamp(min=1, max=max_speakers) - 1).clamp(0, max_speakers - 1)
    return F.cross_entropy(count_logits, targets)


def residual_regularization_loss(residual: torch.Tensor) -> torch.Tensor:
    """
    L2 penalty on fusion residual magnitude (P2-B4).

    Keeps corrections small on clean audio so fusion does not override
    high-quality SR-CorrNet outputs.
    """
    return (residual * residual).mean()


class SpeakerConsistencyLoss(nn.Module):
    """
    ArcFace-style angular-margin speaker embedding loss (P2-B8).

    After uPIT assignment, pushes matched-stream embeddings toward their
    reference speaker identity and apart from other speakers in the batch.
    """

    def __init__(self, embedding_dim: int = 64, margin: float = 0.35, scale: float = 30.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        stream_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            stream_embeddings: [B, K, D] separated-stream embeddings.
            reference_embeddings: [B, K, D] ground-truth speaker embeddings.

        Returns:
            Scalar ArcFace-style classification loss.
        """
        if stream_embeddings.shape != reference_embeddings.shape:
            raise ValueError("stream and reference embedding shapes must match")
        b, k, d = stream_embeddings.shape
        if k < 2:
            return stream_embeddings.sum() * 0.0

        z = F.normalize(self.projection(stream_embeddings.reshape(b * k, d)), dim=-1)
        y = F.normalize(reference_embeddings.reshape(b * k, d), dim=-1)

        logits = self.scale * (z @ y.T)
        labels = torch.arange(b * k, device=z.device)
        logits_m = logits.clone()
        logits_m[torch.arange(b * k, device=z.device), labels] -= self.scale * self.margin
        return F.cross_entropy(logits_m, labels)


class CompositeLoss(nn.Module):
    """
    Weighted sum of all seven CA-MoSE training losses (P2-B5).
    """

    def __init__(self, weights: LossWeights | None = None, embedding_dim: int = 192) -> None:
        """
        Args:
            embedding_dim: Dimensionality of the stream/reference speaker
                embeddings fed to the speaker-consistency loss. Defaults to
                192, the real output size of SpeechBrain's ECAPA-TDNN
                (speechbrain/spkrec-ecapa-voxceleb) — the embedder actually
                used by MossFormer2Expert/ECAPAEmbedder. Must match whatever
                embedder produces stream_embeddings/reference_embeddings, or
                the projection layer's matmul raises a shape mismatch (as it
                did against the old hardcoded default of 64 on the first real
                Kaggle training run — CI's synthetic 64-dim fakes never
                caught it).
        """
        super().__init__()
        self.weights = weights or LossWeights()
        self.mrstft = MultiResolutionSTFTLoss()
        self.speaker_loss = SpeakerConsistencyLoss(embedding_dim=embedding_dim)

    def forward(
        self,
        *,
        estimates: torch.Tensor,
        references: torch.Tensor,
        count_logits: torch.Tensor,
        true_counts: torch.Tensor,
        router_weights: torch.Tensor,
        trivial_mask: torch.Tensor,
        null_index: int,
        fusion_residual: torch.Tensor | None = None,
        stream_embeddings: torch.Tensor | None = None,
        reference_embeddings: torch.Tensor | None = None,
    ) -> LossBreakdown:
        """
        Compute the full composite objective.

        Args:
            estimates: Fused or primary output [B, K, T].
            references: Ground-truth stems [B, K, T].
            count_logits: Scene analyzer coarse count [B, max_speakers].
            true_counts: True speaker counts [B] (values in {2,3,4,5}).
            router_weights: Router output [B, S, E].
            trivial_mask: Per-segment triviality targets [B, S].
            null_index: Null expert index for sparsity loss.
            fusion_residual: Fusion correction [B, K, T] when escalated.
            stream_embeddings: Optional [B, K, D] for speaker consistency.
            reference_embeddings: Optional [B, K, D] matched references.

        Returns:
            LossBreakdown with total weighted loss and per-term values.
        """
        l_si_sdr = pit_si_sdr_loss(estimates, references, true_counts=true_counts)
        l_mrstft = self.mrstft(estimates, references)
        l_count = count_bce_loss(count_logits, true_counts)
        l_lb = load_balance_loss(router_weights)
        l_null = null_sparsity_loss(router_weights, trivial_mask, null_index)
        l_res = (
            residual_regularization_loss(fusion_residual)
            if fusion_residual is not None
            else estimates.sum() * 0.0
        )
        if stream_embeddings is not None and reference_embeddings is not None:
            l_spk = self.speaker_loss(stream_embeddings, reference_embeddings)
        else:
            l_spk = estimates.sum() * 0.0

        w = self.weights
        total = (
            w.si_sdr_upit * l_si_sdr
            + w.multi_res_stft * l_mrstft
            + w.count_bce * l_count
            + w.load_balance * l_lb
            + w.null_sparsity * l_null
            + w.residual_reg * l_res
            + w.speaker_consistency * l_spk
        )

        weighted = {
            "si_sdr_upit": float((w.si_sdr_upit * l_si_sdr).detach().item()),
            "multi_res_stft": float((w.multi_res_stft * l_mrstft).detach().item()),
            "count_bce": float((w.count_bce * l_count).detach().item()),
            "load_balance": float((w.load_balance * l_lb).detach().item()),
            "null_sparsity": float((w.null_sparsity * l_null).detach().item()),
            "residual_reg": float((w.residual_reg * l_res).detach().item()),
            "speaker_consistency": float((w.speaker_consistency * l_spk).detach().item()),
        }

        return LossBreakdown(
            total=total,
            si_sdr_upit=l_si_sdr,
            multi_res_stft=l_mrstft,
            count_bce=l_count,
            load_balance=l_lb,
            null_sparsity=l_null,
            residual_reg=l_res,
            speaker_consistency=l_spk,
            weighted=weighted,
        )
