"""Two-level condition analyzer for CALM-Sep (BLUEPRINT §5.4).

Level 1: raw STFT DSP (SNR, codec family/bitrate, voiced density) — no training.
Level 2: E(0) heads for T60 and speaker-count prior — trained in Stage 3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.vad_features import voiced_frame_density

CALMSEP_SR = 8000
N_COUNT_CLASSES = 4  # N in {2,3,4,5}
CODEC_FAMILIES = ("none", "opus", "aac", "amr-nb", "amr-wb")


@dataclass
class ConditionVector:
    """Inspectable condition embedding (BLUEPRINT §5.4)."""

    snr_db: float = 40.0
    t60_s: float = 0.0
    codec_class: str = "none"
    codec_class_idx: int = 0
    codec_bitrate_bps: float = 0.0
    voiced_density: float = 0.0
    count_prior: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])

    def to_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Flat feature vector for the gate MLP: [snr, t60, codec_idx, bitrate, voiced, *count_prior]."""
        vals = [
            self.snr_db,
            self.t60_s,
            float(self.codec_class_idx),
            self.codec_bitrate_bps / 48_000.0,
            self.voiced_density,
            *self.count_prior,
        ]
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stft_mag(wav: np.ndarray, n_fft: int = 128, hop: int = 64) -> np.ndarray:
    wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    spec = torch.stft(
        wav_t,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        return_complex=True,
        center=True,
    )
    return spec.abs().numpy()


def estimate_snr_db(wav: np.ndarray, sr: int = CALMSEP_SR) -> float:
    """Voiced-frame mean energy over noise-floor mean energy (Level-1)."""
    mag = _stft_mag(wav)
    frame_e = (mag**2).mean(axis=0)
    if frame_e.size == 0:
        return 40.0
    thr = float(np.percentile(frame_e, 30.0))
    voiced = frame_e >= max(thr, 1e-10)
    noise = ~voiced
    if not voiced.any() or not noise.any():
        # Fallback: top/bottom energy ratio.
        hi = float(np.percentile(frame_e, 90.0))
        lo = float(np.percentile(frame_e, 10.0))
        return float(10.0 * np.log10((hi + 1e-10) / (lo + 1e-10)))
    snr = 10.0 * np.log10((frame_e[voiced].mean() + 1e-10) / (frame_e[noise].mean() + 1e-10))
    return float(np.clip(snr, -10.0, 60.0))


def estimate_codec(wav: np.ndarray, sr: int = CALMSEP_SR) -> tuple[str, int, float]:
    """Spectral bandwidth heuristic → (family, class_idx, bitrate_bps)."""
    mag = _stft_mag(wav)
    # Frequency axis for 8 kHz / 128-FFT: bin k → k * sr/n_fft
    n_fft = 128
    freqs = np.arange(mag.shape[0]) * (sr / n_fft)
    band_e = mag.mean(axis=1)
    total = float(band_e.sum() + 1e-10)
    cum = np.cumsum(band_e) / total
    # Bandwidth containing 95% energy.
    idx = int(np.searchsorted(cum, 0.95))
    bw = float(freqs[min(idx, len(freqs) - 1)])
    # Hard cutoffs suggest codec family.
    if bw < 3500:
        return "amr-nb", CODEC_FAMILIES.index("amr-nb"), 12_200.0
    if bw < 5500:
        # Could be AMR-WB or low Opus.
        if band_e[-3:].mean() < band_e.mean() * 0.05:
            return "opus", CODEC_FAMILIES.index("opus"), 12_000.0
        return "amr-wb", CODEC_FAMILIES.index("amr-wb"), 23_850.0
    if bw < 7000 and band_e[-2:].mean() < 1e-4:
        return "aac", CODEC_FAMILIES.index("aac"), 32_000.0
    return "none", 0, 0.0


class ReverbHead(nn.Module):
    """Attention-pooled 1-D CNN over time-averaged E(0) → T60 seconds."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.attn = nn.Linear(64, 1)
        self.out = nn.Linear(64, 1)

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        # e0: (B, T, F, C) → pool F → (B, C, T)
        x = e0.mean(dim=2).transpose(1, 2)
        h = self.conv(x).transpose(1, 2)  # (B, T, 64)
        w = torch.softmax(self.attn(h), dim=1)
        pooled = (w * h).sum(dim=1)
        t60 = F.softplus(self.out(pooled)).squeeze(-1)
        return t60


class CountPriorHead(nn.Module):
    """Soft classification over N∈{2,3,4,5} from pooled E(0)+SNR+voiced."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 2, 128),
            nn.GELU(),
            nn.Linear(128, N_COUNT_CLASSES),
        )

    def forward(self, e0: torch.Tensor, snr_db: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
        pooled = e0.mean(dim=(1, 2))  # (B, C)
        feats = torch.cat([pooled, snr_db.view(-1, 1), voiced.view(-1, 1)], dim=-1)
        return self.mlp(feats)


class ConditionAnalyzer(nn.Module):
    """Two-level condition analyzer."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.reverb_head = ReverbHead(d_model)
        self.count_head = CountPriorHead(d_model)

    def forward_level1(self, wav: np.ndarray, sr: int = CALMSEP_SR) -> dict[str, Any]:
        snr = estimate_snr_db(wav, sr)
        family, idx, bitrate = estimate_codec(wav, sr)
        voiced = float(voiced_frame_density(wav, sr))
        return {
            "snr_db": snr,
            "codec_class": family,
            "codec_class_idx": idx,
            "codec_bitrate_bps": bitrate,
            "voiced_density": voiced,
        }

    def forward_level2(
        self,
        e0: torch.Tensor,
        level1: dict[str, Any],
    ) -> dict[str, Any]:
        if e0.dim() != 4:
            raise ValueError(f"e0 must be (B,T,F,C), got {tuple(e0.shape)}")
        device = e0.device
        snr = torch.tensor([level1["snr_db"]], dtype=torch.float32, device=device)
        voiced = torch.tensor([level1["voiced_density"]], dtype=torch.float32, device=device)
        t60 = self.reverb_head(e0)
        logits = self.count_head(e0, snr, voiced)
        prior = torch.softmax(logits, dim=-1)
        return {
            "t60_s": float(t60[0].detach().cpu()),
            "count_prior": prior[0].detach().cpu().tolist(),
            "count_logits": logits,
            "t60_tensor": t60,
        }

    def forward(
        self,
        wav: np.ndarray,
        e0: torch.Tensor | None = None,
        sr: int = CALMSEP_SR,
    ) -> ConditionVector:
        level1 = self.forward_level1(wav, sr)
        if e0 is None:
            return ConditionVector(
                snr_db=level1["snr_db"],
                codec_class=level1["codec_class"],
                codec_class_idx=level1["codec_class_idx"],
                codec_bitrate_bps=level1["codec_bitrate_bps"],
                voiced_density=level1["voiced_density"],
            )
        level2 = self.forward_level2(e0, level1)
        return ConditionVector(
            snr_db=level1["snr_db"],
            t60_s=level2["t60_s"],
            codec_class=level1["codec_class"],
            codec_class_idx=level1["codec_class_idx"],
            codec_bitrate_bps=level1["codec_bitrate_bps"],
            voiced_density=level1["voiced_density"],
            count_prior=level2["count_prior"],
        )

    def condition_losses(
        self,
        e0: torch.Tensor,
        level1: dict[str, Any],
        target_t60: torch.Tensor,
        target_n: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Supervised Level-2 losses. target_n is class index 0..3 for N=2..5."""
        level2 = self.forward_level2(e0, level1)
        t60_loss = F.l1_loss(level2["t60_tensor"], target_t60)
        count_loss = F.cross_entropy(level2["count_logits"], target_n.long())
        return {"t60_l1": t60_loss, "count_ce": count_loss, "total": t60_loss + count_loss}
