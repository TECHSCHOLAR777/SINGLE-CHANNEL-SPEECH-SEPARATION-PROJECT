"""
Two-level condition analyzer (Dev B, P2-B1 / P2-B2).

Level 1 — raw STFT DSP features, deterministic, no training:
  • SNR estimate via SileroVAD voiced-frame energy ratio
  • Codec bandwidth: fraction of energy in 3-4 kHz band (drops with codec damage)
  • Voiced-frame density: fraction of frames flagged as voiced by SileroVAD
    (fallback: voiced-energy fraction when SileroVAD is not installed)

Level 2 — pooled E(0) heads, trained alongside the gate:
  • T60 reverb head: predict log-T60 from temporal mean of E(0) over voiced frames
  • Count prior MLP: predict N ∈ {2,3,4,5} from E(0) statistics

The combined feature vector drives the gate network (models/gate.py). During
inference both levels run sequentially; during gate training the Level-1 outputs
are fixed (no grad) and Level-2 heads are trained.

Free supervision from MixtureRecipe.condition_vector() (BLUEPRINT §5.4):
  snr_db, t60_s, codec_class, codec_bitrate_kbps, n_speakers.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORALSEP_SR = 8_000
STFT_WIN = 128
STFT_HOP = 64
STFT_BINS = 65  # n_fft//2 + 1

# Codec bandwidth sentinel: energy above this fraction of Nyquist drops at low bitrate.
_BANDWIDTH_CUTOFF_HZ = 3_200
_BANDWIDTH_BIN = int(round(_BANDWIDTH_CUTOFF_HZ / (CORALSEP_SR / 2) * (STFT_BINS - 1)))

_EPS = 1e-10
_LOG_T60_MIN = float(np.log(0.05))
_LOG_T60_MAX = float(np.log(2.0))


# ---------------------------------------------------------------------------
# SileroVAD wrapper (optional)
# ---------------------------------------------------------------------------

_silero_model: object | None = None
_silero_available: bool | None = None


def _try_load_silero() -> bool:
    global _silero_model, _silero_available
    if _silero_available is not None:
        return _silero_available
    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        _silero_model = model.cpu()  # keep VAD on CPU; SR-CorrNet needs GPU VRAM
        _silero_available = True
    except Exception:
        _silero_available = False
    return _silero_available  # type: ignore[return-value]


def voiced_density_silero(waveform: torch.Tensor, sr: int = CORALSEP_SR) -> float:
    """
    Fraction of 30 ms frames SileroVAD classifies as voiced.

    Returns float in [0, 1]. Falls back to voiced-energy fraction if SileroVAD
    is not installed.
    """
    if not _try_load_silero():
        return voiced_density_energy(waveform)
    try:
        assert _silero_model is not None
        wav = waveform.float().squeeze()
        if wav.ndim != 1:
            wav = wav.mean(0)
        if sr != 16_000 and sr != 8_000:
            warnings.warn(
                f"SileroVAD prefers 8kHz or 16kHz; got {sr} Hz", RuntimeWarning, stacklevel=2
            )
        # Silero returns probabilities per 30ms window
        frame_len = int(0.030 * sr)
        hop = frame_len  # non-overlapping
        n_frames = wav.shape[0] // hop
        voiced = 0
        for i in range(n_frames):
            seg = wav[i * hop : (i + 1) * hop].unsqueeze(0)
            with torch.no_grad():
                prob = float(_silero_model(seg.cpu(), sr).item())  # type: ignore[operator]
            voiced += int(prob > 0.5)
        return voiced / max(n_frames, 1)
    except Exception:
        return voiced_density_energy(waveform)


def voiced_density_energy(waveform: torch.Tensor) -> float:
    """
    Voiced-energy fraction: fraction of 30ms frames whose energy exceeds
    the median frame energy. Fallback for when SileroVAD is unavailable.
    """
    wav = waveform.float().squeeze()
    frame_len = int(0.030 * CORALSEP_SR)
    n_frames = wav.shape[0] // frame_len
    if n_frames == 0:
        return 0.5
    frames = wav[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = frames.pow(2).mean(dim=1)
    median = energies.median()
    return float((energies > median).float().mean().item())


# ---------------------------------------------------------------------------
# Level 1: DSP features (no parameters, deterministic)
# ---------------------------------------------------------------------------


def level1_features(mixture_8k: torch.Tensor) -> dict[str, float]:
    """
    Extract raw DSP features from an 8 kHz mixture waveform.

    Args:
        mixture_8k: [T] or [1, T] float32 mono waveform at 8 kHz.

    Returns:
        Dict with keys: snr_est_db, codec_bw_ratio, voiced_density, total_energy_db.
        All values are scalar floats usable as gate inputs.
    """
    wav = mixture_8k.float().squeeze()
    if wav.ndim != 1:
        wav = wav.mean(0)

    # STFT at 8 kHz with SR-CorrNet window/hop
    spec = torch.stft(
        wav,
        n_fft=STFT_WIN,
        hop_length=STFT_HOP,
        win_length=STFT_WIN,
        window=torch.hann_window(STFT_WIN, device=wav.device),
        return_complex=True,
        center=True,
    )  # [F, T_frames], F=65

    power = spec.abs().pow(2)  # [65, T_frames]
    total_power = float(power.sum().item())

    # --- SNR estimate via voiced/unvoiced energy split ---
    voiced_den = voiced_density_silero(wav)
    # Proxy SNR: ratio of frame energies above/below voiced threshold
    frame_energies = power.sum(0)  # [T_frames]
    if frame_energies.numel() > 1:
        threshold = frame_energies.median()
        voiced_energy = float(frame_energies[frame_energies > threshold].mean().item())
        unvoiced_energy = float(frame_energies[frame_energies <= threshold].mean().item())
        snr_est = 10.0 * np.log10(max(voiced_energy, _EPS) / max(unvoiced_energy, _EPS))
    else:
        snr_est = 0.0

    # --- Codec bandwidth ratio ---
    # Energy above _BANDWIDTH_CUTOFF_HZ relative to full-band energy.
    # This drops when codec damage removes high-frequency content.
    highband_power = float(power[_BANDWIDTH_BIN:, :].sum().item())
    bw_ratio = highband_power / max(total_power, _EPS)

    # --- Total energy in dB ---
    total_db = 10.0 * np.log10(max(total_power / max(power.numel(), 1), _EPS))

    return {
        "snr_est_db": float(snr_est),
        "codec_bw_ratio": float(np.clip(bw_ratio, 0.0, 1.0)),
        "voiced_density": float(voiced_den),
        "total_energy_db": float(total_db),
    }


def level1_tensor(mixture_8k: torch.Tensor) -> torch.Tensor:
    """Return Level-1 features as a [4] float32 tensor (for gate input)."""
    feats = level1_features(mixture_8k)
    return torch.tensor(
        [
            feats["snr_est_db"],
            feats["codec_bw_ratio"],
            feats["voiced_density"],
            feats["total_energy_db"],
        ],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Level 2: Learned heads on pooled E(0)
# ---------------------------------------------------------------------------


class T60Head(nn.Module):
    """
    Predict log-T60 from the temporal mean of encoder output E(0).

    E(0) shape: (1, T, 65, 128) from model.encoder forward hook.
    Pooled to (128,) by averaging over T and F, then through a 2-layer MLP.
    """

    def __init__(self, d_model: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e0: (B, T, F, D) encoder output.
        Returns:
            (B,) predicted log-T60 in seconds.
        """
        # Mean over T and F.
        pooled = e0.mean(dim=(1, 2))  # (B, D)
        log_t60 = self.net(pooled).squeeze(-1)  # (B,)
        return log_t60

    def t60_seconds(self, e0: torch.Tensor) -> torch.Tensor:
        """Return predicted T60 in seconds (clamped to plausible range)."""
        log_t60 = self.forward(e0)
        return torch.exp(log_t60.clamp(_LOG_T60_MIN, _LOG_T60_MAX))


class CountPriorMLP(nn.Module):
    """
    Speaker count prior from E(0): P(N | E(0)), N ∈ {2, 3, 4, 5}.

    Returns logits over 4 classes; maps to N by argmax + 2.
    """

    _N_CLASSES = 4  # N in {2,3,4,5}

    def __init__(self, d_model: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self._N_CLASSES),
        )

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e0: (B, T, F, D)
        Returns:
            (B, 4) logits for N ∈ {2,3,4,5}
        """
        pooled = e0.mean(dim=(1, 2))  # (B, D)
        return self.net(pooled)

    def count_estimate(self, e0: torch.Tensor) -> torch.Tensor:
        """Return (B,) integer count estimates in {2,3,4,5}."""
        logits = self.forward(e0)
        return logits.argmax(dim=-1) + 2

    def count_probs(self, e0: torch.Tensor) -> torch.Tensor:
        """Return (B, 4) softmax probabilities over {2,3,4,5}."""
        return F.softmax(self.forward(e0), dim=-1)


class Level2Analyzer(nn.Module):
    """
    Combines T60Head and CountPriorMLP into one module.

    Produces a (B, 6) feature vector:
      [log_t60, t60_seconds, count_logit_2, count_logit_3, count_logit_4, count_logit_5]
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.t60_head = T60Head(d_model)
        self.count_prior = CountPriorMLP(d_model)

    def forward(self, e0: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            e0: (B, T, F, D) encoder output from hook.
        Returns:
            Dict with keys: log_t60 (B,), t60_s (B,), count_logits (B,4), count_probs (B,4).
        """
        log_t60 = self.t60_head(e0)
        t60_s = torch.exp(log_t60.clamp(_LOG_T60_MIN, _LOG_T60_MAX))
        count_logits = self.count_prior(e0)
        count_probs = F.softmax(count_logits, dim=-1)
        return {
            "log_t60": log_t60,
            "t60_s": t60_s,
            "count_logits": count_logits,
            "count_probs": count_probs,
        }

    def feature_vector(self, e0: torch.Tensor) -> torch.Tensor:
        """Return (B, 6) float tensor for gate input."""
        out = self.forward(e0)
        return torch.cat(
            [out["log_t60"].unsqueeze(-1), out["t60_s"].unsqueeze(-1), out["count_probs"]], dim=-1
        )


# ---------------------------------------------------------------------------
# Supervised loss for Level-2 heads
# ---------------------------------------------------------------------------


def level2_loss(
    analyzer: Level2Analyzer,
    e0: torch.Tensor,
    recipe_vectors: list[dict[str, float]],
) -> torch.Tensor:
    """
    Supervised loss on both heads using free condition labels from the recipe.

    t60 head: L1 loss on log-T60 (robust to scale).
    count prior: cross-entropy over {2,3,4,5}.

    Args:
        analyzer: Level2Analyzer module.
        e0: (B, T, F, D) encoder features.
        recipe_vectors: List of B condition_vector() dicts from MixtureRecipe.

    Returns:
        Scalar loss tensor.
    """
    out = analyzer.forward(e0)
    device = out["log_t60"].device

    t60_targets = torch.tensor(
        [np.log(max(rv["t60_s"], 0.05)) for rv in recipe_vectors],
        dtype=torch.float32,
        device=device,
    )
    count_targets = torch.tensor(
        [int(rv["n_speakers"]) - 2 for rv in recipe_vectors],
        dtype=torch.long,
        device=device,
    )

    loss_t60 = F.l1_loss(out["log_t60"], t60_targets)
    loss_count = F.cross_entropy(out["count_logits"], count_targets)
    return loss_t60 + loss_count
