"""
Scene Analyzer — P2-A1 (Dev A, Phase 2).

Full ~1.5 M-parameter scene analysis front-end that replaces the Dev B interim
stub.  Produces three outputs consumed downstream:

  segment_features [B, S, feature_dim]  → Two-level Adaptive Router (P2-C1)
  count_logits     [B, max_speakers]    → count-BCE loss term (P2-B5)
  scene_weights    [B, 3]              → w_TD / w_TF / w_NULL routing hints

Architecture
------------
1. Pure-PyTorch log-mel spectrogram via torch.stft + triangular mel filterbank
   (no torchaudio dependency; filterbank stored as a non-learnable buffer).
2. Per-segment 1-D conv frontend: three layers 80→128 channels, 4× temporal
   downsampling.
3. Single bidirectional GRU (hidden=384) for within-segment temporal context;
   output size 768.
4. Five handcrafted features per segment computed from the raw waveform chunk:
     reverb_proxy      tail-to-head energy ratio (high ↔ long reverb tail)
     noise_floor       10th-percentile amplitude / RMS (high ↔ noisy)
     overlap_density   fraction of short frames above mean energy (high ↔ many
                       speakers active simultaneously)
     spectral_flatness geometric / arithmetic mean of frame energy (near 1 ↔
                       noise-like; near 0 ↔ tonal / speech)
     modulation_rate   normalised variance of the frame-energy envelope
5. Feature projection: concat(GRU [768] + handcrafted [5]) → feature_dim (256)
   followed by LayerNorm + ReLU.
6. Count head:  mean-pool segment features → 2-layer MLP → max_speakers logits.
7. Scene-weight head: mean handcrafted features → 2-layer MLP → 3-way softmax.

Parameter budget (defaults): ~1.55 M — matches MASTER §4.1 spec of ~1.5 M.

Interface contract (unchanged from stub):
  forward(mixture: Tensor[B, T]) → {
      "segment_features": Tensor[B, S, feature_dim],
      "count_logits":     Tensor[B, max_speakers],
      "scene_weights":    Tensor[B, 3],
  }
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8
_LOG_EPS = 1e-6


# ── Mel filterbank helpers ────────────────────────────────────────────────────


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(n_mels: int, n_fft: int, sr: int) -> torch.Tensor:
    """
    Build a triangular mel filterbank matrix [n_mels, n_fft//2+1].

    Uses HTK mel scale (2595·log10(1+f/700)).  Called once at init; the result
    is registered as a non-learnable buffer on the module.
    """
    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(sr / 2.0)

    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = torch.tensor([_mel_to_hz(m.item()) for m in mel_pts])
    bins = torch.floor((n_fft + 1) * hz_pts / sr).long().clamp(0, n_freqs - 1)

    k = torch.arange(n_freqs, dtype=torch.float32)
    fb = torch.zeros(n_mels, n_freqs)
    for m in range(n_mels):
        left = bins[m].float()
        center = bins[m + 1].float()
        right = bins[m + 2].float()
        up = (k - left) / (center - left + _EPS)
        down = (right - k) / (right - center + _EPS)
        fb[m] = torch.clamp(torch.min(up, down), min=0.0)
    return fb


# ── Scene Analyzer ────────────────────────────────────────────────────────────


class SceneAnalyzer(nn.Module):
    """
    Full scene analysis front-end (~1.55 M params).

    Parameters
    ----------
    feature_dim:
        Output width of segment_features (consumed by the Router).  Default 256.
    n_mels:
        Log-mel bins.  Default 80.
    n_fft:
        FFT size for torch.stft.  Default 512 (32 ms at 16 kHz).
    hop:
        STFT hop in samples.  Default 128 (8 ms).
    max_speakers:
        Upper bound for the coarse count head.  Default 5.
    segment_samples:
        Samples per analysis segment.  Default 32 000 (2 s at 16 kHz).
    sr:
        Expected sample rate in Hz.  Default 16 000.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        n_mels: int = 80,
        n_fft: int = 512,
        hop: int = 128,
        max_speakers: int = 5,
        segment_samples: int = 32_000,
        sr: int = 16_000,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop = hop
        self.max_speakers = max_speakers
        self.segment_samples = segment_samples
        self.sr = sr

        # Non-learnable buffers: mel filterbank + STFT window
        self.register_buffer("_mel_fb", _build_mel_filterbank(n_mels, n_fft, sr))
        self.register_buffer("_window", torch.hann_window(n_fft))

        # 1-D conv frontend: n_mels → 128 channels, 4× temporal downsampling
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Bidirectional GRU: 128 → 768 (384 × 2)
        _gru_hidden = 384
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=_gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        _gru_out = _gru_hidden * 2  # 768

        # Feature projection: [GRU 768 + handcrafted 5] → feature_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(_gru_out + 5, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        # Count head: mean-pooled features → max_speakers logits
        self.count_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, max_speakers),
        )

        # Scene-weight head: 5 handcrafted → 3-way softmax (w_TD, w_TF, w_NULL)
        self.scene_weight_head = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _log_mel(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk [B, T] → log-mel [B, n_mels, frames].

        Uses torch.stft with a Hann window (center=True pads by n_fft//2 so
        even very short chunks produce at least one frame).
        """
        b, t = chunk.shape
        if t < self.n_fft:
            chunk = F.pad(chunk, (0, self.n_fft - t))

        window = self._window.to(chunk.device)
        # [B, n_fft//2+1, frames] complex
        stft = torch.stft(
            chunk,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitude = stft.abs()  # [B, n_fft//2+1, frames]
        mel_fb = self._mel_fb.to(chunk.device)  # [n_mels, n_fft//2+1]
        mel = torch.matmul(mel_fb, magnitude)  # [B, n_mels, frames]
        return torch.log(mel + _LOG_EPS)

    def _handcrafted_features(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk [B, segment_samples] → features [B, 5], all values in [0, 1].

        Computed entirely in the time domain from the raw waveform; no STFT.
        Feature index mapping:
            0  reverb_proxy      tail / head energy ratio
            1  noise_floor       low-percentile amplitude / RMS
            2  overlap_density   fraction of active short frames
            3  spectral_flatness geometric / arithmetic mean of frame energy
            4  modulation_rate   normalised frame-energy variance
        """
        b, t = chunk.shape
        frame = 512
        n_f = t // frame

        # 1. reverb_proxy — energy in second half vs first half
        half = t // 2
        e_head = chunk[:, :half].pow(2).mean(dim=-1).clamp(min=_EPS)
        e_tail = chunk[:, half:].pow(2).mean(dim=-1)
        reverb_proxy = (e_tail / e_head).clamp(0.0, 1.0)

        # 2. noise_floor — 10th-percentile amplitude relative to RMS
        rms = chunk.pow(2).mean(dim=-1).sqrt().clamp(min=_EPS)
        noise_floor = (torch.quantile(chunk.abs(), 0.1, dim=-1) / rms).clamp(0.0, 1.0)

        if n_f > 1:
            frames = chunk[:, : n_f * frame].reshape(b, n_f, frame)
            frame_energy = frames.pow(2).mean(dim=-1) + _EPS  # [B, n_f]

            # 3. overlap_density — fraction of frames above mean energy
            threshold = frame_energy.mean(dim=-1, keepdim=True)
            overlap_density = (frame_energy > threshold).float().mean(dim=-1)

            # 4. spectral_flatness — geometric / arithmetic mean of frame energy
            geom = torch.exp(torch.log(frame_energy).mean(dim=-1))
            arith = frame_energy.mean(dim=-1)
            spectral_flatness = (geom / (arith + _EPS)).clamp(0.0, 1.0)

            # 5. modulation_rate — normalised variance of the energy envelope
            mu_sq = frame_energy.mean(dim=-1).pow(2) + _EPS
            modulation_rate = (frame_energy.var(dim=-1) / mu_sq).clamp(0.0, 1.0)
        else:
            half_val = torch.full((b,), 0.5, device=chunk.device)
            zero = torch.zeros(b, device=chunk.device)
            overlap_density = half_val
            spectral_flatness = half_val
            modulation_rate = zero

        return torch.stack(
            [reverb_proxy, noise_floor, overlap_density, spectral_flatness, modulation_rate],
            dim=-1,
        )  # [B, 5]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, mixture: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extract scene features from mono mixtures.

        Parameters
        ----------
        mixture:
            [B, T] mono waveforms at ``self.sr`` Hz.

        Returns
        -------
        dict with keys:
            segment_features : [B, S, feature_dim] — per-segment features for Router.
            count_logits     : [B, max_speakers]   — coarse speaker-count logits.
            scene_weights    : [B, 3]              — softmax (w_TD, w_TF, w_NULL).
        """
        if mixture.ndim != 2:
            raise ValueError(f"expected [B, T], got {tuple(mixture.shape)}")

        b, t = mixture.shape
        n_seg = max(1, math.ceil(t / self.segment_samples))
        pad_len = n_seg * self.segment_samples
        if t < pad_len:
            mixture = F.pad(mixture, (0, pad_len - t))

        seg_feats_list: list[torch.Tensor] = []
        handcrafted_list: list[torch.Tensor] = []

        for s in range(n_seg):
            chunk = mixture[:, s * self.segment_samples : (s + 1) * self.segment_samples]

            # Log-mel per segment [B, n_mels, frames]
            mel = self._log_mel(chunk)

            # Conv encoder [B, 128, T']
            conv_out = self.conv_encoder(mel)

            # BiGRU [B, T', 768] → mean-pool → [B, 768]
            gru_out, _ = self.gru(conv_out.transpose(1, 2))
            seg_feat = gru_out.mean(dim=1)

            seg_feats_list.append(seg_feat)
            handcrafted_list.append(self._handcrafted_features(chunk))

        seg_feats = torch.stack(seg_feats_list, dim=1)      # [B, S, 768]
        handcrafted = torch.stack(handcrafted_list, dim=1)  # [B, S, 5]

        # Project to feature_dim
        fused = torch.cat([seg_feats, handcrafted], dim=-1)  # [B, S, 773]
        segment_features = self.feature_proj(fused)           # [B, S, feature_dim]

        # Count logits
        count_logits = self.count_head(segment_features.mean(dim=1))  # [B, max_speakers]

        # Scene weights from handcrafted features
        scene_weights = torch.softmax(
            self.scene_weight_head(handcrafted.mean(dim=1)), dim=-1
        )  # [B, 3]

        return {
            "segment_features": segment_features,
            "count_logits": count_logits,
            "scene_weights": scene_weights,
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
