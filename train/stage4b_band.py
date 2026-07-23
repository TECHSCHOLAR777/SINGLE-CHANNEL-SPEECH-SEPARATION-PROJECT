"""
Stage 4b: Band recovery head training (Dev C, P3-C1).

Trains BandRecoveryHead to predict a soft mask over the 4-8 kHz high-band.

Pipeline per sample:
  1. Load 16 kHz utterances from LibriSpeech FLAC/WAV, mix N speakers on-the-fly.
  2. Downsample 16kHz → 8kHz (torchaudio), run frozen SR-CorrNet to separate.
  3. Compute 8kHz STFT of each separated stream  → low-band input  (K, 65, T_f).
  4. Compute 16kHz STFT of mixture high-band (4-8 kHz, bins 128-256) → (K, 129, T_f).
  5. Target: 16kHz STFT high-band magnitude of each PIT-matched clean reference.
  6. Loss: MSE(mask * mixture_highband, target_highband) — permutation-invariant.

No gradients flow through SR-CorrNet. Head trains in ~30 epochs on T4/P100/RTX6000.
Saves: best_band.pt, final_band.pt  (BandRecoveryHead.state_dict()).
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.band_recovery import (
    _HIGH_BAND_END,
    _HIGH_BAND_START,
    _STFT_HOP,
    _STFT_N_FFT,
    _STFT_WIN,
    BandRecoveryHead,
)
from train.stage1_single import _get_inner_module, _load_model, _seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_CALMSEP_SR = 8_000
_BAND_SR = 16_000
_STFT_8K_WIN = 128
_STFT_8K_HOP = 64
_STFT_8K_BINS = 65
_HIGH_BAND_BINS = _HIGH_BAND_END - _HIGH_BAND_START + 1  # 129
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def _build_speaker_pool(root: Path) -> dict[str, list[Path]]:
    """Scan FLAC/WAV under root; group by speaker (3rd-from-last path segment)."""
    files = sorted(root.rglob("*.flac")) + sorted(root.rglob("*.wav"))
    pool: dict[str, list[Path]] = {}
    for f in files:
        # LibriSpeech layout: root/<speaker>/<chapter>/<file>.flac
        spk = f.parts[-3] if len(f.parts) >= 3 else "unknown"
        pool.setdefault(spk, []).append(f)
    total = sum(len(v) for v in pool.values())
    log.info("Speaker pool: %d speakers, %d files", len(pool), total)
    return pool


def _load_16k(path: Path, max_samples: int) -> np.ndarray | None:
    """Load mono 16 kHz clip. Returns None on any error or wrong SR."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav[:, 0]
        if sr != _BAND_SR:
            return None
        if len(wav) < _BAND_SR:  # skip clips shorter than 1s
            return None
        wav = wav[:max_samples]
        rms = float(np.sqrt(np.maximum(np.mean(wav**2), _EPS)))
        return (wav / rms).astype(np.float32)
    except Exception:
        return None


def _mix_batch(
    pool: dict[str, list[Path]],
    n_spks: int,
    max_16k: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sample n_spks speakers, load one utterance each, mix.
    Returns (mixture_16k [T], refs_16k [N, T]) or None.
    """
    speaker_ids = rng.sample(list(pool.keys()), min(n_spks, len(pool)))
    clips: list[np.ndarray] = []
    for spk in speaker_ids:
        for _ in range(12):  # retry if file fails
            path = rng.choice(pool[spk])
            clip = _load_16k(path, max_16k)
            if clip is not None:
                gain_db = rng.uniform(-3.0, 3.0)
                clips.append(clip * (10.0 ** (gain_db / 20.0)))
                break
    if len(clips) < 2:
        return None

    max_len = max(len(c) for c in clips)
    refs = np.stack([np.pad(c, (0, max_len - len(c))) for c in clips], axis=0)
    mix = refs.sum(axis=0)
    peak = float(np.abs(mix).max())
    if peak > 1e-8:
        refs /= peak
        mix /= peak
    return mix.astype(np.float32), refs.astype(np.float32)


# ---------------------------------------------------------------------------
# PIT helper (magnitudes only, fast)
# ---------------------------------------------------------------------------


def _pit_mse_loss(
    pred: torch.Tensor,  # (K, C, T)
    target: torch.Tensor,  # (N, C, T)
) -> torch.Tensor:
    """MSE PIT loss. Tries all K! permutations; best permutation minimised."""
    K = pred.shape[0]
    N = target.shape[0]

    if K != N:
        # Mismatch: match each separated stream to its nearest reference (greedy)
        total = torch.tensor(0.0, device=pred.device)
        for k in range(K):
            dists = torch.stack([F.mse_loss(pred[k], target[n]) for n in range(N)])
            total = total + dists.min()
        return total / K

    best_loss = None
    for perm in permutations(range(N)):
        tgt_perm = torch.stack([target[perm[k]] for k in range(K)])
        loss = F.mse_loss(pred, tgt_perm)
        if best_loss is None or loss < best_loss:
            best_loss = loss
    return best_loss  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_band(args: argparse.Namespace) -> None:
    _seed_everything(getattr(args, "seed", 42))
    device = torch.device(getattr(args, "device", "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Frozen SR-CorrNet (base only; no LoRA) ──────────────────────────────
    hf_model = getattr(args, "hf_model", "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    ss_model = _load_model(hf_model, device)
    inner = _get_inner_module(ss_model)
    inner.eval()
    for p in inner.parameters():
        p.requires_grad_(False)
    log.info("SR-CorrNet loaded and frozen (base, no LoRA)")

    # ── Band recovery head ──────────────────────────────────────────────────
    head = BandRecoveryHead(hidden=getattr(args, "head_hidden", 64)).to(device)
    lr = getattr(args, "lr", 1e-3)
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    epochs = getattr(args, "epochs", 30)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log.info(
        "Head params: %d  lr=%.2e  epochs=%d",
        sum(p.numel() for p in head.parameters()),
        lr,
        epochs,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    pool = _build_speaker_pool(Path(args.librispeech_16k))
    rng = random.Random(getattr(args, "seed", 42))
    max_16k = getattr(args, "max_audio_samples_16k", 64000)  # 4 s at 16 kHz
    samples_per_epoch = getattr(args, "samples_per_epoch", 2000)

    # Pre-allocate STFT windows on device
    win_8k = torch.hann_window(_STFT_8K_WIN, device=device)
    win_16k = torch.hann_window(_STFT_N_FFT, device=device)

    # torchaudio resampler (16k → 8k); lazy import so the stub works without it
    try:
        import torchaudio

        _resamp = torchaudio.functional.resample
    except ImportError:
        _resamp = None

    def _to_8k(wav_16k_np: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(wav_16k_np).float().to(device).unsqueeze(0)  # (1, T)
        if _resamp is not None:
            return _resamp(t, _BAND_SR, _CALMSEP_SR)
        # Fallback: simple decimation (good enough for training)
        return t[:, ::2]

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        head.train()
        epoch_loss = 0.0
        t0 = time.time()
        n_steps = 0

        for _ in range(samples_per_epoch):
            n_spks = rng.randint(2, min(4, len(pool)))  # N = 2, 3 or 4
            sample = _mix_batch(pool, n_spks, max_16k, rng)
            if sample is None:
                continue
            mix_16k_np, refs_16k_np = sample

            # ── Separate at 8kHz ────────────────────────────────────────────
            mix_8k = _to_8k(mix_16k_np)  # (1, T8)
            with torch.no_grad():
                out = ss_model.process_waveform(mix_8k, n_spks=torch.tensor(n_spks))
                waves_list = out.get("waveforms", [])

            if not waves_list:
                continue

            # Stack separated streams → (K, T8) float32 on device
            sep_pieces = []
            for w in waves_list:
                if isinstance(w, np.ndarray):
                    sep_pieces.append(torch.from_numpy(w.astype(np.float32)).squeeze())
                else:
                    sep_pieces.append(w.float().squeeze())
            sep_8k = torch.stack(sep_pieces, dim=0).to(device)  # (K, T8)
            K = sep_8k.shape[0]

            # ── STFTs ───────────────────────────────────────────────────────
            with torch.no_grad():
                # Low-band: 8kHz STFT magnitude (K, 65, T_f8)
                stft_sep = torch.stft(
                    sep_8k,
                    n_fft=_STFT_8K_WIN,
                    hop_length=_STFT_8K_HOP,
                    win_length=_STFT_8K_WIN,
                    window=win_8k,
                    return_complex=True,
                    center=True,
                ).abs()  # (K, 65, T_f8)

                mix_16k_t = torch.from_numpy(mix_16k_np).float().to(device)  # (T16,)
                # High-band: 16kHz STFT of mixture (257, T_f16) → expand to (K, 129, T_f16)
                stft_mix_full = torch.stft(
                    mix_16k_t,
                    n_fft=_STFT_N_FFT,
                    hop_length=_STFT_HOP,
                    win_length=_STFT_WIN,
                    window=win_16k,
                    return_complex=True,
                    center=True,
                )  # (257, T_f16)
                stft_mix_hb = stft_mix_full[_HIGH_BAND_START:, :].abs()  # (129, T_f16)
                stft_mix_hb_k = stft_mix_hb.unsqueeze(0).expand(K, -1, -1)  # (K, 129, T_f16)

                # Target: 16kHz STFT high-band of clean references (N, 129, T_f16)
                refs_t = torch.from_numpy(refs_16k_np).float().to(device)  # (N, T16)
                stft_refs_full = torch.stft(
                    refs_t,
                    n_fft=_STFT_N_FFT,
                    hop_length=_STFT_HOP,
                    win_length=_STFT_WIN,
                    window=win_16k,
                    return_complex=True,
                    center=True,
                ).abs()  # (N, 257, T_f16)
                stft_refs_hb = stft_refs_full[:, _HIGH_BAND_START:, :]  # (N, 129, T_f16)

            # Align T_f across both STFT grids (should match since 8ms/frame for both)
            T_f = min(stft_sep.shape[-1], stft_mix_hb_k.shape[-1], stft_refs_hb.shape[-1])
            stft_sep_clip = stft_sep[:, :, :T_f]  # (K, 65, T_f)
            stft_hb_clip = stft_mix_hb_k[:, :, :T_f]  # (K, 129, T_f)
            stft_ref_clip = stft_refs_hb[:, :, :T_f]  # (N, 129, T_f)

            # ── BandRecoveryHead forward + PIT loss ─────────────────────────
            optimizer.zero_grad()
            masks = head(stft_sep_clip, stft_hb_clip)  # (K, 129, T_f)
            pred_hb = masks * stft_hb_clip  # (K, 129, T_f)
            loss = _pit_mse_loss(pred_hb, stft_ref_clip)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        scheduler.step()
        avg = epoch_loss / max(n_steps, 1)
        log.info(
            "Epoch %d/%d  loss=%.6f  steps=%d  time=%.1fs",
            epoch,
            epochs,
            avg,
            n_steps,
            time.time() - t0,
        )

        if avg < best_loss:
            best_loss = avg
            torch.save(head.state_dict(), out_dir / "best_band.pt")
            log.info("Saved best_band.pt (loss=%.6f)", best_loss)

    torch.save(head.state_dict(), out_dir / "final_band.pt")
    log.info("Band recovery training done. Best loss: %.6f", best_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--librispeech-16k", required=True, help="Root of LibriSpeech 16 kHz (FLAC/WAV tree)"
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hf-model", default="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--max-audio-samples-16k", type=int, default=64000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train_band(_parse_args())
