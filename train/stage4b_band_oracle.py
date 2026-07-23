"""
Stage 4b (oracle): BandRecoveryHead training WITHOUT SR-CorrNet.

Uses clean reference speakers directly as the "separated" input (oracle
supervision), bypassing the 32s/sample SR-CorrNet bottleneck.

Head sees: (downsample(clean_ref_16k → 8k), mix_16k) → predict high-band mask.
At inference the head still receives imperfect SR-CorrNet output, but for a
lightweight mask network this transfer works well.

Speed: ~50ms/sample vs 32s/sample with SR-CorrNet. ~5 min for 30 epochs × 200.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
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
from train.stage1_single import _seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=__import__("sys").stdout,
    force=True,
)
log = logging.getLogger(__name__)

_CALMSEP_SR = 8_000
_BAND_SR = 16_000
_STFT_8K_WIN = 128
_STFT_8K_HOP = 64
_HIGH_BAND_BINS = _HIGH_BAND_END - _HIGH_BAND_START + 1  # 129
_EPS = 1e-10


def _build_speaker_pool(root: Path) -> dict[str, list[Path]]:
    files = sorted(root.rglob("*.flac")) + sorted(root.rglob("*.wav"))
    pool: dict[str, list[Path]] = {}
    for f in files:
        spk = f.parts[-3] if len(f.parts) >= 3 else "unknown"
        pool.setdefault(spk, []).append(f)
    total = sum(len(v) for v in pool.values())
    log.info("Speaker pool: %d speakers, %d files", len(pool), total)
    return pool


def _load_16k(path: Path, max_samples: int) -> np.ndarray | None:
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav[:, 0]
        if sr != _BAND_SR:
            return None
        if len(wav) < _BAND_SR:
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
    speaker_ids = rng.sample(list(pool.keys()), min(n_spks, len(pool)))
    clips: list[np.ndarray] = []
    for spk in speaker_ids:
        for _ in range(12):
            clip = _load_16k(rng.choice(pool[spk]), max_16k)
            if clip is not None:
                gain_db = rng.uniform(-3.0, 3.0)
                clips.append(clip * (10.0 ** (gain_db / 20.0)))
                break
    if len(clips) < 2:
        return None
    max_len = max(len(c) for c in clips)
    refs = np.stack([np.pad(c, (0, max_len - len(c))) for c in clips])
    mix = refs.sum(axis=0)
    peak = float(np.abs(mix).max())
    if peak > 1e-8:
        refs /= peak
        mix /= peak
    return mix.astype(np.float32), refs.astype(np.float32)


def _to_8k(wav_np: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(wav_np).float().to(device)
    try:
        import torchaudio

        return torchaudio.functional.resample(t.unsqueeze(0), _BAND_SR, _CALMSEP_SR).squeeze(0)
    except ImportError:
        return t[::2]


def train_band_oracle(args: argparse.Namespace) -> None:
    _seed_everything(getattr(args, "seed", 42))
    device = torch.device(getattr(args, "device", "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    head = BandRecoveryHead(hidden=getattr(args, "head_hidden", 64)).to(device)
    lr = getattr(args, "lr", 1e-3)
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    epochs = getattr(args, "epochs", 30)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log.info(
        "Head params: %d  lr=%.2e  epochs=%d  device=%s",
        sum(p.numel() for p in head.parameters()),
        lr,
        epochs,
        device,
    )

    pool = _build_speaker_pool(Path(args.librispeech_16k))
    rng = random.Random(getattr(args, "seed", 42))
    max_16k = getattr(args, "max_audio_samples_16k", 32000)  # 2s @16kHz default
    spe = getattr(args, "samples_per_epoch", 200)

    win_8k = torch.hann_window(_STFT_8K_WIN, device=device)
    win_16k = torch.hann_window(_STFT_N_FFT, device=device)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        head.train()
        epoch_loss = 0.0
        t0 = time.time()
        n_steps = 0

        for _ in range(spe):
            n_spks = rng.randint(2, min(4, len(pool)))
            sample = _mix_batch(pool, n_spks, max_16k, rng)
            if sample is None:
                continue
            mix_16k_np, refs_16k_np = sample
            N = refs_16k_np.shape[0]

            with torch.no_grad():
                # Oracle: use clean refs downsampled to 8kHz as "separated" input
                refs_8k = torch.stack([_to_8k(refs_16k_np[n], device) for n in range(N)])  # (N, T8)

                mix_t = torch.from_numpy(mix_16k_np).float().to(device)  # (T16,)

                # 8kHz STFT of oracle-separated streams: (N, 65, T_f)
                stft_sep = torch.stft(
                    refs_8k,
                    n_fft=_STFT_8K_WIN,
                    hop_length=_STFT_8K_HOP,
                    win_length=_STFT_8K_WIN,
                    window=win_8k,
                    return_complex=True,
                    center=True,
                ).abs()

                # 16kHz STFT of mixture
                stft_mix_full = torch.stft(
                    mix_t,
                    n_fft=_STFT_N_FFT,
                    hop_length=_STFT_HOP,
                    win_length=_STFT_WIN,
                    window=win_16k,
                    return_complex=True,
                    center=True,
                )  # (257, T_f16)
                stft_mix_hb = stft_mix_full[_HIGH_BAND_START:, :].abs()  # (129, T_f16)
                stft_mix_hb_k = stft_mix_hb.unsqueeze(0).expand(N, -1, -1)  # (N, 129, T_f16)

                # Target: 16kHz high-band of clean refs
                refs_t = torch.from_numpy(refs_16k_np).float().to(device)  # (N, T16)
                stft_refs = torch.stft(
                    refs_t,
                    n_fft=_STFT_N_FFT,
                    hop_length=_STFT_HOP,
                    win_length=_STFT_WIN,
                    window=win_16k,
                    return_complex=True,
                    center=True,
                ).abs()  # (N, 257, T_f16)
                stft_refs_hb = stft_refs[:, _HIGH_BAND_START:, :]  # (N, 129, T_f16)

            T_f = min(stft_sep.shape[-1], stft_mix_hb_k.shape[-1], stft_refs_hb.shape[-1])
            sep_clip = stft_sep[:, :, :T_f]
            hb_clip = stft_mix_hb_k[:, :, :T_f]
            tgt_clip = stft_refs_hb[:, :, :T_f]

            optimizer.zero_grad()
            masks = head(sep_clip, hb_clip)  # (N, 129, T_f)
            pred_hb = masks * hb_clip
            loss = F.mse_loss(pred_hb, tgt_clip)  # no PIT needed: oracle order matches
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        scheduler.step()
        avg = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - t0
        log.info(
            "Epoch %d/%d  loss=%.6f  steps=%d  time=%.1fs", epoch, epochs, avg, n_steps, elapsed
        )

        if avg < best_loss:
            best_loss = avg
            torch.save(head.state_dict(), out_dir / "best_band.pt")
            log.info("Saved best_band.pt (loss=%.6f)", best_loss)

    torch.save(head.state_dict(), out_dir / "final_band.pt")
    log.info("Done. best_loss=%.6f", best_loss)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-16k", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--samples-per-epoch", type=int, default=200)
    p.add_argument("--max-audio-samples-16k", type=int, default=32000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train_band_oracle(_parse_args())
