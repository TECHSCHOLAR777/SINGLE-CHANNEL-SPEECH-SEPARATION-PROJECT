"""CLI: train band-recovery head (BLUEPRINT §8.5).

USER RUNS TRAINING after joint polish:

    python -m train.train_band_recovery --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.band_recovery import BandRecoveryHead, apply_band_recovery, zero_pad_8k_to_16k
from models.preprocess import compute_stft, resample_audio
from utils.hashing import hash_config
from utils.logging import get_logger

log = get_logger("train_band_recovery")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/band_recovery.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out-dir", default="artifacts/band_recovery")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _synthetic_pair(seconds: float = 1.0):
    """Synthetic 8 kHz stream + 16 kHz mixture high-band target."""
    n8 = int(8000 * seconds)
    stream_8k = (np.random.randn(n8) * 0.05).astype(np.float32)
    # Fake high-band content at 16 kHz.
    target_16k = resample_audio(stream_8k, 8000, 16000)
    mix_16k = target_16k + (np.random.randn(len(target_16k)) * 0.01).astype(np.float32)
    return stream_8k, mix_16k, target_16k


def main() -> None:
    args = parse_args()
    from utils.config import load_config

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    head = BandRecoveryHead(channels=int(cfg.get("channels", 32))).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg.get("lr", 1e-3)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1 if args.dry_run else args.epochs):
        stream_8k, mix_16k, target_16k = _synthetic_pair()
        low = torch.from_numpy(np.abs(compute_stft(stream_8k, 128, 64))).unsqueeze(0).to(args.device)
        mix_stft = compute_stft(mix_16k, 256, 128)
        high = torch.from_numpy(np.abs(mix_stft[65:, :])).unsqueeze(0).to(args.device)
        mask = head(low, high)
        # Proxy loss: encourage mask energy (replace with SI-SNR on reconstructed wav in full run).
        loss = F.mse_loss(mask, torch.ones_like(mask) * 0.5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log.info("epoch_end", epoch=epoch, loss=float(loss.detach().cpu()))
        if args.dry_run:
            break

    # Dual-metric guard smoke on synthetic (no refs → applies head).
    head_cpu = head.cpu()
    streams = np.stack([_synthetic_pair()[0], _synthetic_pair()[0]], axis=0)
    mix16 = zero_pad_8k_to_16k(streams.sum(0))
    br = apply_band_recovery(streams, mix16, head_cpu)
    torch.save(
        {
            "head": head_cpu.state_dict(),
            "config_sha256": hash_config(cfg),
            "guard_smoke_applied": br.applied,
        },
        out / "band_recovery.pt",
    )
    log.info("checkpoint_written", path=str(out / "band_recovery.pt"))


if __name__ == "__main__":
    main()
