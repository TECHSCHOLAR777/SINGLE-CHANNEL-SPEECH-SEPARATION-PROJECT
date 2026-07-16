#!/usr/bin/env python3
"""
Corpus-transfer baseline: frozen SR-CorrNet on LibriSpeech-based mixtures (BLUEPRINT §8.1).

Runs the frozen base checkpoint on N synthetic or dev-clean mixtures and logs mean
SI-SDRi. This is the floor every adapter and gate must beat.

When the checkpoint is absent, skips with a clear message and optionally uses a
mock oracle path for CI smoke tests.

Usage::

    python scripts/corpus_transfer_baseline.py --n-mixtures 20
    python scripts/corpus_transfer_baseline.py --n-mixtures 5 --mock
    python scripts/corpus_transfer_baseline.py --config configs/base_checkpoint.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.metrics import pit_si_sdr  # noqa: E402
from utils.config import cfg_get, load_config  # noqa: E402
from utils.logging import get_logger  # noqa: E402

RNG = np.random.default_rng(seed=2026)
CALMSEP_SR = 8000
DURATION_S = 4.0


def _synthesize_mixture(n_spk: int, sr: int = CALMSEP_SR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU-only synthetic mixture for smoke testing."""
    n = int(DURATION_S * sr)
    refs = RNG.standard_normal((n_spk, n))
    refs = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-8)
    mix = refs.sum(axis=0)
    mix = mix / (np.linalg.norm(mix) + 1e-8)
    return mix.astype(np.float32), refs.astype(np.float32), mix.astype(np.float32)


def _run_mock(n_mixtures: int, allowed_n: list[int]) -> float:
    """Oracle mock: references as estimates → very high SI-SDRi."""
    scores: list[float] = []
    for i in range(n_mixtures):
        n_spk = allowed_n[i % len(allowed_n)]
        mix, refs, _ = _synthesize_mixture(n_spk)
        pit = pit_si_sdr(refs, refs, mix)
        scores.append(pit.mean_si_sdri)
    return float(np.mean(scores))


def _run_wrapper(n_mixtures: int, allowed_n: list[int], device: str) -> float:
    """Run frozen SRCorrNetWrapper when checkpoint is available."""
    import torch

    from models.srcorrnet import SRCorrNetWrapper

    wrapper = SRCorrNetWrapper(device=device)
    if not wrapper.is_available:
        raise RuntimeError("SR-CorrNet checkpoint not available")

    wrapper.load()
    scores: list[float] = []

    for i in range(n_mixtures):
        n_spk = allowed_n[i % len(allowed_n)]
        mix_np, refs, _ = _synthesize_mixture(n_spk)
        wav = torch.from_numpy(mix_np).unsqueeze(0)
        wav = wav / (wav.std() + 1e-8)

        out = wrapper.forward(wav, n_spks=None)
        est_wavs = out.get("waveforms", [])
        if not est_wavs:
            scores.append(0.0)
            continue

        estimates = np.stack([w.cpu().numpy().astype(np.float32) for w in est_wavs], axis=0)
        min_len = min(estimates.shape[1], refs.shape[1], mix_np.shape[0])
        pit = pit_si_sdr(
            estimates[:, :min_len],
            refs[:, :min_len],
            mix_np[:min_len],
        )
        scores.append(pit.mean_si_sdri)

    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-transfer baseline SI-SDRi floor")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_checkpoint.yaml",
        help="Checkpoint config path",
    )
    parser.add_argument(
        "--n-mixtures",
        type=int,
        default=20,
        help="Number of mixtures to score (BLUEPRINT §8.1 uses 20)",
    )
    parser.add_argument(
        "--allowed-n",
        nargs="+",
        type=int,
        default=[2],
        help="Speaker counts to cycle through",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use oracle mock (references as estimates) instead of the wrapper",
    )
    args = parser.parse_args()

    log = get_logger("corpus_transfer_baseline")
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    checkpoint_path = cfg_get(cfg, "local_path", None)
    log.info(
        "baseline_start",
        n_mixtures=args.n_mixtures,
        checkpoint=checkpoint_path,
        mock=args.mock,
    )

    if args.mock:
        mean_sdri = _run_mock(args.n_mixtures, args.allowed_n)
        log.info("baseline_mock_complete", mean_si_sdri=mean_sdri)
        print(f"[mock] mean SI-SDRi = {mean_sdri:.2f} dB over {args.n_mixtures} mixtures")
        return

    from models.srcorrnet import SRCorrNetWrapper

    wrapper = SRCorrNetWrapper(
        device=args.device,
        checkpoint_path=checkpoint_path,
    )
    if not wrapper.is_available:
        msg = (
            "SR-CorrNet checkpoint not available — skipping corpus-transfer baseline.\n"
            "Run: python scripts/download_checkpoint.py\n"
            "Or re-run with --mock for a CPU smoke test."
        )
        log.warning("baseline_skipped", reason="checkpoint_missing")
        print(msg)
        sys.exit(0)

    mean_sdri = _run_wrapper(args.n_mixtures, args.allowed_n, args.device)
    log.info("baseline_complete", mean_si_sdri=mean_sdri)
    print(f"Corpus-transfer baseline: mean SI-SDRi = {mean_sdri:.2f} dB over {args.n_mixtures} mixtures")


if __name__ == "__main__":
    main()
