"""Frozen-base corpus-transfer baseline runner (P0-B7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import pit_si_sdr
from models.preprocess import CALMSEP_SAMPLE_RATE
from models.srcorrnet import SRCorrNetWrapper


def run_corpus_transfer_baseline(
    mixtures: list[np.ndarray],
    references: list[np.ndarray],
    *,
    device: str = "cpu",
    wrapper: Any | None = None,
) -> dict[str, float]:
    """Score frozen base SI-SDRi on a small set of mixtures."""
    if wrapper is None:
        wrapper = SRCorrNetWrapper(device=device)
    if not wrapper.is_available:
        raise RuntimeError("SR-CorrNet unavailable for baseline")
    wrapper.load()

    scores: list[float] = []
    for mix, refs in zip(mixtures, references, strict=True):
        import torch

        out = wrapper.forward(torch.from_numpy(mix).float(), n_spks=None)
        waves = out.get("waveforms") or []
        if not waves:
            continue
        est = np.stack([np.asarray(w, dtype=np.float32).reshape(-1) for w in waves], axis=0)
        ref = np.atleast_2d(np.asarray(refs, dtype=np.float32))
        n = min(est.shape[-1], ref.shape[-1], mix.shape[-1])
        result = pit_si_sdr(est[:, :n], ref[:, :n], mix[:n])
        scores.append(result.mean_si_sdri)

    if not scores:
        raise RuntimeError("no mixtures scored")
    return {
        "mean_si_sdri": float(np.mean(scores)),
        "std_si_sdri": float(np.std(scores)),
        "n": float(len(scores)),
        "sample_rate": float(CALMSEP_SAMPLE_RATE),
    }


def write_baseline_log(stats: dict[str, float], path: str | Path) -> None:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
