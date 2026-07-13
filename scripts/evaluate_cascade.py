"""
Evaluate a trained CA-MoSE cascade against its single experts (P2-INT4/INT5).

Runs the trainable heads over a cached dev/test set and reports, all on the same
mixtures with PIT SI-SDRi against the clean references:

  * cascade      — the full routed/fused output
  * mossformer2  — the cheap expert alone (cached moss_streams)
  * expensive    — the expensive expert alone (cached sr_streams)
  * escalation_rate and the P2-INT4 verdict (cascade beats best single expert)

No experts are loaded — everything comes from the cache built by
scripts/build_train_cache.py, so this is fast and CPU-friendly.

Example::

    python -m scripts.evaluate_cascade \
        --cache-dir cache/dev --checkpoint outputs/training/checkpoint.pt \
        --report outputs/eval/cascade_vs_single.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.metrics import pit_si_sdr
from models.cascade_gate import CascadeGate
from train.cached_dataset import CachedExpertDataset
from train.losses import CompositeLoss, LossWeights
from train.trainer import CAMoSETrainable, CAMoSETrainer, _collate_train_batch


def _mean_pit_sdri(
    estimates: np.ndarray, references: np.ndarray, mixture: np.ndarray, n_true: int
) -> float:
    """PIT SI-SDRi of the first n_true estimate rows against n_true references."""
    est = estimates[:n_true]
    ref = references[:n_true]
    return pit_si_sdr(est, ref, mixture).mean_si_sdri


def load_trained_model(
    checkpoint: str | Path | None,
    device: str = "cpu",
    feature_dim: int = 64,
    num_experts: int = 3,
    null_index: int = 2,
    fusion_hidden: int = 256,
    tau: float = 12.0,
) -> tuple[CAMoSETrainer, dict]:
    """Build a trainer wrapping a (possibly untrained) model + gate for eval."""
    model = CAMoSETrainable(
        feature_dim=feature_dim,
        num_experts=num_experts,
        null_index=null_index,
        fusion_hidden=fusion_hidden,
    )
    meta: dict = {}
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["model"])
        tau = float(payload.get("gate_tau", tau))
        meta = {"checkpoint": str(checkpoint), "gate_tau": tau}
    gate = CascadeGate(tau=tau)
    trainer = CAMoSETrainer(
        model=model,
        gate=gate,
        loss_fn=CompositeLoss(LossWeights()),
        device=device,
    )
    return trainer, meta


def evaluate_cache(
    cache_dir: str | Path,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 8,
    tau: float | None = None,
    sr_primary: bool = False,
) -> dict[str, Any]:
    """
    Score cascade vs single experts over a cached set.

    ``tau`` overrides the checkpoint's gate threshold — the escalation decision
    is made at inference (REAL-M quality < tau), so sweeping tau re-scores the
    SAME trained checkpoint with no retraining. Lower tau -> more escalation.

    ``sr_primary`` bypasses the learned CRRR fusion: escalated samples output the
    RAW expensive-expert (SR-CorrNet) streams, non-escalated output MossFormer2.
    This is the honest cascade when the fusion degrades a strong expert — it
    isolates the routing decision from fusion and needs no retraining (the
    trained heads are unused; only the quality gate routes).
    """
    ds = CachedExpertDataset(cache_dir)
    trainer, meta = load_trained_model(checkpoint, device=device)
    if tau is not None:
        trainer.gate.tau = float(tau)
        meta = {**meta, "gate_tau": float(tau), "gate_tau_overridden": True}
    trainer.model.eval()
    gate_tau = float(trainer.gate.tau)

    cascade, moss, expensive = [], [], []
    n_esc = 0
    n_total = 0

    order = list(range(len(ds)))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            items = [ds[i] for i in order[start : start + batch_size]]
            batch = _collate_train_batch(items)
            moss_np = batch.moss_streams.cpu().numpy()
            sr_np = batch.sr_streams.cpu().numpy()
            mix_np = batch.mixture.cpu().numpy()
            ref_np = batch.references.cpu().numpy()
            counts = batch.true_count.cpu().numpy().astype(int)

            if sr_primary:
                # Route on the quality gate; escalated -> raw SR-CorrNet, else moss.
                quality = batch.quality_scores_db.cpu().numpy().reshape(-1)
                esc = quality < gate_tau
                est = np.where(esc[:, None, None], sr_np, moss_np)
                n_esc += int(esc.sum())
            else:
                out = trainer.forward_batch(batch, compute_loss=False)
                est = out.estimates.cpu().numpy()
                n_esc += int(out.escalated_mask.sum().item())
            n_total += est.shape[0]

            for b in range(est.shape[0]):
                n_true = int(counts[b])
                cascade.append(_mean_pit_sdri(est[b], ref_np[b], mix_np[b], n_true))
                moss.append(_mean_pit_sdri(moss_np[b], ref_np[b], mix_np[b], n_true))
                expensive.append(_mean_pit_sdri(sr_np[b], ref_np[b], mix_np[b], n_true))

    cascade_m = float(np.mean(cascade))
    moss_m = float(np.mean(moss))
    exp_m = float(np.mean(expensive))
    best_single = max(moss_m, exp_m)
    best_single_name = "mossformer2" if moss_m >= exp_m else "expensive"

    report = {
        "n_samples": n_total,
        "escalation_rate": n_esc / max(n_total, 1),
        "si_sdri_db": {
            "cascade": cascade_m,
            "mossformer2": moss_m,
            "expensive": exp_m,
        },
        "best_single_expert": best_single_name,
        "cascade_minus_best_single_db": cascade_m - best_single,
        "cascade_beats_single_expert": bool(cascade_m > best_single),  # P2-INT4 verdict
        "mode": "sr_primary" if sr_primary else "fusion",
        **meta,
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate CA-MoSE cascade vs single experts")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--report", default=None, help="Optional JSON output path")
    p.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override the gate threshold (re-scores the same checkpoint; lower = more escalation)",
    )
    p.add_argument(
        "--sr-primary",
        action="store_true",
        help="Bypass fusion: escalated -> raw SR-CorrNet, else MossFormer2 (no retrain)",
    )
    args = p.parse_args()

    report = evaluate_cache(
        args.cache_dir,
        args.checkpoint,
        args.device,
        args.batch_size,
        tau=args.tau,
        sr_primary=args.sr_primary,
    )
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
