"""CLI: Stage-3 condition analyzer + gate training (BLUEPRINT §8.4).

USER RUNS TRAINING on GPU after adapters exist:

    python -m train.train_gate --config configs/gate.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from models.condition import ConditionAnalyzer
from models.gate import GateMLP
from utils.config import load_config
from utils.hashing import hash_config
from utils.logging import get_logger

log = get_logger("train_gate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/gate.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out-dir", default="artifacts/gate")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    epochs = args.epochs or int(cfg.get("epochs", 15))
    sparsity = float(cfg.get("sparsity_weight", 1e-3))

    analyzer = ConditionAnalyzer().to(args.device)
    gate = GateMLP(mode=cfg.get("mode", "per_layer")).to(args.device)
    gate.sparsity_weight = sparsity
    opt = torch.optim.AdamW(
        list(analyzer.parameters()) + list(gate.parameters()),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=0.01,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1 if args.dry_run else epochs):
        # Synthetic supervised batch — replace with compound-condition DataLoader.
        e0 = torch.randn(1, 16, 65, 128, device=args.device)
        level1 = {
            "snr_db": 5.0,
            "codec_class": "none",
            "codec_class_idx": 0,
            "codec_bitrate_bps": 0.0,
            "voiced_density": 0.6,
        }
        t60 = torch.tensor([0.4], device=args.device)
        n_cls = torch.tensor([0], device=args.device)  # N=2
        losses = analyzer.condition_losses(e0, level1, t60, n_cls)
        from models.condition import ConditionVector

        cond = ConditionVector(
            snr_db=5.0, t60_s=0.4, voiced_density=0.6, count_prior=[0.7, 0.1, 0.1, 0.1]
        )
        c = cond.to_tensor(args.device)
        gates_t = gate.raw_forward(c)
        # Clean target near-zero gates + sparsity.
        target_gates = torch.zeros_like(gates_t)
        gate_l1 = F.l1_loss(gates_t, target_gates) + gate.sparsity_loss(gates_t)
        loss = losses["total"] + gate_l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log.info("epoch_end", epoch=epoch, loss=float(loss.detach().cpu()))
        if args.dry_run:
            break

    torch.save(
        {
            "analyzer": analyzer.state_dict(),
            "gate": gate.state_dict(),
            "config_sha256": hash_config(cfg),
        },
        out / "condition_gate.pt",
    )
    log.info("checkpoint_written", path=str(out / "condition_gate.pt"))


if __name__ == "__main__":
    main()
