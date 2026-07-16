"""CLI: Stage-4 mandatory joint polish of adapters + gate (BLUEPRINT §8.5).

USER RUNS TRAINING:

    python -m train.train_joint_polish --config configs/gate.yaml --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.gate import GateMLP
from models.lora import orthogonal_penalty, register_lora
from pipeline.infer import MockCalmSepWrapper
from models.srcorrnet import SRCorrNetWrapper
from train.lora_harness import adapter_training_step
from utils.config import load_config
from utils.hashing import hash_config
from utils.logging import get_logger

log = get_logger("train_joint_polish")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/gate.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr-scale", type=float, default=0.1, help="Fraction of Stage-1 LR")
    p.add_argument("--o-lora-weight", type=float, default=0.0)
    p.add_argument("--out-dir", default="artifacts/joint_polish")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    wrapper = SRCorrNetWrapper(device=args.device)
    if not wrapper.is_available:
        if not args.dry_run:
            raise SystemExit("checkpoint required (or pass --dry-run)")
        wrapper = MockCalmSepWrapper()  # type: ignore[assignment]
    wrapper.load()
    library = register_lora(wrapper.base_nn)
    gate = GateMLP().to(args.device)
    # Joint: all adapter params + gate at 0.1x LR.
    params = library.parameters() + list(gate.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4 * args.lr_scale, weight_decay=0.01)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1 if args.dry_run else args.epochs):
        mix = torch.randn(16000)
        targets = [torch.randn(16000) * 0.05 for _ in range(2)]
        # Rotate active adapter emphasis each step for compound polish.
        active = ("reverb", "noise", "codec")[epoch % 3]
        stats = adapter_training_step(
            library=library,
            active_adapter=active,
            model_forward=lambda w, n_spks=None: wrapper.forward(w, n_spks=n_spks),
            mixture_wav=mix,
            target_wavs=targets,
            n_speakers=2,
            optimizer=opt,
            use_coactivation=True,
            o_lora_weight=args.o_lora_weight,
        )
        log.info("epoch_end", epoch=epoch, **{k: v for k, v in stats.items() if k.startswith("loss")})
        if args.dry_run:
            break

    payload = {
        "adapters": {n: library.state_dict_adapter(n) for n in library.adapter_names},
        "gate": gate.state_dict(),
        "config_sha256": hash_config(cfg),
        "o_lora_weight": args.o_lora_weight,
    }
    torch.save(payload, out / "joint_polish.pt")
    log.info("checkpoint_written", path=str(out / "joint_polish.pt"))


if __name__ == "__main__":
    main()
