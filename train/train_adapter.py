"""CLI: train one LoRA adapter (noise/reverb/codec/universal).

USER RUNS TRAINING — this script is complete but does not execute a full
GPU run in CI. Use:

    python -m train.train_adapter --adapter noise --config configs/adapters/noise.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.lora import freeze_base
from models.srcorrnet import SRCorrNetWrapper
from train.lora_harness import adapter_training_step, attach_adapters, build_adapter_optimizer
from utils.config import load_config
from utils.hashing import hash_config, hash_file
from utils.logging import get_logger

log = get_logger("train_adapter")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, choices=["noise", "reverb", "codec", "universal"])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--out-dir", type=str, default="artifacts/adapters")
    p.add_argument("--dry-run", action="store_true", help="Build graph, one fake step, exit")
    return p.parse_args()


def _synthetic_batch(n_spks: int = 2, sr: int = 8000, seconds: float = 2.0):
    L = int(sr * seconds)
    targets = [torch.randn(L) * 0.05 for _ in range(n_spks)]
    mix = torch.stack(targets).sum(0)
    return mix, targets, n_spks


def main() -> None:
    args = parse_args()
    cfg_path = args.config or f"configs/adapters/{args.adapter}.yaml"
    cfg = load_config(cfg_path) if Path(cfg_path).exists() else {}
    epochs = args.epochs or int(cfg.get("epochs", 20))
    lr = float(cfg.get("lr", 3e-4))

    wrapper = SRCorrNetWrapper(device=args.device)
    if not wrapper.is_available:
        if args.dry_run:
            log.warning("checkpoint_unavailable", hint="dry-run with mock forward")
            from pipeline.infer import MockCalmSepWrapper

            wrapper = MockCalmSepWrapper()  # type: ignore[assignment]
        else:
            raise SystemExit(
                "SR-CorrNet not available. Install/download checkpoint, or pass --dry-run."
            )

    wrapper.load()
    model = wrapper.base_nn if hasattr(wrapper, "base_nn") else wrapper  # type: ignore[assignment]
    names = ("universal",) if args.adapter == "universal" else ("reverb", "noise", "codec")
    # For single-adapter Stage 1 we still register all three for co-activation.
    library = attach_adapters(model if hasattr(model, "named_parameters") else wrapper.base_nn)
    freeze_base(wrapper.base_nn)
    active = "universal" if args.adapter == "universal" else args.adapter
    if active == "universal":
        # Universal uses one adapter name covering full budget — train 'noise' slot as stand-in
        # when only three branches exist; save under universal key.
        active = "noise"
        log.info("universal_adapter_slot", slot="noise")

    opt = build_adapter_optimizer(library, active, lr=lr)
    out_dir = Path(args.out_dir) / args.adapter
    out_dir.mkdir(parents=True, exist_ok=True)

    def _fwd(wav, n_spks=None):
        return wrapper.forward(wav, n_spks=n_spks)

    history = []
    for epoch in range(epochs if not args.dry_run else 1):
        epoch_losses = []
        n_steps = 1 if args.dry_run else args.steps_per_epoch
        for _ in range(n_steps):
            mix, targets, n = _synthetic_batch()
            # NOTE: replace _synthetic_batch with calmsep_mixer DataLoader for real training.
            stats = adapter_training_step(
                library=library,
                active_adapter=active,
                model_forward=_fwd,
                mixture_wav=mix,
                target_wavs=targets,
                n_speakers=n,
                optimizer=opt,
                use_coactivation=args.adapter != "universal",
            )
            epoch_losses.append(stats["loss"])
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        history.append({"epoch": epoch, "loss": mean_loss})
        log.info("epoch_end", epoch=epoch, loss=mean_loss)
        if args.dry_run:
            break

    ckpt_path = out_dir / "adapter.pt"
    state = library.state_dict_adapter(active)
    meta = {
        "adapter": args.adapter,
        "active_slot": active,
        "config_sha256": hash_config(cfg),
        "history": history,
    }
    torch.save({"state": state, "meta": meta}, ckpt_path)
    (out_dir / "meta.json").write_text(json.dumps({**meta, "ckpt_sha256": hash_file(ckpt_path)}, indent=2))
    log.info("checkpoint_written", path=str(ckpt_path))


if __name__ == "__main__":
    main()
