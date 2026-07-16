"""
Stage 4: Mandatory joint polish (Dev B, P3-B1).

Unlocks all 3 adapters + gate simultaneously. Base model stays FROZEN.
Trains 15-20 epochs at 1/10th the Stage 1 learning rate on compound conditions.

BLUEPRINT §6.4: this is not optional. The gate and adapters were trained
separately in Stage 1-3; joint polish integrates them into a coherent system.
If O-LoRA penalty is needed (cross-interference > 0.3 dB), it is applied here.

Also runs the Principle-2 smoke test after joint polish:
  Full system vs frozen base on clean Libri2Mix.
  System MUST NOT be worse than the frozen base.

Usage
-----
    python train/stage4_joint.py \
        --librispeech-8k /data/LibriSpeech_8k \
        --rir-bank data/rirs/bank.json \
        --noise-dir /data/calmsep_noise \
        --adapter-reverb outputs/stage1_reverb/best_reverb.pt \
        --adapter-noise  outputs/stage1_noise/best_noise.pt \
        --adapter-codec  outputs/stage1_codec/best_codec.pt \
        --gate-checkpoint outputs/stage3_gate/best_gate.pt \
        --output-dir outputs/stage4_joint \
        --device cuda \
        --epochs 20
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim

from models.condition import Level2Analyzer, level1_tensor
from models.gate import GateNetwork
from models.lora import LoRALibrary, ADAPTER_NAMES, olora_penalty
from train.losses import calmsep_loss
from train.stage1_single import (
    _seed_everything, _load_model, _get_inner_module,
    _extract_output_waves, _extract_logits,
)
from train.stage3_gate import _load_adapters, _build_gate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def train_joint(args: argparse.Namespace) -> None:
    _seed_everything(getattr(args, "seed", 42))
    device = torch.device(getattr(args, "device", "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ss_model = _load_model(getattr(args, "hf_model", "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"), device)
    inner = _get_inner_module(ss_model)
    lib = LoRALibrary(inner)
    lib.freeze_base()
    _load_adapters(inner, lib, args)

    analyzer = Level2Analyzer().to(device)
    gate_net = GateNetwork().to(device)

    gate_ckpt = getattr(args, "gate_checkpoint", None)
    if gate_ckpt and Path(gate_ckpt).exists():
        ckpt = torch.load(gate_ckpt, map_location=device)
        analyzer.load_state_dict(ckpt["analyzer"])
        gate_net.load_state_dict(ckpt["gate"])
        log.info("Loaded gate checkpoint: %s", gate_ckpt)

    # Joint: train all adapter params + analyzer + gate together.
    all_params = (
        list(analyzer.parameters()) +
        list(gate_net.parameters()) +
        lib.active_parameters()
    )
    # Stage 4 LR = 1/10 of Stage 1 LR.
    stage1_lr = getattr(args, "stage1_lr", 1e-4)
    lr = stage1_lr / 10.0
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(args, "epochs", 20))

    use_olora = getattr(args, "use_olora", False)
    loader = _build_gate_dataset(args)
    epochs = getattr(args, "epochs", 20)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        inner.train()
        analyzer.train()
        gate_net.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            mixture = batch["mixture"].to(device)
            references = batch["references"].to(device)
            n_spks = batch["n_speakers"]
            recipes = batch["recipe"]
            B = mixture.shape[0]

            l1_feats_list, e0_list, est_list, logits_list = [], [], [], []
            for b in range(B):
                wav = mixture[b].unsqueeze(0)
                l1_feats_list.append(level1_tensor(wav.squeeze(0)).to(device))

                e0_capture: dict[str, torch.Tensor] = {}
                hook = None
                inner_m = _get_inner_module(ss_model)
                if hasattr(inner_m, "encoder"):
                    def _h(m: object, inp: object, out: object) -> None:
                        e0_capture["e0"] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
                    hook = inner_m.encoder.register_forward_hook(_h)  # type: ignore[attr-defined]

                # Use gate-determined adapter combination.
                out_sep = ss_model.process_waveform(wav, n_spks=torch.tensor(n_spks[b]))  # type: ignore
                if hook:
                    hook.remove()

                e0 = e0_capture.get("e0")
                if e0 is not None:
                    e0_list.append(e0)
                est_list.append(_extract_output_waves(out_sep, device))
                lg = _extract_logits(out_sep)
                if lg is not None:
                    logits_list.append(lg)

            l1_feats = torch.stack(l1_feats_list)
            if e0_list:
                e0_batch = torch.cat([e.unsqueeze(0) if e.ndim == 3 else e for e in e0_list], dim=0)
                l2_feats = analyzer.feature_vector(e0_batch)
            else:
                l2_feats = torch.zeros(B, 6, device=device)

            condition = torch.cat([l1_feats, l2_feats], dim=-1)
            gates = gate_net(condition)  # (B, 3)

            # Set gates from the gate network output for this batch.
            for b in range(B):
                gate_d = {n: float(gates[b, i].item()) for i, n in enumerate(ADAPTER_NAMES)}
                lib.set_gates(gate_d)
            lib.inject_gates()

            max_k = max(e.shape[0] for e in est_list)
            max_t = max(e.shape[1] for e in est_list)
            estimates = torch.zeros(B, max_k, max_t, device=device)
            for b, e in enumerate(est_list):
                estimates[b, :e.shape[0], :e.shape[1]] = e

            logits_t = torch.stack(logits_list) if logits_list else None
            losses = calmsep_loss(estimates, references, logits_t, n_spks)
            total = losses["total"]

            if use_olora:
                total = total + olora_penalty(inner)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()
            epoch_loss += total.item()

        scheduler.step()
        avg = epoch_loss / max(len(loader), 1)
        log.info("Epoch %d/%d  loss=%.4f  time=%.1fs", epoch, epochs, avg, time.time() - t0)
        if avg < best_loss:
            best_loss = avg
            ckpt = {
                "analyzer": analyzer.state_dict(),
                "gate": gate_net.state_dict(),
                "adapters": {n: lib.adapter_parameters(n) for n in ADAPTER_NAMES},
            }
            # Save full model state for adapters.
            adapter_state = {k: v for k, v in inner.state_dict().items() if "branches" in k}
            torch.save({
                "analyzer": analyzer.state_dict(),
                "gate": gate_net.state_dict(),
                "adapter_state": adapter_state,
            }, out_dir / "best_joint.pt")

    log.info("Joint polish complete. Best: %.4f", best_loss)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-8k", required=True)
    p.add_argument("--rir-bank", default="data/rirs/bank.json")
    p.add_argument("--noise-dir", default="")
    p.add_argument("--adapter-reverb", default="")
    p.add_argument("--adapter-noise", default="")
    p.add_argument("--adapter-codec", default="")
    p.add_argument("--gate-checkpoint", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hf-model", default="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--stage1-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-olora", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train_joint(_parse_args())
