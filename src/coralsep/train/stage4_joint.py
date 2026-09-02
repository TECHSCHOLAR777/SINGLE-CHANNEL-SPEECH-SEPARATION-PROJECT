"""
Stage 4: Joint gated end-to-end polishing (Dev B, P3-B1).

Gate + Level-2 analyzer continue from Stage 3. LoRA adapters from Stage 1
are now updated by gradients flowing from the separation loss through the
differentiable (tensor) gate weights:

    y = W0 x + g_r·Br(Ar x) + g_n·Bn(An x) + g_c·Bc(Ac x)
    where (g_r, g_n, g_c) = gate_net(condition) — grad-enabled tensors.

Critical differences from Stage 3:
  • No torch.no_grad() around the SR-CorrNet forward → sep_loss has grad
  • Gate values stay as Tensors (not converted to float) → grad flows into adapters
  • F.pad + stack for estimates instead of in-place zeros → gradient graph intact
  • inner.eval() keeps base BN/dropout deterministic; LoRA params still get grad
  • O-LoRA penalty always active (adapter A-matrix orthogonality)

Initialisation:
  gate + analyzer   ← Stage 3 best_gate.pt   (--stage3-dir or --gate-checkpoint)
  adapters          ← Stage 1 best_*.pt       (--stage1-dir)
  SR-CorrNet base   FROZEN throughout

Optimiser: AdamW, two param groups
  adapters          lr = 1e-5   wd = 1e-4
  gate + analyzer   lr = 2e-5   wd = 1e-5

Saves per epoch (when loss improves):
  best_joint.pt   — { gate, analyzer, adapter_state (flat A/B dict) }
  final_joint.pt  — same, after last epoch
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from coralsep.models.condition import Level2Analyzer, level1_tensor, level2_loss
from coralsep.models.gate import GateNetwork, oracle_gate
from coralsep.models.lora import ADAPTER_NAMES, LoRALibrary, LoRALinear, olora_penalty
from coralsep.train.losses import coralsep_loss
from coralsep.train.stage1_single import (
    _forward_with_grad,
    _get_inner_module,
    _load_model,
    _seed_everything,
)
from coralsep.train.stage3_gate import _build_gate_dataset, _load_adapters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_OLORA_ALPHA: float = 1e-3
_L1_LAMBDA: float = 1e-3


def _load_stage3_ckpt(
    path: Path,
    analyzer: Level2Analyzer,
    gate_net: GateNetwork,
    device: torch.device,
) -> None:
    ckpt = torch.load(str(path), map_location=device)
    analyzer.load_state_dict(ckpt["analyzer"])
    gate_net.load_state_dict(ckpt["gate"])
    log.info("Loaded Stage 3 checkpoint: %s", path)


def _save_joint_ckpt(
    path: Path,
    gate_net: GateNetwork,
    analyzer: Level2Analyzer,
    inner: torch.nn.Module,
) -> None:
    """Save gate, analyzer, and all refined LoRA A/B weights."""
    adapter_state: dict[str, torch.Tensor] = {}
    for mod_name, mod in inner.named_modules():
        if isinstance(mod, LoRALinear):
            for adapter_name, branch in mod.branches.items():
                for param_name, param in branch.named_parameters():
                    key = f"{mod_name}.branches.{adapter_name}.{param_name}"
                    adapter_state[key] = param.data.clone()
    torch.save(
        {
            "gate": gate_net.state_dict(),
            "analyzer": analyzer.state_dict(),
            "adapter_state": adapter_state,
        },
        path,
    )
    log.info("Saved: %s  (%d adapter tensors)", path, len(adapter_state))


def train_joint(args: argparse.Namespace) -> None:
    _seed_everything(getattr(args, "seed", 42))
    device = torch.device(getattr(args, "device", "cpu"))
    _want_amp = getattr(args, "bf16", True) and device.type == "cuda"
    use_bf16 = _want_amp and torch.cuda.is_bf16_supported()
    _amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if _want_amp else None)
    log.info("Precision: %s", f"AMP {_amp_dtype}" if _amp_dtype else "FP32")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen SR-CorrNet + attach LoRA branches
    ss_model = _load_model(
        getattr(args, "hf_model", "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"), device
    )
    inner = _get_inner_module(ss_model)
    lib = LoRALibrary(inner)
    lib.freeze_base()
    _load_adapters(inner, lib, args)
    inner.to(device)
    engine = getattr(ss_model, "engine", None)
    if engine is not None:
        for _attr in ("stft", "istft"):
            _mod = getattr(engine, _attr, None)
            if _mod is not None and hasattr(_mod, "to"):
                _mod.to(device)

    # Gate + Analyzer — warm-start from Stage 3
    analyzer = Level2Analyzer().to(device)
    gate_net = GateNetwork().to(device)
    gate_ckpt_path = Path(getattr(args, "gate_checkpoint", ""))
    if gate_ckpt_path.exists():
        _load_stage3_ckpt(gate_ckpt_path, analyzer, gate_net, device)
    else:
        log.warning(
            "Stage 3 checkpoint not found at %s; starting gate from scratch", gate_ckpt_path
        )

    # Two param groups: adapters get very low LR (fine-grained correction)
    adapter_params: list[torch.nn.Parameter] = []
    for name in ADAPTER_NAMES:
        adapter_params += lib.adapter_parameters(name)

    lr_adapter = getattr(args, "lr_adapter", 1e-5)
    lr_gate = getattr(args, "lr_gate", 2e-5)
    # Support legacy --stage1-lr / --lr aliases from older notebooks
    if getattr(args, "lr", 0.0) > 0:
        lr_gate = args.lr
        lr_adapter = args.lr / 2.0
    elif getattr(args, "stage1_lr", 0.0) > 0:
        lr_gate = args.stage1_lr / 10.0
        lr_adapter = lr_gate / 2.0

    optimizer = optim.AdamW(
        [
            {"params": adapter_params, "lr": lr_adapter, "weight_decay": 1e-4},
            {
                "params": list(analyzer.parameters()) + list(gate_net.parameters()),
                "lr": lr_gate,
                "weight_decay": 1e-5,
            },
        ]
    )
    epochs = getattr(args, "epochs", 20)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = _build_gate_dataset(args)
    best_loss = float("inf")

    log.info(
        "Stage 4 | epochs=%d  samples/epoch=%d  lr_adapter=%.2e  lr_gate=%.2e",
        epochs,
        getattr(args, "samples_per_epoch", 1000),
        lr_adapter,
        lr_gate,
    )

    for epoch in range(1, epochs + 1):
        # eval(): keeps base BN/dropout deterministic. LoRA params still receive grad
        # because requires_grad=True is independent of train/eval mode.
        inner.eval()
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

            # Per-sample backward: zero once, backward per sample, step once.
            # This keeps peak memory at 1 graph at a time instead of B graphs.
            optimizer.zero_grad()
            batch_loss = 0.0
            e0_list: list[torch.Tensor] = []  # float32, for level-2 causal init

            for b in range(B):
                wav = mixture[b].unsqueeze(0)
                # Clip audio — gradient tape for SR-CorrNet is large; 2 s fits 16 GB GPU
                _MAX_SAMPLES = getattr(args, "max_audio_samples", 8000)
                if wav.shape[-1] > _MAX_SAMPLES:
                    wav = wav[..., :_MAX_SAMPLES]

                l1_feat = level1_tensor(wav.squeeze(0)).to(device)

                # Hook: capture E(0) as float32 regardless of autocast context
                e0_capture: dict = {}

                def _e0_hook(m: object, inp: object, out: object, _cap: dict = e0_capture) -> None:
                    t = out if isinstance(out, torch.Tensor) else out[0]
                    _cap["e0"] = t.detach().float()

                hook_handle = None
                if hasattr(inner, "encoder"):
                    hook_handle = inner.encoder.register_forward_hook(_e0_hook)

                # Level-2 causal init from previous sample's e0
                with torch.no_grad():
                    if e0_list:
                        prev = e0_list[-1]
                        l2_init = analyzer.feature_vector(
                            prev.unsqueeze(0) if prev.ndim == 3 else prev
                        ).squeeze(0)
                    else:
                        l2_init = torch.zeros(6, device=device)

                cond_b = torch.cat([l1_feat.float(), l2_init.float()], dim=-1).unsqueeze(
                    0
                )  # (1, 10)

                # Gate + model forward inside autocast (bf16 activations → half memory)
                with torch.autocast(
                    "cuda", dtype=_amp_dtype or torch.float32, enabled=_amp_dtype is not None
                ):
                    gate_b = gate_net(cond_b).squeeze(0)  # (3,) bf16 with grad
                    lib.set_gates({ADAPTER_NAMES[i]: gate_b[i] for i in range(3)})
                    lib.inject_gates()
                    waves_sep, lg_sep = _forward_with_grad(
                        ss_model, wav, n_spks=torch.tensor(n_spks[b])
                    )

                if hook_handle:
                    hook_handle.remove()

                e0 = e0_capture.get("e0")
                if e0 is not None:
                    e0_list.append(e0)

                # --- Per-sample losses (fp32 after explicit casts) ---
                waves_b = waves_sep.unsqueeze(0)  # (1, K, T)
                ref_b = references[b : b + 1]  # (1, K_ref, T) — coralsep_loss trims
                lg_b = lg_sep.float().unsqueeze(0) if lg_sep is not None else None
                sep_loss_b = coralsep_loss(waves_b, ref_b, lg_b, n_spks[b : b + 1])["total"]

                gate_b_f = gate_b.float()  # bf16 → fp32; grad still flows
                oracle_b = oracle_gate([recipes[b]], device=device)
                bce_b = F.binary_cross_entropy(
                    gate_b_f.unsqueeze(0) / gate_net.gate_scale, oracle_b
                )
                l1_b = _L1_LAMBDA * gate_b_f.abs().mean()

                l2_b = torch.tensor(0.0, device=device)
                if e0 is not None:
                    with torch.autocast(
                        "cuda", dtype=_amp_dtype or torch.float32, enabled=_amp_dtype is not None
                    ):
                        e0_in = e0.unsqueeze(0) if e0.ndim == 3 else e0
                        l2_b = level2_loss(analyzer, e0_in, [recipes[b]])

                # Normalise by B so effective step equals a full-batch backward
                loss_b = (sep_loss_b + bce_b + l1_b + l2_b) / B
                loss_b.backward()  # frees this sample's graph immediately
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                batch_loss += loss_b.item()

            # Clear stale injected gates
            lib.set_gates({n: 0.0 for n in ADAPTER_NAMES})
            lib.inject_gates()

            # O-LoRA penalty — model weights are fp32, computed once per batch
            olo = olora_penalty(inner, alpha=_OLORA_ALPHA)
            olo.backward()
            batch_loss += olo.item()

            torch.nn.utils.clip_grad_norm_(
                adapter_params + list(analyzer.parameters()) + list(gate_net.parameters()),
                5.0,
            )
            optimizer.step()
            epoch_loss += batch_loss

        scheduler.step()
        avg = epoch_loss / max(len(loader), 1)
        log.info("Epoch %d/%d  loss=%.4f  time=%.1fs", epoch, epochs, avg, time.time() - t0)

        if avg < best_loss:
            best_loss = avg
            _save_joint_ckpt(out_dir / "best_joint.pt", gate_net, analyzer, inner)

    _save_joint_ckpt(out_dir / "final_joint.pt", gate_net, analyzer, inner)
    log.info("Joint training done. Best loss: %.4f", best_loss)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-8k", default="")
    p.add_argument("--data-root", default="")
    p.add_argument("--rir-bank", default="datasets/rirs/bank.json")
    p.add_argument("--noise-dir", default="")
    p.add_argument("--adapter-reverb", default="")
    p.add_argument("--adapter-noise", default="")
    p.add_argument("--adapter-codec", default="")
    p.add_argument("--stage1-dir", default="")
    p.add_argument("--stage3-dir", default="")
    p.add_argument("--gate-checkpoint", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--checkpoint-dir", default="")
    p.add_argument("--hf-model", default="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr-adapter", type=float, default=1e-5)
    p.add_argument("--lr-gate", type=float, default=2e-5)
    # Legacy aliases kept for backwards-compat with older notebook versions
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--stage1-lr", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-epoch", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--max-audio-samples", type=int, default=8000)
    args = p.parse_args()
    if not args.librispeech_8k and args.data_root:
        args.librispeech_8k = args.data_root
    if not args.output_dir and args.checkpoint_dir:
        args.output_dir = args.checkpoint_dir
    if args.stage1_dir:
        s1 = Path(args.stage1_dir)
        if not args.adapter_reverb:
            args.adapter_reverb = str(s1 / "best_reverb.pt")
        if not args.adapter_noise:
            args.adapter_noise = str(s1 / "best_noise.pt")
        if not args.adapter_codec:
            args.adapter_codec = str(s1 / "best_codec.pt")
    if args.stage3_dir and not args.gate_checkpoint:
        args.gate_checkpoint = str(Path(args.stage3_dir) / "best_gate.pt")
    if not args.librispeech_8k:
        p.error("--librispeech-8k or --data-root is required")
    if not args.output_dir:
        p.error("--output-dir or --checkpoint-dir is required")
    return args


if __name__ == "__main__":
    train_joint(_parse_args())
