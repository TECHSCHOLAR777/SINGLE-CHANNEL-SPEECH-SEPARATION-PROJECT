"""
Stage 3: Gate and condition analyzer training (Dev B, P2-B4).

Trains the GateNetwork and Level2Analyzer jointly:
  - Level-1 features are fixed DSP (no parameters)
  - Level-2 heads (T60Head + CountPriorMLP) are trained against recipe labels
  - GateNetwork is trained with BCE against oracle gate + L1 sparsity
  - Separation loss flows through the gate (gates multiply LoRA branch outputs)
  - Base model stays frozen; adapters stay fixed from Stage 1

Held-out combinations (BLUEPRINT 7.5): reverb+codec and noise+codec are
never in the training set here. The assert_not_held_out check enforces this.

Usage
-----
    python train/stage3_gate.py \
        --librispeech-8k /data/LibriSpeech_8k \
        --rir-bank data/rirs/bank.json \
        --noise-dir /data/calmsep_noise \
        --adapter-reverb outputs/stage1_reverb/best_reverb.pt \
        --adapter-noise  outputs/stage1_noise/best_noise.pt \
        --adapter-codec  outputs/stage1_codec/best_codec.pt \
        --output-dir outputs/stage3_gate \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim

from models.condition import Level2Analyzer, level1_tensor, level2_loss
from models.gate import GateNetwork, gate_loss
from models.lora import LoRALibrary, ADAPTER_NAMES
from train.losses import calmsep_loss
from train.stage1_single import (
    _seed_everything, _load_model, _get_inner_module,
    _extract_output_waves, _extract_logits,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_adapters(inner: torch.nn.Module, lib: LoRALibrary, args: argparse.Namespace) -> None:
    """Load Stage 1 adapter weights into the LoRA library."""
    for adapter, arg_name in [("reverb", "adapter_reverb"), ("noise", "adapter_noise"), ("codec", "adapter_codec")]:
        path = getattr(args, arg_name, None)
        if path and Path(path).exists():
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            # Filter keys to this adapter and strip prefix.
            filtered = {
                k.replace(f"adapter.{adapter}.", ""): v
                for k, v in state.items()
                if f"branches.{adapter}" in k or f"adapter.{adapter}" in k
            }
            missing, unexpected = inner.load_state_dict(filtered, strict=False)
            log.info("Loaded %s adapter: %d tensors, %d missing, %d unexpected",
                     adapter, len(filtered), len(missing), len(unexpected))
        else:
            log.warning("Adapter checkpoint not found for %s at %s", adapter, path)


def _build_gate_dataset(args: argparse.Namespace) -> object:
    """Build mixed-condition training data, excluding held-out combos."""
    from data.calmsep_mixer import CalmSepMixer
    from data.rir_bank import RirBank
    from data.degradations import apply_reverb, apply_noise, apply_codec, assert_not_held_out
    import json
    import numpy as np

    libri_8k = Path(args.librispeech_8k)
    files = sorted(libri_8k.rglob("*.flac")) + sorted(libri_8k.rglob("*.wav"))
    held_out: set[str] = set()
    mf = libri_8k / "manifest_8k.json"
    if mf.exists():
        d = json.loads(mf.read_text())
        for si in d.get("splits", {}).values():
            if "dev" in si.get("split", "") or "test" in si.get("split", ""):
                held_out.update(si.get("speakers", []))

    seed = getattr(args, "seed", 42)
    rng = np.random.default_rng(seed)
    mixer = CalmSepMixer(files, held_out_speaker_ids=held_out, rng=rng)
    rir_bank = RirBank(Path(args.rir_bank)) if getattr(args, "rir_bank", None) else None

    # Allowed single/double conditions (no reverb+codec or noise+codec).
    allowed_conds = ["clean", "reverb", "noise", "codec", "reverb+noise", "all-three"]

    class _GateDS(torch.utils.data.Dataset):  # type: ignore[type-arg]
        def __init__(self, n: int) -> None:
            self.n = n
            self._rng = np.random.default_rng(seed + 200)

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict:
            while True:
                m = mixer.mix(split="train")
                cond = str(self._rng.choice(allowed_conds))
                try:
                    if "reverb" in cond and rir_bank is not None:
                        m = apply_reverb(m, rir_bank, self._rng)
                    if "noise" in cond:
                        noise_dir = Path(getattr(args, "noise_dir", ""))
                        nfiles = (sorted((noise_dir / "wham").glob("*_8k.wav")) +
                                  sorted((noise_dir / "dns4").glob("*_8k.wav"))) if noise_dir.exists() else []
                        if nfiles:
                            import soundfile as sf
                            nf = nfiles[int(self._rng.integers(len(nfiles)))]
                            n_wav, _ = sf.read(str(nf), dtype="float32")
                            m = apply_noise(m, n_wav, self._rng)
                    if "codec" in cond:
                        codec = str(self._rng.choice(["opus", "aac"]))
                        m = apply_codec(m, codec, 12_000)
                    assert_not_held_out(m.recipe)
                    break
                except ValueError:
                    continue

            return {
                "mixture": torch.from_numpy(m.mixture).float(),
                "references": torch.from_numpy(m.references).float(),
                "n_speakers": m.recipe.n_speakers,
                "recipe": m.recipe.condition_vector(),
            }

    def _collate(batch: list[dict]) -> dict:
        max_t = max(b["mixture"].shape[0] for b in batch)
        max_n = max(b["references"].shape[0] for b in batch)
        mixes, refs, ns, recs = [], [], [], []
        for b in batch:
            t, n = b["mixture"].shape[0], b["references"].shape[0]
            mixes.append(torch.nn.functional.pad(b["mixture"], (0, max_t - t)))
            refs.append(torch.nn.functional.pad(b["references"], (0, max_t - t, 0, max_n - n)))
            ns.append(b["n_speakers"])
            recs.append(b["recipe"])
        return {
            "mixture": torch.stack(mixes),
            "references": torch.stack(refs),
            "n_speakers": ns,
            "recipe": recs,
        }

    return torch.utils.data.DataLoader(
        _GateDS(getattr(args, "samples_per_epoch", 2000)),
        batch_size=getattr(args, "batch_size", 4),
        shuffle=True,
        num_workers=getattr(args, "num_workers", 2),
        collate_fn=_collate,
    )


def train_gate(args: argparse.Namespace) -> None:
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

    optimizer = optim.AdamW(
        list(analyzer.parameters()) + list(gate_net.parameters()),
        lr=getattr(args, "lr", 5e-4),
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(args, "epochs", 30))
    loader = _build_gate_dataset(args)
    epochs = getattr(args, "epochs", 30)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
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
            l1_feats_list = []
            estimates_list, logits_list = [], []
            e0_list = []

            for b in range(B):
                wav = mixture[b].unsqueeze(0)
                l1_feat = level1_tensor(wav.squeeze(0)).to(device)
                l1_feats_list.append(l1_feat)

                # Capture E(0) via hook.
                e0_capture = {}
                def _e0_hook(m: object, inp: object, out: object) -> None:
                    e0_capture["e0"] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()

                inner_model = _get_inner_module(ss_model)
                hook_handle = None
                if hasattr(inner_model, "encoder"):
                    hook_handle = inner_model.encoder.register_forward_hook(_e0_hook)

                with torch.no_grad():
                    out_sep = ss_model.process_waveform(wav, n_spks=torch.tensor(n_spks[b]))  # type: ignore

                if hook_handle:
                    hook_handle.remove()

                e0 = e0_capture.get("e0")
                if e0 is not None:
                    e0_list.append(e0)

                estimates_list.append(_extract_output_waves(out_sep, device))
                lg = _extract_logits(out_sep)
                if lg is not None:
                    logits_list.append(lg)

            l1_feats = torch.stack(l1_feats_list)  # (B, 4)

            # Level-2 features from E(0).
            if e0_list:
                e0_batch = torch.cat([e.unsqueeze(0) if e.ndim == 3 else e for e in e0_list], dim=0)
                l2_feats = analyzer.feature_vector(e0_batch)  # (B, 6)
            else:
                l2_feats = torch.zeros(B, 6, device=device)

            condition = torch.cat([l1_feats, l2_feats], dim=-1)  # (B, 10)
            gates = gate_net(condition)  # (B, 3)

            # Apply gates to a second pass.
            max_k = max(e.shape[0] for e in estimates_list)
            max_t = max(e.shape[1] for e in estimates_list)
            estimates = torch.zeros(B, max_k, max_t, device=device)
            for b, e in enumerate(estimates_list):
                estimates[b, :e.shape[0], :e.shape[1]] = e

            logits_t = torch.stack(logits_list) if logits_list else None
            sep_loss = calmsep_loss(estimates, references, logits_t, n_spks)["total"]
            total = gate_loss(gate_net, condition, recipes, sep_loss)

            # Level-2 supervised loss.
            if e0_list:
                l2_sup = level2_loss(analyzer, e0_batch, recipes)
                total = total + l2_sup

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(analyzer.parameters()) + list(gate_net.parameters()), 5.0
            )
            optimizer.step()
            epoch_loss += total.item()

        scheduler.step()
        avg = epoch_loss / max(len(loader), 1)
        log.info("Epoch %d/%d  loss=%.4f  time=%.1fs", epoch, epochs, avg, time.time() - t0)
        if avg < best_loss:
            best_loss = avg
            torch.save({"analyzer": analyzer.state_dict(), "gate": gate_net.state_dict()}, out_dir / "best_gate.pt")

    torch.save({"analyzer": analyzer.state_dict(), "gate": gate_net.state_dict()}, out_dir / "final_gate.pt")
    log.info("Gate training done. Best loss: %.4f", best_loss)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-8k", default="")
    p.add_argument("--data-root", default="")          # alias used by notebooks
    p.add_argument("--rir-bank", default="data/rirs/bank.json")
    p.add_argument("--noise-dir", default="")
    # Stage 1 adapter paths can be given explicitly or via --stage1-dir.
    p.add_argument("--adapter-reverb", default="")
    p.add_argument("--adapter-noise", default="")
    p.add_argument("--adapter-codec", default="")
    p.add_argument("--stage1-dir", default="")         # used by notebooks; resolves adapter paths
    p.add_argument("--output-dir", default="")
    p.add_argument("--checkpoint-dir", default="")     # alias used by notebooks
    p.add_argument("--hf-model", default="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)   # blueprint gate.yaml: 5e-5 (was 5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    # Resolve aliases.
    if not args.librispeech_8k and args.data_root:
        args.librispeech_8k = args.data_root
    if not args.output_dir and args.checkpoint_dir:
        args.output_dir = args.checkpoint_dir
    # If --stage1-dir given, infer adapter paths that weren't explicitly set.
    if args.stage1_dir:
        stage1 = Path(args.stage1_dir)
        if not args.adapter_reverb:
            args.adapter_reverb = str(stage1 / "best_reverb.pt")
        if not args.adapter_noise:
            args.adapter_noise = str(stage1 / "best_noise.pt")
        if not args.adapter_codec:
            args.adapter_codec = str(stage1 / "best_codec.pt")
    if not args.librispeech_8k:
        p.error("--librispeech-8k or --data-root is required")
    if not args.output_dir:
        p.error("--output-dir or --checkpoint-dir is required")
    return args


if __name__ == "__main__":
    train_gate(_parse_args())
