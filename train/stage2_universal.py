"""
Stage 2: Universal adapter baseline (Dev B, P1-B4 alternative).

Trains a SINGLE combined adapter on all three conditions simultaneously.
BLUEPRINT: if this is within 0.5 dB of the per-condition adapters on the
primary benchmark (noisy-reverberant LibriMix 2-spk SI-SDRi), adopt it.
Evaluated before gate is built; result determines whether Stage 3-4 proceeds.

Usage
-----
    python train/stage2_universal.py \
        --librispeech-8k /data/LibriSpeech_8k \
        --rir-bank data/rirs/bank.json \
        --noise-dir /data/calmsep_noise \
        --output-dir outputs/stage2_universal \
        --device cuda \
        --epochs 40
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from models.lora import LoRALibrary, ADAPTER_NAMES, lora_summary
from train.losses import calmsep_loss
from train.stage1_single import (
    _seed_everything, _load_model, _get_inner_module,
    _extract_output_waves, _extract_logits, _save_adapter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_UNIVERSAL_ADAPTER = "reverb"  # one adapter used for all conditions; re-using "reverb" slot


def _build_universal_dataset(args: argparse.Namespace) -> object:
    """Build a dataset that randomly draws from all three conditions."""
    from data.calmsep_mixer import CalmSepMixer
    from data.rir_bank import RirBank
    from data.degradations import apply_reverb, apply_noise, apply_codec
    import json

    libri_8k = Path(args.librispeech_8k)
    source_files = sorted(libri_8k.rglob("*.flac")) + sorted(libri_8k.rglob("*.wav"))

    held_out_spks: set[str] = set()
    manifest = libri_8k / "manifest_8k.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        for split_info in data.get("splits", {}).values():
            if "dev" in split_info.get("split", "") or "test" in split_info.get("split", ""):
                held_out_spks.update(split_info.get("speakers", []))

    seed = getattr(args, "seed", 42)
    rng = np.random.default_rng(seed)
    mixer = CalmSepMixer(source_files, held_out_speaker_ids=held_out_spks, rng=rng)
    rir_bank = RirBank(Path(args.rir_bank)) if getattr(args, "rir_bank", None) else None

    class _UniversalDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
        def __init__(self, n: int) -> None:
            self.n = n
            self._rng = np.random.default_rng(seed + 99)

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict:
            m = mixer.mix(split="train")
            cond = self._rng.choice(["reverb", "noise", "codec"])
            if cond == "reverb" and rir_bank is not None:
                m = apply_reverb(m, rir_bank, self._rng)
            elif cond == "noise":
                noise_dir = Path(getattr(args, "noise_dir", ""))
                noise_files = (
                    sorted((noise_dir / "wham").glob("*_8k.wav")) +
                    sorted((noise_dir / "dns4").glob("*_8k.wav"))
                ) if noise_dir.exists() else []
                if noise_files:
                    import soundfile as sf
                    nf = noise_files[int(self._rng.integers(len(noise_files)))]
                    noise_wav, _ = sf.read(str(nf), dtype="float32")
                    m = apply_noise(m, noise_wav, self._rng)
            elif cond == "codec":
                codec = str(self._rng.choice(["opus", "aac"]))
                m = apply_codec(m, codec, 12_000)
            return {
                "mixture": torch.from_numpy(m.mixture).float(),
                "references": torch.from_numpy(m.references).float(),
                "n_speakers": m.recipe.n_speakers,
                "recipe": m.recipe.condition_vector(),
            }

    def _collate(batch: list[dict]) -> dict:
        max_t = max(b["mixture"].shape[0] for b in batch)
        max_n = max(b["references"].shape[0] for b in batch)
        mixtures, refs_list, ns, recipes = [], [], [], []
        for b in batch:
            t = b["mixture"].shape[0]
            n = b["references"].shape[0]
            mixtures.append(torch.nn.functional.pad(b["mixture"], (0, max_t - t)))
            r = torch.nn.functional.pad(b["references"], (0, max_t - t, 0, max_n - n))
            refs_list.append(r)
            ns.append(b["n_speakers"])
            recipes.append(b["recipe"])
        return {
            "mixture": torch.stack(mixtures),
            "references": torch.stack(refs_list),
            "n_speakers": ns,
            "recipe": recipes,
        }

    n_samples = getattr(args, "samples_per_epoch", 2000)
    return torch.utils.data.DataLoader(
        _UniversalDataset(n_samples),
        batch_size=getattr(args, "batch_size", 4),
        shuffle=True,
        num_workers=getattr(args, "num_workers", 2),
        collate_fn=_collate,
    )


def train_universal(args: argparse.Namespace) -> None:
    _seed_everything(getattr(args, "seed", 42))
    device = torch.device(getattr(args, "device", "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ss_model = _load_model(getattr(args, "hf_model", "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"), device)
    inner = _get_inner_module(ss_model)

    # Universal adapter: reuse the "reverb" name since only one adapter is trained.
    lib = LoRALibrary(inner, adapter_names=["universal"])
    lib.freeze_base()
    log.info("Universal adapter attached: %d modules", lib.n_attached)

    optimizer = optim.AdamW(
        lib.adapter_parameters("universal"), lr=getattr(args, "lr", 1e-4), weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(args, "epochs", 40))
    loader = _build_universal_dataset(args)
    epochs = getattr(args, "epochs", 40)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        inner.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in loader:
            mixture = batch["mixture"].to(device)
            references = batch["references"].to(device)
            n_spks = batch["n_speakers"]

            with lib.forward_context("universal", co_activate=False):
                B = mixture.shape[0]
                estimates_list, logits_list = [], []
                for b in range(B):
                    out = ss_model.process_waveform(  # type: ignore[attr-defined]
                        mixture[b].unsqueeze(0), n_spks=torch.tensor(n_spks[b])
                    )
                    estimates_list.append(_extract_output_waves(out, device))
                    lg = _extract_logits(out)
                    if lg is not None:
                        logits_list.append(lg)

                max_k = max(e.shape[0] for e in estimates_list)
                max_t = max(e.shape[1] for e in estimates_list)
                estimates = torch.zeros(B, max_k, max_t, device=device)
                for b, e in enumerate(estimates_list):
                    estimates[b, :e.shape[0], :e.shape[1]] = e

                logits_t = torch.stack(logits_list) if logits_list else None
                losses = calmsep_loss(estimates, references, logits_t, n_spks)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(lib.adapter_parameters("universal"), 5.0)
            optimizer.step()
            epoch_loss += losses["total"].item()

        scheduler.step()
        avg = epoch_loss / max(len(loader), 1)
        log.info("Epoch %d/%d  loss=%.4f  time=%.1fs", epoch, epochs, avg, time.time() - t0)
        if avg < best_loss:
            best_loss = avg
            _save_adapter(lib, inner, "universal", output_dir / "best_universal.pt")

    _save_adapter(lib, inner, "universal", output_dir / "final_universal.pt")
    log.info("Universal adapter done. Best: %.4f", best_loss)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-8k", required=True)
    p.add_argument("--rir-bank", default="data/rirs/bank.json")
    p.add_argument("--noise-dir", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hf-model", default="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    train_universal(_parse_args())
