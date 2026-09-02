"""
Stage 2: Universal adapter baseline (Dev B, P1-B4 alternative).

Trains a SINGLE combined adapter on all three conditions simultaneously.
BLUEPRINT: if this is within 0.5 dB of the per-condition adapters on the
primary benchmark (noisy-reverberant LibriMix 2-spk SI-SDRi), adopt it.
Evaluated before gate is built; result determines whether Stage 3-4 proceeds.

Usage
-----
    python src/coralsep/train/stage2_universal.py \
        --librispeech-8k /data/LibriSpeech_8k \
        --rir-bank datasets/rirs/bank.json \
        --noise-dir /data/calmsep_noise \
        --output-dir outputs/stage2_universal \
        --device cuda \
        --epochs 40
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from coralsep.models.lora import LoRALibrary
from coralsep.train.losses import coralsep_loss
from coralsep.train.stage1_single import (
    _forward_with_grad,
    _get_inner_module,
    _load_model,
    _save_adapter,
    _seed_everything,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_UNIVERSAL_ADAPTER = "reverb"  # one adapter used for all conditions; re-using "reverb" slot


def _build_universal_dataset(args: argparse.Namespace) -> object:
    """Build a dataset that randomly draws from all three conditions."""
    import json

    import soundfile as sf

    from coralsep.data.condition_mixer import CoralSepMixer
    from coralsep.data.degradations import apply_codec, apply_noise, apply_reverb
    from coralsep.data.rir_bank import RirBank

    libri_8k = Path(args.librispeech_8k)
    source_files = sorted(libri_8k.rglob("*.flac")) + sorted(libri_8k.rglob("*.wav"))
    if not source_files:
        raise FileNotFoundError(f"No audio files in {libri_8k}")

    # Speaker holdout: handle both list and dict manifest formats
    held_out_spks: set[str] = set()
    manifest = libri_8k / "manifest_8k.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        items = data if isinstance(data, list) else list(data.get("splits", {}).values())
        for split_info in items:
            if isinstance(split_info, dict):
                if "dev" in split_info.get("split", "") or "test" in split_info.get("split", ""):
                    held_out_spks.update(
                        split_info.get("speaker_ids", split_info.get("speakers", []))
                    )

    seed = getattr(args, "seed", 42)
    rng = np.random.default_rng(seed)
    mixer = CoralSepMixer(source_files, held_out_speaker_ids=held_out_spks, rng=rng)

    # RirBank: strip bank.json suffix to get the directory (matches stage1 pattern)
    rir_bank = None
    _rb_arg = getattr(args, "rir_bank", None)
    if _rb_arg:
        _rb_path = Path(_rb_arg)
        _rb_dir = _rb_path.parent if _rb_path.suffix == ".json" else _rb_path
        if (_rb_dir / "bank.json").exists():
            rir_bank = RirBank(_rb_dir)
        else:
            log.warning("bank.json not found at %s; reverb will use clean audio", _rb_dir)

    # Pre-compute noise files ONCE — avoid 28k glob calls per epoch inside __getitem__
    noise_files: list[Path] = []
    noise_dir = Path(getattr(args, "noise_dir", ""))
    if noise_dir.exists():
        noise_files = sorted((noise_dir / "wham").glob("*_8k.wav")) + sorted(
            (noise_dir / "dns4").glob("*_8k.wav")
        )
        if not noise_files:
            noise_files = sorted(noise_dir.rglob("*_8k.wav"))
    log.info("Noise files found: %d", len(noise_files))

    max_clip = getattr(args, "max_clip_samples", 16000)

    class _UniversalDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
        def __init__(self, n: int) -> None:
            self.n = n
            self._rng = np.random.default_rng(seed + 99)

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict:
            import dataclasses

            from coralsep.data.mixer_stub import MixtureSample

            m = mixer.mix(split="train")

            # Clip before degradation — reverb on 30s is ~15× slower than on 2s
            if m.mixture.shape[0] > max_clip:
                _start = int(self._rng.integers(0, m.mixture.shape[0] - max_clip))
                m = dataclasses.replace(
                    m,
                    sample=MixtureSample(
                        mixture=m.sample.mixture[_start : _start + max_clip],
                        references=m.sample.references[:, _start : _start + max_clip],
                        sample_rate=m.sample.sample_rate,
                        utterance_id=m.sample.utterance_id,
                    ),
                )

            cond = self._rng.choice(["reverb", "noise", "codec"])
            if cond == "reverb" and rir_bank is not None:
                m = apply_reverb(m, rir_bank, self._rng)
            elif cond == "noise" and noise_files:
                nf = noise_files[int(self._rng.integers(len(noise_files)))]
                noise_wav, _ = sf.read(str(nf), dtype="float32")
                m = apply_noise(m, noise_wav, self._rng)
            elif cond == "codec":
                codec = str(self._rng.choice(["opus", "aac"]))
                m = apply_codec(m, codec, 12_000)

            mixture = torch.from_numpy(m.mixture).float()
            refs = torch.from_numpy(m.references).float()
            if mixture.shape[0] > max_clip:
                mixture = mixture[:max_clip]
                refs = refs[:, :max_clip]
            return {
                "mixture": mixture,
                "references": refs,
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
    use_bf16 = getattr(args, "bf16", True) and device.type == "cuda"
    log.info("Precision: %s", "BF16 autocast" if use_bf16 else "FP32")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ss_model = _load_model(
        getattr(args, "hf_model", "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"), device
    )
    inner = _get_inner_module(ss_model)

    lib = LoRALibrary(inner, adapter_names=["universal"])
    lib.freeze_base()
    # Move to device AFTER LoRA attachment so branches land on the right device
    inner.to(device)
    engine = getattr(ss_model, "engine", None)
    if engine is not None:
        for _attr in ("stft", "istft"):
            _m = getattr(engine, _attr, None)
            if _m is not None and hasattr(_m, "to"):
                _m.to(device)
    log.info("Universal adapter attached: %d modules  device: %s", lib.n_attached, device)

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
        n_batches = len(loader)
        for batch_idx, batch in enumerate(loader, 1):
            mixture = batch["mixture"].to(device)  # [B, T]
            references = batch["references"].to(device)  # [B, N, T]
            n_spks = batch["n_speakers"]
            B = mixture.shape[0]

            # Per-sample backward: forward one sample, backward immediately, free graph.
            # Avoids holding B activation graphs at once — safe even on large GPUs.
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            lib.set_adapter("universal", co_activate=False)
            lib.inject_gates()
            for b in range(B):
                wav = mixture[b].unsqueeze(0)  # [1, T]
                ref = references[b].unsqueeze(0)  # [1, N, T]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    waves, logits = _forward_with_grad(
                        ss_model,
                        wav,
                        n_spks=torch.tensor(n_spks[b]).to(device),
                    )
                est = waves.unsqueeze(0)  # [1, K, T]
                lg = logits.unsqueeze(0) if logits is not None else None
                losses = coralsep_loss(est, ref, lg, [n_spks[b]])
                (losses["total"] / B).backward()
                batch_loss += losses["total"].item() / B

            torch.nn.utils.clip_grad_norm_(lib.adapter_parameters("universal"), 5.0)
            optimizer.step()
            epoch_loss += batch_loss

            if batch_idx % max(1, n_batches // 10) == 0 or batch_idx == n_batches:
                log.info(
                    "  Epoch %d/%d  batch %d/%d  batch_loss=%.4f",
                    epoch,
                    epochs,
                    batch_idx,
                    n_batches,
                    batch_loss,
                )

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        log.info("Epoch %d/%d  loss=%.4f  time=%.1fs", epoch, epochs, avg, time.time() - t0)
        if avg < best_loss:
            best_loss = avg
            _save_adapter(lib, inner, "universal", output_dir / "best_universal.pt")

    _save_adapter(lib, inner, "universal", output_dir / "final_universal.pt")
    log.info("Universal adapter done. Best: %.4f", best_loss)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-8k", default="")
    p.add_argument("--data-root", default="")  # alias used by notebooks
    p.add_argument("--rir-bank", default="datasets/rirs/bank.json")
    p.add_argument("--noise-dir", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--checkpoint-dir", default="")  # alias used by notebooks
    p.add_argument("--stage1-dir", default="")  # accepted but unused in stage2
    p.add_argument("--hf-model", default="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--bf16", action="store_true", default=True, help="Use BF16 autocast (default: True)"
    )
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    args = p.parse_args()
    if not args.librispeech_8k and args.data_root:
        args.librispeech_8k = args.data_root
    if not args.output_dir and args.checkpoint_dir:
        args.output_dir = args.checkpoint_dir
    if not args.librispeech_8k:
        p.error("--librispeech-8k or --data-root is required")
    if not args.output_dir:
        p.error("--output-dir or --checkpoint-dir is required")
    return args


if __name__ == "__main__":
    train_universal(_parse_args())
