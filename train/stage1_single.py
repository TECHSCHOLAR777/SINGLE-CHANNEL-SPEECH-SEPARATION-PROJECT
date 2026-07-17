"""
Stage 1: Single adapter training (Dev B, P1-B4/B5/B6).

Trains one adapter (reverb | noise | codec) at a time on its dedicated condition.
The frozen base model provides the separation backbone; LoRA branches add
condition-specific residual corrections.

Co-activation warm-up is always on: other adapters are active at U(0.0, 0.2)
so Stage 4 joint polish sees a model that already tolerates composition.

Usage
-----
    python train/stage1_single.py \
        --adapter reverb \
        --librispeech-8k /data/LibriSpeech_8k \
        --rir-bank data/rirs/bank.json \
        --noise-dir /data/calmsep_noise \     # only for noise adapter
        --output-dir outputs/stage1_reverb \
        --device cuda \
        --epochs 40 \
        --batch-size 4 \
        --lr 1e-4

For Kaggle: run with --device cuda --batch-size 4 on a T4 instance.
Each adapter takes ~6-8 h on one T4 GPU with 40 epochs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from models.lora import LoRALibrary, ADAPTER_NAMES, lora_summary
from train.losses import calmsep_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(hf_model: str, device: torch.device) -> object:
    """Load the frozen SR-CorrNet checkpoint."""
    import sys
    try:
        from sr_corrnet import SSInference  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "SR-CorrNet-SS not installed. Run:\n"
            "  git clone https://github.com/dmlguq456/SR_CorrNet_SS.git\n"
            '  cd SR_CorrNet_SS && pip install -e ".[hub]"'
        )
    log.info("Loading frozen checkpoint: %s", hf_model)
    model = SSInference.from_pretrained(checkpoint_path=hf_model, device=str(device))
    return model


def _get_inner_module(model: object) -> torch.nn.Module:
    """Extract the nn.Module from SSInference wrapper."""
    for attr in ("model", "net", "separator", "_model"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Module):
            return m
    if isinstance(model, torch.nn.Module):
        return model  # type: ignore[return-value]
    raise RuntimeError("Cannot extract nn.Module from SSInference object.")


def _build_dataset(adapter: str, args: argparse.Namespace) -> object:
    """Build a DataLoader for the given adapter condition."""
    # Import here so the training script works even if data modules
    # are on a separate branch (they will be merged before Kaggle run).
    from data.calmsep_mixer import CalmSepMixer, CALMSEP_SAMPLE_RATE
    from data.rir_bank import RirBank
    from data.degradations import apply_reverb, apply_noise, apply_codec

    libri_8k = Path(args.librispeech_8k)
    source_files = sorted(libri_8k.rglob("*.flac")) + sorted(libri_8k.rglob("*.wav"))
    if not source_files:
        raise FileNotFoundError(f"No audio files in {libri_8k}")

    # Speaker holdout: dev-clean and test-clean speakers
    held_out_spks: set[str] = set()
    manifest = libri_8k / "manifest_8k.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        for split_info in data.get("splits", {}).values():
            if "dev" in split_info.get("split", "") or "test" in split_info.get("split", ""):
                held_out_spks.update(split_info.get("speakers", []))

    seed = getattr(args, "seed", 42)
    rng = np.random.default_rng(seed)
    mixer = CalmSepMixer(
        source_files,
        held_out_speaker_ids=held_out_spks,
        rng=rng,
        seed=seed,
    )
    mixer.assert_speaker_isolation()

    rir_bank = None
    if adapter == "reverb":
        rir_bank = RirBank(Path(args.rir_bank)) if args.rir_bank else None

    class _DynDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
        def __init__(self, n_samples: int) -> None:
            self.n = n_samples
            self._rng = np.random.default_rng(seed + 1)

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            m = mixer.mix(split="train")
            if adapter == "reverb" and rir_bank is not None:
                m = apply_reverb(m, rir_bank, self._rng)
            elif adapter == "noise":
                noise_dir = Path(args.noise_dir)
                noise_files = sorted((noise_dir / "wham").glob("*_8k.wav")) + \
                              sorted((noise_dir / "dns4").glob("*_8k.wav"))
                if noise_files:
                    import soundfile as sf
                    nf = noise_files[self._rng.integers(len(noise_files))]
                    noise_wav, _ = sf.read(str(nf), dtype="float32")
                    m = apply_noise(m, noise_wav, self._rng)
            elif adapter == "codec":
                codec = self._rng.choice(["opus", "aac", "amr-nb"])
                bitrates = {"opus": 12_000, "aac": 16_000, "amr-nb": 7_950}
                m = apply_codec(m, codec, bitrates[codec])

            mixture = torch.from_numpy(m.mixture).float()   # [T]
            refs = torch.from_numpy(m.references).float()   # [N, T]
            return {
                "mixture": mixture,
                "references": refs,
                "n_speakers": m.recipe.n_speakers,
                "recipe": m.recipe.condition_vector(),
            }

    n_train = getattr(args, "samples_per_epoch", 2000)
    dataset = _DynDataset(n_train)

    def _collate(batch: list[dict]) -> dict[str, object]:
        # Pad to max length in batch.
        max_t = max(b["mixture"].shape[0] for b in batch)
        mixtures, refs_list, ns, recipes = [], [], [], []
        for b in batch:
            t = b["mixture"].shape[0]
            mix = torch.nn.functional.pad(b["mixture"], (0, max_t - t))
            rf = torch.nn.functional.pad(b["references"], (0, max_t - t))
            # Pad references to max N in batch.
            mixtures.append(mix)
            refs_list.append(rf)
            ns.append(b["n_speakers"])
            recipes.append(b["recipe"])
        # Stack references: all batches have same N (drawn uniform); if not, pad.
        max_n = max(r.shape[0] for r in refs_list)
        refs_padded = []
        for r in refs_list:
            if r.shape[0] < max_n:
                r = torch.nn.functional.pad(r, (0, 0, 0, max_n - r.shape[0]))
            refs_padded.append(r)
        return {
            "mixture": torch.stack(mixtures),
            "references": torch.stack(refs_padded),
            "n_speakers": ns,
            "recipe": recipes,
        }

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=getattr(args, "batch_size", 4),
        shuffle=True,
        num_workers=getattr(args, "num_workers", 2),
        collate_fn=_collate,
    )
    return loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_single_adapter(args: argparse.Namespace) -> None:
    _seed_everything(getattr(args, "seed", 42))
    device = torch.device(getattr(args, "device", "cpu"))
    adapter = args.adapter
    if adapter not in ADAPTER_NAMES:
        raise ValueError(f"adapter must be one of {ADAPTER_NAMES}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen model and attach LoRA.
    hf_model = getattr(args, "hf_model", "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    ss_model = _load_model(hf_model, device)
    inner = _get_inner_module(ss_model)
    inner.to(device)

    lib = LoRALibrary(inner)
    lib.freeze_base()
    log.info("LoRA attached: %d modules", lib.n_attached)
    counts = lora_summary(inner)
    log.info("LoRA params: %s", counts)

    optimizer = optim.AdamW(
        lib.adapter_parameters(adapter),
        lr=getattr(args, "lr", 1e-4),
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=getattr(args, "epochs", 40)
    )

    loader = _build_dataset(adapter, args)
    epochs = getattr(args, "epochs", 40)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        inner.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            mixture = batch["mixture"].to(device)     # [B, T]
            references = batch["references"].to(device)  # [B, N, T]
            n_spks = batch["n_speakers"]

            with lib.forward_context(adapter, co_activate=True):
                # Forward pass through the full inference engine.
                # Note: SSInference.process_waveform returns separated streams.
                # We pass each sample through the engine individually (batch_size=1
                # constraint from the attractor BLUEPRINT §15.8).
                B = mixture.shape[0]
                all_estimates = []
                all_logits = []
                for b in range(B):
                    wav = mixture[b].unsqueeze(0)  # [1, T]
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):  # type: ignore[attr-defined]
                        out = ss_model.process_waveform(  # type: ignore[attr-defined]
                            wav, n_spks=torch.tensor(n_spks[b])
                        )
                    waves = _extract_output_waves(out, device)   # [K, T]
                    logits = _extract_logits(out)
                    all_estimates.append(waves)
                    if logits is not None:
                        all_logits.append(logits)

                # Pad all estimates to same (K, T).
                max_k = max(e.shape[0] for e in all_estimates)
                max_t = max(e.shape[1] for e in all_estimates)
                estimates = torch.zeros(B, max_k, max_t, device=device)
                for b, e in enumerate(all_estimates):
                    estimates[b, :e.shape[0], :e.shape[1]] = e

                logits_tensor = torch.stack(all_logits, dim=0) if all_logits else None
                losses = calmsep_loss(estimates, references, logits_tensor, n_spks)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(lib.adapter_parameters(adapter), 5.0)
            optimizer.step()
            epoch_loss += losses["total"].item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)
        elapsed = time.time() - t0
        log.info("Epoch %d/%d  loss=%.4f  time=%.1fs", epoch, epochs, avg_loss, elapsed)

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_adapter(lib, inner, adapter, output_dir / f"best_{adapter}.pt")

    _save_adapter(lib, inner, adapter, output_dir / f"final_{adapter}.pt")
    log.info("Training complete. Best loss: %.4f", best_loss)


def _extract_output_waves(out: object, device: torch.device) -> torch.Tensor:
    """Extract separated waveforms from model output as [K, T] tensor."""
    if isinstance(out, dict):
        waves = out.get("waveforms") or out.get("est_sources") or out.get("sources")
    else:
        waves = out
    if isinstance(waves, (list, tuple)):
        rows = [w.squeeze().to(device) for w in waves]
        return torch.stack(rows, dim=0)
    arr = waves.squeeze().to(device)  # type: ignore[union-attr]
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    return arr


def _extract_logits(out: object) -> torch.Tensor | None:
    if isinstance(out, dict) and "pres" in out:
        pres = out["pres"]
        if isinstance(pres, dict) and "logits" in pres:
            return pres["logits"]
    return None


def _save_adapter(lib: LoRALibrary, model: torch.nn.Module, adapter: str, path: Path) -> None:
    """Save only the adapter parameters (not the full model)."""
    state = {
        f"adapter.{adapter}.{name}": param
        for name, param in model.state_dict().items()
        if f"branches.{adapter}" in name
    }
    torch.save({"adapter": adapter, "state_dict": state}, path)
    log.info("Saved adapter checkpoint: %s (%d tensors)", path, len(state))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a single CALM-Sep LoRA adapter")
    p.add_argument("--adapter", required=True, choices=list(ADAPTER_NAMES))
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
    train_single_adapter(_parse_args())
