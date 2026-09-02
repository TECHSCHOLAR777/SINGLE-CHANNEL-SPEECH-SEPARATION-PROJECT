"""
Stage 1: Single adapter training (Dev B, P1-B4/B5/B6).

Trains one adapter (reverb | noise | codec) at a time on its dedicated condition.
The frozen base model provides the separation backbone; LoRA branches add
condition-specific residual corrections.

Co-activation warm-up is always on: other adapters are active at U(0.0, 0.2)
so Stage 4 joint polish sees a model that already tolerates composition.

Usage
-----
    python src/coralsep/train/stage1_single.py \
        --adapter reverb \
        --librispeech-8k /data/LibriSpeech_8k \
        --rir-bank datasets/rirs/bank.json \
        --noise-dir /data/calmsep_noise \
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

from coralsep.models.lora import ADAPTER_NAMES, LoRALibrary, lora_summary
from coralsep.train.losses import coralsep_loss

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
    """Load the frozen SR-CorrNet checkpoint.

    Always loads on CPU first (model.pt contains float64 tensors which MPS
    cannot receive via map_location='mps'). The caller is responsible for
    moving the extracted inner module to the target device.
    """
    try:
        from sr_corrnet import SSInference  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "SR-CorrNet-SS not installed. Run:\n"
            "  git clone https://github.com/dmlguq456/SR_CorrNet_SS.git\n"
            '  cd SR_CorrNet_SS && pip install -e ".[hub]"'
        ) from exc
    log.info("Loading frozen checkpoint: %s (on cpu, will move to %s)", hf_model, device)
    model = SSInference.from_pretrained(checkpoint_path=hf_model, device="cpu")
    return model


def _get_inner_module(model: object) -> torch.nn.Module:
    """Extract the nn.Module from SSInference wrapper.

    SSInference nests the separator at: SSInference → engine (EngineInfer) → model (nn.Module).
    """
    if isinstance(model, torch.nn.Module):
        return model  # type: ignore[return-value]
    # One level deep
    for attr in ("model", "engine", "net", "separator", "_model"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Module):
            return m
    # Two levels deep: SSInference.engine.model
    for attr1 in ("engine", "model", "net", "separator", "_model"):
        wrapper = getattr(model, attr1, None)
        if wrapper is not None:
            for attr2 in ("model", "net", "separator", "_model", "engine"):
                m = getattr(wrapper, attr2, None)
                if isinstance(m, torch.nn.Module):
                    return m
    raise RuntimeError("Cannot extract nn.Module from SSInference object.")


def _collate(batch: list[dict]) -> dict:
    """Pad a batch of variable-length samples to the longest in the batch."""
    max_t = max(b["mixture"].shape[0] for b in batch)
    mixtures, refs_list, ns, recipes = [], [], [], []
    for b in batch:
        t = b["mixture"].shape[0]
        mix = torch.nn.functional.pad(b["mixture"], (0, max_t - t))
        rf = torch.nn.functional.pad(b["references"], (0, max_t - t))
        mixtures.append(mix)
        refs_list.append(rf)
        ns.append(b["n_speakers"])
        recipes.append(b["recipe"])
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


def _worker_init_fn(worker_id: int) -> None:
    """Re-seed each DataLoader worker's RNG so workers produce unique samples."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    ds = worker_info.dataset
    worker_seed = ds.seed + 1 + worker_id * 99991
    ds._rng = np.random.default_rng(worker_seed)
    if hasattr(ds, "mixer") and hasattr(ds.mixer, "_rng"):
        ds.mixer._rng = np.random.default_rng(worker_seed + 1)


class _DynDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Module-level (picklable) dataset for single-adapter Stage 1 training.

    Must be at module scope so DataLoader workers can pickle it when num_workers>0.
    Only picklable state is stored; data imports happen inside __getitem__.
    """

    def __init__(
        self,
        n_samples: int,
        adapter: str,
        mixer: object,
        rir_bank: object | None,
        noise_files: list,
        seed: int,
        max_clip: int,
    ) -> None:
        self.n = n_samples
        self.adapter = adapter
        self.mixer = mixer
        self.rir_bank = rir_bank
        self._noise_files = noise_files  # list[Path] — picklable
        self.seed = seed
        self.max_clip = max_clip
        # BUG FIX: re-seeded per worker via worker_init_fn
        self._rng = np.random.default_rng(seed + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        # Import inside __getitem__ so module objects don't need to be pickled.
        import soundfile as sf

        from coralsep.data.degradations import apply_codec, apply_noise, apply_reverb

        m = self.mixer.mix(split="train")

        # Clip raw audio BEFORE applying degradations — reverb/noise on the full
        # LibriSpeech utterance (up to 30s) is ~15× slower than on a 2s clip.
        if m.mixture.shape[0] > self.max_clip:
            import dataclasses

            from coralsep.data.mixer_stub import MixtureSample

            _start = int(self._rng.integers(0, m.mixture.shape[0] - self.max_clip))
            clipped_sample = MixtureSample(
                mixture=m.sample.mixture[_start : _start + self.max_clip],
                references=m.sample.references[:, _start : _start + self.max_clip],
                sample_rate=m.sample.sample_rate,
                utterance_id=m.sample.utterance_id,
            )
            m = dataclasses.replace(m, sample=clipped_sample)

        if self.adapter == "reverb" and self.rir_bank is not None:
            m = apply_reverb(m, self.rir_bank, self._rng)
        elif self.adapter == "noise" and self._noise_files:
            nf = self._noise_files[self._rng.integers(len(self._noise_files))]
            noise_wav, _ = sf.read(str(nf), dtype="float32")
            m = apply_noise(m, noise_wav, self._rng)
        elif self.adapter == "codec":
            codec = self._rng.choice(["opus", "aac", "amr-nb"])
            bitrates = {"opus": 12_000, "aac": 16_000, "amr-nb": 7_950}
            m = apply_codec(m, codec, bitrates[codec])

        mixture = torch.from_numpy(m.mixture).float()
        refs = torch.from_numpy(m.references).float()
        # Secondary clip in case degradation changed length (e.g. reverb tail)
        if mixture.shape[0] > self.max_clip:
            mixture = mixture[: self.max_clip]
            refs = refs[:, : self.max_clip]
        return {
            "mixture": mixture,
            "references": refs,
            "n_speakers": m.recipe.n_speakers,
            "recipe": m.recipe.condition_vector(),
        }


def _build_dataset(adapter: str, args: argparse.Namespace) -> object:
    """Build a DataLoader for the given adapter condition."""
    # Import here so the training script works even if data modules
    # are on a separate branch (they will be merged before Kaggle run).
    from coralsep.data.condition_mixer import CoralSepMixer
    from coralsep.data.rir_bank import RirBank

    libri_8k = Path(args.librispeech_8k)
    source_files = sorted(libri_8k.rglob("*.flac")) + sorted(libri_8k.rglob("*.wav"))
    if not source_files:
        raise FileNotFoundError(f"No audio files in {libri_8k}")

    # Speaker holdout: dev-clean and test-clean speakers
    held_out_spks: set[str] = set()
    manifest = libri_8k / "manifest_8k.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        items = data if isinstance(data, list) else list(data.get("splits", {}).values())
        for split_info in items:
            if "dev" in split_info.get("split", "") or "test" in split_info.get("split", ""):
                held_out_spks.update(split_info.get("speaker_ids", split_info.get("speakers", [])))

    seed = getattr(args, "seed", 42)
    rng = np.random.default_rng(seed)
    mixer = CoralSepMixer(
        source_files,
        held_out_speaker_ids=held_out_spks,
        rng=rng,
        seed=seed,
    )
    mixer.assert_speaker_isolation()

    rir_bank = None
    if adapter == "reverb":
        _rb_path = Path(args.rir_bank)
        rir_bank = (
            RirBank(_rb_path.parent if _rb_path.suffix == ".json" else _rb_path)
            if args.rir_bank
            else None
        )

    # BUG FIX: pre-compute noise files once (was: glob called inside every __getitem__,
    # causing 28 000 filesystem stat calls per training sample — ~40x slower than needed).
    noise_files: list[Path] = []
    if adapter == "noise":
        noise_dir = Path(args.noise_dir)
        # Accept files in named sub-dirs (wham/, dns4/) OR directly in noise_dir.
        noise_files = sorted((noise_dir / "wham").glob("*_8k.wav")) + sorted(
            (noise_dir / "dns4").glob("*_8k.wav")
        )
        if not noise_files:
            # Fallback: any .wav directly under noise_dir
            noise_files = sorted(noise_dir.rglob("*_8k.wav"))
        if not noise_files:
            raise FileNotFoundError(
                f"No noise files (*_8k.wav) found under {noise_dir}. "
                "Run data/prepare_noise_staging.py first."
            )
        log.info("Noise adapter: found %d noise files in %s", len(noise_files), noise_dir)

    # BUG FIX: fail fast for codec adapter when ffmpeg is absent (Lightning AI).
    # Without ffmpeg every sample silently falls back to mu-law (G.711), which
    # is a qualitatively different degradation — the adapter learns mu-law, not
    # real codec artifacts. Error out so the user installs ffmpeg first.
    if adapter == "codec":
        from coralsep.data.codec_augmentation import is_ffmpeg_available

        if not is_ffmpeg_available():
            raise RuntimeError(
                "ffmpeg is required for the codec adapter but was not found on PATH.\n"
                "On Lightning AI: conda install -y -c conda-forge ffmpeg\n"
                "  or: apt-get install -y ffmpeg"
            )
        log.info("Codec adapter: ffmpeg found at %s", __import__("shutil").which("ffmpeg"))

    n_train = getattr(args, "samples_per_epoch", 2000)
    max_clip = getattr(args, "max_clip_samples", 16000)
    dataset = _DynDataset(
        n_samples=n_train,
        adapter=adapter,
        mixer=mixer,
        rir_bank=rir_bank,
        noise_files=noise_files,
        seed=seed,
        max_clip=max_clip,
    )

    num_workers = getattr(args, "num_workers", 2)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=getattr(args, "batch_size", 4),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_single_adapter(args: argparse.Namespace) -> None:
    _seed_everything(getattr(args, "seed", 42))
    device = torch.device(getattr(args, "device", "cpu"))
    # BF16 only on CUDA; MPS supports FP16 autocast; CPU stays FP32
    # M5 Pro MPS (PyTorch 2.13+) supports BF16 — prefer it over FP16.
    use_bf16 = getattr(args, "bf16", True) and device.type in ("cuda", "mps")
    use_fp16 = getattr(args, "fp16", False) and device.type == "mps" and not use_bf16
    _prec = "BF16" if use_bf16 else ("FP16" if use_fp16 else "FP32")
    log.info("Device: %s  Precision: %s", device, _prec)
    adapter = args.adapter
    if adapter not in ADAPTER_NAMES:
        raise ValueError(f"adapter must be one of {ADAPTER_NAMES}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen model and attach LoRA.
    hf_model = getattr(args, "hf_model", "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    ss_model = _load_model(hf_model, device)
    inner = _get_inner_module(ss_model)

    lib = LoRALibrary(inner)
    lib.freeze_base()
    inner.to(device)  # move after LoRA attachment so branches land on device too

    # Also move engine.stft to device — it has learnable/fixed conv weights that
    # must match the device of model_input when _forward_with_grad runs.
    engine = getattr(ss_model, "engine", None)
    if engine is not None:
        stft_mod = getattr(engine, "stft", None)
        istft_mod = getattr(engine, "istft", None)
        if stft_mod is not None and hasattr(stft_mod, "to"):
            stft_mod.to(device)
        if istft_mod is not None and hasattr(istft_mod, "to"):
            istft_mod.to(device)

    log.info("LoRA attached: %d modules", lib.n_attached)
    counts = lora_summary(inner)
    log.info("LoRA params: %s", counts)

    optimizer = optim.AdamW(
        lib.adapter_parameters(adapter),
        lr=getattr(args, "lr", 1e-4),
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(args, "epochs", 40))

    loader = _build_dataset(adapter, args)
    epochs = getattr(args, "epochs", 40)
    best_loss = float("inf")
    epoch_times: list[float] = []  # rolling history for ETA

    # ── MPS Metal shader warm-up ───────────────────────────────────────────────
    # On Apple MPS, the first forward+backward for each unique n_spks value
    # (2-5) triggers Metal shader compilation that takes 3-60s per variant.
    # Pre-compiling all variants here means the training loop never stalls on
    # compilation. The compiled shaders are cached by the OS across restarts.
    if device.type == "mps":
        log.info("MPS warm-up: compiling Metal shaders for n_spks=2..5 (one-time, ~30s)...")
        _ac_device = "mps"
        _ac_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
        _ac_enabled = use_bf16 or use_fp16
        inner.eval()
        _t_wu = time.time()
        for _n in [2, 3, 4, 5]:
            _wav = torch.zeros(1, 16000, device=device)
            _ref = torch.zeros(1, _n, 16000, device=device)
            optimizer.zero_grad(set_to_none=True)
            with lib.forward_context(adapter, co_activate=True):
                with torch.autocast(_ac_device, dtype=_ac_dtype, enabled=_ac_enabled):
                    _waves, _logits = _forward_with_grad(ss_model, _wav, n_spks=torch.tensor(_n))
            _est = _waves.unsqueeze(0)
            _lg = _logits.unsqueeze(0) if _logits is not None else None
            from coralsep.train.losses import coralsep_loss as _csep

            _ld = _csep(_est, _ref[..., : _est.shape[-1]], _lg, [_n])
            _ld["total"].backward()
            optimizer.step()
            torch.mps.synchronize()
            torch.mps.empty_cache()
            log.info("  warm-up n_spks=%d done (%.1fs)", _n, time.time() - _t_wu)
        optimizer.zero_grad(set_to_none=True)
        inner.train()
        log.info("MPS warm-up complete (%.1fs total)", time.time() - _t_wu)
    # ──────────────────────────────────────────────────────────────────────────

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

            # Per-sample backward accumulation: process each sample, call
            # backward immediately, then free the activation graph. A grouped
            # batched forward (_forward_batch) was tried on 2026-07-18 and OOMs
            # on 24 GB unified memory — holding 4 activation graphs at once
            # exceeds the ~30 GiB MPS ceiling. Per-sample is the memory-safe path.
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            with lib.forward_context(adapter, co_activate=True):
                for b in range(B):
                    wav = mixture[b].unsqueeze(0)  # [1, T]
                    ref = references[b].unsqueeze(0)  # [1, N, T]
                    _ac_device = device.type if device.type in ("cuda", "mps") else "cpu"
                    _ac_dtype = (
                        torch.bfloat16
                        if use_bf16
                        else (torch.float16 if use_fp16 else torch.float32)
                    )
                    _ac_enabled = use_bf16 or use_fp16
                    with torch.autocast(_ac_device, dtype=_ac_dtype, enabled=_ac_enabled):
                        waves, logits = _forward_with_grad(
                            ss_model, wav, n_spks=torch.tensor(n_spks[b])
                        )
                    est = waves.unsqueeze(0)  # [1, K, T]
                    logits_t = logits.unsqueeze(0) if logits is not None else None
                    losses = coralsep_loss(est, ref, logits_t, [n_spks[b]])
                    # Scale by 1/B so accumulated grads equal a true batch mean
                    (losses["total"] / B).backward()
                    batch_loss += losses["total"].item() / B

            torch.nn.utils.clip_grad_norm_(lib.adapter_parameters(adapter), 5.0)
            optimizer.step()
            if device.type == "mps":
                torch.mps.empty_cache()
            epoch_loss += batch_loss

            # Intra-epoch progress every 10% of batches
            if batch_idx % max(1, n_batches // 10) == 0 or batch_idx == n_batches:
                frac = batch_idx / n_batches
                elapsed_now = time.time() - t0
                eta_epoch = elapsed_now / frac * (1 - frac)
                log.info(
                    "  Epoch %d/%d  batch %d/%d (%.0f%%)  " "batch_loss=%.4f  epoch_eta=%.0fs",
                    epoch,
                    epochs,
                    batch_idx,
                    n_batches,
                    frac * 100,
                    batch_loss,
                    eta_epoch,
                )

        scheduler.step()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_loss = epoch_loss / max(n_batches, 1)

        # ETA for remaining epochs (use rolling last-5 average)
        recent = epoch_times[-5:]
        avg_epoch_time = sum(recent) / len(recent)
        remaining_epochs = epochs - epoch
        eta_total = avg_epoch_time * remaining_epochs
        eta_h = int(eta_total // 3600)
        eta_m = int((eta_total % 3600) // 60)

        marker = " *** NEW BEST ***" if avg_loss < best_loss else ""
        log.info(
            "Epoch %d/%d  loss=%.4f  time=%.1fs  ETA=%dh%02dm%s",
            epoch,
            epochs,
            avg_loss,
            elapsed,
            eta_h,
            eta_m,
            marker,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_adapter(lib, inner, adapter, output_dir / f"best_{adapter}.pt")

    _save_adapter(lib, inner, adapter, output_dir / f"final_{adapter}.pt")
    log.info("Training complete. Best loss: %.4f", best_loss)


def _forward_with_grad(
    ss_model: object,
    wav: torch.Tensor,
    n_spks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gradient-capable forward pass through SSInference.

    Bypasses SSInference.process_waveform / engine.infer_chunk which are both
    decorated with @torch.inference_mode() and would detach the graph.
    Replicates the exact same computation without that decorator.

    Returns:
        waves: [K, T] separated waveforms (on same device as input)
        logits: [K, 7] attractor logits or None
    """
    engine = ss_model.engine  # type: ignore[attr-defined]
    # Use the input tensor's device (engine.device may still say "cpu" after
    # the model was loaded on CPU then moved to MPS via inner.to(device)).
    target_device = wav.device
    # SS-specific std normalisation
    wav_norm = wav / (wav.std(dim=-1, keepdim=True) + 1e-8)

    # STFT and iSTFT use torch.complex which does NOT support BF16 or FP16 on MPS.
    # Disable autocast around these ops so they always run in float32.
    ac_device = target_device.type if target_device.type in ("cuda", "mps") else "cpu"
    with torch.autocast(ac_device, enabled=False):
        stft = engine.stft(wav_norm.float().to(target_device), cplx=True)

    # Real/imag concat → (2M, F, T)  float32
    model_input = torch.cat([stft.real, stft.imag], dim=0)
    # Model forward — no inference_mode so gradients flow through LoRA branches.
    # Autocast (if any) from the outer training loop context applies here.
    out_list, _aux, pres = engine.model(model_input, n_spks=n_spks)
    if not out_list:
        return torch.zeros(1, stft.shape[-1], device=target_device), None

    # (B=1, M_o, F, T, 2) → complex float32 (N, M_o, F, T)
    with torch.autocast(ac_device, enabled=False):
        estim_stft = torch.cat(
            [torch.complex(e[..., 0].float(), e[..., 1].float()) for e in out_list], dim=0
        )
        # Select reference channel: (N, F, T)
        stft_out = estim_stft[:, engine.ref_ch, :, :]
        # iSTFT → list of 1-D waveforms
        waveforms = [
            engine.istft(stft_out[i], cplx=True, squeeze=True) for i in range(stft_out.shape[0])
        ]

    waves = torch.stack(waveforms, dim=0)  # [K, T]
    logits = pres.get("logits") if isinstance(pres, dict) else None
    return waves, logits


def _extract_output_waves(out: object, device: torch.device) -> torch.Tensor:
    """Extract [K, T] waveform tensor from process_waveform output dict."""
    wavs = out.get("waveforms", []) if isinstance(out, dict) else []  # type: ignore[union-attr]
    if not wavs:
        return torch.zeros(1, 1, device=device)
    waves = []
    for w in wavs:
        t = w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
        t = t.squeeze().to(device)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        waves.append(t)
    max_t = max(w.shape[-1] for w in waves)
    padded = [torch.nn.functional.pad(w, (0, max_t - w.shape[-1])) for w in waves]
    return torch.stack(padded)  # [K, T]


def _extract_logits(out: object) -> torch.Tensor | None:
    """Extract (1, 7) attractor logits from process_waveform output, or None."""
    if not isinstance(out, dict):
        return None
    pres = out.get("pres")
    if not isinstance(pres, dict):
        return None
    logits = pres.get("logits")
    if logits is None:
        return None
    return logits.squeeze(0) if logits.ndim == 3 else logits


def _forward_batch(
    ss_model: object,
    wav: torch.Tensor,  # [B, T]
    n_spks: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Batched forward for B samples that all have the same n_spks.

    Runs a single GPU kernel launch for all B samples instead of B sequential
    launches. Requires n_spks to be the same across the batch (use groups).

    Returns:
        waves:  [B, K, T] separated waveforms
        logits: [B, K+2] presence logits or None
    """
    engine = ss_model.engine  # type: ignore[attr-defined]
    target_device = wav.device
    B = wav.shape[0]

    wav_norm = wav / (wav.std(dim=-1, keepdim=True) + 1e-8)  # [B, T]

    ac_device = target_device.type if target_device.type in ("cuda", "mps") else "cpu"
    with torch.autocast(ac_device, enabled=False):
        stft = engine.stft(wav_norm.float().to(target_device), cplx=True)  # [B, F, T_stft]

    # [B, 2*M=2, F, T_stft] — model expects (B, 2M, F, T), unsqueezes if 3D
    model_input = torch.stack([stft.real, stft.imag], dim=1)

    # Pass 1-d tensor so AttractorSplit.forward takes the vector path (B>1)
    n_spks_t = torch.tensor([n_spks] * B, device=target_device)
    out_list, _aux, pres = engine.model(model_input, n_spks=n_spks_t)

    if not out_list:
        return torch.zeros(B, 1, stft.shape[-1], device=target_device), None

    # out_list: K tensors each [B, M_o, F, T, 2]
    with torch.autocast(ac_device, enabled=False):
        # Stack speakers → [K, B, M_o, F, T] complex
        stft_spk = torch.stack(
            [torch.complex(e[..., 0].float(), e[..., 1].float()) for e in out_list],
            dim=0,
        )
        # Select ref channel → [K, B, F, T], then transpose to [B, K, F, T]
        stft_out = stft_spk[:, :, engine.ref_ch, :, :].permute(1, 0, 2, 3)  # [B, K, F, T]

        # iSTFT — flatten to [B*K, F, T], istft each, reshape back
        BK = B * stft_out.shape[1]
        stft_flat = stft_out.reshape(BK, *stft_out.shape[2:])
        waveforms = [engine.istft(stft_flat[i], cplx=True, squeeze=True) for i in range(BK)]
        T_out = waveforms[0].shape[0]
        waves = torch.stack(waveforms).reshape(B, -1, T_out)  # [B, K, T]

    logits = pres.get("logits") if isinstance(pres, dict) else None
    # logits from model: [B, K+2] or similar; return as-is for per-sample indexing
    return waves, logits


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
    p = argparse.ArgumentParser(description="Train a single CoRAL-Sep LoRA adapter")
    p.add_argument("--adapter", required=True, choices=list(ADAPTER_NAMES))
    # Accept both --librispeech-8k (direct) and --data-root (Kaggle notebook convention).
    p.add_argument("--librispeech-8k", default="")
    p.add_argument("--data-root", default="")  # alias used by notebooks
    p.add_argument("--rir-bank", default="datasets/rirs/bank.json")
    p.add_argument("--noise-dir", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--checkpoint-dir", default="")  # alias used by notebooks
    p.add_argument("--config", default="")  # accepted but unused (config baked in)
    p.add_argument("--hf-model", default="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    _default_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    p.add_argument("--device", default=_default_device)
    p.add_argument(
        "--fp16", action="store_true", default=False, help="Use FP16 autocast on MPS (Apple GPU)"
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--max-clip-samples",
        type=int,
        default=16000,
        help="Max waveform length in samples (default 16000 = 2s @ 8kHz)",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 autocast (default: True, L40S/A100/H100 supported)",
    )
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    args = p.parse_args()
    # Resolve aliases: notebook passes --data-root and --checkpoint-dir.
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
    train_single_adapter(_parse_args())
