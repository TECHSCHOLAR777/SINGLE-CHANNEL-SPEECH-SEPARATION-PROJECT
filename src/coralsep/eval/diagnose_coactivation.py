"""
I-043 diagnostic: does the reverb adapter degrade further under deployment-like
co-activation than under its own trained regime?

Stage 1 trains each adapter with the other two barely switched on, at most 20
percent (train/stage1_single.py module docstring). The deployed gate blends
all three near 50 percent (I-003). This script loads all three Stage 1
checkpoints into one LoRALibrary and compares the reverb adapter's SI-SNR
under its trained regime (reverb=1.0, others=0.0) against a 0.5/0.5/0.5 blend,
on the same reverberant mixture, so the two conditions differ only in
co-activation and nothing else.

This does not replace I-025's own finding (the adapter is harmful even alone,
at reverb=1.0/others=0.0). It answers a narrower question: does co-activation
make an already-bad situation measurably worse, which would mean retraining
with a wider co-activation range is worth trying before anything else.

Usage:
    python -m coralsep.eval.diagnose_coactivation \
        --reverb-ckpt best_reverb.pt --noise-ckpt best_noise.pt --codec-ckpt best_codec.pt \
        --librispeech-8k <dir> --rir-bank <dir> --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sr_corrnet import SSInference  # type: ignore

from coralsep.data.condition_mixer import CoralSepMixture, MixtureRecipe
from coralsep.data.degradations import apply_reverb
from coralsep.data.mixer_stub import MixtureSample
from coralsep.data.rir_bank import RirBank
from coralsep.eval.eval_reverb_adapter import (
    HF_CKPT,
    _model_device,
    _move_stft_to_device,
    build_test_mixture,
    pit_si_snr_np,
)
from coralsep.models.lora import LoRALibrary
from coralsep.train.stage1_single import _get_inner_module


def load_all_three(
    reverb_ckpt: Path, noise_ckpt: Path, codec_ckpt: Path, device: str
) -> tuple[SSInference, LoRALibrary]:
    """Load the frozen backbone with all three Stage 1 adapters attached."""
    ss = SSInference.from_pretrained(checkpoint_path=HF_CKPT, device=device)
    inner = _get_inner_module(ss)
    lib = LoRALibrary(inner)
    lib.freeze_base()
    inner.to(device)
    _move_stft_to_device(ss, device)

    inner_sd = inner.state_dict()
    for name, ckpt_path in (
        ("reverb", reverb_ckpt),
        ("noise", noise_ckpt),
        ("codec", codec_ckpt),
    ):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = ckpt["state_dict"]
        loaded = 0
        for full_key, tensor in state.items():
            parts = full_key.split(".", 2)
            local_key = parts[2] if len(parts) >= 3 else full_key
            if local_key in inner_sd:
                inner_sd[local_key] = tensor
                loaded += 1
        print(f"  loaded {loaded} tensors for adapter {name!r} from {ckpt_path}")
    inner.load_state_dict(inner_sd, strict=False)
    return ss, lib


def run_with_gates(ss, lib, wav_np, n_spks, gates: dict[str, float]) -> list[np.ndarray]:
    lib.set_gates(gates)
    lib.inject_gates()
    wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(_model_device(ss))
    with torch.inference_mode():
        out = ss.process_waveform(wav, n_spks=torch.tensor(n_spks))
    return [w.squeeze().cpu().float().numpy() for w in out["waveforms"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reverb-ckpt", required=True, type=Path)
    p.add_argument("--noise-ckpt", required=True, type=Path)
    p.add_argument("--codec-ckpt", required=True, type=Path)
    p.add_argument("--librispeech-8k", required=True, type=Path)
    p.add_argument("--rir-bank", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t60", type=float, default=0.5)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print("[SETUP] Loading backbone with all three Stage 1 adapters ...")
    ss, lib = load_all_three(args.reverb_ckpt, args.noise_ckpt, args.codec_ckpt, args.device)

    print(f"[SETUP] Loading RIR bank from {args.rir_bank} ...")
    rir_bank = RirBank(args.rir_bank)

    print(f"[SETUP] Building test mixture from {args.librispeech_8k} ...")
    mixture_clean, refs_clean, n_spks = build_test_mixture(args.librispeech_8k, rng)

    max_t = max(len(r) for r in refs_clean)
    refs_arr = np.stack([np.pad(r, (0, max_t - len(r))) for r in refs_clean])
    sample = MixtureSample(
        mixture=refs_arr.sum(0).astype(np.float32),
        references=refs_arr,
        sample_rate=8000,
        utterance_id="coact-diag",
    )
    calmix = CoralSepMixture(sample=sample, recipe=MixtureRecipe(n_speakers=n_spks))
    rev = apply_reverb(calmix, rir_bank, rng, t60_s=args.t60)
    mix = rev.mixture
    wet_refs = [rev.references[i] for i in range(rev.references.shape[0])]

    print(f"  T60 = {rev.recipe.t60_s:.2f}s, {n_spks} speakers")

    trained_regime = {"reverb": 1.0, "noise": 0.0, "codec": 0.0}
    deployed_regime = {"reverb": 0.5, "noise": 0.5, "codec": 0.5}
    off_regime = {"reverb": 0.0, "noise": 0.0, "codec": 0.0}

    out_off = run_with_gates(ss, lib, mix, n_spks, off_regime)
    out_trained = run_with_gates(ss, lib, mix, n_spks, trained_regime)
    out_deployed = run_with_gates(ss, lib, mix, n_spks, deployed_regime)

    snr_off = pit_si_snr_np(out_off, wet_refs)
    snr_trained = pit_si_snr_np(out_trained, wet_refs)
    snr_deployed = pit_si_snr_np(out_deployed, wet_refs)

    print("\n" + "=" * 60)
    print("  I-043: co-activation regime comparison (reverb condition)")
    print("=" * 60)
    print(f"  All gates off (frozen backbone)     : SI-SNR = {snr_off:.2f} dB")
    print(f"  Trained regime  (1.0, 0.0, 0.0)      : SI-SNR = {snr_trained:.2f} dB")
    print(f"  Deployed regime (0.5, 0.5, 0.5)      : SI-SNR = {snr_deployed:.2f} dB")
    print(f"\n  Trained vs off   : {snr_trained - snr_off:+.2f} dB")
    print(f"  Deployed vs off  : {snr_deployed - snr_off:+.2f} dB")
    print(f"  Deployed vs trained (co-activation cost): {snr_deployed - snr_trained:+.2f} dB")
    print("=" * 60)

    if snr_deployed < snr_trained - 0.5:
        print(
            "\n  Verdict: co-activation makes it measurably worse. Widening the "
            "Stage 1 co-activation range is worth trying before anything else."
        )
    elif snr_deployed > snr_trained + 0.5:
        print(
            "\n  Verdict: co-activation does not explain the harm; the deployed "
            "blend is no worse (or better) than the adapter's own trained regime."
        )
    else:
        print(
            "\n  Verdict: co-activation makes little difference here. The harm "
            "in I-025 is not primarily a co-activation mismatch."
        )


if __name__ == "__main__":
    main()
