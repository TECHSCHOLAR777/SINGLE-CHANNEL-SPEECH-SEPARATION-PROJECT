"""
I-003 / I-042 diagnostic: does the trained gate actually discriminate conditions,
and does forcing Level-2 to zero (the production reality per I-042) collapse it?

I-003 found the deployed gate blends all three adapters near 0.5 regardless of
condition, using a temperature fitted in Stage 4c. This script does not have
that fitted temperature (it lives with the Stage 4 joint checkpoint, not the
Stage 3 gate checkpoint this diagnostic loads), so it measures a different,
earlier point in the pipeline: the raw Stage 3 GateNetwork output, before
calibration, under two conditions on the same four audio mixtures:

  1. Real Level-2 features, computed from the actual encoder output E(0) via
     the trained Level2Analyzer (what Stage 3 training itself used).
  2. Level-2 forced to zero, which is what pipeline/infer.py actually does in
     production (I-042: the gate runs once per utterance, before any chunk
     has been separated, so there is no prior E(0) to draw Level-2 from).

If the gate discriminates conditions with real Level-2 but collapses toward a
constant with Level-2 zeroed, that is direct evidence I-042 is a contributing
cause of I-003's flat gate, independent of the Stage 4c temperature question.

Usage:
    python -m coralsep.eval.diagnose_gate_flatness \
        --gate-ckpt best_gate.pt --librispeech-8k <dir> --rir-bank <dir> --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from coralsep.data.condition_mixer import CoralSepMixture, MixtureRecipe
from coralsep.data.degradations import apply_noise, apply_reverb
from coralsep.data.mixer_stub import MixtureSample
from coralsep.data.rir_bank import RirBank
from coralsep.eval.eval_reverb_adapter import build_test_mixture
from coralsep.models.condition import Level2Analyzer, level1_tensor
from coralsep.models.experts.srcorrnet import SRCorrNetExpert
from coralsep.models.gate import GateNetwork


def _build_conditions(
    refs_clean: list[np.ndarray], n_spks: int, rir_bank: RirBank, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    max_t = max(len(r) for r in refs_clean)
    refs_arr = np.stack([np.pad(r, (0, max_t - len(r))) for r in refs_clean])
    sample = MixtureSample(
        mixture=refs_arr.sum(0).astype(np.float32),
        references=refs_arr,
        sample_rate=8000,
        utterance_id="gate-diag",
    )
    clean = CoralSepMixture(sample=sample, recipe=MixtureRecipe(n_speakers=n_spks))

    mild = apply_reverb(clean, rir_bank, rng, t60_s=0.3)
    strong = apply_reverb(clean, rir_bank, rng, t60_s=0.9)

    noise = rng.standard_normal(max_t).astype(np.float32)
    noisy = apply_noise(clean, noise, rng, snr_db=-3.0)

    return {
        "clean": clean.mixture,
        "reverb_mild (T60 0.3s)": mild.mixture,
        "reverb_strong (T60 0.9s)": strong.mixture,
        "noisy (SNR -3dB)": noisy.mixture,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate-ckpt", required=True, type=Path)
    p.add_argument("--librispeech-8k", required=True, type=Path)
    p.add_argument("--rir-bank", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print("[SETUP] Loading gate checkpoint ...")
    ckpt = torch.load(args.gate_ckpt, map_location=args.device, weights_only=False)
    gate = GateNetwork().to(args.device)
    gate.load_state_dict(ckpt["gate"])
    gate.eval()
    analyzer = Level2Analyzer().to(args.device)
    analyzer.load_state_dict(ckpt["analyzer"])
    analyzer.eval()

    print("[SETUP] Loading frozen backbone (SRCorrNetExpert, Patch B for E(0)) ...")
    expert = SRCorrNetExpert(device=args.device)

    print(f"[SETUP] Loading RIR bank from {args.rir_bank} ...")
    rir_bank = RirBank(args.rir_bank)

    print(f"[SETUP] Building test mixture from {args.librispeech_8k} ...")
    _, refs_clean, n_spks = build_test_mixture(args.librispeech_8k, rng)
    conditions = _build_conditions(refs_clean, n_spks, rir_bank, rng)

    print("\n" + "=" * 92)
    print("  I-003 / I-042: gate output, real Level-2 vs Level-2 forced to zero")
    print("=" * 92)
    header = f"  {'Condition':<28} {'reverb (real)':>14} {'noise (real)':>13} {'codec (real)':>13}"
    header += f"  |  {'reverb (0)':>11} {'noise (0)':>10} {'codec (0)':>10}"
    print(header)
    print("  " + "-" * 88)

    real_gates: dict[str, dict[str, float]] = {}
    zero_gates: dict[str, dict[str, float]] = {}

    for name, mix in conditions.items():
        result = expert.separate(mix, sample_rate=8000, n_spks=n_spks)
        e0 = result.encoder_e0
        if e0 is None:
            print(f"  {name:<28}  (no E(0) captured, skipping)")
            continue

        l1 = level1_tensor(torch.from_numpy(mix).float()).to(args.device)
        with torch.no_grad():
            l2_real = analyzer.feature_vector(e0.to(args.device)).squeeze(0)
        l2_zero = torch.zeros(6, device=args.device)

        with torch.no_grad():
            g_real = gate.gate_dict(torch.cat([l1, l2_real]))
            g_zero = gate.gate_dict(torch.cat([l1, l2_zero]))

        real_gates[name] = g_real
        zero_gates[name] = g_zero

        print(
            f"  {name:<28} {g_real['reverb']:>14.3f} {g_real['noise']:>13.3f} "
            f"{g_real['codec']:>13.3f}  |  {g_zero['reverb']:>11.3f} {g_zero['noise']:>10.3f} "
            f"{g_zero['codec']:>10.3f}"
        )

    print("=" * 92)

    def _spread(gates: dict[str, dict[str, float]], adapter: str) -> float:
        vals = [g[adapter] for g in gates.values()]
        return float(np.std(vals)) if vals else float("nan")

    print("\n  Standard deviation of each adapter's gate across the four conditions:")
    for adapter in ("reverb", "noise", "codec"):
        real_std = _spread(real_gates, adapter)
        zero_std = _spread(zero_gates, adapter)
        print(
            f"    {adapter:<8} real Level-2: {real_std:.4f}   Level-2 forced to zero: {zero_std:.4f}"
        )

    print(
        "\n  A near-zero std under 'forced to zero' next to a clearly larger std under "
        "'real Level-2' means the gate can discriminate conditions when given the "
        "signal it was trained on, and I-042 (the production path never supplying "
        "that signal) is a real contributing cause of the flat gate in I-003, not "
        "just the Stage 4c temperature. If both are near-zero, the gate was not "
        "discriminative even with real Level-2, and I-042 is not the explanation."
    )


if __name__ == "__main__":
    main()
