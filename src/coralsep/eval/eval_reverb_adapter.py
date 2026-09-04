"""
Comprehensive evaluation and diagnosis of the reverb LoRA adapter.

Runs 5 diagnostic passes:
  1. SANITY, does _forward_with_grad == process_waveform? (pipeline bug check)
  2. GATE, are LoRA B-matrices non-zero? Does gate=1 change the output?
  3. SI-SNR, base vs adapted on clean / reverb-0.4s / reverb-0.8s mixtures
  4. TARGET, SI-SNR vs wet reference vs anechoic reference (explains loss numbers)
  5. SUMMARY, verdict and root-cause

Usage:
  cd ~/Desktop/SINGLE-CHANNEL-SPEECH-SEPARATION-PROJECT
  .venv/bin/python src/coralsep/eval/eval_reverb_adapter.py \
    --checkpoint checkpoints/stage1_reverb/best_reverb.pt \
    --librispeech-8k data/calmsep-8k/librispeech-8k \
    --rir-bank data/calmsep-8k/rirs \
    --output-dir eval_outputs

History: the run that produced the I-025 finding (see EXPERIMENT_REGISTRY.md
EXP-003) executed on a Lightning AI workspace under
/teamspace/studios/this_studio/. That platform was banned permanently on
2026-07-18 after the account was deleted; nothing in this script depends on
it any more. See docs/PROJECT_HISTORY.md and docs/restoration/ISSUE_LEDGER.md
I-033.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------


def _add_paths() -> None:
    """Add SR-CorrNet and coralsep to sys.path regardless of CWD."""
    repo = Path(__file__).resolve().parent.parent
    candidates = [
        repo,
        repo.parent / "SR_CorrNet_SS",
    ]
    for p in candidates:
        s = str(p)
        if s not in sys.path and p.exists():
            sys.path.insert(0, s)


_add_paths()

from sr_corrnet import SSInference  # type: ignore

from coralsep.data.condition_mixer import CoralSepMixture, MixtureRecipe  # type: ignore
from coralsep.data.degradations import apply_reverb  # type: ignore
from coralsep.data.mixer_stub import MixtureSample  # type: ignore
from coralsep.data.rir_bank import RirBank  # type: ignore
from coralsep.models.lora import LoRALayer, LoRALibrary  # type: ignore
from coralsep.train.stage1_single import _forward_with_grad, _get_inner_module  # type: ignore

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_EPS = 1e-10


def si_snr_np(estimate: np.ndarray, target: np.ndarray) -> float:
    min_t = min(len(estimate), len(target))
    e, t = estimate[:min_t].astype(np.float64), target[:min_t].astype(np.float64)
    e -= e.mean()
    t -= t.mean()
    s_tgt = (e @ t) / (t @ t + _EPS) * t
    noise = e - s_tgt
    return float(10 * np.log10((s_tgt @ s_tgt) / (noise @ noise + _EPS) + _EPS))


def pit_si_snr_np(estimates: list[np.ndarray], references: list[np.ndarray]) -> float:
    """Best permutation SI-SNR, averaged over matched pairs."""
    from itertools import permutations

    K = min(len(estimates), len(references))
    best = -999.0
    for perm in permutations(range(len(estimates)), K):
        val = np.mean([si_snr_np(estimates[perm[j]], references[j]) for j in range(K)])
        if val > best:
            best = val
    return best


def si_snr_mixture(mixture: np.ndarray, references: list[np.ndarray]) -> float:
    """SI-SNR of the raw mixture against each clean reference (oracle lower bound)."""
    return np.mean([si_snr_np(mixture, r) for r in references]).item()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

HF_CKPT = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"


def _move_stft_to_device(ss: SSInference, device: str) -> None:
    """Move engine.stft / engine.istft to device.

    SSInference.from_pretrained(device=...) does not move these: the STFT
    kernel is a plain buffer created at construction time, not part of the
    checkpoint state dict, so it stays wherever it was built regardless of
    the device argument. train/stage1_single.py already learned this
    (`_move_stft_to_device`'s counterpart there, lines around
    train_single_adapter's "Also move engine.stft to device" comment); this
    script never did, so it worked on CPU (its only device until now) and
    crashed on CUDA with a device-mismatch RuntimeError in engine.stft's
    conv1d. See I-050.
    """
    engine = getattr(ss, "engine", None)
    if engine is None:
        return
    for attr in ("stft", "istft"):
        mod = getattr(engine, attr, None)
        if mod is not None and hasattr(mod, "to"):
            mod.to(device)


def load_base(device: str) -> SSInference:
    print(f"  Loading base SR-CorrNet ({HF_CKPT}) on {device} ...")
    ss = SSInference.from_pretrained(checkpoint_path=HF_CKPT, device=device)
    _move_stft_to_device(ss, device)
    return ss


def load_adapted(checkpoint: Path, device: str, rank: int = 8) -> tuple[SSInference, LoRALibrary]:
    """Load SR-CorrNet with LoRA reverb adapter from checkpoint.

    rank must match the rank the checkpoint was trained at (default 8, the
    BLUEPRINT value); a mismatch raises a shape error from load_state_dict,
    since LoRA A/B tensor sizes depend on it. Needed for the I-025 rank
    ablation, where a checkpoint trained at a non-default rank is evaluated
    with this same script.
    """
    print(f"  Loading adapted model from {checkpoint} ...")
    ss = SSInference.from_pretrained(checkpoint_path=HF_CKPT, device=device)
    inner = _get_inner_module(ss)
    lib = LoRALibrary(inner, attn_rank=rank)
    lib.freeze_base()
    inner.to(device)
    _move_stft_to_device(ss, device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state = ckpt["state_dict"]
    # Remap keys: "adapter.reverb.branches.reverb.A" → "branches.reverb.A"
    inner_sd = inner.state_dict()
    loaded = 0
    for full_key, tensor in state.items():
        # Strip "adapter.reverb." prefix written by _save_adapter
        parts = full_key.split(".", 2)
        if len(parts) >= 3:
            local_key = parts[2]
        else:
            local_key = full_key
        if local_key in inner_sd:
            inner_sd[local_key] = tensor
            loaded += 1
    inner.load_state_dict(inner_sd, strict=False)
    print(f"  Loaded {loaded} LoRA tensors from checkpoint.")
    return ss, lib


def _model_device(ss: SSInference) -> torch.device:
    """The device the model's own parameters live on, so callers never have
    to thread a device string through every diagnostic pass separately."""
    return next(_get_inner_module(ss).parameters()).device


def run_base_model(ss: SSInference, wav_np: np.ndarray, n_spks: int) -> list[np.ndarray]:
    """Run original process_waveform (inference_mode, no LoRA)."""
    wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(_model_device(ss))
    with torch.inference_mode():
        out = ss.process_waveform(wav, n_spks=torch.tensor(n_spks))
    return [w.squeeze().cpu().float().numpy() for w in out["waveforms"]]


def run_adapted_model(
    ss: SSInference, lib: LoRALibrary, wav_np: np.ndarray, n_spks: int, gate: float = 1.0
) -> list[np.ndarray]:
    """Run adapted model with reverb gate set explicitly."""
    lib.set_gates({"reverb": gate, "noise": 0.0, "codec": 0.0})
    lib.inject_gates()
    wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(_model_device(ss))
    with torch.inference_mode():
        out = ss.process_waveform(wav, n_spks=torch.tensor(n_spks))
    return [w.squeeze().cpu().float().numpy() for w in out["waveforms"]]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def build_test_mixture(
    libri_dir: Path,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Return (mixture, clean_references, n_spks) from dev-clean speakers."""
    dev_clean = libri_dir / "dev-clean"
    if not dev_clean.exists():
        dev_clean = libri_dir  # fallback

    files = sorted(dev_clean.rglob("*.wav")) + sorted(dev_clean.rglob("*.flac"))
    if len(files) < 2:
        # Fallback to train-clean-100
        fallback = libri_dir / "train-clean-100"
        files = sorted(fallback.rglob("*.wav")) + sorted(fallback.rglob("*.flac"))
    assert len(files) >= 2, f"Need at least 2 audio files in {libri_dir}"

    n_spks = 2
    idxs = rng.choice(len(files), size=n_spks, replace=False)
    chosen = [files[int(i)] for i in idxs]

    waveforms = []
    for p in chosen:
        audio, sr = sf.read(str(p), dtype="float32")
        assert sr == 8000, f"Expected 8kHz, got {sr} Hz in {p}"
        waveforms.append(audio.flatten())

    max_t = max(len(w) for w in waveforms)
    waveforms = [np.pad(w, (0, max_t - len(w))) for w in waveforms]
    mixture = sum(waveforms).astype(np.float32)
    return mixture, waveforms, n_spks


# ---------------------------------------------------------------------------
# Diagnostic passes
# ---------------------------------------------------------------------------


def diag_sanity(
    ss_base: SSInference,
    ss_adapted: SSInference,
    lib: LoRALibrary,
    mixture: np.ndarray,
    n_spks: int,
) -> dict:
    """
    PASS 1: Does process_waveform == _forward_with_grad with gate=0?
    If they differ, our training forward pass computes something different
    from the model we actually care about, that's a pipeline bug.
    """
    print("\n[PASS 1] Sanity: process_waveform vs _forward_with_grad (gate=0)")

    base_out = run_base_model(ss_base, mixture, n_spks)

    # Gate=0 adapted model via process_waveform, should equal base
    device = _model_device(ss_adapted)
    lib.set_gates({"reverb": 0.0, "noise": 0.0, "codec": 0.0})
    lib.inject_gates()
    wav = torch.from_numpy(mixture).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        out_g0 = ss_adapted.process_waveform(wav, n_spks=torch.tensor(n_spks))
    g0_out = [w.squeeze().cpu().float().numpy() for w in out_g0["waveforms"]]

    # _forward_with_grad with gate=0
    lib.set_gates({"reverb": 0.0, "noise": 0.0, "codec": 0.0})
    lib.inject_gates()
    wav_t = torch.from_numpy(mixture).float().unsqueeze(0).to(device)
    waves_fwg, _ = _forward_with_grad(ss_adapted, wav_t, n_spks=torch.tensor(n_spks))
    fwg_out = waves_fwg.detach().cpu().float().numpy()

    # Compare base vs gate=0 adapted
    min_k = min(len(base_out), len(g0_out))
    diffs_pw = [np.abs(base_out[k] - g0_out[k][: len(base_out[k])]).max() for k in range(min_k)]
    diffs_fwg = [
        (
            np.abs(base_out[k] - fwg_out[k][: len(base_out[k])]).max()
            if k < len(fwg_out)
            else float("nan")
        )
        for k in range(min_k)
    ]

    mean_diff_pw = float(np.mean(diffs_pw))
    mean_diff_fwg = float(np.nanmean(diffs_fwg))

    print(f"  base vs adapted(gate=0) via process_waveform: max diff = {mean_diff_pw:.6f}")
    print(f"  base vs _forward_with_grad(gate=0):           max diff = {mean_diff_fwg:.6f}")

    pw_ok = mean_diff_pw < 1e-3
    fwg_ok = mean_diff_fwg < 1e-3
    print(f"  process_waveform match: {'✅ OK' if pw_ok else '❌ MISMATCH, LoRA init non-zero?'}")
    print(
        f"  _forward_with_grad match: {'✅ OK' if fwg_ok else '❌ MISMATCH, training forward has a bug!'}"
    )
    return {
        "pw_match": pw_ok,
        "fwg_match": fwg_ok,
        "diff_pw": mean_diff_pw,
        "diff_fwg": mean_diff_fwg,
    }


def diag_gate_effect(
    ss_adapted: SSInference,
    lib: LoRALibrary,
    mixture: np.ndarray,
    refs: list[np.ndarray],
    n_spks: int,
) -> dict:
    """
    PASS 2: Do B-matrices have non-zero weights? Does gate=1 actually change output?
    If gate=0 and gate=1 produce identical outputs → B stayed at zero → no learning happened.
    """
    print("\n[PASS 2] Gate effect: are LoRA weights non-zero?")

    # Check B norms
    b_norms = []
    a_norms = []
    for mod in _get_inner_module(ss_adapted).modules():
        if isinstance(mod, LoRALayer):
            b_norms.append(mod.B.data.norm().item())
            a_norms.append(mod.A.data.norm().item())

    mean_b = float(np.mean(b_norms))
    mean_a = float(np.mean(a_norms))
    print(f"  LoRA A mean norm: {mean_a:.4f}  (should be ~1–3 from kaiming init)")
    print(f"  LoRA B mean norm: {mean_b:.4f}  (0.0 at init; >0.01 means learning happened)")
    b_learned = mean_b > 0.01

    # gate=0 vs gate=1 output diff
    g0 = run_adapted_model(ss_adapted, lib, mixture, n_spks, gate=0.0)
    g1 = run_adapted_model(ss_adapted, lib, mixture, n_spks, gate=1.0)

    min_k = min(len(g0), len(g1))
    output_diffs = [np.abs(g0[k][: len(g1[k])] - g1[k][: len(g0[k])]).max() for k in range(min_k)]
    mean_output_diff = float(np.mean(output_diffs))
    print(f"  gate=0 vs gate=1 output max diff: {mean_output_diff:.6f}")

    snr_g0 = pit_si_snr_np(g0, refs)
    snr_g1 = pit_si_snr_np(g1, refs)
    delta = snr_g1 - snr_g0
    print(
        f"  SI-SNR(gate=0) = {snr_g0:.2f} dB   SI-SNR(gate=1) = {snr_g1:.2f} dB   Δ = {delta:+.2f} dB"
    )

    if not b_learned:
        print("  ❌ B matrices are near-zero, LoRA did not learn. Training had a gradient problem.")
    elif mean_output_diff < 1e-5:
        print("  ❌ B non-zero but output unchanged, gate injection not working during inference.")
    elif abs(delta) < 0.1:
        print(
            "  ⚠️  Adapter changes output but SI-SNR unchanged, learning did not improve separation."
        )
    else:
        print(f"  ✅ Adapter active and changes SI-SNR by {delta:+.2f} dB vs gate=0.")

    return {
        "b_learned": b_learned,
        "mean_b_norm": mean_b,
        "output_diff": mean_output_diff,
        "snr_g0": snr_g0,
        "snr_g1": snr_g1,
        "snr_delta": delta,
    }


def diag_sisnr(
    ss_base: SSInference,
    ss_adapted: SSInference,
    lib: LoRALibrary,
    rir_bank: RirBank,
    mixture_clean: np.ndarray,
    refs_clean: list[np.ndarray],
    n_spks: int,
    rng: np.random.Generator,
) -> dict:
    """
    PASS 3: SI-SNR comparison on clean / reverb-mild / reverb-strong conditions.
    Reports both absolute SI-SNR and SI-SNRi (improvement over mixture).
    """
    print("\n[PASS 3] SI-SNR comparison: base vs adapted")

    results = {}
    conditions = [
        ("clean", None),
        ("reverb_mild", 0.4),
        ("reverb_strong", 0.8),
    ]

    for cond_name, t60 in conditions:
        print(f"\n  --- Condition: {cond_name} ---")
        if t60 is None:
            mix, refs = mixture_clean.copy(), [r.copy() for r in refs_clean]
        else:
            recipe = MixtureRecipe(n_speakers=n_spks)
            max_t = max(len(r) for r in refs_clean)
            refs_arr = np.stack([np.pad(r, (0, max_t - len(r))) for r in refs_clean])
            mix_arr = refs_arr.sum(axis=0).astype(np.float32)
            sample = MixtureSample(
                mixture=mix_arr, references=refs_arr, sample_rate=8000, utterance_id="eval"
            )
            calmix = CoralSepMixture(sample=sample, recipe=recipe)
            rev = apply_reverb(calmix, rir_bank, rng, t60_s=t60)
            mix = rev.mixture
            # Score against the wet reference apply_reverb returns, not the dry
            # source. The reverb adapter is trained to separate but not
            # dereverberate (data/degradations.py, BLUEPRINT 7.6), so scoring
            # against the dry source here would grade it on a task it was never
            # asked to do. See I-025 and I-040.
            refs = [rev.references[i] for i in range(rev.references.shape[0])]

        # Mixture baseline, against the same reference the model is scored against.
        snr_mix_ref = si_snr_mixture(mix, refs)
        print(f"  SI-SNR(mixture vs reference) = {snr_mix_ref:.2f} dB  [lower bound]")

        # Base model
        base_out = run_base_model(ss_base, mix, n_spks)
        snr_base = pit_si_snr_np(base_out, refs)
        snri_base = snr_base - snr_mix_ref
        print(f"  Base  → SI-SNR = {snr_base:.2f} dB   SI-SNRi = {snri_base:+.2f} dB")

        # Adapted model (reverb gate=1)
        adp_out = run_adapted_model(ss_adapted, lib, mix, n_spks, gate=1.0)
        snr_adp = pit_si_snr_np(adp_out, refs)
        snri_adp = snr_adp - snr_mix_ref
        delta = snr_adp - snr_base
        print(
            f"  Adapt → SI-SNR = {snr_adp:.2f} dB   SI-SNRi = {snri_adp:+.2f} dB   Δ = {delta:+.2f} dB"
        )

        results[cond_name] = {
            "snr_mix": snr_mix_ref,
            "snr_base": snr_base,
            "snr_adp": snr_adp,
            "snri_base": snri_base,
            "snri_adp": snri_adp,
            "delta": delta,
        }

    return results


def diag_target(
    ss_base: SSInference,
    ss_adapted: SSInference,
    lib: LoRALibrary,
    rir_bank: RirBank,
    refs_clean: list[np.ndarray],
    n_spks: int,
    rng: np.random.Generator,
) -> dict:
    """
    PASS 4: Training loss decomposition.
    Shows SI-SNR(output, wet_ref) vs SI-SNR(output, anechoic_ref).
    Explains why training loss values (~8.6) looked bad vs WHAMR numbers.
    """
    print("\n[PASS 4] Training target analysis: wet reference vs anechoic reference")

    recipe = MixtureRecipe(n_speakers=n_spks)
    max_t = max(len(r) for r in refs_clean)
    refs_arr = np.stack([np.pad(r, (0, max_t - len(r))) for r in refs_clean])
    sample = MixtureSample(
        mixture=refs_arr.sum(0).astype(np.float32),
        references=refs_arr,
        sample_rate=8000,
        utterance_id="eval",
    )
    calmix = CoralSepMixture(sample=sample, recipe=MixtureRecipe(n_speakers=n_spks))
    rev = apply_reverb(calmix, rir_bank, rng, t60_s=0.5)

    mix = rev.mixture
    wet_refs = [rev.references[i] for i in range(rev.references.shape[0])]
    anec_refs = refs_clean

    print(f"  T60 = {rev.recipe.t60_s:.2f}s")

    base_out = run_base_model(ss_base, mix, n_spks)
    adp_out = run_adapted_model(ss_adapted, lib, mix, n_spks, gate=1.0)

    snr_base_wet = pit_si_snr_np(base_out, wet_refs)
    snr_adp_wet = pit_si_snr_np(adp_out, wet_refs)
    snr_base_anec = pit_si_snr_np(base_out, anec_refs)
    snr_adp_anec = pit_si_snr_np(adp_out, anec_refs)

    # Training loss ≈ -SI-SNR(output, wet_ref) - 0.5 * SI-SNR_freq
    # (freq SI-SNR is usually ~1-2 dB lower, so loss ≈ 1.5 × |SI-SNR_wet|)
    approx_loss_base = -snr_base_wet  # approximate
    approx_loss_adp = -snr_adp_wet

    print(f"\n  {'Metric':<35} {'Base':>10} {'Adapted':>10}")
    print(f"  {'-'*55}")
    print(
        f"  {'SI-SNR (vs wet reference)':<35} {snr_base_wet:>10.2f} {snr_adp_wet:>10.2f}  ← training target"
    )
    print(
        f"  {'SI-SNR (vs anechoic = true quality)':<35} {snr_base_anec:>10.2f} {snr_adp_anec:>10.2f}  ← real performance"
    )
    print(
        f"  {'Approx training loss (-SI-SNR_wet)':<35} {approx_loss_base:>10.2f} {approx_loss_adp:>10.2f}  ← what we logged"
    )

    gap_base = snr_base_anec - snr_base_wet
    print(f"\n  anechoic vs wet reference gap: {gap_base:+.2f} dB")
    # abs(), not a plain > check: the gap is informative in either direction.
    # A large negative gap (anechoic scores far lower than wet) means the
    # output is close to the wet target and far from the dry one, exactly
    # what a working separate-but-do-not-dereverberate adapter should look
    # like; a large positive gap would mean the reverse. Either way, a small
    # gap is the case that means the reference choice barely matters, not a
    # gap of either sign. See I-025 and I-040.
    if abs(gap_base) > 2:
        direction = "lower" if gap_base < 0 else "higher"
        print(
            f"  → Large gap: anechoic scores {direction} than wet by {abs(gap_base):.1f} dB. "
            "The reference choice materially changes what these numbers mean; "
            "see I-025 for which one is the correct measure for this adapter."
        )
    else:
        print(
            "  → Small gap: the loss values reflect real separation quality, not just target choice."
        )

    return {
        "snr_base_wet": snr_base_wet,
        "snr_adp_wet": snr_adp_wet,
        "snr_base_anec": snr_base_anec,
        "snr_adp_anec": snr_adp_anec,
        "anec_vs_wet_gap": gap_base,
    }


def save_audio(path: Path, wav: np.ndarray, sr: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), sr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Eval reverb adapter vs base SR-CorrNet")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--librispeech-8k", required=True)
    p.add_argument("--rir-bank", required=True)
    p.add_argument("--output-dir", default="eval_outputs")
    p.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank the checkpoint was trained at (must match, see load_adapted).",
    )
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    libri_dir = Path(args.librispeech_8k)
    rir_dir = Path(args.rir_bank)
    out_dir = Path(args.output_dir)
    device = args.device
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("  CoRAL-Sep Reverb Adapter Evaluation & Diagnosis")
    print("=" * 60)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")

    # Load models
    print("\n[SETUP] Loading models...")
    ss_base = load_base(device)
    ss_adapted, lib = load_adapted(ckpt_path, device, rank=args.rank)

    # Load RIR bank
    print(f"[SETUP] Loading RIR bank from {rir_dir} ...")
    rir_bank = RirBank(rir_dir)

    # Build one test mixture from dev-clean (held-out speakers)
    print(f"[SETUP] Building test mixture from {libri_dir} ...")
    mixture_clean, refs_clean, n_spks = build_test_mixture(libri_dir, rng)
    print(f"  {n_spks} speakers, mixture length = {len(mixture_clean)/8000:.1f}s")

    # Run all diagnostic passes
    r1 = diag_sanity(ss_base, ss_adapted, lib, mixture_clean, n_spks)
    r2 = diag_gate_effect(ss_adapted, lib, mixture_clean, refs_clean, n_spks)
    r3 = diag_sisnr(ss_base, ss_adapted, lib, rir_bank, mixture_clean, refs_clean, n_spks, rng)
    r4 = diag_target(ss_base, ss_adapted, lib, rir_bank, refs_clean, n_spks, rng)

    # Save audio samples (reverb_mild condition)
    print(f"\n[AUDIO] Saving audio to {out_dir} ...")
    max_t = max(len(r) for r in refs_clean)
    refs_arr = np.stack([np.pad(r, (0, max_t - len(r))) for r in refs_clean])
    sample = MixtureSample(
        mixture=refs_arr.sum(0).astype(np.float32),
        references=refs_arr,
        sample_rate=8000,
        utterance_id="eval",
    )
    rev = apply_reverb(
        CoralSepMixture(sample=sample, recipe=MixtureRecipe(n_speakers=n_spks)),
        rir_bank,
        rng,
        t60_s=0.4,
    )
    mix_rev = rev.mixture
    base_out = run_base_model(ss_base, mix_rev, n_spks)
    adp_out = run_adapted_model(ss_adapted, lib, mix_rev, n_spks, gate=1.0)

    save_audio(out_dir / "00_mixture_reverb.wav", mix_rev)
    save_audio(out_dir / "01_ref_speaker0_clean.wav", refs_clean[0])
    save_audio(out_dir / "02_ref_speaker1_clean.wav", refs_clean[1])
    for i, w in enumerate(base_out):
        save_audio(out_dir / f"03_base_speaker{i}.wav", w)
    for i, w in enumerate(adp_out):
        save_audio(out_dir / f"04_adapted_speaker{i}.wav", w)
    print(f"  Saved {2 + 2 + len(base_out) + len(adp_out)} files.")

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    reverb_delta = r3.get("reverb_mild", {}).get("delta", 0.0)
    b_learned = r2["b_learned"]
    fwg_ok = r1["fwg_match"]

    issues = []
    positives = []

    if not fwg_ok:
        issues.append(
            "_forward_with_grad differs from process_waveform, training loss computed on wrong output"
        )
    else:
        positives.append("Training forward pass matches inference forward pass ✅")

    if not b_learned:
        issues.append("LoRA B matrices are near-zero, gradients did not reach LoRA branches")
    else:
        positives.append(f"LoRA learned (B norm = {r2['mean_b_norm']:.4f}) ✅")

    if reverb_delta > 0.5:
        positives.append(f"Adapter improves reverb SI-SNR by {reverb_delta:+.2f} dB ✅")
    elif reverb_delta > 0:
        issues.append(f"Adapter improves by only {reverb_delta:+.2f} dB, marginal gain")
    else:
        issues.append(f"Adapter makes reverb SI-SNR worse by {reverb_delta:.2f} dB")

    gap = r4["anec_vs_wet_gap"]
    if gap > 2:
        positives.append(
            f"High training loss ({gap:.1f} dB anec/wet gap) is partly due to wet-reference "
            f"target, not model failure ✅"
        )

    print("\n  Positive findings:")
    for s in positives:
        print(f"    + {s}")
    if issues:
        print("\n  Issues found:")
        for s in issues:
            print(f"    ✗ {s}")

    print("\n  Root-cause summary:")
    if not b_learned:
        print(textwrap.dedent("""
            CRITICAL: LoRA B matrices never left zero. Despite the loss decreasing
            slightly, the adapter weights are not learning. Most likely cause:
            the freeze_base() bug (now fixed) prevented grads from reaching B.
            The checkpoint predates the fix. Retrain from scratch with current code.
        """))
    elif not fwg_ok:
        print(textwrap.dedent("""
            CRITICAL: _forward_with_grad computes something different from
            process_waveform. The training loss was measuring the wrong output.
            Fix _forward_with_grad and retrain.
        """))
    elif reverb_delta < 0.1:
        print(textwrap.dedent("""
            LoRA weights exist but don't improve SI-SNR. Possible causes:
            1. LoRA rank (8) too small for this task
            2. Only 500 samples/epoch, try 2000
            3. Wet-reference target confuses the adapter (switch to anechoic)
        """))
    else:
        print(textwrap.dedent(f"""
            Training is working. Adapter improves reverb SI-SNR by {reverb_delta:+.2f} dB.
            High training loss values were partly explained by the wet-reference
            target (anec/wet gap = {gap:.1f} dB). The real quality (vs anechoic)
            is better than the loss numbers suggested.
        """))

    print("=" * 60)


if __name__ == "__main__":
    main()
