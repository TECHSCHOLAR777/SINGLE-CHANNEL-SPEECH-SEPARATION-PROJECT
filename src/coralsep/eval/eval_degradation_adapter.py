"""
Evaluation and diagnosis of the noise and codec LoRA adapters.

I-025 found the reverb adapter was harmful, and no equivalent diagnostic had
ever been run against the noise or codec adapters (CKPT-002, CKPT-003 in
docs/restoration/DATA_AND_MODEL_INVENTORY.md): both had only ever been loaded
for the I-043 co-activation check, never independently scored against the
frozen backbone. This script closes that gap.

Unlike the reverb adapter, noise and codec damage do not touch the
references (data/degradations.py apply_noise and apply_codec both leave
mixture.references untouched), so there is no wet/anechoic ambiguity here:
one reference, one score, no PASS 4 equivalent to eval_reverb_adapter.py's
wet-vs-anechoic decomposition.

Runs 3 diagnostic passes:
  1. SANITY, does gate=0 match the frozen backbone exactly?
  2. GATE, are LoRA B-matrices non-zero? Does gate=1 change the output?
  3. SI-SNR, base vs adapted on clean / mild / severe conditions

Usage:
  python src/coralsep/eval/eval_degradation_adapter.py \
    --adapter noise \
    --checkpoint ablations/.../best_noise.pt \
    --librispeech-8k <dir> \
    --noise-dir <staged noise dir> \
    --device cuda

  python src/coralsep/eval/eval_degradation_adapter.py \
    --adapter codec \
    --checkpoint ablations/.../best_codec.pt \
    --librispeech-8k <dir> \
    --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Path setup, same pattern as eval_reverb_adapter.py
# ---------------------------------------------------------------------------


def _add_paths() -> None:
    repo = Path(__file__).resolve().parent.parent
    candidates = [repo, repo.parent / "SR_CorrNet_SS"]
    for p in candidates:
        s = str(p)
        if s not in sys.path and p.exists():
            sys.path.insert(0, s)


_add_paths()

from sr_corrnet import SSInference  # type: ignore

from coralsep.data.condition_mixer import CoralSepMixture, MixtureRecipe  # type: ignore
from coralsep.data.degradations import apply_codec, apply_noise  # type: ignore
from coralsep.data.mixer_stub import MixtureSample  # type: ignore
from coralsep.models.lora import LoRALayer, LoRALibrary  # type: ignore
from coralsep.train.stage1_single import _forward_with_grad, _get_inner_module  # type: ignore

_EPS = 1e-10
HF_CKPT = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"

_ZERO_GATES = {"reverb": 0.0, "noise": 0.0, "codec": 0.0}


def _gate_dict(adapter: str, value: float) -> dict:
    gates = dict(_ZERO_GATES)
    gates[adapter] = value
    return gates


# ---------------------------------------------------------------------------
# Metrics, identical to eval_reverb_adapter.py
# ---------------------------------------------------------------------------


def si_snr_np(estimate: np.ndarray, target: np.ndarray) -> float:
    min_t = min(len(estimate), len(target))
    e, t = estimate[:min_t].astype(np.float64), target[:min_t].astype(np.float64)
    e -= e.mean()
    t -= t.mean()
    s_tgt = (e @ t) / (t @ t + _EPS) * t
    noise = e - s_tgt
    return float(10 * np.log10((s_tgt @ s_tgt) / (noise @ noise + _EPS) + _EPS))


def pit_si_snr_np(estimates: list[np.ndarray], references: list[np.ndarray]) -> float:
    from itertools import permutations

    K = min(len(estimates), len(references))
    best = -999.0
    for perm in permutations(range(len(estimates)), K):
        val = np.mean([si_snr_np(estimates[perm[j]], references[j]) for j in range(K)])
        if val > best:
            best = val
    return best


def si_snr_mixture(mixture: np.ndarray, references: list[np.ndarray]) -> float:
    return np.mean([si_snr_np(mixture, r) for r in references]).item()


# ---------------------------------------------------------------------------
# Model helpers, identical to eval_reverb_adapter.py
# ---------------------------------------------------------------------------


def _move_stft_to_device(ss: SSInference, device: str) -> None:
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
    print(f"  Loading adapted model from {checkpoint} ...")
    ss = SSInference.from_pretrained(checkpoint_path=HF_CKPT, device=device)
    inner = _get_inner_module(ss)
    lib = LoRALibrary(inner, attn_rank=rank)
    lib.freeze_base()
    inner.to(device)
    _move_stft_to_device(ss, device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state = ckpt["state_dict"]
    inner_sd = inner.state_dict()
    loaded = 0
    for full_key, tensor in state.items():
        parts = full_key.split(".", 2)
        local_key = parts[2] if len(parts) >= 3 else full_key
        if local_key in inner_sd:
            inner_sd[local_key] = tensor
            loaded += 1
    inner.load_state_dict(inner_sd, strict=False)
    print(f"  Loaded {loaded} LoRA tensors from checkpoint.")
    return ss, lib


def _model_device(ss: SSInference) -> torch.device:
    return next(_get_inner_module(ss).parameters()).device


def run_base_model(ss: SSInference, wav_np: np.ndarray, n_spks: int) -> list[np.ndarray]:
    wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(_model_device(ss))
    with torch.inference_mode():
        out = ss.process_waveform(wav, n_spks=torch.tensor(n_spks))
    return [w.squeeze().cpu().float().numpy() for w in out["waveforms"]]


def run_adapted_model(
    ss: SSInference,
    lib: LoRALibrary,
    wav_np: np.ndarray,
    n_spks: int,
    adapter: str,
    gate: float = 1.0,
) -> list[np.ndarray]:
    lib.set_gates(_gate_dict(adapter, gate))
    lib.inject_gates()
    wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(_model_device(ss))
    with torch.inference_mode():
        out = ss.process_waveform(wav, n_spks=torch.tensor(n_spks))
    return [w.squeeze().cpu().float().numpy() for w in out["waveforms"]]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def build_test_mixture(
    libri_dir: Path, rng: np.random.Generator
) -> tuple[np.ndarray, list[np.ndarray], int]:
    dev_clean = libri_dir / "dev-clean"
    if not dev_clean.exists():
        dev_clean = libri_dir

    files = sorted(dev_clean.rglob("*.wav")) + sorted(dev_clean.rglob("*.flac"))
    if len(files) < 2:
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


def load_one_noise_file(noise_dir: Path, rng: np.random.Generator) -> np.ndarray:
    files = sorted(noise_dir.rglob("*.wav"))
    assert files, f"No .wav noise files found under {noise_dir}"
    idx = int(rng.integers(0, len(files)))
    audio, sr = sf.read(str(files[idx]), dtype="float32")
    assert sr == 8000, f"Expected 8kHz noise, got {sr} Hz in {files[idx]}"
    return audio.flatten()


def _make_mixture(refs: list[np.ndarray], n_spks: int) -> CoralSepMixture:
    max_t = max(len(r) for r in refs)
    refs_arr = np.stack([np.pad(r, (0, max_t - len(r))) for r in refs])
    sample = MixtureSample(
        mixture=refs_arr.sum(axis=0).astype(np.float32),
        references=refs_arr,
        sample_rate=8000,
        utterance_id="eval",
    )
    return CoralSepMixture(sample=sample, recipe=MixtureRecipe(n_speakers=n_spks))


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
    print("\n[PASS 1] Sanity: process_waveform vs _forward_with_grad (gate=0)")

    base_out = run_base_model(ss_base, mixture, n_spks)

    device = _model_device(ss_adapted)
    lib.set_gates(dict(_ZERO_GATES))
    lib.inject_gates()
    wav = torch.from_numpy(mixture).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        out_g0 = ss_adapted.process_waveform(wav, n_spks=torch.tensor(n_spks))
    g0_out = [w.squeeze().cpu().float().numpy() for w in out_g0["waveforms"]]

    lib.set_gates(dict(_ZERO_GATES))
    lib.inject_gates()
    wav_t = torch.from_numpy(mixture).float().unsqueeze(0).to(device)
    waves_fwg, _ = _forward_with_grad(ss_adapted, wav_t, n_spks=torch.tensor(n_spks))
    fwg_out = waves_fwg.detach().cpu().float().numpy()

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
    print(f"  process_waveform match: {'OK' if pw_ok else 'MISMATCH'}")
    print(f"  _forward_with_grad match: {'OK' if fwg_ok else 'MISMATCH'}")
    return {"pw_match": pw_ok, "fwg_match": fwg_ok}


def diag_gate_effect(
    ss_adapted: SSInference,
    lib: LoRALibrary,
    mixture: np.ndarray,
    refs: list[np.ndarray],
    n_spks: int,
    adapter: str,
) -> dict:
    print("\n[PASS 2] Gate effect: are LoRA weights non-zero?")

    b_norms, a_norms = [], []
    for mod in _get_inner_module(ss_adapted).modules():
        if isinstance(mod, LoRALayer):
            b_norms.append(mod.B.data.norm().item())
            a_norms.append(mod.A.data.norm().item())

    mean_b = float(np.mean(b_norms))
    mean_a = float(np.mean(a_norms))
    print(f"  LoRA A mean norm: {mean_a:.4f}")
    print(f"  LoRA B mean norm: {mean_b:.4f}  (0.0 at init; >0.01 means learning happened)")
    b_learned = mean_b > 0.01

    g0 = run_adapted_model(ss_adapted, lib, mixture, n_spks, adapter, gate=0.0)
    g1 = run_adapted_model(ss_adapted, lib, mixture, n_spks, adapter, gate=1.0)

    min_k = min(len(g0), len(g1))
    output_diffs = [np.abs(g0[k][: len(g1[k])] - g1[k][: len(g0[k])]).max() for k in range(min_k)]
    mean_output_diff = float(np.mean(output_diffs))
    print(f"  gate=0 vs gate=1 output max diff: {mean_output_diff:.6f}")

    snr_g0 = pit_si_snr_np(g0, refs)
    snr_g1 = pit_si_snr_np(g1, refs)
    delta = snr_g1 - snr_g0
    print(
        f"  SI-SNR(gate=0) = {snr_g0:.2f} dB   SI-SNR(gate=1) = {snr_g1:.2f} dB   Delta = {delta:+.2f} dB"
    )

    if not b_learned:
        print("  FAIL: B matrices are near-zero, LoRA did not learn.")
    elif mean_output_diff < 1e-5:
        print("  FAIL: B non-zero but output unchanged, gate injection not working.")
    else:
        print(f"  OK: adapter active and changes SI-SNR by {delta:+.2f} dB vs gate=0.")

    return {
        "b_learned": b_learned,
        "mean_b_norm": mean_b,
        "output_diff": mean_output_diff,
        "snr_delta": delta,
    }


def diag_sisnr_noise(
    ss_base: SSInference,
    ss_adapted: SSInference,
    lib: LoRALibrary,
    mixture_clean: np.ndarray,
    refs_clean: list[np.ndarray],
    n_spks: int,
    noise_dir: Path,
    rng: np.random.Generator,
) -> dict:
    print("\n[PASS 3] SI-SNR comparison: base vs adapted (noise conditions)")
    conditions = [("clean", None), ("noise_mild", 5.0), ("noise_severe", -6.0)]
    return _run_sisnr_conditions(
        ss_base,
        ss_adapted,
        lib,
        mixture_clean,
        refs_clean,
        n_spks,
        "noise",
        conditions,
        lambda calmix, snr_db: apply_noise(
            calmix, load_one_noise_file(noise_dir, rng), rng, snr_db=snr_db
        ),
    )


def diag_sisnr_codec(
    ss_base: SSInference,
    ss_adapted: SSInference,
    lib: LoRALibrary,
    mixture_clean: np.ndarray,
    refs_clean: list[np.ndarray],
    n_spks: int,
) -> dict:
    print("\n[PASS 3] SI-SNR comparison: base vs adapted (codec conditions)")
    conditions = [
        ("clean", None),
        ("codec_opus_16k", ("opus", 16000)),
        ("codec_amr_nb_4750", ("amr-nb", 4750)),
    ]
    return _run_sisnr_conditions(
        ss_base,
        ss_adapted,
        lib,
        mixture_clean,
        refs_clean,
        n_spks,
        "codec",
        conditions,
        lambda calmix, spec: apply_codec(calmix, spec[0], spec[1]),
    )


def _run_sisnr_conditions(
    ss_base, ss_adapted, lib, mixture_clean, refs_clean, n_spks, adapter, conditions, degrade_fn
) -> dict:
    results = {}
    for cond_name, param in conditions:
        print(f"\n  --- Condition: {cond_name} ---")
        if param is None:
            mix, refs = mixture_clean.copy(), [r.copy() for r in refs_clean]
        else:
            calmix = _make_mixture(refs_clean, n_spks)
            damaged = degrade_fn(calmix, param)
            mix = damaged.mixture
            refs = [damaged.references[i] for i in range(damaged.references.shape[0])]

        snr_mix_ref = si_snr_mixture(mix, refs)
        print(f"  SI-SNR(mixture vs reference) = {snr_mix_ref:.2f} dB  [lower bound]")

        base_out = run_base_model(ss_base, mix, n_spks)
        snr_base = pit_si_snr_np(base_out, refs)
        snri_base = snr_base - snr_mix_ref
        print(f"  Base  -> SI-SNR = {snr_base:.2f} dB   SI-SNRi = {snri_base:+.2f} dB")

        adp_out = run_adapted_model(ss_adapted, lib, mix, n_spks, adapter, gate=1.0)
        snr_adp = pit_si_snr_np(adp_out, refs)
        snri_adp = snr_adp - snr_mix_ref
        delta = snr_adp - snr_base
        print(
            f"  Adapt -> SI-SNR = {snr_adp:.2f} dB   SI-SNRi = {snri_adp:+.2f} dB   Delta = {delta:+.2f} dB"
        )

        results[cond_name] = {
            "snr_mix": snr_mix_ref,
            "snr_base": snr_base,
            "snr_adp": snr_adp,
            "delta": delta,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Eval noise/codec adapter vs base SR-CorrNet")
    p.add_argument("--adapter", required=True, choices=["noise", "codec"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--librispeech-8k", required=True)
    p.add_argument("--noise-dir", default="", help="Required for --adapter noise")
    p.add_argument("--device", default="cpu")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.adapter == "noise" and not args.noise_dir:
        p.error("--noise-dir is required for --adapter noise")

    ckpt_path = Path(args.checkpoint)
    libri_dir = Path(args.librispeech_8k)
    device = args.device
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print(f"  CoRAL-Sep {args.adapter.upper()} Adapter Evaluation")
    print("=" * 60)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")

    print("\n[SETUP] Loading models...")
    ss_base = load_base(device)
    ss_adapted, lib = load_adapted(ckpt_path, device, rank=args.rank)

    print(f"[SETUP] Building test mixture from {libri_dir} ...")
    mixture_clean, refs_clean, n_spks = build_test_mixture(libri_dir, rng)
    print(f"  {n_spks} speakers, mixture length = {len(mixture_clean)/8000:.1f}s")

    r1 = diag_sanity(ss_base, ss_adapted, lib, mixture_clean, n_spks)
    r2 = diag_gate_effect(ss_adapted, lib, mixture_clean, refs_clean, n_spks, args.adapter)

    if args.adapter == "noise":
        r3 = diag_sisnr_noise(
            ss_base,
            ss_adapted,
            lib,
            mixture_clean,
            refs_clean,
            n_spks,
            Path(args.noise_dir),
            rng,
        )
    else:
        r3 = diag_sisnr_codec(ss_base, ss_adapted, lib, mixture_clean, refs_clean, n_spks)

    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    if not r1["pw_match"] or not r1["fwg_match"]:
        print("  FAIL: training forward pass does not match inference forward pass.")
    if not r2["b_learned"]:
        print("  FAIL: LoRA did not learn (B norm near zero).")
    for cond, res in r3.items():
        if cond == "clean":
            continue
        verdict = "helps" if res["delta"] > 0 else "hurts"
        print(f"  {cond}: adapter {verdict} by {res['delta']:+.2f} dB vs base")
    print("=" * 60)


if __name__ == "__main__":
    main()
