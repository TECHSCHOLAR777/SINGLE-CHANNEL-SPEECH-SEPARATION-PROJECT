"""
CoRAL-Sep final evaluation: Baseline SR-CorrNet vs CoRAL-Sep (gate + LoRA adapters).

Runs on LibriMix test splits at 8 kHz (the native SR-CorrNet sample rate).
Band recovery is evaluated separately as SI-SDR would require 16 kHz references
which the LibriMix test set does not include.

Metrics:
  SI-SDR  (dB), scale-invariant SDR
  SI-SDRi (dB), improvement over the unprocessed mixture

Output: console table + JSON at results/eval_outputs/calmsep_eval.json

Usage:
    PYTHONPATH=/path/to/sr_corrnet_src:. python src/coralsep/eval/run_eval.py \
        --librimix /path/to/librimix_generated \
        --checkpoint-dir checkpoints/ \
        [--n-per-split 100]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from coralsep.eval.metrics import count_accuracy, pit_si_sdr
from coralsep.eval.stats import bootstrap_ci

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

_SR = 8_000
_LIBRIMIX_SPLITS = ["Libri2Mix", "Libri3Mix", "Libri4Mix", "Libri5Mix"]
_HF_MODEL = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _iter_test_samples(librimix_root: Path, split: str, n: int):
    """Yield (mix_wav, refs_wav, uid) for up to n samples from one split."""
    base = librimix_root / split / "wav8k" / "min" / "test"
    mix_dir = base / "mix_both"
    if not mix_dir.exists():
        log.warning("Test directory not found: %s", mix_dir)
        return

    files = sorted(mix_dir.glob("*.wav"))[:n]
    # Detect speaker count from existing sN directories
    n_spks = sum(1 for i in range(1, 6) if (base / f"s{i}").exists())

    for f in files:
        uid = f.stem
        try:
            mix, sr = sf.read(str(f), dtype="float32", always_2d=True)
            if sr != _SR:
                continue
            mix = mix[:, 0]
            refs = []
            for i in range(1, n_spks + 1):
                rp = base / f"s{i}" / f"{uid}.wav"
                if not rp.exists():
                    break
                r, _ = sf.read(str(rp), dtype="float32", always_2d=True)
                refs.append(r[:, 0])
            if len(refs) < 2:
                continue
            min_len = min(len(mix), *(len(r) for r in refs))
            yield mix[:min_len].astype(np.float32), np.stack([r[:min_len] for r in refs]).astype(
                np.float32
            ), uid
        except Exception as e:
            log.debug("Skip %s: %s", uid, e)


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------


def _load_base_model(device: torch.device):
    from coralsep.train.stage1_single import _get_inner_module, _load_model

    ss = _load_model(_HF_MODEL, device)
    inner = _get_inner_module(ss)
    inner.eval()
    for p in inner.parameters():
        p.requires_grad_(False)
    return ss


def _run_baseline(
    ss_model, mix_wav: np.ndarray, n_spks: int | None, device: torch.device
) -> np.ndarray:
    """Run base SR-CorrNet (no adapters). Returns (K, T) float32.

    n_spks=None uses the model's own attractor path (models/experts/srcorrnet.py
    Patch A documents the same call shape): the number of streams the model
    returns is its own count estimate, not a supplied answer. See I-002.
    """
    wav_t = torch.from_numpy(mix_wav).float().to(device).unsqueeze(0)
    with torch.no_grad():
        if n_spks is None:
            out = ss_model.process_waveform(wav_t)
        else:
            out = ss_model.process_waveform(wav_t, n_spks=torch.tensor(n_spks))
    waves = out.get("waveforms", [])
    if not waves:
        return mix_wav[np.newaxis]
    return np.stack(
        [
            (w.squeeze().cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w).squeeze())
            for w in waves
        ]
    )


def _load_universal_ckpt(path: Path) -> dict:
    """Load best_universal from either a .pt file or a PyTorch zip directory."""
    if path.is_dir():
        import io
        import zipfile

        buf = io.BytesIO()
        fixed = (2024, 1, 1, 0, 0, 0)
        with zipfile.ZipFile(buf, "w") as zf:
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    zi = zipfile.ZipInfo("archive/" + str(f.relative_to(path)), date_time=fixed)
                    zi.compress_type = zipfile.ZIP_STORED
                    zf.writestr(zi, f.read_bytes())
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=False)
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _load_calmsep(ckpt_dir: Path, device: torch.device, universal_ckpt: Path | None = None):
    from coralsep.models.condition import Level2Analyzer, level1_tensor
    from coralsep.models.gate import GateNetwork
    from coralsep.models.lora import ADAPTER_NAMES, LoRALibrary, LoRALinear
    from coralsep.train.stage1_single import _get_inner_module, _load_model

    ss = _load_model(_HF_MODEL, device)
    inner = _get_inner_module(ss)
    lib = LoRALibrary(inner)
    lib.freeze_base()

    analyzer = Level2Analyzer().to(device)
    gate_net = GateNetwork().to(device)

    joint_ckpt = ckpt_dir / "stage4_joint" / "best_joint.pt"
    if joint_ckpt.exists():
        ckpt = torch.load(str(joint_ckpt), map_location="cpu", weights_only=False)
        gate_net.load_state_dict(ckpt["gate"])
        analyzer.load_state_dict(ckpt["analyzer"])
        adapter_state = ckpt.get("adapter_state", {})
        loaded = 0
        for mod_name, mod in inner.named_modules():
            if not isinstance(mod, LoRALinear):
                continue
            for adapter_name, branch in mod.branches.items():
                for param_name, param in branch.named_parameters():
                    key = f"{mod_name}.branches.{adapter_name}.{param_name}"
                    if key in adapter_state:
                        param.data.copy_(adapter_state[key].to(param.device))
                        loaded += 1
        log.info("Loaded best_joint.pt: %d adapter tensors", loaded)
    else:
        log.warning("best_joint.pt not found at %s, using random gate", joint_ckpt)

    if universal_ckpt is not None and universal_ckpt.exists():
        univ = _load_universal_ckpt(universal_ckpt)
        univ_state = univ.get("state_dict", univ)
        loaded_u = 0
        for mod_name, mod in inner.named_modules():
            if not isinstance(mod, LoRALinear):
                continue
            if "universal" not in mod.branches:
                continue
            branch = mod.branches["universal"]
            for param_name, param in branch.named_parameters():
                key = f"adapter.universal.{mod_name}.branches.universal.{param_name}"
                alt = f"{mod_name}.branches.universal.{param_name}"
                src = univ_state.get(key) or univ_state.get(alt)
                if src is not None:
                    param.data.copy_(src.to(param.device))
                    loaded_u += 1
        log.info("Loaded universal adapter: %d tensors from %s", loaded_u, universal_ckpt)

    temperature = 1.0
    calib = ckpt_dir / "stage4c" / "calibration.pt"
    if calib.exists():
        c = torch.load(str(calib), map_location="cpu", weights_only=False)
        temperature = float(c["temperature"].item())
        log.info("Gate temperature T=%.4f", temperature)

    inner.to(device).eval()
    gate_net.eval()
    analyzer.eval()

    # Move STFT/iSTFT modules (engine-level) and sync wrapper device reference.
    # Mirrors stage1_single.py lines 352-361 and 524-535.
    engine = getattr(ss, "engine", None)
    if engine is not None:
        for attr in ("stft", "istft"):
            mod = getattr(engine, attr, None)
            if mod is not None and hasattr(mod, "to"):
                mod.to(device)
        try:
            engine.device = device
        except Exception:
            pass
    try:
        ss.device = device
    except Exception:
        pass

    return ss, inner, lib, gate_net, temperature, level1_tensor, ADAPTER_NAMES


def _run_calmsep(
    ss_model,
    inner,
    lib,
    gate_net,
    temperature,
    level1_tensor,
    ADAPTER_NAMES,
    mix_wav: np.ndarray,
    n_spks: int | None,
    device: torch.device,
) -> np.ndarray:
    """Run full CoRAL-Sep pipeline. Returns (K, T) float32 at 8 kHz.

    n_spks=None uses the model's own attractor path instead of a supplied
    count. See I-002 and _run_baseline's docstring.
    """
    from coralsep.models.lora import LoRALinear  # noqa: F401

    wav_t = torch.from_numpy(mix_wav).float().to(device)
    l1 = level1_tensor(wav_t).to(device)
    l2 = torch.zeros(6, device=device)
    cond = torch.cat([l1, l2]).unsqueeze(0)

    with torch.no_grad():
        gate_prob = gate_net(cond).squeeze(0).float().clamp(1e-6, 1 - 1e-6)

    if temperature != 1.0:
        logit = torch.log(gate_prob / (1 - gate_prob))
        gate_prob = torch.sigmoid(logit / temperature)

    lib.set_gates({ADAPTER_NAMES[i]: gate_prob[i].item() for i in range(3)})
    lib.inject_gates()

    wav_in = wav_t.unsqueeze(0)
    with torch.no_grad():
        if n_spks is None:
            out = ss_model.process_waveform(wav_in)
        else:
            out = ss_model.process_waveform(wav_in, n_spks=torch.tensor(n_spks))

    lib.set_gates({n: 0.0 for n in ADAPTER_NAMES})
    lib.inject_gates()

    waves = out.get("waveforms", [])
    if not waves:
        return mix_wav[np.newaxis]
    return np.stack(
        [
            (w.squeeze().cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w).squeeze())
            for w in waves
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _score_split(
    split: str,
    librimix_root: Path,
    n: int,
    ss_base,
    ss_calm,
    inner,
    lib,
    gate_net,
    temperature,
    l1fn,
    ADAPTER_NAMES,
    device: torch.device,
    oracle_count: bool = False,
) -> dict:
    """Score one LibriMix split.

    oracle_count=False (the default, see I-002) never tells either model how
    many speakers are in the mixture; each model's own attractor path decides,
    and N_hat is compared against the true count read from the split name.
    oracle_count=True reproduces the original behaviour, supplying the true
    count directly, kept only so old numbers stay reproducible for comparison.
    """
    base_sisdrs, base_sisdris = [], []
    calm_sisdrs, calm_sisdris = [], []
    base_n_true, base_n_hat = [], []
    calm_n_true, calm_n_hat = [], []
    base_per_sample: list[dict] = []
    calm_per_sample: list[dict] = []
    n_true = int(split.replace("Libri", "").replace("Mix", ""))
    n_spks_arg = n_true if oracle_count else None

    for i, (mix, refs, uid) in enumerate(_iter_test_samples(librimix_root, split, n)):

        def _align(est: np.ndarray, ref: np.ndarray, mix_: np.ndarray):
            """Trim all arrays to the shortest common length."""
            T = min(est.shape[-1], ref.shape[-1], mix_.shape[-1])
            return est[..., :T], ref[..., :T], mix_[:T]

        # Baseline
        try:
            est_b = _run_baseline(ss_base, mix, n_spks_arg, device)
            base_n_true.append(n_true)
            base_n_hat.append(est_b.shape[0])
            est_b, refs_a, mix_a = _align(est_b, refs, mix)
            r_b = pit_si_sdr(est_b, refs_a, mix_a)
            base_sisdrs.append(r_b.mean_si_sdr)
            base_sisdris.append(r_b.mean_si_sdri)
            base_per_sample.append(
                {
                    "uid": uid,
                    "n_true": n_true,
                    "n_hat": int(est_b.shape[0]),
                    "si_sdr": r_b.mean_si_sdr,
                    "si_sdri": r_b.mean_si_sdri,
                }
            )
        except Exception as e:
            log.debug("[%s] baseline skip %s: %s", split, uid, e)

        # CoRAL-Sep
        try:
            est_c = _run_calmsep(
                ss_calm,
                inner,
                lib,
                gate_net,
                temperature,
                l1fn,
                ADAPTER_NAMES,
                mix,
                n_spks_arg,
                device,
            )
            calm_n_true.append(n_true)
            calm_n_hat.append(est_c.shape[0])
            est_c, refs_a, mix_a = _align(est_c, refs, mix)
            r_c = pit_si_sdr(est_c, refs_a, mix_a)
            calm_sisdrs.append(r_c.mean_si_sdr)
            calm_sisdris.append(r_c.mean_si_sdri)
            calm_per_sample.append(
                {
                    "uid": uid,
                    "n_true": n_true,
                    "n_hat": int(est_c.shape[0]),
                    "si_sdr": r_c.mean_si_sdr,
                    "si_sdri": r_c.mean_si_sdri,
                }
            )
        except Exception as e:
            log.debug("[%s] coralsep skip %s: %s", split, uid, e)

        if (i + 1) % 10 == 0:
            log.info(
                "%s [%d/%d] base SI-SDRi=%.2f | calm SI-SDRi=%.2f",
                split,
                i + 1,
                n,
                float(np.mean(base_sisdris)) if base_sisdris else 0,
                float(np.mean(calm_sisdris)) if calm_sisdris else 0,
            )

    def _ci(vals: list[float]) -> dict | None:
        """BCa 95% CI on SI-SDRi, or None below the resample floor (I-026)."""
        if len(vals) < 8:
            return None
        mean, lo, hi = bootstrap_ci(np.array(vals, dtype=np.float64))
        return {"mean": mean, "ci_low": lo, "ci_high": hi}

    return {
        "split": split,
        "n_samples": len(base_sisdrs),
        "oracle_count": oracle_count,
        "baseline": {
            "si_sdr": float(np.mean(base_sisdrs)) if base_sisdrs else None,
            "si_sdri": float(np.mean(base_sisdris)) if base_sisdris else None,
            "si_sdri_ci": _ci(base_sisdris),
            "count_accuracy": (count_accuracy(base_n_true, base_n_hat) if base_n_true else None),
            "per_sample": base_per_sample,
        },
        "coralsep": {
            "si_sdr": float(np.mean(calm_sisdrs)) if calm_sisdrs else None,
            "si_sdri": float(np.mean(calm_sisdris)) if calm_sisdris else None,
            "si_sdri_ci": _ci(calm_sisdris),
            "count_accuracy": (count_accuracy(calm_n_true, calm_n_hat) if calm_n_true else None),
            "per_sample": calm_per_sample,
        },
        "delta_si_sdri": (
            float(np.mean(calm_sisdris)) - float(np.mean(base_sisdris))
            if base_sisdris and calm_sisdris
            else None
        ),
        "delta_si_sdr": (
            float(np.mean(calm_sisdrs)) - float(np.mean(base_sisdrs))
            if base_sisdrs and calm_sisdrs
            else None
        ),
    }


def _print_table(results: list[dict]) -> None:
    oracle = results[0].get("oracle_count", True) if results else True
    print("\n" + "=" * 90)
    if oracle:
        print("  WARNING: --oracle-count was used. Count accuracy is not meaningful, see I-002.")
    print(
        f"{'Split':<12} {'N':>5}  {'Base SI-SDR':>11} {'Base SI-SDRi':>12} {'Base N-acc':>10} "
        f"{'Calm SI-SDR':>11} {'Calm SI-SDRi':>12} {'Calm N-acc':>10} {'Δ SI-SDRi':>10}"
    )
    print("-" * 90)
    for r in results:
        b = r["baseline"]
        c = r["coralsep"]
        d = r["delta_si_sdri"]
        b_acc = b.get("count_accuracy")
        c_acc = c.get("count_accuracy")
        print(
            f"{r['split']:<12} {r['n_samples']:>5}  "
            f"{b['si_sdr']:>10.2f}  {b['si_sdri']:>11.2f}  "
            f"{b_acc if b_acc is None else f'{b_acc:.2f}':>10} "
            f"{c['si_sdr']:>10.2f}  {c['si_sdri']:>11.2f}  "
            f"{c_acc if c_acc is None else f'{c_acc:.2f}':>10} "
            f"{d:>+9.2f}"
        )
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--librimix",
        required=True,
        help="Root of librimix_generated (contains Libri2Mix/, Libri3Mix/)",
    )
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument(
        "--n-per-split",
        type=int,
        default=100,
        help="Samples per split (default 100; set lower for quick smoke-test)",
    )
    p.add_argument(
        "--universal-ckpt",
        default=None,
        help="Path to Stage 2 best_universal checkpoint (file or directory)",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Which splits to run (default: all available in --librimix dir)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results/eval_outputs/calmsep_eval.json")
    p.add_argument(
        "--oracle-count",
        action="store_true",
        default=False,
        help=(
            "Supply the true speaker count to both models (the original, invalid "
            "behaviour, kept only to reproduce old numbers for comparison). "
            "Default: neither model is told the count; each estimates it from "
            "its own attractor path. See I-002."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    librimix = Path(args.librimix)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading baseline SR-CorrNet …")
    ss_base = _load_base_model(device)

    log.info("Loading CoRAL-Sep …")
    universal_ckpt = Path(args.universal_ckpt) if args.universal_ckpt else None
    ss_calm, inner, lib, gate_net, temperature, l1fn, ADAPTER_NAMES = _load_calmsep(
        ckpt_dir, device, universal_ckpt
    )

    # Auto-detect available splits from the librimix directory
    available = sorted(
        d.name
        for d in librimix.iterdir()
        if d.is_dir()
        and d.name.startswith("Libri")
        and d.name.endswith("Mix")
        and (d / "wav8k" / "min" / "test" / "mix_both").exists()
    )
    splits_to_run = args.splits if args.splits else [s for s in _LIBRIMIX_SPLITS if s in available]
    log.info("Splits found: %s | running: %s", available, splits_to_run)

    results = []
    for split in splits_to_run:
        log.info("─── %s (n=%d) ───", split, args.n_per_split)
        t0 = time.time()
        r = _score_split(
            split,
            librimix,
            args.n_per_split,
            ss_base,
            ss_calm,
            inner,
            lib,
            gate_net,
            temperature,
            l1fn,
            ADAPTER_NAMES,
            device,
            oracle_count=args.oracle_count,
        )
        r["wall_time_s"] = round(time.time() - t0, 1)
        results.append(r)
        log.info("%s done in %.0fs", split, r["wall_time_s"])

    _print_table(results)

    payload = {"results": results, "model": _HF_MODEL, "checkpoint_dir": str(ckpt_dir)}
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
