"""
Stage 4c: Gate temperature scaling (CPU-only, ~5-10 min).

Post-hoc calibration: find scalar T so that
    sigmoid(gate_logit / T) → well-calibrated binary probabilities.

Steps:
  1. Load gate_net weights from best_joint.pt (Stage 4 output).
  2. Draw N calibration samples via the same gate dataset as Stage 3/4.
     Level-2 init is zeros (first-sample approximation — fine for temp scaling).
  3. Collect (gate_prob, oracle_label) pairs for all 3 adapters.
  4. Golden-section search over T ∈ (0.05, 10.0) minimising BCE.
  5. Save calibration.pt = {"temperature": T_tensor}.

No GPU, no SR-CorrNet. Runs in under 10 minutes on CPU.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from models.condition import level1_tensor
from models.gate import GateNetwork, oracle_gate
from train.stage3_gate import _build_gate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEVICE = torch.device("cpu")


def _load_gate(ckpt_path: Path, gate_net: GateNetwork) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=_DEVICE, weights_only=False)
    key = "gate" if "gate" in ckpt else next(k for k in ckpt if "gate" in k.lower())
    gate_net.load_state_dict(ckpt[key])
    log.info("Loaded gate from %s (key=%s)", ckpt_path, key)


def _collect(
    loader: object,
    gate_net: GateNetwork,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward gate on each batch mixture; collect (prob, oracle_label) pairs.
    Returns (probs [M*3], labels [M*3]) where M <= n_samples.
    """
    gate_net.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    collected = 0
    l2_zeros = torch.zeros(6)

    for batch in loader:  # type: ignore[union-attr]
        if collected >= n_samples:
            break
        mixture = batch["mixture"]  # (B, T) float32
        recipes = batch["recipe"]  # list[dict[str,float]]
        B = mixture.shape[0]

        for b in range(B):
            if collected >= n_samples:
                break
            try:
                wav = mixture[b]
                l1_feat = level1_tensor(wav)  # (4,)
                cond = torch.cat([l1_feat, l2_zeros]).unsqueeze(0)  # (1,10)

                with torch.no_grad():
                    gate_prob = gate_net(cond).squeeze(0).float()  # (3,) ∈ [0,1]

                oracle_b = oracle_gate([recipes[b]], device=_DEVICE).squeeze(0)  # (3,)

                all_probs.append(gate_prob)
                all_labels.append(oracle_b.float())
                collected += 1
            except Exception as e:
                log.debug("Sample skipped: %s", e)

        if collected % 50 == 0 and collected > 0:
            log.info("Collected %d / %d calibration samples", collected, n_samples)

    log.info("Calibration set: %d samples → %d pairs", collected, collected * 3)
    return torch.stack(all_probs).view(-1), torch.stack(all_labels).view(-1)


def _bce_at_temp(T: float, probs: torch.Tensor, labels: torch.Tensor) -> float:
    """BCE after re-scaling pre-sigmoid logits by temperature T."""
    with torch.no_grad():
        eps = 1e-6
        logits = torch.log(probs.clamp(eps, 1 - eps) / (1 - probs.clamp(eps, 1 - eps)))
        scaled = torch.sigmoid(logits / T).clamp(eps, 1 - eps)
        return float(F.binary_cross_entropy(scaled, labels).item())


def calibrate(args: argparse.Namespace) -> None:
    # ── Load gate ──────────────────────────────────────────────────────────
    gate_net = GateNetwork()
    ckpt_path = Path(getattr(args, "gate_checkpoint", ""))
    if ckpt_path.exists():
        _load_gate(ckpt_path, gate_net)
    else:
        log.warning("Checkpoint not found at %s — using random gate (sanity only)", ckpt_path)

    # ── Build calibration loader (same as Stage 3 training) ───────────────
    # Reduce dataset size to n_calibration samples
    n_cal = getattr(args, "n_calibration", 200)
    _orig_spe = getattr(args, "samples_per_epoch", n_cal)
    args.samples_per_epoch = n_cal
    args.batch_size = getattr(args, "batch_size", 4)
    args.num_workers = 0
    loader = _build_gate_dataset(args)
    args.samples_per_epoch = _orig_spe  # restore

    # ── Collect (gate_prob, oracle_label) pairs ────────────────────────────
    probs, labels = _collect(loader, gate_net, n_cal)

    # ── Golden-section search for optimal temperature T ────────────────────
    bce_T1 = _bce_at_temp(1.0, probs, labels)
    result = minimize_scalar(
        lambda T: _bce_at_temp(T, probs, labels),
        bounds=(0.05, 10.0),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 300},
    )
    T_opt = float(result.x)
    bce_opt = _bce_at_temp(T_opt, probs, labels)
    log.info("Temperature T=%.4f  BCE: %.4f → %.4f (T=1)", T_opt, bce_T1, bce_opt)

    # ── Save ───────────────────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"temperature": torch.tensor(T_opt)}, out / "calibration.pt")
    log.info("Saved calibration.pt  T=%.4f", T_opt)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-8k", required=True)
    p.add_argument("--rir-bank", default="")
    p.add_argument("--noise-dir", default="")
    p.add_argument(
        "--gate-checkpoint", required=True, help="Path to best_joint.pt (Stage 4 output)"
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-calibration", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    calibrate(_parse_args())
