"""Fit and hash the post-hoc calibration artifacts (BLUEPRINT §8.5).

Run after the joint stage, on held-out predictions:

    python -m train.calibrate --held-out runs/stage4/heldout.npz --out-dir calibration/artifacts

Every artifact this writes is recorded in a manifest with its SHA-256, so a
later run can be proven identical to the one behind a published number or
proven different. `utils.hashing.verify_manifest` does the checking.

Relationship to `train/stage4c_calib.py`: that script fits the gate temperature
during Stage 4c and is the source of the T value in `calibration.pt`. This one
fits the four post-hoc calibrators that consume Stage 4 outputs, and is the only
place that hashes them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from calibration.completeness import CompletenessCalibrator
from calibration.confidence import ConfidenceCalibrator
from calibration.ood import OODCalibrator
from calibration.temperature import TemperatureScaler
from utils.hashing import write_manifest
from utils.logging import get_logger

log = get_logger("calibrate")

REQUIRED_ARRAYS: tuple[str, ...] = (
    "count_logits",
    "count_labels",
    "conf_scores",
    "conf_labels",
    "comp_probs",
    "comp_labels",
    "ood_features",
)
"""Arrays a held-out bundle must contain. Named here so the error is specific."""

# Each calibrator picks its own serialisation format: TemperatureScaler uses
# torch.save, ConfidenceCalibrator and OODCalibrator use pickle, and
# CompletenessCalibrator uses np.save, which appends .npy to whatever path it is
# given. The extensions below match what each class actually writes, so the
# manifest hashes real files. Unifying those formats is tracked separately.
TEMPERATURE_FILE = "count_temperature.pt"
CONFIDENCE_FILE = "stream_confidence.pkl"
COMPLETENESS_FILE = "completeness.npy"
OOD_FILE = "ood.pkl"

ARTIFACT_FILES: tuple[str, ...] = (
    TEMPERATURE_FILE,
    CONFIDENCE_FILE,
    COMPLETENESS_FILE,
    OOD_FILE,
)


def load_held_out(path: str | Path) -> dict[str, np.ndarray]:
    """Load a held-out prediction bundle saved as .npz.

    Raises:
        FileNotFoundError: if the bundle does not exist.
        KeyError: naming every array that is missing, rather than the first.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"held-out bundle not found: {p}")
    with np.load(p) as data:
        missing = [k for k in REQUIRED_ARRAYS if k not in data]
        if missing:
            raise KeyError(f"held-out bundle {p} is missing arrays: {missing}")
        return {k: np.asarray(data[k]) for k in REQUIRED_ARRAYS}


def synthetic_held_out(n: int = 256, seed: int = 0) -> dict[str, np.ndarray]:
    """Random stand-in data, for exercising the fitting path only.

    The artifacts this produces are meaningless as calibration. Anything fitted
    from it is written with `synthetic: true` in the manifest so it can never be
    mistaken for a measurement.
    """
    rng = np.random.default_rng(seed)
    return {
        "count_logits": rng.normal(size=(n, 4)),
        "count_labels": rng.integers(0, 4, size=n),
        "conf_scores": rng.random(n),
        "conf_labels": (rng.random(n) > 0.4).astype(np.float64),
        "comp_probs": rng.random(n).clip(1e-3, 1 - 1e-3),
        "comp_labels": (rng.random(n) > 0.3).astype(np.float64),
        "ood_features": rng.normal(size=(n, 4)),
    }


def fit_all(
    held_out: dict[str, np.ndarray],
    out_dir: str | Path,
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Fit all four calibrators, save them, and write a hashed manifest.

    Args:
        held_out: Arrays named by `REQUIRED_ARRAYS`.
        out_dir: Directory to write artifacts and the manifest into.
        synthetic: Recorded in the manifest. Set when the inputs are not real
            held-out predictions.

    Returns:
        The manifest dict, including per-file SHA-256 digests.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    temperature = TemperatureScaler()
    nll = temperature.calibrate(
        torch.as_tensor(held_out["count_logits"], dtype=torch.float32),
        torch.as_tensor(held_out["count_labels"], dtype=torch.int64),
    )
    temperature.save(out / TEMPERATURE_FILE)

    confidence = ConfidenceCalibrator()
    confidence.fit(held_out["conf_scores"], held_out["conf_labels"])
    confidence.save(out / CONFIDENCE_FILE)
    ece = confidence.expected_calibration_error(held_out["conf_scores"], held_out["conf_labels"])

    completeness = CompletenessCalibrator()
    completeness.fit(held_out["comp_probs"], held_out["comp_labels"])
    # CompletenessCalibrator uses np.save, which appends .npy to the path.
    completeness.save(out / COMPLETENESS_FILE.replace(".npy", ""))

    ood = OODCalibrator()
    ood.fit(held_out["ood_features"])
    ood.calibrate_threshold(held_out["ood_features"])
    ood.save(out / OOD_FILE)

    artifacts = [out / name for name in ARTIFACT_FILES]
    return write_manifest(
        artifacts,
        out / "manifest.json",
        root=out,
        extra={
            "synthetic": synthetic,
            "n_held_out": int(len(held_out["count_labels"])),
            "count_temperature": float(temperature.temperature.item()),
            "count_nll": float(nll),
            "confidence_ece": float(ece),
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--held-out",
        default=None,
        help="Path to a .npz bundle of held-out predictions from Stage 4",
    )
    p.add_argument("--out-dir", default="calibration/artifacts")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Fit on random data to exercise the path. Artifacts are marked synthetic.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.held_out:
        held_out, synthetic = load_held_out(args.held_out), False
    elif args.synthetic:
        held_out, synthetic = synthetic_held_out(), True
    else:
        raise SystemExit(
            "Pass --held-out with a Stage 4 prediction bundle, or --synthetic to "
            "exercise the fitting path with random data."
        )

    manifest = fit_all(held_out, args.out_dir, synthetic=synthetic)
    log.info(
        "calibration_fitted",
        out_dir=args.out_dir,
        synthetic=synthetic,
        set_hash=manifest.get("set_hash"),
        temperature=manifest.get("count_temperature"),
    )


if __name__ == "__main__":
    main()
