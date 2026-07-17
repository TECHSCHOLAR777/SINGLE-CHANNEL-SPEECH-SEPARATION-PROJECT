"""CLI: fit calibration artifacts on held-out features (BLUEPRINT §8.5).

USER RUNS after joint polish with real held-out logits/labels:

    python -m train.calibrate --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from calibration.fit import fit_all
from utils.logging import get_logger

log = get_logger("calibrate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="calibration/artifacts")
    p.add_argument("--dry-run", action="store_true", help="Fit on synthetic held-out data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(0)
    # Synthetic stand-in; replace with held-out dumps from Stage 4.
    n = 256
    count_logits = rng.normal(size=(n, 4))
    count_labels = rng.integers(0, 4, size=n)
    conf_features = rng.normal(size=(n, 3))
    conf_labels = (rng.random(n) > 0.4).astype(np.float64)
    comp_features = rng.normal(size=(n, 3))
    comp_labels = (rng.random(n) > 0.3).astype(np.float64)

    manifest = fit_all(
        count_logits=count_logits,
        count_labels=count_labels,
        conf_features=conf_features,
        conf_labels=conf_labels,
        comp_features=comp_features,
        comp_labels=comp_labels,
        out_dir=args.out_dir,
    )
    log.info("calibration_fitted", out_dir=args.out_dir, manifest=manifest)


if __name__ == "__main__":
    main()
