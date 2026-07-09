"""
Train the stop-classifier (Dev C, Phase 3).

Consumes a feature JSONL produced by running the separation pipeline over
Libri2Mix through Libri5Mix, where each line holds the feature vector for one
peel decision plus the ground-truth label (1 = more speakers remained,
0 = that was the last one). Trains the MLP, calibrates with temperature
scaling on a held-out split, saves the checkpoint, and prints the counting
report from eval.reporting.

Feature line schema:
    {"features": [f1, f2, f3, f4, f5], "label": 1, "n_true": 4, "mixture_id": "..."}

Self-test mode (--self-test) needs no data or GPU: it synthesizes two
separable Gaussian feature clusters, trains, asserts accuracy above 0.9, and
round-trips save/load. CI runs this to prove the training mechanics work
before real features exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.reporting import calibration_curve, iter_jsonl
from models.stop_classifier import FEATURE_NAMES, StopClassifier


def load_feature_jsonl(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load features [N, F] and labels [N] from the pipeline's feature log."""
    feats: list[list[float]] = []
    labels: list[float] = []
    for row in iter_jsonl(path):
        feats.append([float(v) for v in row["features"]])
        labels.append(float(row["label"]))
    if not feats:
        raise ValueError(f"no rows in {path}")
    x = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    if x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"feature width {x.shape[1]} does not match FEATURE_NAMES ({len(FEATURE_NAMES)})"
        )
    return x, y


def synth_features(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two separable Gaussian clusters in feature space, for the self-test."""
    rng = np.random.default_rng(seed)
    dim = len(FEATURE_NAMES)
    continue_mean = np.array([0.45, 0.8, 0.7, 0.5, 1.0])[:dim]
    stop_mean = np.array([0.05, 0.1, 0.15, 0.08, -1.0])[:dim]
    half = n // 2
    x1 = rng.normal(continue_mean, 0.12, size=(half, dim))
    x0 = rng.normal(stop_mean, 0.12, size=(n - half, dim))
    x = np.vstack([x1, x0]).astype(np.float32)
    y = np.concatenate([np.ones(half), np.zeros(n - half)]).astype(np.float32)
    order = rng.permutation(n)
    return torch.from_numpy(x[order]), torch.from_numpy(y[order])


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    val_fraction: float,
    seed: int,
    device: str,
) -> tuple[StopClassifier, dict]:
    """
    Train, calibrate, and evaluate the stop-classifier.

    Returns:
        The trained model and a metrics dict (val_accuracy, temperature, ece).
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    n_val = max(1, int(len(x) * val_fraction))
    x_train, y_train = x[:-n_val].to(dev), y[:-n_val].to(dev)
    x_val, y_val = x[-n_val:].to(dev), y[-n_val:].to(dev)

    model = StopClassifier().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x_train), device=dev)
        for start in range(0, len(x_train), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        val_pred = (torch.sigmoid(val_logits) > 0.5).float()
        val_accuracy = float((val_pred == y_val).float().mean().item())

    temperature = model.fit_temperature(val_logits, y_val)
    with torch.no_grad():
        proba = model.predict_proba(x_val).cpu().numpy()
    correct = (proba > 0.5) == y_val.cpu().numpy().astype(bool)
    confidence = np.where(proba > 0.5, proba, 1.0 - proba)
    calib = calibration_curve(confidence.tolist(), correct.tolist(), n_bins=10)

    return model, {
        "val_accuracy": val_accuracy,
        "temperature": temperature,
        "ece": calib["ece"],
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=str, help="Feature JSONL from the pipeline")
    parser.add_argument("--out", type=str, default="checkpoints/stop_classifier.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Train on synthetic separable features and assert accuracy > 0.9",
    )
    args = parser.parse_args()

    if args.self_test:
        x, y = synth_features(n=4000, seed=args.seed)
    elif args.features:
        x, y = load_feature_jsonl(args.features)
    else:
        parser.error("provide --features or --self-test")

    model, metrics = train(
        x,
        y,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(json.dumps({"checkpoint": str(out), **metrics}, indent=2))

    if args.self_test:
        assert metrics["val_accuracy"] > 0.9, f"self-test failed: {metrics['val_accuracy']:.3f}"
        reloaded = StopClassifier.load(out)
        with torch.no_grad():
            p1 = model.predict_proba(x[:8].to(args.device)).cpu()
            p2 = reloaded.predict_proba(x[:8])
        assert torch.allclose(p1, p2, atol=1e-5), "save/load round-trip mismatch"
        print("self-test passed: accuracy > 0.9 and checkpoint round-trip verified")


if __name__ == "__main__":
    main()
