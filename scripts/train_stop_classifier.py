"""
Train the speaker-count stop-classifier on peel-off examples from the cache (P3-C5).

Builds (features, continue/stop) examples from a frozen-expert cache (ideally the
mixed 2-5 speaker cache), trains the StopClassifier MLP with BCE, then fits
temperature scaling on a held-out split so the counting confidences are honest
(P3-C1 calibration). Saves a single torch checkpoint.

Example::

    python -m scripts.train_stop_classifier \
        --cache-dir /kaggle/working/cache_mixed/train \
        --val-cache-dir /kaggle/working/cache_mixed/dev \
        --epochs 60 --out /kaggle/working/outputs/counting/stop_classifier.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eval.counting_infer import build_peel_dataset
from models.stop_classifier import StopClassifier


def train_stop_classifier(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    x_train, y_train = build_peel_dataset(args.cache_dir)
    if args.val_cache_dir:
        x_val, y_val = build_peel_dataset(args.val_cache_dir)
    else:
        # Hold out a slice of the training cache for calibration.
        idx = rng.permutation(len(x_train))
        cut = max(1, int(0.15 * len(idx)))
        x_val, y_val = x_train[idx[:cut]], y_train[idx[:cut]]
        x_train, y_train = x_train[idx[cut:]], y_train[idx[cut:]]

    # Features are already O(1) (ratios / probs / distances), so the MLP's first
    # layer learns per-feature scale — no separate normalization to keep in sync
    # between training and inference.
    xt = torch.from_numpy(x_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    xv = torch.from_numpy(x_val).float().to(device)
    yv = torch.from_numpy(y_val).float().to(device)

    model = StopClassifier(in_features=x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Class-balance the BCE — stop (0) vs continue (1) are usually imbalanced.
    pos = float(yt.sum())
    neg = float(len(yt) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(xt)
        loss = bce(logits, yt)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = ((model(xt) > 0).float() == yt).float().mean().item()
            print(f"epoch {epoch}: loss={loss.item():.4f} train_acc={acc:.3f}")

    # Temperature scaling on the held-out split (post-hoc calibration).
    model.eval()
    with torch.no_grad():
        val_logits = model(xv).detach()
    temp = model.fit_temperature(val_logits, yv)

    with torch.no_grad():
        val_acc = ((val_logits > 0).float() == yv).float().mean().item()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)

    report = {
        "n_train_examples": int(len(y_train)),
        "n_val_examples": int(len(y_val)),
        "temperature": temp,
        "val_step_accuracy": val_acc,
        "checkpoint": str(out_path),
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Train the speaker-count stop-classifier")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--val-cache-dir", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/counting/stop_classifier.pt")
    train_stop_classifier(p.parse_args())


if __name__ == "__main__":
    main()
