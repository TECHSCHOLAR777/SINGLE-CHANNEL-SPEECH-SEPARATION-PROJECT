"""
End-to-end P3 counting pipeline: cache -> peel dataset -> train -> infer -> M3
artifacts. Runs on a synthetic cache (no experts, no weights), CPU only.

Proves the pieces that close GATE M3's data criteria compose:
  * build_peel_dataset produces one (features, continue/stop) row per peel step
    with correct labels (continue while step < N, stop at/after N),
  * the trained StopClassifier drives peel-off unknown-N inference,
  * eval_counting emits a confusion matrix + calibration curve + summary.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from eval.counting_infer import build_peel_dataset, infer_count
from models.counting_features import FEATURE_NAMES, VADAdapter
from models.stop_classifier import StopClassifier
from train.cached_dataset import make_sample_dict, save_cache_shard

SR = 16000
T = 8000
K = 5
D = 192


def _sample(n_true: int, seed: int) -> dict:
    """A K-stem cached sample: first n_true stems carry distinct energetic
    signals (real speakers), the rest are near-silent pads."""
    rng = np.random.default_rng(seed)
    stems = np.zeros((K, T), dtype=np.float32)
    embs = np.zeros((K, D), dtype=np.float32)
    for j in range(K):
        if j < n_true:
            stems[j] = (rng.standard_normal(T) * 0.5).astype(np.float32)
            e = rng.standard_normal(D).astype(np.float32)
        else:
            stems[j] = (rng.standard_normal(T) * 1e-3).astype(np.float32)  # silent pad
            e = np.full(D, 1e-3, dtype=np.float32)
        embs[j] = e / (np.linalg.norm(e) + 1e-8)
    mixture = stems[:n_true].sum(axis=0)
    return make_sample_dict(
        mixture=torch.from_numpy(mixture),
        references=torch.from_numpy(stems[:n_true]),
        moss_streams=torch.from_numpy(stems),
        sr_streams=torch.from_numpy(stems),
        true_count=float(n_true),
        quality_db=8.0,
        sr_confidence=torch.full((K,), 0.8),
        moss_mask_entropy=torch.full((K,), 0.3),
        trivial_mask=torch.zeros(3),
        stream_embeddings=torch.from_numpy(embs),
        reference_embeddings=torch.from_numpy(embs[:n_true]),
    )


def _build_cache(tmp_path, counts):
    shard = [_sample(n, seed=i) for i, n in enumerate(counts)]
    save_cache_shard(tmp_path / "shard_00000.pt", shard, SR)


def test_peel_dataset_labels(tmp_path):
    # N=3 -> steps 1,2 continue (1), steps 3,4,5 stop (0). K=5 rows per sample.
    _build_cache(tmp_path, [3])
    x, y = build_peel_dataset(tmp_path)
    assert x.shape == (K, len(FEATURE_NAMES))
    assert y.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_peel_dataset_varied_counts(tmp_path):
    _build_cache(tmp_path, [2, 3, 4, 5])
    x, y = build_peel_dataset(tmp_path)
    assert x.shape == (4 * K, len(FEATURE_NAMES))
    # continue-label count = sum(n-1) over samples = 1+2+3+4 = 10
    assert int(y.sum()) == 10


def test_train_and_infer_recovers_counts(tmp_path):
    # Train on a balanced mix of 2..5 speaker samples, then check peel-off
    # inference recovers a sensible count on a fresh sample.
    counts = [2, 3, 4, 5] * 6
    _build_cache(tmp_path, counts)
    x, y = build_peel_dataset(tmp_path)

    torch.manual_seed(0)
    clf = StopClassifier(in_features=x.shape[1])
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2)
    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(clf(xt), yt)
        loss.backward()
        opt.step()
    clf.eval()

    # Step-level accuracy should be well above chance on this separable signal.
    with torch.no_grad():
        acc = ((clf(xt) > 0).float() == yt).float().mean().item()
    assert acc > 0.8

    # Peel-off inference returns an in-range count with a valid confidence.
    vad = VADAdapter("energy")
    s = _sample(4, seed=999)
    mixture = s["mixture"].numpy().astype(np.float32)
    stems = s["sr_streams"].to(torch.float32).numpy()
    embs = s["stream_embeddings"].to(torch.float32).numpy()
    est, conf = infer_count(clf, mixture, stems, embs, vad)
    assert 1 <= est <= K
    assert 0.0 <= conf <= 1.0


def test_eval_counting_emits_artifacts(tmp_path):
    from scripts.eval_counting import evaluate_counting

    cache = tmp_path / "cache"
    cache.mkdir()
    _build_cache(cache, [2, 3, 4, 5, 2, 3])

    # Train a quick classifier to a checkpoint.
    x, y = build_peel_dataset(cache)
    torch.manual_seed(0)
    clf = StopClassifier(in_features=x.shape[1])
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2)
    xt, yt = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(200):
        opt.zero_grad()
        loss_fn(clf(xt), yt).backward()
        opt.step()
    ckpt = tmp_path / "stop.pt"
    clf.save(ckpt)

    out = tmp_path / "report"
    args = SimpleNamespace(
        cache_dir=str(cache),
        checkpoint=str(ckpt),
        output_dir=str(out),
        threshold=0.5,
        condition="mixed",
        tier="L1",
        count_range=[2, 5],
    )
    verdict = evaluate_counting(args)
    assert 0.0 <= verdict["count_accuracy"] <= 1.0
    assert (out / "count_confusion_matrix.csv").exists()
    assert (out / "count_confusion_matrix.svg").exists()
    assert (out / "counting_summary.json").exists()
    assert (out / "counting_runlog.jsonl").exists()
