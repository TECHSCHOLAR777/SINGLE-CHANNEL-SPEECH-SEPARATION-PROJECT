"""
Stop-classifier training data + unknown-N inference, driven off the cache (P3).

The frozen-expert cache (`scripts/build_train_cache.py`) already holds, per
mixture: the mixture waveform, the aligned expensive-expert stems, the true
speaker count, and ECAPA stream embeddings. That is exactly what the
stop-classifier's peel-off counting needs — so both training-example extraction
and unknown-N inference run straight off the cache with no experts reloaded.

Peel semantics (`models/stop_classifier.py`): a decision at step k (1-indexed)
asks "after accepting the k-th stem, do MORE speakers remain?". The label is
``continue`` (1) while k < N and ``stop`` (0) once k >= N. At inference we peel
until the classifier says stop; the count is the number of stems accepted then.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.counting_features import FEATURE_NAMES, VADAdapter, compute_stop_features
from models.stop_classifier import StopClassifier
from train.cached_dataset import CachedExpertDataset


def _min_cos_distance(candidate_emb: np.ndarray, accepted_embs: np.ndarray) -> float:
    """Min ECAPA cosine distance from candidate to accepted embeddings; 1.0 if
    none accepted yet (a novel speaker has no duplicate to collide with)."""
    if accepted_embs.shape[0] == 0:
        return 1.0
    cand = candidate_emb / (np.linalg.norm(candidate_emb) + 1e-8)
    acc = accepted_embs / (np.linalg.norm(accepted_embs, axis=1, keepdims=True) + 1e-8)
    sims = acc @ cand
    return float(1.0 - float(np.max(sims)))


def _sample_arrays(batch) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mixture = batch.mixture.detach().cpu().numpy().astype(np.float32)
    stems = batch.sr_streams.detach().cpu().numpy().astype(np.float32)
    k = stems.shape[0]
    if batch.stream_embeddings is not None:
        embs = batch.stream_embeddings.detach().cpu().numpy().astype(np.float32)
    else:
        embs = np.zeros((k, 192), dtype=np.float32)
    n_true = int(round(float(batch.true_count)))
    return mixture, stems, embs, n_true


def _peel_feature(mixture, stems, embs, k, vad) -> np.ndarray:
    """Feature vector for the decision at step k (1-indexed): candidate = k-th
    stem, accepted = the k-1 stems before it."""
    accepted = stems[: k - 1]
    candidate = stems[k - 1]
    emb_dist = _min_cos_distance(embs[k - 1], embs[: k - 1])
    return compute_stop_features(
        mixture,
        accepted,
        candidate,
        min_embedding_distance=emb_dist,
        attractor_stop_logit=0.0,  # the published SR-CorrNet API exposes no TDA logit
        vad=vad,
    )


def build_peel_dataset(cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble (X [M, F], y [M]) stop-classifier training examples from a cache.

    One example per (sample, peel-step). y = 1 (continue) while step < true N,
    else 0 (stop). M = sum over samples of K peel steps.
    """
    ds = CachedExpertDataset(cache_dir)
    vad = VADAdapter("energy")
    feats: list[np.ndarray] = []
    labels: list[float] = []
    for i in range(len(ds)):
        mixture, stems, embs, n_true = _sample_arrays(ds[i])
        k_max = stems.shape[0]
        for k in range(1, k_max + 1):
            feats.append(_peel_feature(mixture, stems, embs, k, vad))
            labels.append(1.0 if k < n_true else 0.0)
    if not feats:
        raise RuntimeError(f"no peel examples built from {cache_dir}")
    return np.stack(feats, axis=0).astype(np.float32), np.asarray(labels, dtype=np.float32)


def infer_count(
    clf: StopClassifier,
    mixture: np.ndarray,
    stems: np.ndarray,
    embs: np.ndarray,
    vad: VADAdapter,
    threshold: float = 0.5,
    min_count: int = 2,
) -> tuple[int, float]:
    """
    Peel-off unknown-N inference for one mixture.

    Returns (estimated_count, confidence). Peels until P(continue) < threshold
    (or the stems run out); the count is the number of stems accepted at that
    point, and confidence is the calibrated P(stop) of the deciding step.
    """
    k_max = stems.shape[0]
    last_p_more = 1.0
    for k in range(1, k_max + 1):
        feat = _peel_feature(mixture, stems, embs, k, vad)
        p_more = float(clf.predict_proba(torch.from_numpy(feat).float().unsqueeze(0)).item())
        last_p_more = p_more
        if p_more < threshold and k >= min_count:
            return k, 1.0 - p_more
    return k_max, 1.0 - last_p_more


def infer_counts_over_cache(
    clf: StopClassifier,
    cache_dir: str | Path,
    threshold: float = 0.5,
) -> list[dict]:
    """Run peel-off counting over every cached sample; return per-sample dicts
    with n_true, n_estimated, confidence (feeds RunRecord / counting_report)."""
    ds = CachedExpertDataset(cache_dir)
    vad = VADAdapter("energy")
    out: list[dict] = []
    for i in range(len(ds)):
        mixture, stems, embs, n_true = _sample_arrays(ds[i])
        est, conf = infer_count(clf, mixture, stems, embs, vad, threshold=threshold)
        out.append({"n_true": n_true, "n_estimated": est, "confidence": conf})
    return out


N_FEATURES = len(FEATURE_NAMES)
