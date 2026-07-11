"""
Hungarian stream alignment (Dev C, Phase 1).

Matches separated streams across two SeparationResults (or raw arrays) so
"speaker 1" means the same person everywhere: expert-vs-expert fusion,
chunk-to-chunk stitching, and estimate-vs-reference display all consume this.

Primary cost is speaker-embedding cosine distance (StreamMetadata.embedding,
typically ECAPA-TDNN). When embeddings are absent the fallback is normalized
waveform cross-correlation, so alignment degrades gracefully instead of
crashing. Which signal was used is always returned, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from schemas.separation_result import SeparationResult

_EPS = 1e-8


@dataclass
class AlignmentResult:
    """Outcome of aligning streams of B onto the order of A.

    Attributes:
        assignment: (index_in_a, index_in_b) matched pairs.
        cost_matrix: The full cost matrix used, shape [K_a, K_b].
        method: "embedding" or "xcorr", whichever cost was actually used.
        unmatched_a: Indices in A with no partner (K_a > K_b case).
        unmatched_b: Indices in B with no partner (K_b > K_a case).
    """

    assignment: list[tuple[int, int]]
    cost_matrix: np.ndarray
    method: str
    unmatched_a: list[int]
    unmatched_b: list[int]


def _safe_row_normalize(rows: np.ndarray, *, center: bool = False) -> np.ndarray:
    """Normalize rows without magnifying silent, invalid, or extreme inputs.

    Each row is first scaled by its largest finite absolute value. This avoids
    overflow when squaring values near the float64 limit. Rows that are empty,
    non-finite, or effectively silent become all zeros; their dot products are
    therefore neutral rather than NaN/Inf.
    """
    arr = np.atleast_2d(np.asarray(rows, dtype=np.float64)).copy()
    if arr.shape[1] == 0:
        raise ValueError("rows must contain at least one feature/sample")

    finite_rows = np.isfinite(arr).all(axis=1)
    arr[~np.isfinite(arr)] = 0.0
    if center:
        arr -= arr.mean(axis=1, keepdims=True)

    peak = np.max(np.abs(arr), axis=1, keepdims=True)
    valid_peak = finite_rows[:, None] & np.isfinite(peak) & (peak > _EPS)
    scaled = np.divide(arr, peak, out=np.zeros_like(arr), where=valid_peak)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    valid_norm = valid_peak & np.isfinite(norm) & (norm > _EPS)
    return np.divide(scaled, norm, out=np.zeros_like(scaled), where=valid_norm)


def cosine_cost_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Cost = 1 - cosine similarity between embedding rows.

    Invalid or near-zero rows receive neutral cost 1.0 against every row.
    """
    a = np.asarray(emb_a, dtype=np.float64)
    b = np.asarray(emb_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"embedding shapes incompatible: {a.shape} vs {b.shape}")
    a_n = _safe_row_normalize(a)
    b_n = _safe_row_normalize(b)
    similarity = np.einsum("id,jd->ij", a_n, b_n, optimize=True)
    return np.clip(1.0 - similarity, 0.0, 2.0)


def xcorr_cost_matrix(streams_a: np.ndarray, streams_b: np.ndarray) -> np.ndarray:
    """Cost = 1 - absolute normalized zero-lag cross-correlation.

    Zero-lag suffices because both stream sets come from the same mixture
    timeline. Silent, invalid, and extreme rows produce finite neutral cost.
    """
    a = np.atleast_2d(np.asarray(streams_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(streams_b, dtype=np.float64))
    t = min(a.shape[1], b.shape[1])
    if t == 0:
        raise ValueError("waveforms must contain at least one sample")
    a_n = _safe_row_normalize(a[:, :t], center=True)
    b_n = _safe_row_normalize(b[:, :t], center=True)
    similarity = np.abs(np.einsum("it,jt->ij", a_n, b_n, optimize=True))
    return np.clip(1.0 - similarity, 0.0, 1.0)


def _embeddings_of(result: SeparationResult) -> np.ndarray | None:
    embs = [m.embedding for m in result.metadata]
    if any(e is None for e in embs):
        return None
    return np.stack([np.asarray(e, dtype=np.float64) for e in embs], axis=0)


def align_results(a: SeparationResult, b: SeparationResult) -> AlignmentResult:
    """Align streams of result B onto the speaker order of result A."""
    emb_a, emb_b = _embeddings_of(a), _embeddings_of(b)
    if emb_a is not None and emb_b is not None:
        cost = cosine_cost_matrix(emb_a, emb_b)
        method = "embedding"
    else:
        cost = xcorr_cost_matrix(a.streams, b.streams)
        method = "xcorr"

    if not np.isfinite(cost).all():
        raise RuntimeError("alignment cost matrix contains non-finite values")
    ia, ib = linear_sum_assignment(cost)
    assignment = list(zip(ia.tolist(), ib.tolist(), strict=True))
    matched_a = {int(i) for i in ia}
    matched_b = {int(j) for j in ib}
    return AlignmentResult(
        assignment=assignment,
        cost_matrix=cost,
        method=method,
        unmatched_a=[i for i in range(a.num_streams) if i not in matched_a],
        unmatched_b=[j for j in range(b.num_streams) if j not in matched_b],
    )


def reorder_result(b: SeparationResult, alignment: AlignmentResult) -> SeparationResult:
    """Return a copy of B reordered to A's speaker order."""
    order = [j for _, j in sorted(alignment.assignment, key=lambda pair: pair[0])]
    order += alignment.unmatched_b
    streams = b.streams[order]
    metadata = [replace(b.metadata[j]) for j in order]
    return SeparationResult(
        streams=streams,
        sample_rate=b.sample_rate,
        speaker_count=streams.shape[0],
        metadata=metadata,
        mixture=b.mixture,
        escalated=b.escalated,
        expert_used=b.expert_used,
    )
