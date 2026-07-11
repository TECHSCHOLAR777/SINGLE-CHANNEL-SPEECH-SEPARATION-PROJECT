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


def cosine_cost_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Cost = 1 - cosine similarity between embedding rows.

    Args:
        emb_a: [K_a, D] embeddings.
        emb_b: [K_b, D] embeddings.

    Returns:
        [K_a, K_b] cost matrix in [0, 2].
    """
    a = np.asarray(emb_a, dtype=np.float64)
    b = np.asarray(emb_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"embedding shapes incompatible: {a.shape} vs {b.shape}")
    a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)
    b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)
    return 1.0 - a_n @ b_n.T


def xcorr_cost_matrix(streams_a: np.ndarray, streams_b: np.ndarray) -> np.ndarray:
    """
    Cost = 1 - |normalized zero-lag cross-correlation| between waveforms.

    Fallback signal when embeddings are unavailable. Zero-lag suffices because
    both stream sets come from the same mixture timeline.
    """
    a = np.atleast_2d(np.asarray(streams_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(streams_b, dtype=np.float64))
    t = min(a.shape[1], b.shape[1])
    a = a[:, :t] - a[:, :t].mean(axis=1, keepdims=True)
    b = b[:, :t] - b[:, :t].mean(axis=1, keepdims=True)
    a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)
    b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)
    return 1.0 - np.abs(a_n @ b_n.T)


def _embeddings_of(result: SeparationResult) -> np.ndarray | None:
    embs = [m.embedding for m in result.metadata]
    if any(e is None for e in embs):
        return None
    return np.stack([np.asarray(e, dtype=np.float64) for e in embs], axis=0)


def align_results(a: SeparationResult, b: SeparationResult) -> AlignmentResult:
    """
    Align streams of result B onto the speaker order of result A.

    Uses embedding cosine cost when both results carry embeddings for every
    stream, otherwise waveform cross-correlation. Rectangular cases are
    handled by the Hungarian solver; unmatched indices are reported.

    Args:
        a: Reference-order result (e.g. the cheap expert's output).
        b: Result to reorder (e.g. the expensive expert's output).

    Returns:
        AlignmentResult; apply with reorder_result(b, alignment).
    """
    emb_a, emb_b = _embeddings_of(a), _embeddings_of(b)
    if emb_a is not None and emb_b is not None:
        cost = cosine_cost_matrix(emb_a, emb_b)
        method = "embedding"
    else:
        cost = xcorr_cost_matrix(a.streams, b.streams)
        method = "xcorr"

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
    """
    Return a copy of B with streams and metadata reordered to A's speaker order.

    Matched streams take the slot of their partner in A; B streams unmatched
    in A (over-count) are appended after, preserving all audio. The returned
    object is a new SeparationResult; B is not mutated.
    """
    order = [j for _, j in sorted(alignment.assignment, key=lambda p: p[0])]
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
