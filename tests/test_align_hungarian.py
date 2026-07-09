"""Tests for align/hungarian.py: embedding and xcorr alignment on the schema."""

import numpy as np

from align.hungarian import (
    align_results,
    cosine_cost_matrix,
    reorder_result,
    xcorr_cost_matrix,
)
from schemas.separation_result import SeparationResult, StreamMetadata

RNG = np.random.default_rng(seed=7)


def make_result(streams: np.ndarray, embeddings: list | None = None) -> SeparationResult:
    k = streams.shape[0]
    metadata = None
    if embeddings is not None:
        metadata = [
            StreamMetadata(expert_source="test", embedding=np.asarray(e)) for e in embeddings
        ]
    return SeparationResult(
        streams=streams.astype(np.float32),
        sample_rate=16000,
        speaker_count=k,
        metadata=metadata or [],
        expert_used="test",
    )


def test_cosine_cost_identity_embeddings_near_zero() -> None:
    e = RNG.standard_normal((3, 16))
    cost = cosine_cost_matrix(e, e)
    assert np.allclose(np.diag(cost), 0.0, atol=1e-9)


def test_embedding_alignment_recovers_permutation() -> None:
    embs = RNG.standard_normal((3, 16))
    streams = RNG.standard_normal((3, 4000))
    a = make_result(streams, embeddings=list(embs))
    perm = [2, 0, 1]
    b = make_result(streams[perm], embeddings=list(embs[perm]))
    alignment = align_results(a, b)
    assert alignment.method == "embedding"
    reordered = reorder_result(b, alignment)
    assert np.allclose(reordered.streams, a.streams)


def test_xcorr_fallback_when_embeddings_missing() -> None:
    streams = RNG.standard_normal((3, 4000))
    a = make_result(streams)
    b = make_result(streams[[1, 2, 0]])
    alignment = align_results(a, b)
    assert alignment.method == "xcorr"
    reordered = reorder_result(b, alignment)
    assert np.allclose(reordered.streams, a.streams)


def test_rectangular_alignment_reports_unmatched() -> None:
    embs = RNG.standard_normal((3, 16))
    streams = RNG.standard_normal((3, 4000))
    a = make_result(streams[:2], embeddings=list(embs[:2]))
    b = make_result(streams, embeddings=list(embs))
    alignment = align_results(a, b)
    assert len(alignment.assignment) == 2
    assert len(alignment.unmatched_b) == 1
    reordered = reorder_result(b, alignment)
    assert reordered.num_streams == 3  # unmatched appended, audio preserved


def test_xcorr_cost_shape_and_range() -> None:
    cost = xcorr_cost_matrix(RNG.standard_normal((2, 3000)), RNG.standard_normal((4, 3000)))
    assert cost.shape == (2, 4)
    assert np.all(cost >= 0.0) and np.all(cost <= 1.0 + 1e-9)
