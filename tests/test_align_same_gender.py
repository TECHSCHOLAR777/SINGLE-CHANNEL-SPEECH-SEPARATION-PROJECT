"""P1-C4 stress tests: alignment under same-gender (similar-embedding) conditions."""

import numpy as np

from coralsep.align.chunking import ChunkStitcher
from coralsep.align.hungarian import align_results, cosine_cost_matrix, reorder_result
from tests.test_align_hungarian import make_result

RNG = np.random.default_rng(seed=42)


def similar_embeddings(n: int, dim: int = 64, spread: float = 0.25) -> np.ndarray:
    """Embeddings clustered around one direction: the same-gender regime.

    spread=0.25 gives pairwise cosine similarity around 0.94-0.97, matching
    ECAPA behavior for same-gender similar-timbre speakers.
    """
    base = RNG.standard_normal(dim)
    base /= np.linalg.norm(base)
    out = []
    for _ in range(n):
        perturb = RNG.standard_normal(dim)
        perturb -= perturb.dot(base) * base  # orthogonal component
        perturb /= np.linalg.norm(perturb)
        vec = base + spread * perturb
        out.append(vec / np.linalg.norm(vec))
    return np.stack(out)


def test_same_gender_costs_are_small_but_ordered() -> None:
    embs = similar_embeddings(3)
    cost = cosine_cost_matrix(embs, embs)
    off_diag = cost[~np.eye(3, dtype=bool)]
    assert np.all(np.diag(cost) < 1e-9)  # self-match exactly zero
    assert np.all(off_diag > 1e-4)  # rivals close but never zero
    assert np.all(off_diag < 0.15)  # genuinely the hard regime


def test_hungarian_recovers_permutation_with_similar_voices() -> None:
    embs = similar_embeddings(3)
    streams = RNG.standard_normal((3, 4000))
    a = make_result(streams, embeddings=list(embs))
    perm = [1, 2, 0]
    b = make_result(streams[perm], embeddings=list(embs[perm]))
    alignment = align_results(a, b)
    reordered = reorder_result(b, alignment)
    # Global Hungarian optimization resolves it even when margins are tiny.
    assert np.allclose(reordered.streams, a.streams)


def test_identical_embeddings_degenerate_but_complete() -> None:
    """Truly indistinguishable voices: assignment is arbitrary yet total.

    Documents the known limit: with identical embeddings the aligner cannot
    know the true pairing, but it must still return a complete assignment
    without crashing or dropping streams.
    """
    emb = similar_embeddings(1)[0]
    embs = np.stack([emb, emb, emb])
    streams = RNG.standard_normal((3, 4000))
    a = make_result(streams, embeddings=list(embs))
    b = make_result(streams[[2, 0, 1]], embeddings=list(embs))
    alignment = align_results(a, b)
    assert len(alignment.assignment) == 3
    assert alignment.unmatched_a == [] and alignment.unmatched_b == []


def test_chunk_stitcher_keeps_similar_speakers_separate() -> None:
    """Two same-gender speakers must not merge into one track across chunks."""
    sr = 16000
    st = ChunkStitcher(sample_rate=sr, chunk_sec=1.0, overlap_sec=0.25, match_threshold=0.35)
    e1, e2 = similar_embeddings(2, spread=0.3)
    chunk = RNG.standard_normal((2, sr)).astype(np.float32)

    first = st.add_chunk(chunk, np.stack([e1, e2]))
    second = st.add_chunk(chunk[[1, 0]], np.stack([e2, e1]))  # swapped arrival
    assert st.num_tracks == 2  # no merge
    assert second == [first[1], first[0]]  # identities tracked through swap


def test_chunk_stitcher_similar_new_voice_spawns_when_over_threshold() -> None:
    """A third similar voice beyond the cost threshold opens a new track."""
    sr = 16000
    st = ChunkStitcher(sample_rate=sr, chunk_sec=1.0, overlap_sec=0.25, match_threshold=0.02)
    e1, e2 = similar_embeddings(2, spread=0.3)
    st.add_chunk(RNG.standard_normal((1, sr)).astype(np.float32), e1[None])
    st.add_chunk(RNG.standard_normal((1, sr)).astype(np.float32), e2[None])
    # With a tight threshold, the similar-but-different voice is a new track,
    # never silently absorbed into speaker 1.
    assert st.num_tracks == 2
