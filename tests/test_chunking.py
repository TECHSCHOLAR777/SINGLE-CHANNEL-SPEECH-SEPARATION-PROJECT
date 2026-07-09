"""Tests for align/chunking.py: identity lock across chunks and stitching."""

import numpy as np
import pytest

from align.chunking import ChunkStitcher

RNG = np.random.default_rng(seed=11)
SR = 16000


def make_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(16)
    return e / np.linalg.norm(e)


def test_consistent_ids_across_permuted_chunks() -> None:
    st = ChunkStitcher(sample_rate=SR, chunk_sec=1.0, overlap_sec=0.25)
    e1, e2, e3 = make_embedding(1), make_embedding(2), make_embedding(3)
    chunk = RNG.standard_normal((3, SR)).astype(np.float32)

    ids_first = st.add_chunk(chunk, np.stack([e1, e2, e3]))
    ids_second = st.add_chunk(chunk[[2, 0, 1]], np.stack([e3, e1, e2]))
    # Permuted arrival order, same people: same track ids after reorder.
    assert ids_second == [ids_first[2], ids_first[0], ids_first[1]]
    assert st.num_tracks == 3


def test_new_speaker_spawns_new_track() -> None:
    st = ChunkStitcher(sample_rate=SR, chunk_sec=1.0, overlap_sec=0.25, match_threshold=0.3)
    chunk2 = RNG.standard_normal((2, SR)).astype(np.float32)
    st.add_chunk(chunk2, np.stack([make_embedding(1), make_embedding(2)]))
    chunk3 = RNG.standard_normal((3, SR)).astype(np.float32)
    ids = st.add_chunk(chunk3, np.stack([make_embedding(1), make_embedding(2), make_embedding(99)]))
    assert st.num_tracks == 3
    assert ids[2] == 2  # the new voice opened track 2, old tracks intact


def test_silent_chunk_track_persists() -> None:
    st = ChunkStitcher(sample_rate=SR, chunk_sec=1.0, overlap_sec=0.25)
    e1, e2 = make_embedding(1), make_embedding(2)
    both = RNG.standard_normal((2, SR)).astype(np.float32)
    st.add_chunk(both, np.stack([e1, e2]))
    only_first = RNG.standard_normal((1, SR)).astype(np.float32)
    st.add_chunk(only_first, np.stack([e1]))  # speaker 2 silent this chunk
    ids = st.add_chunk(both, np.stack([e1, e2]))  # speaker 2 returns
    assert ids == [0, 1]  # matched against full history, not just previous chunk
    assert st.num_tracks == 2


def test_stitch_constant_signal_reconstructed_across_overlap() -> None:
    st = ChunkStitcher(sample_rate=100, chunk_sec=1.0, overlap_sec=0.5)
    e = make_embedding(5)
    ones = np.ones((1, 100), dtype=np.float32)
    st.add_chunk(ones, e[None])
    st.add_chunk(ones, e[None])
    out = st.stitch()
    assert out.shape == (1, 150)
    # Crossfade of identical signals must reconstruct the constant.
    assert np.allclose(out[0, 10:140], 1.0, atol=1e-3)


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        ChunkStitcher(sample_rate=SR, chunk_sec=1.0, overlap_sec=1.5)
