"""Tests for pipeline/stitcher.py (Dev C)."""

import numpy as np
import pytest

from pipeline.stitcher import (
    ChunkStitcher,
    _hungarian_correlation,
    _hungarian_cosine,
    _pad_or_trim_k,
)
from pipeline.chunker import CHUNK_SAMPLES_8K, STEP_SAMPLES_8K, SR_8K


def _rand_chunk(K: int, T: int = CHUNK_SAMPLES_8K, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((K, T)).astype(np.float32)


class TestChunkStitcher:
    def test_single_chunk_passthrough(self):
        chunk = _rand_chunk(2)
        stitcher = ChunkStitcher(n_speakers=2)
        stitcher.feed_chunk(chunk)
        out = stitcher.finalize(total_samples=CHUNK_SAMPLES_8K)
        assert out.waveforms.shape == (2, CHUNK_SAMPLES_8K)
        assert out.speaker_count == 2
        assert out.n_chunks == 1

    def test_two_chunks_output_length(self):
        K, T_out = 3, STEP_SAMPLES_8K + CHUNK_SAMPLES_8K
        c1 = _rand_chunk(K, seed=1)
        c2 = _rand_chunk(K, seed=2)
        stitcher = ChunkStitcher(n_speakers=K)
        stitcher.feed_chunk(c1)
        stitcher.feed_chunk(c2)
        out = stitcher.finalize(total_samples=T_out)
        assert out.waveforms.shape == (K, T_out)

    def test_permutation_recorded(self):
        K = 2
        c1 = _rand_chunk(K, seed=10)
        c2 = _rand_chunk(K, seed=11)
        stitcher = ChunkStitcher(n_speakers=K)
        stitcher.feed_chunk(c1)
        stitcher.feed_chunk(c2)
        out = stitcher.finalize()
        # First chunk always has identity permutation.
        assert out.permutations[0] == [0, 1]
        # Second chunk has a valid permutation (each element in {0,1}).
        perm = out.permutations[1]
        assert sorted(perm) == [0, 1]

    def test_reset_clears_state(self):
        K = 2
        stitcher = ChunkStitcher(n_speakers=K)
        stitcher.feed_chunk(_rand_chunk(K, seed=0))
        stitcher.reset()
        assert stitcher._chunks == []
        out = stitcher.finalize()
        assert out.n_chunks == 0

    def test_n_speakers_inferred_from_first_chunk(self):
        stitcher = ChunkStitcher()  # n_speakers=None
        stitcher.feed_chunk(_rand_chunk(3, seed=0))
        assert stitcher.n_speakers == 3

    def test_extra_speakers_trimmed(self):
        stitcher = ChunkStitcher(n_speakers=2)
        stitcher.feed_chunk(_rand_chunk(4, seed=0))  # 4 > 2
        assert stitcher._chunks[0].shape[0] == 2

    def test_ecapa_permutation_used_when_available(self):
        K = 3
        D = 192
        rng = np.random.default_rng(42)
        stitcher = ChunkStitcher(n_speakers=K, use_ecapa=True)

        c1 = _rand_chunk(K, seed=1)
        emb1 = rng.standard_normal((K, D)).astype(np.float32)
        stitcher.feed_chunk(c1, embeddings=emb1)

        c2 = _rand_chunk(K, seed=2)
        emb2 = rng.standard_normal((K, D)).astype(np.float32)
        stitcher.feed_chunk(c2, embeddings=emb2)

        out = stitcher.finalize()
        assert len(out.permutations) == 2


class TestHungarian:
    def test_cosine_identity_perm(self):
        K, D = 3, 64
        emb = np.eye(K, D, dtype=np.float32)
        perm = _hungarian_cosine(emb, emb)
        assert perm == [0, 1, 2]

    def test_cosine_reverse_perm(self):
        K, D = 2, 4
        ref = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        new = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        perm = _hungarian_cosine(ref, new)
        assert perm == [1, 0]

    def test_correlation_identity(self):
        K = 2
        T = CHUNK_SAMPLES_8K
        chunk = _rand_chunk(K, T=T, seed=0)
        # Same chunk in both → identity.
        perm = _hungarian_correlation(chunk, chunk, overlap_samples=T // 4)
        assert sorted(perm) == [0, 1]


class TestPadOrTrimK:
    def test_no_change_needed(self):
        arr = np.zeros((3, 100), dtype=np.float32)
        result = _pad_or_trim_k(arr, 3)
        assert result.shape == (3, 100)

    def test_trim_extra_rows(self):
        arr = np.zeros((5, 100), dtype=np.float32)
        result = _pad_or_trim_k(arr, 3)
        assert result.shape == (3, 100)

    def test_pad_missing_rows(self):
        arr = np.ones((2, 100), dtype=np.float32)
        result = _pad_or_trim_k(arr, 4)
        assert result.shape == (4, 100)
        assert np.all(result[2:] == 0.0)
