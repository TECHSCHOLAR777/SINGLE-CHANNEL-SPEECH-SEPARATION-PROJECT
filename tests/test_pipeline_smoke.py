"""CPU smoke tests for the full inference pipeline (BLUEPRINT §6)."""

from __future__ import annotations

import numpy as np

from coralsep.models.preprocess import CORALSEP_SAMPLE_RATE, coralsep_preprocess
from coralsep.pipeline.chunker import CHUNK_SAMPLES_8K, Chunker
from coralsep.pipeline.stitcher import ChunkStitcher


def test_coralsep_preprocess_dual_rate():
    mix = np.random.randn(16000).astype(np.float32) * 0.1
    result = coralsep_preprocess(mix, sample_rate=16000)
    assert result.waveform_8k.shape[0] > 0
    assert result.stft_16k is not None
    assert result.waveform_8k.dtype == np.float32


def test_chunker_produces_correct_count():
    sr = CORALSEP_SAMPLE_RATE
    duration_s = 10.0
    mix = np.random.randn(int(duration_s * sr)).astype(np.float32) * 0.1
    chunker = Chunker(mix)
    chunks = list(chunker)
    assert len(chunks) >= 1
    assert chunks[-1].is_last


def test_chunker_chunk_duration():
    mix = np.random.randn(CORALSEP_SAMPLE_RATE * 5).astype(np.float32) * 0.1
    for chunk in Chunker(mix):
        assert chunk.waveform_8k.shape[0] == CHUNK_SAMPLES_8K
        assert chunk.waveform_8k.dtype == np.float32
        break


def test_stitcher_single_chunk():
    K = 2
    streams = np.random.randn(K, CHUNK_SAMPLES_8K).astype(np.float32) * 0.05
    stitcher = ChunkStitcher(n_speakers=K)
    stitcher.feed_chunk(streams)
    result = stitcher.finalize()
    assert result.waveforms.shape[0] == K
    assert result.speaker_count == K


def test_stitcher_two_chunks_shape():
    K = 3
    stitcher = ChunkStitcher(n_speakers=K)
    for _ in range(2):
        streams = np.random.randn(K, CHUNK_SAMPLES_8K).astype(np.float32) * 0.05
        stitcher.feed_chunk(streams)
    result = stitcher.finalize()
    assert result.waveforms.shape[0] == K
    assert result.waveforms.ndim == 2
