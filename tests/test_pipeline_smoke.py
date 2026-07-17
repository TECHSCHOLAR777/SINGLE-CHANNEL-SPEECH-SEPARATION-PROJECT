"""CPU smoke tests for the full inference pipeline (BLUEPRINT §6)."""

from __future__ import annotations

import numpy as np

from models.preprocess import CALMSEP_SAMPLE_RATE, calmsep_preprocess
from pipeline.chunker import CHUNK_DURATION_S, STEP_DURATION_S, Chunker, CHUNK_SAMPLES_8K
from pipeline.stitcher import ChunkStitcher


def test_calmsep_preprocess_dual_rate():
    mix = np.random.randn(16000).astype(np.float32) * 0.1
    result = calmsep_preprocess(mix, sample_rate=16000)
    assert result.waveform_8k.shape[0] > 0
    assert result.stft_16k is not None
    assert result.waveform_8k.dtype == np.float32


def test_chunker_produces_correct_count():
    sr = CALMSEP_SAMPLE_RATE
    duration_s = 10.0
    mix = np.random.randn(int(duration_s * sr)).astype(np.float32) * 0.1
    chunker = Chunker(mix)
    chunks = list(chunker)
    assert len(chunks) >= 1
    assert chunks[-1].is_last


def test_chunker_chunk_duration():
    mix = np.random.randn(CALMSEP_SAMPLE_RATE * 5).astype(np.float32) * 0.1
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
