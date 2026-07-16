"""CPU smoke tests for chunker, stitcher, CalmSepEngine mock path."""

from __future__ import annotations

import numpy as np

from models.preprocess import CALMSEP_SR, preprocess_calmsep
from pipeline.chunker import CHUNK_SEC, STEP_SEC, chunk_audio
from pipeline.infer import CalmSepEngine, MockCalmSepWrapper
from pipeline.stitcher import CalmSepStitcher, upsample_streams_8k_to_16k


def test_preprocess_calmsep_dual_rate():
    wav = np.random.randn(16000).astype(np.float32) * 0.05
    prep = preprocess_calmsep(wav, 16000)
    assert prep.wav_8k.ndim == 1
    assert prep.stft_8k.ndim == 2
    assert prep.stft_16k.ndim == 2
    assert prep.sample_rate_internal == 8000


def test_chunker_sizes():
    wav = np.random.randn(int(CALMSEP_SR * 5.0)).astype(np.float32) * 0.05
    chunks = chunk_audio(wav, CALMSEP_SR)
    assert len(chunks) >= 2
    assert chunks[0].wav_8k.shape[0] == int(CHUNK_SEC * CALMSEP_SR)
    # Step between starts
    assert chunks[1].start_8k == int(STEP_SEC * CALMSEP_SR)


def test_engine_mock_end_to_end():
    engine = CalmSepEngine(wrapper=MockCalmSepWrapper(n_speakers=3), base_only=True)
    wav = np.random.randn(int(CALMSEP_SR * 2.5)).astype(np.float32) * 0.05
    result = engine(wav, CALMSEP_SR)
    assert result.speaker_count >= 2
    assert result.sample_rate == 16000
    assert result.p_k is not None
    assert result.gate_vector is not None
    assert result.completeness is not None
    assert result.condition_estimates is not None


def test_stitcher_two_chunks():
    st = CalmSepStitcher()
    a = np.random.randn(2, 8000).astype(np.float32) * 0.05
    b = np.concatenate([a[:, -int(1.6 * 16000) :], np.random.randn(2, 4000).astype(np.float32) * 0.01], axis=1)
    # Simpler: just add two similar chunks
    st.add_chunk(a)
    st.add_chunk(a * 0.9)
    out = st.finalize()
    assert out.n_global >= 1
    assert out.streams_16k.ndim == 2


def test_upsample_helper():
    s = np.random.randn(2, 800).astype(np.float32)
    u = upsample_streams_8k_to_16k(s)
    assert u.shape[0] == 2
    assert u.shape[1] == 1600
