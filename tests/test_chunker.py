"""Tests for pipeline/chunker.py (Dev C)."""

import numpy as np
import pytest

from pipeline.chunker import (
    CHUNK_SAMPLES_8K,
    STEP_SAMPLES_8K,
    SR_8K,
    AudioChunk,
    Chunker,
    _compute_stft_16k,
)


def _make_sine(duration_s: float, sr: int = SR_8K) -> np.ndarray:
    t = np.arange(int(duration_s * sr), dtype=np.float32) / sr
    return np.sin(2 * np.pi * 440.0 * t)


class TestChunker:
    def test_single_chunk_short_audio(self):
        wav = _make_sine(1.0)  # shorter than one chunk
        chunker = Chunker(wav)
        chunks = list(chunker)
        assert len(chunks) == 1
        assert chunks[0].waveform_8k.shape == (CHUNK_SAMPLES_8K,)
        assert chunks[0].chunk_index == 0
        assert chunks[0].is_last

    def test_multi_chunk_count(self):
        wav = _make_sine(6.0)  # 48000 samples → 4 chunks
        chunker = Chunker(wav)
        chunks = list(chunker)
        # (48000 - 19200) / 6400 + 1 = 4.65 → 5 chunks
        assert len(chunks) >= 4
        assert chunks[-1].is_last

    def test_all_chunks_same_length_with_padding(self):
        wav = _make_sine(5.0)
        chunker = Chunker(wav, pad_last=True)
        for chunk in chunker:
            assert chunk.waveform_8k.shape == (CHUNK_SAMPLES_8K,)

    def test_no_16k_stft_when_no_16k_audio(self):
        wav = _make_sine(3.0)
        chunker = Chunker(wav)
        for chunk in chunker:
            assert chunk.stft_16k is None

    def test_16k_stft_shape_when_provided(self):
        wav8 = _make_sine(3.0, sr=SR_8K)
        wav16 = _make_sine(3.0, sr=16000)
        chunker = Chunker(wav8, wav16)
        for chunk in chunker:
            assert chunk.stft_16k is not None
            # 16 kHz STFT with n_fft=512: 257 bins
            assert chunk.stft_16k.shape[0] == 257
            assert chunk.stft_16k.dtype == np.complex64

    def test_start_samples_monotone(self):
        wav = _make_sine(4.0)
        chunker = Chunker(wav)
        starts = [c.start_sample_8k for c in chunker]
        assert starts == sorted(starts)
        assert starts[0] == 0

    def test_chunk_at_index(self):
        wav = _make_sine(4.0)
        chunker = Chunker(wav)
        all_chunks = list(chunker)
        for i, chunk in enumerate(all_chunks):
            assert chunker.chunk_at(i).chunk_index == i

    def test_n_chunks_property(self):
        wav = _make_sine(4.0)
        chunker = Chunker(wav)
        assert chunker.n_chunks == len(list(chunker))

    def test_1d_input_required(self):
        with pytest.raises(ValueError, match="1-D"):
            Chunker(np.zeros((2, 8000)))

    def test_very_short_audio_one_chunk(self):
        wav = np.zeros(100, dtype=np.float32)
        chunker = Chunker(wav)
        chunks = list(chunker)
        assert len(chunks) == 1
        assert chunks[0].waveform_8k.shape[0] == CHUNK_SAMPLES_8K


class TestStft16k:
    def test_output_shape(self):
        wav = _make_sine(2.4, sr=16000)
        spec = _compute_stft_16k(wav)
        assert spec.shape[0] == 257
        assert spec.dtype == np.complex64

    def test_zero_input(self):
        wav = np.zeros(38400, dtype=np.float32)
        spec = _compute_stft_16k(wav)
        assert np.allclose(np.abs(spec), 0.0)
