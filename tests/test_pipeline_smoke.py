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


def test_separate_chunk_resamples_expert_output_back_to_8k():
    """Regression for I-061.

    SRCorrNetExpert.separate() always returns streams at PROJECT_SAMPLE_RATE
    (16 kHz) regardless of the input rate, its own documented contract, but
    ChunkStitcher is built and configured around CORALSEP_SAMPLE_RATE (8 kHz).
    Before the fix, CoralSepPipeline._separate_chunk fed the stitcher 16 kHz
    data directly, silently reconstructing only half of each chunk's real
    duration. A fake expert here reproduces the exact contract (echoing input
    length unchanged, but always labeling and sizing output as 16 kHz), which
    is the one behavior the shared conftest MockExpert did not replicate
    before this same ticket's fix, and which is why no earlier test caught
    this. Asserts the length CoralSepPipeline actually feeds the stitcher
    matches CHUNK_SAMPLES_8K exactly, not double it.
    """
    from coralsep.models.preprocess import PROJECT_SAMPLE_RATE
    from coralsep.pipeline.chunker import AudioChunk
    from coralsep.pipeline.infer import CoralSepPipeline
    from coralsep.schemas.separation_result import SeparationResult

    class FixedRateFakeExpert:
        """Always returns PROJECT_SAMPLE_RATE-length streams, like the real expert."""

        def separate(self, waveform, sample_rate, n_spks=None):
            k = n_spks or 2
            # The real expert's output length tracks the *upsampled* input
            # length, not the raw sample count it was called with.
            out_len = int(round(len(waveform) * PROJECT_SAMPLE_RATE / sample_rate))
            streams = np.random.randn(k, out_len).astype(np.float32) * 0.05
            return SeparationResult(
                streams=streams,
                sample_rate=PROJECT_SAMPLE_RATE,
                speaker_count=k,
                expert_used="fixed-rate-fake",
            )

    pipeline = CoralSepPipeline(expert=FixedRateFakeExpert())
    chunk = AudioChunk(
        waveform_8k=np.random.randn(CHUNK_SAMPLES_8K).astype(np.float32) * 0.1,
        stft_16k=None,
        chunk_index=0,
        start_sample_8k=0,
        is_last=True,
    )

    result = pipeline._separate_chunk(chunk, gate_vec={}, n_spks=2)

    assert result.sample_rate == CORALSEP_SAMPLE_RATE
    assert result.streams.shape[1] == CHUNK_SAMPLES_8K


def test_stitcher_reset_restores_dynamic_n_speakers_to_none():
    """Regression for I-062.

    feed_chunk() sets self.n_speakers from the first chunk's K when the
    stitcher was built without a fixed count (the dynamic-counting case,
    CoralSepPipeline's Pass 1). Before this fix, reset() cleared the buffered
    chunks/embeddings/permutations but left that learned n_speakers in place,
    so CoralSepPipeline's Pass 2 (a real speaker count correction after Pass 1,
    fed through the same stitcher after reset()) had every one of its
    genuinely-correctly-separated streams silently pad/trimmed to match Pass
    1's stale, possibly wrong, per-chunk count. Confirmed on real GPU hardware
    to explain a real catastrophic quality drop on 3+ speaker mixtures even
    when the final reported speaker count looked correct.
    """
    stitcher = ChunkStitcher(n_speakers=None)
    stitcher.feed_chunk(np.random.randn(3, CHUNK_SAMPLES_8K).astype(np.float32) * 0.05)
    assert stitcher.n_speakers == 3

    stitcher.reset()

    assert stitcher.n_speakers is None, "reset() must restore the dynamic (None) count"

    # A second feeding pass at a genuinely different, correct K must not be
    # force-reshaped to the first pass's stale count.
    stitcher.feed_chunk(np.random.randn(2, CHUNK_SAMPLES_8K).astype(np.float32) * 0.05)
    result = stitcher.finalize()
    assert result.waveforms.shape[0] == 2


def test_stitcher_reset_restores_an_explicit_fixed_n_speakers():
    """A stitcher built with an explicit, fixed n_speakers keeps that value
    across reset(), not None; only the dynamic (None) case should reset to
    None."""
    stitcher = ChunkStitcher(n_speakers=4)
    stitcher.feed_chunk(np.random.randn(4, CHUNK_SAMPLES_8K).astype(np.float32) * 0.05)

    stitcher.reset()

    assert stitcher.n_speakers == 4
