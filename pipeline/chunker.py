"""Chunking for CALM-Sep: 2.4 s window, 0.8 s step at 8 kHz (BLUEPRINT §6.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.preprocess import (
    BAND_HOP,
    BAND_N_FFT,
    CALMSEP_HOP,
    CALMSEP_N_FFT,
    CALMSEP_SR,
    OUTPUT_SR,
    compute_stft,
    preprocess_calmsep,
    resample_audio,
)

CHUNK_SEC = 2.4
STEP_SEC = 0.8


@dataclass
class AudioChunk:
    """One overlapping chunk with dual-rate views."""

    index: int
    start_8k: int
    end_8k: int
    wav_8k: np.ndarray
    stft_8k: np.ndarray
    wav_16k: np.ndarray
    stft_16k: np.ndarray


def chunk_audio(
    mixture: np.ndarray,
    sample_rate: int,
    chunk_sec: float = CHUNK_SEC,
    step_sec: float = STEP_SEC,
) -> list[AudioChunk]:
    """Slice mixture into overlapping CALM-Sep chunks."""
    prep = preprocess_calmsep(mixture, sample_rate)
    wav = prep.wav_8k
    chunk_len = int(round(chunk_sec * CALMSEP_SR))
    step = int(round(step_sec * CALMSEP_SR))
    if chunk_len <= 0 or step <= 0:
        raise ValueError("chunk_sec and step_sec must be positive")
    if len(wav) <= chunk_len:
        return [
            AudioChunk(
                index=0,
                start_8k=0,
                end_8k=len(wav),
                wav_8k=wav,
                stft_8k=prep.stft_8k,
                wav_16k=prep.wav_16k,
                stft_16k=prep.stft_16k,
            )
        ]

    chunks: list[AudioChunk] = []
    start = 0
    idx = 0
    while start < len(wav):
        end = min(start + chunk_len, len(wav))
        piece = wav[start:end]
        if len(piece) < chunk_len // 4 and idx > 0:
            break
        if len(piece) < chunk_len:
            piece = np.pad(piece, (0, chunk_len - len(piece)))
        stft_8k = compute_stft(piece, n_fft=CALMSEP_N_FFT, hop_length=CALMSEP_HOP)
        # Parallel 16 kHz view of the same time span.
        start_16 = int(round(start * OUTPUT_SR / CALMSEP_SR))
        end_16 = int(round(end * OUTPUT_SR / CALMSEP_SR))
        wav_16 = prep.wav_16k[start_16:end_16]
        target_16 = int(round(chunk_sec * OUTPUT_SR))
        if len(wav_16) < target_16:
            wav_16 = np.pad(wav_16, (0, target_16 - len(wav_16)))
        else:
            wav_16 = wav_16[:target_16]
        stft_16k = compute_stft(wav_16, n_fft=BAND_N_FFT, hop_length=BAND_HOP)
        chunks.append(
            AudioChunk(
                index=idx,
                start_8k=start,
                end_8k=min(end, len(wav)),
                wav_8k=piece.astype(np.float32),
                stft_8k=stft_8k,
                wav_16k=wav_16.astype(np.float32),
                stft_16k=stft_16k,
            )
        )
        if end >= len(wav):
            break
        start += step
        idx += 1
    return chunks


def chunk_from_8k(
    wav_8k: np.ndarray,
    chunk_sec: float = CHUNK_SEC,
    step_sec: float = STEP_SEC,
) -> list[AudioChunk]:
    """Chunk an already-8 kHz waveform."""
    return chunk_audio(wav_8k, CALMSEP_SR, chunk_sec, step_sec)
