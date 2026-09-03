"""
Audio chunker for the CoRAL-Sep streaming pipeline (Dev C, P4-C1).

Produces 2.4 s overlapping chunks at 8 kHz (19 200 samples) with a 0.8 s
step (6 400 samples), yielding 66.7% overlap. The overlap is crossfaded by
ChunkStitcher at reassembly time.

Also produces a parallel 16 kHz STFT slice for each chunk, used by the
band recovery module. The 16 kHz slice is computed from the same time region
as the 8 kHz chunk (index conversion: 16000/8000 = 2× sample ratio).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Chunking parameters (BLUEPRINT §6.1)
CHUNK_DURATION_S: float = 2.4
STEP_DURATION_S: float = 0.8
OVERLAP_S: float = CHUNK_DURATION_S - STEP_DURATION_S  # 1.6 s

# 8 kHz (SR-CorrNet branch)
SR_8K: int = 8_000
CHUNK_SAMPLES_8K: int = int(CHUNK_DURATION_S * SR_8K)  # 19 200
STEP_SAMPLES_8K: int = int(STEP_DURATION_S * SR_8K)  # 6 400

# 16 kHz (band recovery branch)
SR_16K: int = 16_000
CHUNK_SAMPLES_16K: int = int(CHUNK_DURATION_S * SR_16K)  # 38 400
STEP_SAMPLES_16K: int = int(STEP_DURATION_S * SR_16K)  # 12 800

# STFT parameters for the 16 kHz branch
_STFT_N_FFT_16K: int = 512
_STFT_HOP_16K: int = 128
_STFT_WIN_16K: int = 512


@dataclass
class AudioChunk:
    """
    One overlapping chunk of audio with its 16 kHz STFT companion.

    Attributes:
        waveform_8k: [CHUNK_SAMPLES_8K] float32 waveform at 8 kHz.
        stft_16k: [257, T_frames] complex64 STFT at 16 kHz, or None if
            a 16 kHz signal was not provided.
        chunk_index: 0-based chunk number.
        start_sample_8k: Start sample in the original 8 kHz signal.
        is_last: True for the final chunk (may be shorter before padding).
    """

    waveform_8k: np.ndarray
    stft_16k: np.ndarray | None
    chunk_index: int
    start_sample_8k: int
    is_last: bool = False

    @property
    def duration_s(self) -> float:
        return CHUNK_DURATION_S


class Chunker:
    """
    Segments an audio recording into overlapping 2.4 s / 0.8 s-step chunks.

    Args:
        waveform_8k: [T] float32 mono waveform at 8 kHz.
        waveform_16k: [T*2] float32 mono waveform at 16 kHz, or None.
            When provided, a 16 kHz STFT slice is included in each chunk.
        pad_last: When True (default), the final short chunk is zero-padded
            to CHUNK_SAMPLES_8K so all chunks have the same length.
    """

    def __init__(
        self,
        waveform_8k: np.ndarray,
        waveform_16k: np.ndarray | None = None,
        pad_last: bool = True,
    ) -> None:
        self.wav8k = np.asarray(waveform_8k, dtype=np.float32).squeeze()
        if self.wav8k.ndim != 1:
            raise ValueError(f"waveform_8k must be 1-D, got {self.wav8k.shape}")

        self.wav16k = (
            np.asarray(waveform_16k, dtype=np.float32).squeeze()
            if waveform_16k is not None
            else None
        )
        self.pad_last = pad_last
        self._chunks: list[AudioChunk] | None = None

    @property
    def n_chunks(self) -> int:
        return len(self._build_chunks())

    def __iter__(self):  # type: ignore[return]
        return iter(self._build_chunks())

    def _build_chunks(self) -> list[AudioChunk]:
        if self._chunks is not None:
            return self._chunks

        n = len(self.wav8k)
        starts = list(range(0, max(n - CHUNK_SAMPLES_8K + 1, 1), STEP_SAMPLES_8K))
        # Ensure at least one chunk even for very short inputs.
        if not starts:
            starts = [0]
        # Add a final chunk if the last regular chunk doesn't reach the end.
        if starts[-1] + CHUNK_SAMPLES_8K < n:
            starts.append(max(0, n - CHUNK_SAMPLES_8K))

        chunks: list[AudioChunk] = []
        for idx, start in enumerate(starts):
            end = start + CHUNK_SAMPLES_8K
            seg8 = self.wav8k[start:end]
            is_last = idx == len(starts) - 1

            if len(seg8) < CHUNK_SAMPLES_8K:
                if self.pad_last:
                    seg8 = np.pad(seg8, (0, CHUNK_SAMPLES_8K - len(seg8)))
                else:
                    seg8 = seg8.copy()

            stft16 = None
            if self.wav16k is not None:
                start16 = start * 2
                end16 = start16 + CHUNK_SAMPLES_16K
                seg16 = self.wav16k[start16:end16]
                if len(seg16) < CHUNK_SAMPLES_16K:
                    seg16 = np.pad(seg16, (0, CHUNK_SAMPLES_16K - len(seg16)))
                stft16 = _compute_stft_16k(seg16)

            chunks.append(
                AudioChunk(
                    waveform_8k=seg8,
                    stft_16k=stft16,
                    chunk_index=idx,
                    start_sample_8k=start,
                    is_last=is_last,
                )
            )

        self._chunks = chunks
        return chunks

    def chunk_at(self, idx: int) -> AudioChunk:
        return self._build_chunks()[idx]


def _compute_stft_16k(waveform_16k: np.ndarray) -> np.ndarray:
    """[T_16k] → [257, frames] complex64 STFT."""
    wav = torch.from_numpy(waveform_16k.astype(np.float32))
    spec = torch.stft(
        wav,
        n_fft=_STFT_N_FFT_16K,
        hop_length=_STFT_HOP_16K,
        win_length=_STFT_WIN_16K,
        window=torch.hann_window(_STFT_WIN_16K),
        return_complex=True,
        center=True,
    )
    return spec.numpy().astype(np.complex64)
