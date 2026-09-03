"""
Chunk stitcher for the CoRAL-Sep streaming pipeline (Dev C, P4-C2).

Overlap-adds separated chunks back into full-length waveforms while solving
the permutation problem across chunk boundaries.

Speaker permutation across chunks is resolved greedily with cosine similarity
on the chunk-level mean embedding (or waveform correlation as fallback). When
ECAPA-TDNN embeddings are available they are used preferentially; otherwise
the stitcher falls back to time-domain cross-correlation (fast but noisier).

Overlap crossfade: linear fade-in/fade-out over the 1.6 s overlap region
(12 800 samples at 8 kHz or 25 600 samples at 16 kHz).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coralsep.pipeline.chunker import (
    CHUNK_SAMPLES_8K,
    OVERLAP_S,
    SR_8K,
    STEP_SAMPLES_8K,
)

_EPS = 1e-10


@dataclass
class StitchedOutput:
    """Full-length waveforms reassembled from per-chunk separation results."""

    waveforms: np.ndarray
    """(K, T) float32 full-length separated streams at 8 kHz."""

    sample_rate: int = SR_8K
    speaker_count: int = 0
    n_chunks: int = 0

    permutations: list[list[int]] = field(default_factory=list)
    """Per-chunk permutation vectors mapping chunk stream → canonical stream."""

    def __post_init__(self) -> None:
        if self.speaker_count == 0:
            self.speaker_count = self.waveforms.shape[0]


class ChunkStitcher:
    """
    Greedy permutation-solving overlap-adder for CoRAL-Sep chunk streams.

    Args:
        n_speakers: Fixed K for this utterance, set after the first chunk.
            Pass None to infer from the first chunk.
        sample_rate: Output sample rate (8000 Hz for the SR-CorrNet branch).
        use_ecapa: When True, use pre-computed ECAPA embeddings for permutation
            solving (passed via feed_chunk). Falls back to correlation if absent.
        overlap_samples: Crossfade region length in samples. Defaults to
            OVERLAP_S * sample_rate.
    """

    def __init__(
        self,
        n_speakers: int | None = None,
        sample_rate: int = SR_8K,
        use_ecapa: bool = True,
        overlap_samples: int | None = None,
    ) -> None:
        self.n_speakers = n_speakers
        self.sample_rate = sample_rate
        self.use_ecapa = use_ecapa
        self.overlap_samples = (
            overlap_samples if overlap_samples is not None else int(OVERLAP_S * sample_rate)
        )

        self._chunks: list[np.ndarray] = []
        """Accumulated list of (K, CHUNK) chunks after permutation alignment."""

        self._embeddings: list[np.ndarray | None] = []
        """Per-chunk mean ECAPA embeddings or None."""

        self._permutations: list[list[int]] = []
        self._step = int(STEP_SAMPLES_8K * sample_rate / SR_8K)
        self._chunk_len = int(CHUNK_SAMPLES_8K * sample_rate / SR_8K)

    def feed_chunk(
        self,
        streams: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> None:
        """
        Append one chunk's separation results.

        Args:
            streams: (K, T_chunk) float32 separated waveforms for this chunk.
            embeddings: (K, D) optional ECAPA embeddings for permutation solving.
        """
        streams = np.asarray(streams, dtype=np.float32)
        if streams.ndim != 2:
            raise ValueError(f"streams must be (K, T_chunk), got {streams.shape}")

        K = streams.shape[0]
        if self.n_speakers is None:
            self.n_speakers = K
        elif K != self.n_speakers:
            streams = _pad_or_trim_k(streams, self.n_speakers)

        if not self._chunks:
            perm = list(range(self.n_speakers))
        else:
            perm = self._solve_permutation(streams, embeddings)
            streams = streams[perm]

        self._chunks.append(streams)
        self._embeddings.append(
            embeddings[perm] if (embeddings is not None and self.use_ecapa) else None
        )
        self._permutations.append(perm)

    def finalize(self, total_samples: int | None = None) -> StitchedOutput:
        """
        Overlap-add all buffered chunks and return the full-length output.

        Args:
            total_samples: Expected total sample count. If provided the output
                is trimmed to this length; otherwise inferred from chunk count.

        Returns:
            StitchedOutput with (K, T) waveforms.
        """
        if not self._chunks:
            K = self.n_speakers or 1
            T = total_samples or 0
            return StitchedOutput(
                waveforms=np.zeros((K, T), dtype=np.float32),
                speaker_count=K,
                n_chunks=0,
            )

        K = self._chunks[0].shape[0]
        n = len(self._chunks)
        inferred_len = self._step * (n - 1) + self._chunk_len
        T = total_samples if total_samples is not None else inferred_len

        output = np.zeros((K, T), dtype=np.float32)
        weight = np.zeros(T, dtype=np.float32)

        fade_in = np.linspace(0.0, 1.0, self.overlap_samples, dtype=np.float32)
        fade_out = fade_in[::-1].copy()

        for i, chunk in enumerate(self._chunks):
            start = i * self._step
            end = start + self._chunk_len

            env = np.ones(self._chunk_len, dtype=np.float32)
            if i > 0:
                env[: self.overlap_samples] = fade_in
            if i < n - 1:
                env[-self.overlap_samples :] = fade_out

            # Clip to output bounds (last chunk may be shorter).
            clip_end = min(end, T)
            seg_len = clip_end - start
            if seg_len <= 0:
                continue

            output[:, start:clip_end] += chunk[:, :seg_len] * env[:seg_len]
            weight[start:clip_end] += env[:seg_len]

        # Normalize overlapping regions.
        nz = weight > _EPS
        output[:, nz] /= weight[nz]

        return StitchedOutput(
            waveforms=output,
            sample_rate=self.sample_rate,
            speaker_count=K,
            n_chunks=n,
            permutations=self._permutations,
        )

    def reset(self) -> None:
        self._chunks.clear()
        self._embeddings.clear()
        self._permutations.clear()

    # ------------------------------------------------------------------
    # Permutation solving
    # ------------------------------------------------------------------

    def _solve_permutation(
        self,
        new_streams: np.ndarray,
        new_embeddings: np.ndarray | None,
    ) -> list[int]:
        """
        Return a permutation p such that new_streams[p[i]] best matches
        canonical stream i established by the previous chunk.

        Uses ECAPA cosine similarity when available; otherwise falls back
        to time-domain overlap cross-correlation.
        """
        prev_emb = self._embeddings[-1]

        if self.use_ecapa and new_embeddings is not None and prev_emb is not None:
            return _hungarian_cosine(prev_emb, new_embeddings)

        return _hungarian_correlation(self._chunks[-1], new_streams, self.overlap_samples)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_or_trim_k(streams: np.ndarray, K: int) -> np.ndarray:
    """Ensure streams has exactly K rows (pad zeros or drop extra)."""
    k_have = streams.shape[0]
    if k_have == K:
        return streams
    if k_have > K:
        return streams[:K]
    pad = np.zeros((K - k_have, streams.shape[1]), dtype=np.float32)
    return np.concatenate([streams, pad], axis=0)


def _hungarian_cosine(ref_emb: np.ndarray, new_emb: np.ndarray) -> list[int]:
    """
    Greedy best-match permutation via cosine similarity matrix.

    Greedy (rather than true Hungarian) is fine for K ≤ 5.
    """
    K = ref_emb.shape[0]
    ref_norm = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + _EPS)
    new_norm = new_emb / (np.linalg.norm(new_emb, axis=1, keepdims=True) + _EPS)
    sim = ref_norm @ new_norm.T  # (K, K)

    perm = [-1] * K
    used = set()
    for _ in range(K):
        best_ref, best_new, best_sim = -1, -1, -999.0
        for r in range(K):
            if perm[r] != -1:
                continue
            for c in range(K):
                if c in used:
                    continue
                if sim[r, c] > best_sim:
                    best_sim = sim[r, c]
                    best_ref, best_new = r, c
        perm[best_ref] = best_new
        used.add(best_new)
    return perm


def _hungarian_correlation(
    prev_chunk: np.ndarray,
    new_chunk: np.ndarray,
    overlap_samples: int,
) -> list[int]:
    """
    Greedy permutation using correlation in the overlap region.

    prev_chunk tail × new_chunk head correlation → closest match.
    """
    K = prev_chunk.shape[0]
    tail = prev_chunk[:, -overlap_samples:]  # (K, overlap)
    head = new_chunk[:, :overlap_samples]  # (K, overlap)

    # Normalise rows.
    t_norm = tail / (np.linalg.norm(tail, axis=1, keepdims=True) + _EPS)
    h_norm = head / (np.linalg.norm(head, axis=1, keepdims=True) + _EPS)
    sim = t_norm @ h_norm.T  # (K, K)

    perm = [-1] * K
    used = set()
    for _ in range(K):
        best_r, best_c, best_v = -1, -1, -999.0
        for r in range(K):
            if perm[r] != -1:
                continue
            for c in range(K):
                if c in used:
                    continue
                if sim[r, c] > best_v:
                    best_v = sim[r, c]
                    best_r, best_c = r, c
        perm[best_r] = best_c
        used.add(best_c)
    return perm
