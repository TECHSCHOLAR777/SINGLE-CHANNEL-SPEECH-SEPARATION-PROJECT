"""
Cross-chunk identity lock for long-form audio (Dev C, Phase 1).

Chunked separation re-permutes speakers between chunks: chunk k's "speaker 1"
silently becomes chunk k+1's "speaker 3", which is invisible in benchmarks and
fatal in demos. The ChunkStitcher maintains a rolling bank of speaker-track
embeddings across the WHOLE recording (not just the previous chunk, so a
speaker silent for one chunk keeps their track), Hungarian-matches each new
chunk against the bank, spawns new tracks for genuinely new voices, and
overlap-adds with a crossfade into continuous per-speaker waveforms.

All thresholds come from configs/devc.yaml via the constructor; nothing is
hardcoded in the logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from coralsep.align.hungarian import cosine_cost_matrix

_EPS = 1e-8


@dataclass
class _Track:
    """One speaker's running state across chunks."""

    embedding: np.ndarray
    """EMA speaker embedding, shape [D]."""

    last_seen_chunk: int
    segments: list[tuple[int, np.ndarray]] = field(default_factory=list)
    """(start_sample, waveform) pieces, crossfaded at stitch time."""


class ChunkStitcher:
    """
    Assigns chunk streams to persistent speaker tracks and stitches waveforms.

    Args:
        sample_rate: Audio sample rate in Hz.
        chunk_sec: Chunk length used by the caller, seconds.
        overlap_sec: Overlap between consecutive chunks, seconds. Crossfade
            length equals the overlap.
        match_threshold: Maximum cosine cost (1 - similarity) for a chunk
            stream to join an existing track; above it a new track spawns.
        ema: Exponential moving average factor for track embeddings; higher
            trusts history more.
        max_tracks: Hard cap on the persistent track bank, normally the
            expected speaker count. Without a cap, any chunk whose embedding
            drifts past match_threshold spawns a brand-new track, so a single
            unstable output slot mints one phantom track per chunk (each alive
            in one chunk and silent elsewhere, which then corrupts every
            downstream alignment metric). At the cap, a stream that clears no
            threshold is force-assigned to its best Hungarian partner instead
            of spawning. Its embedding is deliberately NOT folded into that
            track's EMA, since a match that failed the threshold is suspect and
            would poison the bank. None restores the old uncapped behaviour.
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_sec: float,
        overlap_sec: float,
        match_threshold: float = 0.35,
        ema: float = 0.7,
        max_tracks: int | None = None,
    ) -> None:
        if overlap_sec >= chunk_sec:
            raise ValueError("overlap_sec must be smaller than chunk_sec")
        if not 0.0 <= ema <= 1.0:
            raise ValueError("ema must lie in [0, 1]")
        if max_tracks is not None and max_tracks < 1:
            raise ValueError(f"max_tracks must be >= 1 or None, got {max_tracks}")
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.overlap_sec = overlap_sec
        self.match_threshold = match_threshold
        self.ema = ema
        self.max_tracks = max_tracks
        self._tracks: list[_Track] = []
        self._chunk_index = 0
        self._forced_matches = 0

    @property
    def hop_samples(self) -> int:
        return int(round((self.chunk_sec - self.overlap_sec) * self.sample_rate))

    @property
    def num_tracks(self) -> int:
        return len(self._tracks)

    @property
    def forced_matches(self) -> int:
        """Streams assigned above match_threshold because the bank was at cap.

        A non-zero count is a health signal, not an error: it means the
        embeddings are noisier than match_threshold assumes (common for ECAPA
        on separated-but-still-contaminated overlapping speech). A count near
        the total stream count means the threshold is doing no work and the
        cap is carrying the whole assignment.
        """
        return self._forced_matches

    def add_chunk(self, streams: np.ndarray, embeddings: np.ndarray) -> list[int]:
        """
        Register one chunk's separated streams against the persistent tracks.

        Args:
            streams: [K, T_chunk] separated waveforms for this chunk.
            embeddings: [K, D] speaker embeddings for those streams.

        Returns:
            Track id per input stream, in input order.
        """
        streams = np.atleast_2d(np.asarray(streams, dtype=np.float32))
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        if streams.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"streams ({streams.shape[0]}) and embeddings ({embeddings.shape[0]}) "
                "must agree on stream count"
            )

        start = self._chunk_index * self.hop_samples
        assigned: dict[int, int] = {}
        partner: dict[int, int] = {}

        if self._tracks:
            bank = np.stack([t.embedding for t in self._tracks], axis=0)
            cost = cosine_cost_matrix(embeddings, bank)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
                # Hungarian is one-to-one, so partner[] can never collide two
                # streams of this chunk onto the same track.
                partner[i] = j
                if cost[i, j] <= self.match_threshold:
                    assigned[i] = j

        track_ids: list[int] = []
        for i in range(streams.shape[0]):
            at_cap = self.max_tracks is not None and len(self._tracks) >= self.max_tracks
            if i in assigned:
                tid = assigned[i]
                track = self._tracks[tid]
                track.embedding = self.ema * track.embedding + (1.0 - self.ema) * embeddings[i]
                track.embedding /= max(float(np.linalg.norm(track.embedding)), _EPS)
            elif at_cap and i in partner:
                # Bank is full and this stream cleared no threshold. Spawning
                # would mint a phantom track, so take the best partner instead.
                # No EMA update: the match failed the threshold, so folding its
                # embedding in would drag the track toward a suspect identity.
                tid = partner[i]
                self._forced_matches += 1
            else:
                # Genuinely new voice, or no partner available (K > num_tracks).
                # Audio is never dropped, so a stream with no partner spawns
                # even at cap; that only happens if the expert emits more
                # streams than max_tracks, which is a caller error worth seeing.
                emb = embeddings[i] / max(float(np.linalg.norm(embeddings[i])), _EPS)
                self._tracks.append(_Track(embedding=emb, last_seen_chunk=self._chunk_index))
                tid = len(self._tracks) - 1
            self._tracks[tid].last_seen_chunk = self._chunk_index
            self._tracks[tid].segments.append((start, streams[i].copy()))
            track_ids.append(tid)

        self._chunk_index += 1
        return track_ids

    def stitch(self) -> np.ndarray:
        """
        Overlap-add all tracks into continuous waveforms.

        Overlapping regions are blended with a linear crossfade of length
        overlap_sec. Chunks where a track was silent stay zero, preserving
        timeline alignment across tracks.

        Returns:
            [num_tracks, total_samples] float32 array.
        """
        if not self._tracks:
            return np.zeros((0, 0), dtype=np.float32)

        total = 0
        for track in self._tracks:
            for start, wav in track.segments:
                total = max(total, start + wav.shape[0])

        out = np.zeros((len(self._tracks), total), dtype=np.float64)
        weight = np.zeros((len(self._tracks), total), dtype=np.float64)
        fade_len = int(round(self.overlap_sec * self.sample_rate))

        for tid, track in enumerate(self._tracks):
            for start, wav in track.segments:
                w = np.ones(wav.shape[0], dtype=np.float64)
                ramp = min(fade_len, wav.shape[0])
                if ramp > 0:
                    w[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False) + _EPS
                    w[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=False) + _EPS
                sl = slice(start, start + wav.shape[0])
                out[tid, sl] += wav.astype(np.float64) * w
                weight[tid, sl] += w

        weight = np.where(weight > 0, weight, 1.0)
        return (out / weight).astype(np.float32)
