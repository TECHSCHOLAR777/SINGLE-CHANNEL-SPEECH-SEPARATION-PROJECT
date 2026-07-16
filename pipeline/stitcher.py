"""Stream stitching for CALM-Sep (BLUEPRINT §6.3–6.4).

Primary continuity: max correlation on overlap.
Tie-break: ECAPA-TDNN embedding similarity.
Global count: agglomerative clustering of embeddings (duration > 1.0 s).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

from models.preprocess import CALMSEP_SR, OUTPUT_SR, resample_audio
from pipeline.chunker import CHUNK_SEC, STEP_SEC


@dataclass
class StitchedOutput:
    streams_16k: np.ndarray
    n_global: int
    cluster_ids: list[int]
    per_stream_duration: list[float]


@dataclass
class _Track:
    pieces: list[tuple[int, np.ndarray]] = field(default_factory=list)
    embedding: np.ndarray | None = None


def _xcorr_score(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    x = a[:n] - a[:n].mean()
    y = b[:n] - b[:n].mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
    return float(np.dot(x, y) / denom)


class CalmSepStitcher:
    """Stitch per-chunk streams into global speaker tracks at 16 kHz."""

    def __init__(
        self,
        sample_rate: int = OUTPUT_SR,
        chunk_sec: float = CHUNK_SEC,
        step_sec: float = STEP_SEC,
        embedder: object | None = None,
        cluster_threshold: float = 0.4,
        min_duration_sec: float = 1.0,
        max_tracks: int = 5,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.step_sec = step_sec
        self.overlap_sec = chunk_sec - step_sec
        self.embedder = embedder
        self.cluster_threshold = cluster_threshold
        self.min_duration_sec = min_duration_sec
        self.max_tracks = max(1, int(max_tracks))
        self._tracks: list[_Track] = []
        self._chunk_i = 0

    @property
    def hop_samples(self) -> int:
        return int(round(self.step_sec * self.sample_rate))

    @property
    def overlap_samples(self) -> int:
        return int(round(self.overlap_sec * self.sample_rate))

    def reset(self) -> None:
        self._tracks.clear()
        self._chunk_i = 0

    def _embed(self, wav: np.ndarray) -> np.ndarray | None:
        if self.embedder is None:
            return None
        try:
            emb = self.embedder.embed(wav, self.sample_rate)  # type: ignore[attr-defined]
            return np.asarray(emb, dtype=np.float32).reshape(-1)
        except Exception:
            return None

    def add_chunk(self, streams_16k: np.ndarray) -> None:
        streams = np.atleast_2d(np.asarray(streams_16k, dtype=np.float32))
        start = self._chunk_i * self.hop_samples
        if not self._tracks:
            for s in streams[: self.max_tracks]:
                emb = self._embed(s)
                self._tracks.append(_Track(pieces=[(start, s)], embedding=emb))
            self._chunk_i += 1
            return

        # Build cost: prefer high correlation on overlap; ECAPA as tie-break.
        n_new, n_old = streams.shape[0], len(self._tracks)
        cost = np.ones((n_new, n_old), dtype=np.float64)
        for i, s in enumerate(streams):
            for j, tr in enumerate(self._tracks):
                prev = tr.pieces[-1][1]
                ov = self.overlap_samples
                if ov > 0 and len(prev) >= ov and len(s) >= ov:
                    corr = _xcorr_score(prev[-ov:], s[:ov])
                else:
                    corr = _xcorr_score(prev, s)
                emb_sim = 0.0
                emb = self._embed(s)
                if emb is not None and tr.embedding is not None:
                    ea, eb = tr.embedding, emb
                    emb_sim = float(np.dot(ea, eb) / ((np.linalg.norm(ea) * np.linalg.norm(eb)) + 1e-8))
                score = 0.7 * corr + 0.3 * emb_sim
                cost[i, j] = 1.0 - score

        row, col = linear_sum_assignment(cost)
        assigned_old = set()
        assigned_new = set()
        for i, j in zip(row.tolist(), col.tolist(), strict=True):
            if cost[i, j] < 0.65:  # accept
                tr = self._tracks[j]
                tr.pieces.append((start, streams[i]))
                emb = self._embed(streams[i])
                if emb is not None:
                    if tr.embedding is None:
                        tr.embedding = emb
                    else:
                        tr.embedding = 0.7 * tr.embedding + 0.3 * emb
                assigned_old.add(j)
                assigned_new.add(i)

        for i in range(n_new):
            if i not in assigned_new:
                if len(self._tracks) >= self.max_tracks:
                    # Force-assign to best existing track (no spawn beyond K0=5).
                    j_best = int(np.argmin(cost[i]))
                    self._tracks[j_best].pieces.append((start, streams[i]))
                else:
                    emb = self._embed(streams[i])
                    self._tracks.append(_Track(pieces=[(start, streams[i])], embedding=emb))

        self._chunk_i += 1

    def _crossfade_track(self, track: _Track) -> np.ndarray:
        if not track.pieces:
            return np.zeros(0, dtype=np.float32)
        total_len = track.pieces[-1][0] + len(track.pieces[-1][1])
        out = np.zeros(total_len, dtype=np.float32)
        weight = np.zeros(total_len, dtype=np.float32)
        ov = self.overlap_samples
        for start, wav in track.pieces:
            end = start + len(wav)
            w = np.ones(len(wav), dtype=np.float32)
            if ov > 0 and len(wav) > 2 * ov:
                ramp = np.linspace(0, 1, ov, dtype=np.float32)
                w[:ov] = ramp
                w[-ov:] = ramp[::-1]
            out[start:end] += wav * w
            weight[start:end] += w
        weight = np.maximum(weight, 1e-6)
        return (out / weight).astype(np.float32)

    def finalize(self) -> StitchedOutput:
        waveforms = [self._crossfade_track(t) for t in self._tracks]
        durations = [len(w) / float(self.sample_rate) for w in waveforms]
        embeddings = []
        keep_idx = []
        for i, (w, d, t) in enumerate(zip(waveforms, durations, self._tracks, strict=True)):
            if d < self.min_duration_sec:
                continue
            keep_idx.append(i)
            if t.embedding is not None:
                embeddings.append(t.embedding)
            else:
                embeddings.append(np.zeros(8, dtype=np.float32))

        if not keep_idx:
            # Keep longest track at least.
            if waveforms:
                order = int(np.argmax([len(w) for w in waveforms]))
                arr = np.stack([waveforms[order]], axis=0)
                return StitchedOutput(arr, 1, [0], [durations[order]])
            return StitchedOutput(np.zeros((0, 0), dtype=np.float32), 0, [], [])

        kept_wavs = [waveforms[i] for i in keep_idx]
        kept_dur = [durations[i] for i in keep_idx]
        emb = np.stack(embeddings, axis=0)
        if len(kept_wavs) == 1:
            arr = np.stack(kept_wavs, axis=0)
            return StitchedOutput(arr, 1, [0], kept_dur)

        # Agglomerative clustering on cosine distance.
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb_n = emb / norms
        sim = emb_n @ emb_n.T
        dist = np.clip(1.0 - sim, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        z = linkage(condensed, method="average")
        labels = fcluster(z, t=self.cluster_threshold, criterion="distance")
        # Merge waveforms within cluster by energy-weighted sum / pick longest.
        clusters = sorted(set(int(x) for x in labels))
        merged: list[np.ndarray] = []
        for c in clusters:
            members = [kept_wavs[i] for i, lab in enumerate(labels) if int(lab) == c]
            max_len = max(len(m) for m in members)
            acc = np.zeros(max_len, dtype=np.float32)
            for m in members:
                acc[: len(m)] += m
            acc /= max(len(members), 1)
            merged.append(acc)
        # Pad to common length.
        max_len = max(len(m) for m in merged)
        arr = np.stack([np.pad(m, (0, max_len - len(m))) for m in merged], axis=0)
        n_global = arr.shape[0]
        return StitchedOutput(
            streams_16k=arr.astype(np.float32),
            n_global=n_global,
            cluster_ids=clusters,
            per_stream_duration=[len(m) / float(self.sample_rate) for m in merged],
        )


def upsample_streams_8k_to_16k(streams_8k: np.ndarray) -> np.ndarray:
    return np.stack(
        [resample_audio(s, CALMSEP_SR, OUTPUT_SR) for s in np.atleast_2d(streams_8k)],
        axis=0,
    ).astype(np.float32)
