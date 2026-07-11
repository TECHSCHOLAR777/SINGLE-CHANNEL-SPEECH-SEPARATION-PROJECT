"""
Expert-alignment orchestration for Phase 1 integration (Dev C).

``run_and_align`` executes two experts and reorders the second output into the
first expert's speaker order. ``run_and_align_long`` separates overlapping
chunks, assigns streams to persistent speaker tracks, and overlap-adds them.
Protocols keep real wrappers swappable with deterministic CI fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.signal import resample_poly

from align.chunking import ChunkStitcher
from align.embeddings import EcapaEmbedder
from align.hungarian import AlignmentResult, align_results, reorder_result
from schemas.separation_result import SeparationResult, StreamMetadata


@runtime_checkable
class Engine(Protocol):
    """Minimal contract shared by separation expert wrappers."""

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        """Separate one mono waveform and return the shared schema."""


class ResultEmbedder(Protocol):
    """Embedding adapter accepted by the integration functions."""

    def embed_result(self, result: SeparationResult) -> SeparationResult:
        """Return a result whose stream metadata contain embeddings."""


@dataclass(frozen=True)
class PairedAlignmentResult:
    """Two expert outputs after the second has been aligned to the first."""

    anchor: SeparationResult
    aligned: SeparationResult
    alignment: AlignmentResult

    @property
    def mean_matched_cost(self) -> float:
        """Mean cost over Hungarian-matched stream pairs."""
        if not self.alignment.assignment:
            return float("nan")
        values = [self.alignment.cost_matrix[i, j] for i, j in self.alignment.assignment]
        return float(np.mean(values))


@dataclass(frozen=True)
class LongAlignmentResult:
    """Persistent-track output and per-chunk identity assignments."""

    result: SeparationResult
    chunk_results: tuple[SeparationResult, ...]
    chunk_track_ids: tuple[tuple[int, ...], ...]
    chunk_starts: tuple[int, ...]

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_results)


def _all_embeddings_present(result: SeparationResult) -> bool:
    return bool(result.metadata) and all(meta.embedding is not None for meta in result.metadata)


def ensure_embeddings(
    result: SeparationResult,
    embedder: ResultEmbedder | None = None,
) -> SeparationResult:
    """Attach missing stream embeddings, preserving complete results."""
    if _all_embeddings_present(result):
        return result
    active = embedder if embedder is not None else EcapaEmbedder()
    enriched = active.embed_result(result)
    if not _all_embeddings_present(enriched):
        raise RuntimeError("embedder returned a result with missing stream embeddings")
    return enriched


def run_and_align(
    anchor_engine: Engine,
    other_engine: Engine,
    mixture: np.ndarray,
    sample_rate: int,
    *,
    embedder: ResultEmbedder | None = None,
) -> PairedAlignmentResult:
    """Run two experts and align ``other_engine`` to ``anchor_engine`` order."""
    wave = np.asarray(mixture, dtype=np.float32).squeeze()
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError(f"mixture must be a non-empty mono waveform, got {wave.shape}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    anchor = ensure_embeddings(anchor_engine.separate(wave, sample_rate), embedder)
    other = ensure_embeddings(other_engine.separate(wave, sample_rate), embedder)
    if anchor.sample_rate != other.sample_rate:
        raise ValueError(
            "expert outputs use different sample rates: "
            f"{anchor.sample_rate} vs {other.sample_rate}"
        )

    alignment = align_results(anchor, other)
    return PairedAlignmentResult(
        anchor=anchor,
        aligned=reorder_result(other, alignment),
        alignment=alignment,
    )


def _resample_mono(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(waveform, dtype=np.float32)
    factor = gcd(orig_sr, target_sr)
    out = resample_poly(
        np.asarray(waveform, dtype=np.float64),
        up=target_sr // factor,
        down=orig_sr // factor,
    )
    return out.astype(np.float32)


def run_and_align_long(
    engine: Engine,
    mixture: np.ndarray,
    sample_rate: int,
    *,
    embedder: ResultEmbedder | None = None,
    chunk_sec: float = 4.0,
    overlap_sec: float = 1.0,
    match_threshold: float = 0.35,
    ema: float = 0.7,
) -> LongAlignmentResult:
    """Separate long audio in overlapping chunks with persistent speaker IDs."""
    wave = np.asarray(mixture, dtype=np.float32).squeeze()
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError(f"mixture must be a non-empty mono waveform, got {wave.shape}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if chunk_sec <= 0.0:
        raise ValueError("chunk_sec must be positive")
    if overlap_sec < 0.0 or overlap_sec >= chunk_sec:
        raise ValueError("overlap_sec must satisfy 0 <= overlap_sec < chunk_sec")

    chunk_samples = max(1, int(round(chunk_sec * sample_rate)))
    hop_samples = max(1, int(round((chunk_sec - overlap_sec) * sample_rate)))
    starts = range(0, wave.shape[0], hop_samples)

    stitcher: ChunkStitcher | None = None
    output_sample_rate: int | None = None
    chunk_results: list[SeparationResult] = []
    track_ids: list[tuple[int, ...]] = []
    output_starts: list[int] = []

    for start in starts:
        chunk = wave[start : start + chunk_samples]
        if chunk.size == 0:
            continue
        separated = ensure_embeddings(engine.separate(chunk, sample_rate), embedder)
        embeddings = np.stack(
            [np.asarray(meta.embedding, dtype=np.float64) for meta in separated.metadata],
            axis=0,
        )

        if stitcher is None:
            output_sample_rate = separated.sample_rate
            stitcher = ChunkStitcher(
                sample_rate=output_sample_rate,
                chunk_sec=chunk_sec,
                overlap_sec=overlap_sec,
                match_threshold=match_threshold,
                ema=ema,
            )
        elif separated.sample_rate != output_sample_rate:
            raise ValueError(
                "engine changed output sample rate between chunks: "
                f"{output_sample_rate} vs {separated.sample_rate}"
            )

        ids = stitcher.add_chunk(separated.streams, embeddings)
        chunk_results.append(separated)
        track_ids.append(tuple(ids))
        output_starts.append((len(chunk_results) - 1) * stitcher.hop_samples)

        if start + chunk_samples >= wave.shape[0]:
            break

    if stitcher is None or output_sample_rate is None or not chunk_results:
        raise RuntimeError("engine produced no chunk results")

    streams = stitcher.stitch()
    output_mixture = _resample_mono(wave, sample_rate, output_sample_rate)
    target_len = output_mixture.shape[0]
    if streams.shape[1] > target_len:
        streams = streams[:, :target_len]
    elif streams.shape[1] < target_len:
        streams = np.pad(streams, ((0, 0), (0, target_len - streams.shape[1])))

    first = chunk_results[0]
    metadata = [
        StreamMetadata(
            expert_source=first.expert_used,
            confidence=1.0,
            extra={"persistent_track_id": track_id},
        )
        for track_id in range(streams.shape[0])
    ]
    result = SeparationResult(
        streams=streams,
        sample_rate=output_sample_rate,
        speaker_count=streams.shape[0],
        metadata=metadata,
        mixture=output_mixture,
        escalated=first.escalated,
        expert_used=first.expert_used,
    )
    return LongAlignmentResult(
        result=result,
        chunk_results=tuple(chunk_results),
        chunk_track_ids=tuple(track_ids),
        chunk_starts=tuple(output_starts),
    )
