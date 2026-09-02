"""Integration tests for paired and long-form alignment orchestration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from coralsep.align.integration import run_and_align, run_and_align_long
from coralsep.schemas.separation_result import SeparationResult, StreamMetadata

SR = 100


class FakeEngine:
    """Return deterministic sources, optionally permuted on alternating calls."""

    def __init__(
        self,
        sources: np.ndarray,
        *,
        alternate: bool = False,
        hop_samples: int = 0,
    ) -> None:
        self.sources = np.asarray(sources, dtype=np.float32)
        self.alternate = alternate
        self.hop_samples = hop_samples
        self.calls = 0

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        length = len(mixture)
        start = self.calls * self.hop_samples
        streams = self.sources[:, start : start + length].copy()
        embeddings = np.eye(self.sources.shape[0], dtype=np.float32)
        if self.alternate and self.calls % 2 == 1:
            order = np.arange(self.sources.shape[0])[::-1]
            streams = streams[order]
            embeddings = embeddings[order]
        self.calls += 1
        metadata = [
            StreamMetadata(expert_source="fake", embedding=embedding) for embedding in embeddings
        ]
        return SeparationResult(
            streams=streams,
            sample_rate=sample_rate,
            speaker_count=streams.shape[0],
            metadata=metadata,
            mixture=np.asarray(mixture, dtype=np.float32),
            expert_used="fake",
        )


def _sources(length: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / SR
    return np.stack(
        [
            np.sin(2 * np.pi * 3 * t),
            np.sin(2 * np.pi * 7 * t),
        ]
    ).astype(np.float32)


def test_run_and_align_reorders_second_expert() -> None:
    sources = _sources(200)
    anchor = FakeEngine(sources)
    other = FakeEngine(sources[::-1])
    # Correct the second engine's embeddings so they identify its reversed rows.
    original = other.separate

    def reversed_result(mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        result = original(mixture, sample_rate)
        metadata = [
            replace(result.metadata[0], embedding=np.array([0.0, 1.0])),
            replace(result.metadata[1], embedding=np.array([1.0, 0.0])),
        ]
        return replace(result, metadata=metadata)

    other.separate = reversed_result  # type: ignore[method-assign]
    output = run_and_align(anchor, other, sources.sum(axis=0), SR)
    assert output.alignment.method == "embedding"
    assert np.allclose(output.aligned.streams, output.anchor.streams)
    assert output.mean_matched_cost == pytest.approx(0.0)


def test_run_and_align_long_locks_alternating_permutations() -> None:
    length = 650
    sources = _sources(length)
    engine = FakeEngine(sources, alternate=True, hop_samples=150)
    output = run_and_align_long(
        engine,
        sources.sum(axis=0),
        SR,
        chunk_sec=2.0,
        overlap_sec=0.5,
        match_threshold=0.2,
    )
    assert output.num_chunks >= 3
    assert output.result.num_streams == 2
    assert output.chunk_track_ids[0] == (0, 1)
    assert output.chunk_track_ids[1] == (1, 0)
    assert output.chunk_track_ids[2] == (0, 1)
    assert output.result.streams.shape == sources.shape
    assert np.allclose(output.result.streams, sources, atol=1e-5)


def test_run_and_align_long_rejects_invalid_overlap() -> None:
    sources = _sources(200)
    with pytest.raises(ValueError, match="overlap_sec"):
        run_and_align_long(
            FakeEngine(sources),
            sources.sum(axis=0),
            SR,
            chunk_sec=1.0,
            overlap_sec=1.0,
        )
