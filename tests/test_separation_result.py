"""Tests for the shared SeparationResult schema."""

import numpy as np
import pytest
import torch

from coralsep.schemas.separation_result import SeparationResult, StreamMetadata


def test_separation_result_valid() -> None:
    streams = np.random.randn(3, 16000).astype(np.float32)
    result = SeparationResult(
        streams=streams,
        sample_rate=16000,
        speaker_count=3,
        expert_used="sepformer",
    )
    assert result.num_streams == 3
    assert result.duration_sec == pytest.approx(1.0)
    assert len(result.metadata) == 3


def test_separation_result_count_mismatch_raises() -> None:
    streams = np.random.randn(3, 1000).astype(np.float32)
    with pytest.raises(ValueError, match="speaker_count"):
        SeparationResult(streams=streams, sample_rate=16000, speaker_count=2)


def test_from_torch() -> None:
    t = torch.randn(3, 8000)
    result = SeparationResult.from_torch(t, sample_rate=16000, expert_used="test")
    assert result.streams.shape == (3, 8000)
    assert result.speaker_count == 3


def test_stream_metadata_extra() -> None:
    meta = StreamMetadata(expert_source="srcorrnet", confidence=0.9, extra={"k": 1})
    assert meta.extra["k"] == 1
