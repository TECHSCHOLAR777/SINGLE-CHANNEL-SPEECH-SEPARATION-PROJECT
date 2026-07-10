"""Tests for align/embeddings.py (ECAPA wrapper, speechbrain mocked)."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from align.embeddings import ECAPA_SAMPLE_RATE, EcapaEmbedder
from schemas.separation_result import SeparationResult, StreamMetadata

RNG = np.random.default_rng(seed=21)


@pytest.fixture()
def mocked_embedder(monkeypatch) -> EcapaEmbedder:
    """Embedder whose speechbrain backend is a deterministic mock."""
    import torch

    fake_module = MagicMock()

    class FakeClassifier:
        @staticmethod
        def encode_batch(wavs):
            # Deterministic per-stream embedding: mean/std stats tiled to 192-d.
            k = wavs.shape[0]
            base = torch.stack([torch.linspace(float(i + 1), float(i + 2), 192) for i in range(k)])
            return base.unsqueeze(1)  # [K, 1, 192]

    fake_module.EncoderClassifier.from_hparams.return_value = FakeClassifier()
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", fake_module)
    monkeypatch.setitem(sys.modules, "speechbrain", MagicMock())
    monkeypatch.setitem(sys.modules, "speechbrain.inference", MagicMock())
    return EcapaEmbedder(device="cpu")


def test_embed_shape_and_normalization(mocked_embedder: EcapaEmbedder) -> None:
    streams = RNG.standard_normal((3, ECAPA_SAMPLE_RATE)).astype(np.float32)
    emb = mocked_embedder.embed(streams, ECAPA_SAMPLE_RATE)
    assert emb.shape == (3, 192)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-6)


def test_empty_streams_rejected(mocked_embedder: EcapaEmbedder) -> None:
    with pytest.raises(ValueError):
        mocked_embedder.embed(np.zeros((0, 100)), ECAPA_SAMPLE_RATE)


def test_lazy_load_called_once(mocked_embedder: EcapaEmbedder) -> None:
    streams = RNG.standard_normal((2, 8000)).astype(np.float32)
    mocked_embedder.embed(streams, ECAPA_SAMPLE_RATE)
    mocked_embedder.embed(streams, ECAPA_SAMPLE_RATE)
    assert mocked_embedder._model is not None


def test_embed_result_fills_only_missing(mocked_embedder: EcapaEmbedder) -> None:
    streams = RNG.standard_normal((2, ECAPA_SAMPLE_RATE)).astype(np.float32)
    preset = np.ones(192) / np.sqrt(192)
    result = SeparationResult(
        streams=streams,
        sample_rate=ECAPA_SAMPLE_RATE,
        speaker_count=2,
        metadata=[
            StreamMetadata(expert_source="expert", embedding=preset),
            StreamMetadata(expert_source="expert", embedding=None),
        ],
        expert_used="expert",
    )
    filled = mocked_embedder.embed_result(result)
    assert np.allclose(filled.metadata[0].embedding, preset)  # preserved
    assert filled.metadata[1].embedding is not None  # computed
    assert result.metadata[1].embedding is None  # input not mutated


def test_resample_path_changes_length() -> None:
    pytest.importorskip("torchaudio")
    streams = RNG.standard_normal((1, 8000)).astype(np.float32)  # 1 s at 8 kHz
    out = EcapaEmbedder._resample(streams, sample_rate=8000)
    assert out.shape == (1, ECAPA_SAMPLE_RATE)  # 1 s at 16 kHz
