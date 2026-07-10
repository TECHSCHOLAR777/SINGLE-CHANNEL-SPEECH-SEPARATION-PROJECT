"""Tests for models/counting_features.py (P3-B1 stop-classifier features)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from models.counting_features import (
    CountingFeatureExtractor,
    StopFeatureBundle,
    VADAdapter,
    assemble_stop_features,
    compute_min_embedding_distance,
    compute_residual,
    compute_stop_features,
    cosine_distance,
    mixture_consistency_error,
    residual_energy_ratio,
)
from models.stop_classifier import FEATURE_NAMES

RNG = np.random.default_rng(seed=11)


@dataclass
class _MockEmbedder:
    """Deterministic pseudo-embeddings for unit tests (no SpeechBrain download)."""

    scale: float = 1.0

    def embed_stream(self, waveform: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        del sample_rate
        w = np.asarray(waveform, dtype=np.float64).reshape(-1)
        seed = int(abs(np.sum(w * 1000.0)) % 10_000)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(8)
        return (vec / (np.linalg.norm(vec) + 1e-8) * self.scale).astype(np.float32)


def test_feature_vector_order_and_shape() -> None:
    mix = RNG.standard_normal(8000)
    feats = compute_stop_features(
        mixture=mix,
        accepted_stems=np.zeros((0, 8000)),
        candidate_stem=mix * 0.5,
        vad_speech_prob=0.7,
        min_embedding_distance=1.0,
        attractor_stop_logit=-0.4,
    )
    assert feats.shape == (len(FEATURE_NAMES),)
    assert feats[1] == pytest.approx(0.7)
    assert feats[4] == pytest.approx(-0.4)


def test_perfect_extraction_drives_residual_features_to_zero() -> None:
    a, b = RNG.standard_normal((2, 8000))
    mix = a + b
    feats = compute_stop_features(
        mixture=mix,
        accepted_stems=a[None, :],
        candidate_stem=b,
        vad_speech_prob=0.0,
        min_embedding_distance=1.0,
    )
    assert feats[0] == pytest.approx(0.0, abs=1e-9)
    assert feats[3] == pytest.approx(0.0, abs=1e-6)


def test_unexplained_speaker_leaves_high_residual() -> None:
    a, b, c = RNG.standard_normal((3, 8000))
    mix = a + b + c
    feats = compute_stop_features(
        mixture=mix,
        accepted_stems=a[None, :],
        candidate_stem=b,
        vad_speech_prob=0.9,
        min_embedding_distance=0.8,
    )
    assert feats[0] > 0.1


def test_residual_energy_ratio_matches_manual() -> None:
    mix = RNG.standard_normal(4000)
    stem = mix * 0.6
    residual = mix - stem
    expected = float(np.dot(residual, residual)) / (float(np.dot(mix, mix)) + 1e-8)
    assert residual_energy_ratio(mix, np.zeros((0, len(mix))), stem) == pytest.approx(expected)


def test_mixture_consistency_error_matches_norm_ratio() -> None:
    mix = RNG.standard_normal(4000)
    stem = mix * 0.4
    residual = mix - stem
    expected = float(np.linalg.norm(residual)) / (float(np.linalg.norm(mix)) + 1e-8)
    assert mixture_consistency_error(mix, np.zeros((0, len(mix))), stem) == pytest.approx(expected)


def test_compute_residual_shape() -> None:
    mix = RNG.standard_normal(1000)
    a, b = RNG.standard_normal((2, 1000))
    res = compute_residual(mix, a[None, :], b)
    assert res.shape == (1000,)
    np.testing.assert_allclose(res, mix - a - b, atol=1e-12)


def test_cosine_distance_identical_is_zero() -> None:
    v = RNG.standard_normal(16)
    v = v / np.linalg.norm(v)
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_min_embedding_distance_empty_stems_returns_one() -> None:
    stem = RNG.standard_normal(1600)
    dist = compute_min_embedding_distance(stem, np.zeros((0, len(stem))), _MockEmbedder())
    assert dist == pytest.approx(1.0)


def test_min_embedding_distance_duplicate_stem_is_low() -> None:
    stem = RNG.standard_normal(1600)
    dist = compute_min_embedding_distance(stem, stem[None, :], _MockEmbedder())
    assert dist < 0.05


def test_vad_silence_vs_speech_energy_backend() -> None:
    vad = VADAdapter(backend="energy")
    silence = np.zeros(16_000, dtype=np.float64)
    t = np.linspace(0, 1.0, 16_000, endpoint=False)
    speech = 0.25 * np.sin(2 * np.pi * 200 * t)
    assert vad.speech_prob(silence) < vad.speech_prob(speech)


def test_vad_short_clip_returns_bounded_prob() -> None:
    vad = VADAdapter(backend="energy")
    prob = vad.speech_prob(np.array([0.01, -0.01, 0.02], dtype=np.float64))
    assert 0.0 <= prob <= 1.0


def test_assemble_stop_features_order() -> None:
    vec = assemble_stop_features(
        residual_energy_ratio=0.1,
        vad_speech_prob=0.2,
        min_embedding_distance=0.3,
        mixture_consistency_error=0.4,
        attractor_stop_logit=0.5,
    )
    np.testing.assert_allclose(vec, [0.1, 0.2, 0.3, 0.4, 0.5], rtol=0, atol=1e-6)


def test_stop_feature_bundle_to_vector() -> None:
    bundle = StopFeatureBundle(
        residual_energy_ratio=0.11,
        vad_speech_prob=0.22,
        min_embedding_distance=0.33,
        mixture_consistency_error=0.44,
        attractor_stop_logit=-1.0,
    )
    assert bundle.to_vector().shape == (len(FEATURE_NAMES),)
    assert bundle.to_vector()[4] == pytest.approx(-1.0)


def test_counting_feature_extractor_with_mock_embedder() -> None:
    a, b = RNG.standard_normal((2, 8000))
    mix = a + b
    extractor = CountingFeatureExtractor(embedder=_MockEmbedder())
    vec = extractor.extract(mix, np.zeros((0, 8000)), a, attractor_stop_logit=0.5)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert 0.0 <= vec[1] <= 1.0
    assert vec[4] == pytest.approx(0.5)


def test_compute_stop_features_auto_vad_on_residual() -> None:
    """When VAD is omitted, residual speech should register on unexplained energy."""
    a, b, c = RNG.standard_normal((3, 4000))
    mix = a + b + c
    vad = VADAdapter(backend="energy")
    feats = compute_stop_features(
        mixture=mix,
        accepted_stems=a[None, :],
        candidate_stem=b,
        min_embedding_distance=1.0,
        vad=vad,
    )
    assert feats[0] > 0.05
    assert 0.0 <= feats[1] <= 1.0


def test_length_mismatch_raises() -> None:
    mix = RNG.standard_normal(100)
    with pytest.raises(ValueError, match="length"):
        compute_stop_features(
            mixture=mix,
            accepted_stems=np.zeros((1, 50)),
            candidate_stem=mix,
            vad_speech_prob=0.0,
            min_embedding_distance=1.0,
        )
