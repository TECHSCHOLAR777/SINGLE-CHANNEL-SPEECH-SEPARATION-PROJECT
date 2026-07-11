"""
Phase 1 expert integration test (P1-B6).

Runs MossFormer2 (mock) and SR-CorrNet/SepFormer (mock) on the same synthetic
3-speaker clip, aligns outputs via Hungarian matching, and verifies REAL-M
quality estimation. No pretrained weight downloads required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from align.hungarian import align_results, reorder_result
from models.experts.mossformer2 import MossFormer2Expert
from models.experts.sepformer import SepFormerExpert
from models.realm_quality import REALMQualityEstimator
from schemas.separation_result import SeparationResult, StreamMetadata

RNG = np.random.default_rng(seed=42)


def _synthetic_3speaker_mixture(t: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    """Build a 3-speaker mixture with distinct sinusoids for reliable xcorr alignment."""
    sr = 16000
    time = np.arange(t, dtype=np.float32) / sr
    refs = np.stack(
        [
            np.sin(2 * np.pi * 300 * time),
            np.sin(2 * np.pi * 500 * time),
            np.sin(2 * np.pi * 700 * time),
        ],
        axis=0,
    ).astype(np.float32)
    mixture = refs.sum(axis=0)
    return mixture, refs


@patch("models.experts.mossformer2.attach_ecapa_embeddings", side_effect=lambda r, **kw: r)
@patch.object(MossFormer2Expert, "is_available", return_value=True)
def test_expert_integration_align_and_quality(_avail: MagicMock, _emb: MagicMock) -> None:
    """Both experts on same clip → aligned streams + REAL-M scores."""
    t = 8000
    mixture, refs = _synthetic_3speaker_mixture(t)

    # Mock MossFormer2: return refs in order
    moss_cv = MagicMock(return_value=refs[:, np.newaxis, :])
    moss = MossFormer2Expert(device="cpu", compute_embeddings=False)
    moss._cv = moss_cv
    cheap = moss.separate(mixture, sample_rate=16000)

    # Mock expensive expert via SepFormer with permuted outputs
    perm = [1, 2, 0]
    expensive_streams = refs[perm]
    mock_sep = MagicMock()
    mock_sep.separate_batch.return_value = torch.from_numpy(
        expensive_streams.T[np.newaxis, :, :]
    )
    mock_sep.eval.return_value = mock_sep
    sep = SepFormerExpert(device="cpu")
    sep._model = mock_sep
    expensive = sep.separate(mixture, sample_rate=16000)
    expensive = SeparationResult(
        streams=expensive.streams,
        sample_rate=expensive.sample_rate,
        speaker_count=expensive.speaker_count,
        metadata=[
            StreamMetadata(
                expert_source="srcorrnet",
                confidence=0.9,
                extra={"attractor_index": i, "attractor": RNG.standard_normal(16)},
            )
            for i in range(expensive.num_streams)
        ],
        mixture=expensive.mixture,
        escalated=True,
        expert_used="srcorrnet",
    )

    # Hungarian alignment via waveform xcorr (no embeddings required)
    alignment = align_results(cheap, expensive)
    assert alignment.method == "xcorr"
    aligned_expensive = reorder_result(expensive, alignment)
    assert aligned_expensive.streams.shape == cheap.streams.shape
    assert np.allclose(aligned_expensive.streams, cheap.streams, atol=1e-5)

    # REAL-M quality on cheap output (mocked)
    mock_snr = MagicMock()
    raw = __import__("torch").tensor([0.18, 0.19, 0.17])
    mock_snr.estimate_batch.return_value = raw
    mock_snr.gettrue_snrrange.return_value = raw * 10.0
    realm = REALMQualityEstimator(device="cpu")
    realm._model = mock_snr
    quality = realm.estimate_result(cheap)
    assert len(quality.sisnr_db_per_stream) == 3
    assert quality.mean_sisnr_db > 0

    # Attractor metadata present on expensive expert
    assert all("attractor" in m.extra for m in expensive.metadata)
