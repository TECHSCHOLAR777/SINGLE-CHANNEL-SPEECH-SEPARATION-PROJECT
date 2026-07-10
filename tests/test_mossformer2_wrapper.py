"""Tests for MossFormer2 expert wrapper (mocked, no pretrained weights)."""

from unittest.mock import MagicMock, patch

import numpy as np

from models.experts.mossformer2 import MossFormer2Expert


def test_mossformer2_parse_clearvoice_output_3d() -> None:
    sr = 16000
    t = 8000
    # [spk, batch, length]
    output = np.random.randn(3, 1, t).astype(np.float32)
    streams, conf = MossFormer2Expert._parse_clearvoice_output(output, t)
    assert streams.shape == (3, t)
    assert len(conf) == 3


@patch("models.experts.mossformer2.attach_ecapa_embeddings", side_effect=lambda r, **kw: r)
@patch.object(MossFormer2Expert, "is_available", return_value=True)
def test_mossformer2_separate_shape(_avail: MagicMock, _emb: MagicMock) -> None:
    sr = 16000
    t = 8000
    mixture = np.random.randn(t).astype(np.float32)

    mock_cv = MagicMock(return_value=np.random.randn(3, 1, t).astype(np.float32))

    expert = MossFormer2Expert(device="cpu", compute_embeddings=False)
    expert._cv = mock_cv

    result = expert.separate(mixture, sample_rate=sr)

    assert result.streams.shape == (3, t)
    assert result.speaker_count == 3
    assert result.expert_used == "mossformer2"
    assert result.escalated is False
    assert result.mixture is not None
