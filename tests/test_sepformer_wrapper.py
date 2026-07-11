"""Tests for SepFormer expert wrapper (mocked, no pretrained weights)."""

from unittest.mock import MagicMock

import numpy as np
import torch

from models.experts.sepformer import SepFormerExpert


def test_sepformer_separate_shape() -> None:
    sr = 16000
    t = 8000
    mixture = np.random.randn(t).astype(np.float32)

    est = torch.randn(3, t)
    mock_model = MagicMock()
    mock_model.separate_batch.return_value = est.unsqueeze(0).transpose(1, 2)
    mock_model.eval.return_value = mock_model

    expert = SepFormerExpert(device="cpu")
    expert._model = mock_model  # inject mock, skip download

    result = expert.separate(mixture, sample_rate=sr)

    assert result.streams.shape == (3, t)
    assert result.speaker_count == 3
    assert result.expert_used == "sepformer"
    assert result.sample_rate == 16000
