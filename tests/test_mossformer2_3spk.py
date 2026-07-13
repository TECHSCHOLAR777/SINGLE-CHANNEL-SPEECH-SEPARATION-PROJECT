"""Tests for the MossFormer2 3-speaker wrapper (mocked — no weights/network)."""

from __future__ import annotations

import numpy as np
import torch

from models.experts.mossformer2_3spk import (
    DEFAULT_HF_MODEL,
    MossFormer2ThreeSpkExpert,
    _clean_state,
    _fix_length,
)


def test_default_config() -> None:
    expert = MossFormer2ThreeSpkExpert(device="cpu")
    assert expert.hf_model_id == DEFAULT_HF_MODEL
    assert expert.model_sample_rate == 8000
    assert expert.MAX_SPEAKERS == 3
    assert expert.compute_embeddings is True


def test_fix_length_crops_and_pads() -> None:
    assert _fix_length(np.ones((3, 100), np.float32), 80).shape == (3, 80)
    assert _fix_length(np.ones((3, 50), np.float32), 80).shape == (3, 80)


def test_clean_state_strips_prefixes() -> None:
    raw = {"model.enc.weight": 1, "module.dec.bias": 2, "mask_net.x": 3}
    cleaned = _clean_state(raw)
    assert set(cleaned) == {"enc.weight", "dec.bias", "mask_net.x"}
    # nested state_dict form
    assert _clean_state({"state_dict": {"model.a": 9}}) == {"a": 9}


def test_separate_shape_with_mocked_model(monkeypatch) -> None:
    sr = 16000
    t = 16000
    mixture = np.random.randn(t).astype(np.float32)

    # compute_embeddings=False so the shape test stays offline (no ECAPA load).
    expert = MossFormer2ThreeSpkExpert(device="cpu", compute_embeddings=False)

    # Mock the loaded model: returns a list of 3 streams at the 8 kHz model rate.
    def fake_forward(wav_t):
        n = wav_t.shape[-1]
        return [torch.randn(1, n) for _ in range(3)]

    expert._model = fake_forward  # bypass _load_model
    monkeypatch.setattr(expert, "_load_model", lambda: None)

    result = expert.separate(mixture, sample_rate=sr)
    assert result.streams.shape == (3, t)  # resampled back to 16 kHz, length matched
    assert result.speaker_count == 3
    assert result.expert_used == "mossformer2_3spk"
    assert result.sample_rate == 16000
