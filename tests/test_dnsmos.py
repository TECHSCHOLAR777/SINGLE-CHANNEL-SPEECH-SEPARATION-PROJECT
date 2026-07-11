"""Tests for eval/dnsmos.py: availability gating, never fake numbers."""

import numpy as np
import pytest

from eval.dnsmos import DNSMOS_SAMPLE_RATE, DnsmosScorer
from utils.config import cfg_get, load_config

RNG = np.random.default_rng(seed=5)


def test_unconfigured_scorer_is_unavailable() -> None:
    assert DnsmosScorer().is_available is False
    assert DnsmosScorer(model_path=None).is_available is False


def test_missing_model_file_is_unavailable(tmp_path) -> None:
    assert DnsmosScorer(model_path=tmp_path / "nope.onnx").is_available is False


def test_score_raises_with_download_instructions_when_unavailable() -> None:
    wav = RNG.standard_normal(DNSMOS_SAMPLE_RATE)
    with pytest.raises(RuntimeError, match="model_path"):
        DnsmosScorer().score(wav, DNSMOS_SAMPLE_RATE)


def test_score_or_none_is_silent_and_honest() -> None:
    wav = RNG.standard_normal(DNSMOS_SAMPLE_RATE)
    assert DnsmosScorer().score_or_none(wav, DNSMOS_SAMPLE_RATE) is None


def test_wrong_sample_rate_rejected_when_available(tmp_path) -> None:
    model = tmp_path / "fake.onnx"
    model.write_bytes(b"placeholder")
    scorer = DnsmosScorer(model_path=model)
    assert scorer.is_available is True
    with pytest.raises(ValueError, match="16000"):
        scorer.score(RNG.standard_normal(8000), 8000)


def test_config_carries_dnsmos_key() -> None:
    cfg = load_config("configs/default.yaml", "configs/devc.yaml")
    assert cfg_get(cfg, "eval.dnsmos.model_path", "MISSING") is None
