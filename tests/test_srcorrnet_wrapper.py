"""Tests for SR-CorrNet expert availability checks and output parsing."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from models.experts.srcorrnet import SRCorrNetExpert, _extract_waveforms, _fix_length


def test_srcorrnet_not_available_without_repo() -> None:
    expert = SRCorrNetExpert(device="cpu", repo_path=None)
    assert not expert.is_available


def test_srcorrnet_not_available_missing_checkpoint(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    expert = SRCorrNetExpert(
        device="cpu",
        repo_path=repo,
        checkpoint_path=tmp_path / "missing.pt",
    )
    assert not expert.is_available


def test_srcorrnet_available_with_repo_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    expert = SRCorrNetExpert(device="cpu", repo_path=repo, checkpoint_path=None)
    assert expert.is_available


def test_extract_waveforms_from_dict_of_list() -> None:
    # SSInference.process_waveform returns {"waveforms": [1-D tensor per spk], ...}
    t = 4000
    k = 3
    out = {
        "waveforms": [torch.randn(t) for _ in range(k)],
        "vad": None,
        "doa": None,
    }
    streams = _extract_waveforms(out)
    assert streams.shape == (k, t)
    assert streams.dtype == np.float32


def test_extract_waveforms_from_tensor() -> None:
    t = 2000
    k = 2
    streams = _extract_waveforms(torch.randn(1, k, t))
    assert streams.shape == (k, t)


def test_fix_length_crops_and_pads() -> None:
    assert _fix_length(np.ones((3, 100), dtype=np.float32), 80).shape == (3, 80)
    assert _fix_length(np.ones((3, 50), dtype=np.float32), 80).shape == (3, 80)


def test_srcorrnet_default_hf_model() -> None:
    expert = SRCorrNetExpert(device="cpu", num_speakers=3)
    assert expert.hf_model_id.startswith("shinuh/sr-corrnet-ss")
    assert expert.model_sample_rate == 8000


def test_srcorrnet_loads_hf_model_via_checkpoint_path_not_config(monkeypatch) -> None:
    """
    Regression: SSInference.from_pretrained's `config` kwarg only accepts a
    *local* config name/path (it resolves to "SS/<value>.yaml" and raises
    FileNotFoundError for anything else). An HF Hub id like
    "shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk" must go through `checkpoint_path`
    instead — passing it as `config` failed every single sample on Kaggle.
    """
    mock_inference = MagicMock()
    fake_module = types.ModuleType("sr_corrnet")
    fake_module.SSInference = mock_inference  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sr_corrnet", fake_module)

    expert = SRCorrNetExpert(device="cpu", num_speakers=3, hf_model_id="shinuh/fake-model")
    monkeypatch.setattr(SRCorrNetExpert, "is_available", property(lambda self: True))
    expert._load_model()

    mock_inference.from_pretrained.assert_called_once_with(
        checkpoint_path="shinuh/fake-model", device="cpu"
    )
