"""Tests for SRCorrNetWrapper availability checks and API contract.

Does not require the checkpoint or sr_corrnet to be installed — all live
model calls are guarded by importorskip or mocked out.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from models.srcorrnet import SRCorrNetWrapper

# SRCorrNetExpert is a re-export of SRCorrNetWrapper for backwards compat
from models.experts.srcorrnet import SRCorrNetExpert


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def test_not_available_without_repo_or_package(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "sr_corrnet", raising=False)
    wrapper = SRCorrNetWrapper(device="cpu", repo_path=None)
    if importlib.util.find_spec("sr_corrnet") is None:
        assert not wrapper.is_available


def test_not_available_when_checkpoint_path_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    wrapper = SRCorrNetWrapper(
        device="cpu",
        repo_path=repo,
        checkpoint_path=tmp_path / "missing.pt",
    )
    assert not wrapper.is_available


def test_available_with_existing_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    wrapper = SRCorrNetWrapper(device="cpu", repo_path=repo, checkpoint_path=None)
    assert wrapper.is_available


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------

def test_default_hf_model_is_var_2_5spk() -> None:
    wrapper = SRCorrNetWrapper()
    assert "var-2-5spk" in wrapper.hf_model_id


def test_default_prob_thres() -> None:
    wrapper = SRCorrNetWrapper()
    assert wrapper.prob_thres == 0.5


def test_custom_prob_thres() -> None:
    wrapper = SRCorrNetWrapper(prob_thres=0.7)
    assert wrapper.prob_thres == 0.7


def test_sample_rate_is_8000() -> None:
    assert SRCorrNetWrapper.SAMPLE_RATE == 8000


def test_max_speakers_is_5() -> None:
    assert SRCorrNetWrapper.MAX_SPEAKERS == 5


# ---------------------------------------------------------------------------
# SRCorrNetExpert is the same class
# ---------------------------------------------------------------------------

def test_expert_alias_is_wrapper() -> None:
    assert SRCorrNetExpert is SRCorrNetWrapper


# ---------------------------------------------------------------------------
# Load calls from_pretrained with correct args
# ---------------------------------------------------------------------------

def _make_mock_sr_corrnet():
    fake = types.ModuleType("sr_corrnet")
    mock_infer = MagicMock()
    mock_base_nn = MagicMock()
    mock_infer.engine.model = mock_base_nn
    mock_base_nn.encoder = MagicMock()
    mock_base_nn.dec_block = [MagicMock() for _ in range(4)]
    mock_base_nn.spk_split.forward = MagicMock(return_value=None)
    mock_base_nn.forward = MagicMock(return_value=(None, None, None))
    fake.SSInference = MagicMock(return_value=mock_infer)  # type: ignore[attr-defined]
    return fake, mock_infer


def test_load_uses_checkpoint_path_not_config_for_hf_id(monkeypatch) -> None:
    """HF Hub model id must go through checkpoint_path, not config (BLUEPRINT §3.5)."""
    fake, mock_infer = _make_mock_sr_corrnet()
    monkeypatch.setitem(sys.modules, "sr_corrnet", fake)

    wrapper = SRCorrNetWrapper(hf_model_id="shinuh/fake-model")
    wrapper.load()

    fake.SSInference.from_pretrained.assert_called_once_with(
        checkpoint_path="shinuh/fake-model",
        device="cpu",
    )


def test_load_is_idempotent(monkeypatch) -> None:
    fake, mock_infer = _make_mock_sr_corrnet()
    monkeypatch.setitem(sys.modules, "sr_corrnet", fake)

    wrapper = SRCorrNetWrapper(hf_model_id="shinuh/fake")
    wrapper.load()
    wrapper.load()  # second call must not call from_pretrained again

    assert fake.SSInference.from_pretrained.call_count == 1


# ---------------------------------------------------------------------------
# forward() API
# ---------------------------------------------------------------------------

def _build_wrapper_with_mock(n_active: int = 2) -> tuple[SRCorrNetWrapper, types.ModuleType]:
    fake, mock_infer = _make_mock_sr_corrnet()
    probs = torch.zeros(1, 7)
    for i in range(1, n_active + 1):
        probs[0, i] = 0.9
    pres = {"probs": probs, "logits": torch.zeros(1, 7)}

    # model.forward must return (out, out_aux, pres) — matches Model.forward signature
    mock_infer.engine.model.forward = MagicMock(return_value=(None, None, pres))

    wrapper = SRCorrNetWrapper(device="cpu", hf_model_id="shinuh/fake")
    wrapper._inference = mock_infer
    wrapper._apply_patches()  # wraps base_nn.forward to cache pres

    # process_waveform side-effect: call patched base_nn.forward to seed _pres_cache
    waveforms = [torch.zeros(8000) for _ in range(n_active)]

    def _fake_process_waveform(wav, n_spks=None):
        mock_infer.engine.model.forward()  # fires Patch A wrapper -> pres_cache populated
        return {"waveforms": waveforms}

    mock_infer.process_waveform.side_effect = _fake_process_waveform

    return wrapper, fake


def test_forward_returns_expected_keys() -> None:
    wrapper, fake = _build_wrapper_with_mock()
    with patch.dict(sys.modules, {"sr_corrnet": fake}):
        result = wrapper.forward(torch.randn(1, 8000))
    for key in ("waveforms", "p_k", "n_active", "e0", "dec_stages"):
        assert key in result, f"key '{key}' missing from forward() result"


def test_forward_p_k_shape() -> None:
    wrapper, fake = _build_wrapper_with_mock(n_active=3)
    with patch.dict(sys.modules, {"sr_corrnet": fake}):
        result = wrapper.forward(torch.randn(1, 8000))
    assert result["p_k"] is not None
    assert result["p_k"].shape == (1, 7)


def test_forward_n_active_correct() -> None:
    for n in [2, 3, 4, 5]:
        wrapper, fake = _build_wrapper_with_mock(n_active=n)
        with patch.dict(sys.modules, {"sr_corrnet": fake}):
            result = wrapper.forward(torch.randn(1, 8000))
        assert result["n_active"] == n, f"n_active={result['n_active']}, expected {n}"


def test_forward_accepts_1d_wav() -> None:
    wrapper, fake = _build_wrapper_with_mock()
    with patch.dict(sys.modules, {"sr_corrnet": fake}):
        result = wrapper.forward(torch.randn(8000))  # 1-D, no batch dim
    assert "p_k" in result
