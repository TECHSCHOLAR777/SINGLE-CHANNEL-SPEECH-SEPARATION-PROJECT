"""Tests for SR-CorrNet expert availability checks and output parsing."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from coralsep.models.experts.srcorrnet import SRCorrNetExpert, _extract_waveforms, _fix_length


def test_srcorrnet_not_available_without_repo_or_package(monkeypatch) -> None:
    # is_available falls back to importlib.util.find_spec("sr_corrnet") when no
    # repo_path is given, so this only exercises "not available" when the
    # package genuinely is not importable. Since I-019, sr_corrnet is a real
    # pinned pip dependency, so a plain repo_path=None expert is available
    # whenever the environment has it installed; simulate the uninstalled
    # case instead of asserting on whatever happens to be on this machine.
    import coralsep.models.experts.srcorrnet as mod

    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: None)
    expert = SRCorrNetExpert(device="cpu", repo_path=None)
    assert not expert.is_available


def test_srcorrnet_available_without_repo_when_package_installed(monkeypatch) -> None:
    import coralsep.models.experts.srcorrnet as mod

    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: object())
    expert = SRCorrNetExpert(device="cpu", repo_path=None)
    assert expert.is_available


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
    expert = SRCorrNetExpert(device="cpu")
    assert expert.hf_model_id.startswith("shinuh/sr-corrnet-ss")
    assert expert.model_sample_rate == 8000


def test_srcorrnet_loads_hf_model_via_checkpoint_path_not_config(monkeypatch) -> None:
    """
    Regression: SSInference.from_pretrained's `config` kwarg only accepts a
    *local* config name/path (it resolves to "SS/<value>.yaml" and raises
    FileNotFoundError for anything else). An HF Hub id like
    "shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk" must go through `checkpoint_path`
    instead, passing it as `config` failed every single sample on Kaggle.
    """
    mock_inference = MagicMock()
    fake_module = types.ModuleType("sr_corrnet")
    fake_module.SSInference = mock_inference  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sr_corrnet", fake_module)

    expert = SRCorrNetExpert(device="cpu", hf_model_id="shinuh/fake-model")
    monkeypatch.setattr(SRCorrNetExpert, "is_available", property(lambda self: True))
    expert._load_model()

    mock_inference.from_pretrained.assert_called_once_with(
        checkpoint_path="shinuh/fake-model", device="cpu"
    )


def test_inner_model_finds_the_real_two_level_nesting() -> None:
    """
    Regression for I-051: SSInference nests the separator two levels deep,
    SSInference -> engine (a plain object, not an nn.Module) -> model. The
    old _inner_model only checked one level (self._model.model / .net /
    .separator / ._model directly), which never matches this shape, so
    hasattr(model, "encoder") in _register_hooks always saw None and Patch B
    (E(0)) and Patch C (decoder features) never actually registered their
    hooks against a real SSInference object. Mirrors
    train/stage1_single.py::_get_inner_module, which already handles this.
    """
    real_inner = torch.nn.Linear(4, 4)

    class FakeEngine:
        def __init__(self, inner: torch.nn.Module) -> None:
            self.model = inner

    class FakeSSInference:
        def __init__(self, inner: torch.nn.Module) -> None:
            self.engine = FakeEngine(inner)

    expert = SRCorrNetExpert(device="cpu")
    expert._model = FakeSSInference(real_inner)  # type: ignore[assignment]

    assert expert._inner_model() is real_inner


def test_dec_hook_survives_a_stream_count_other_than_k0() -> None:
    """
    Regression for I-051: the decoder hook used to assume every call produces
    K0=5 streams (feat.shape[0] // K0), which crashed feat.view whenever the
    model was actually asked for a different count, since separate() calls
    process_waveform with a specific n_spks and the decoder emits exactly
    that many streams, not always K0. A crash inside a forward hook aborts
    the whole forward pass it is attached to, so this also checks the fix
    degrades to skipping the stage instead of raising.
    """

    class FakeDecoderModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Identity()
            self.dec_block = torch.nn.ModuleList([torch.nn.Identity()])

    class FakeEngine:
        def __init__(self, inner: torch.nn.Module) -> None:
            self.model = inner

    class FakeSSInference:
        def __init__(self, inner: torch.nn.Module) -> None:
            self.engine = FakeEngine(inner)

    inner = FakeDecoderModel()
    expert = SRCorrNetExpert(device="cpu")
    expert._model = FakeSSInference(inner)  # type: ignore[assignment]
    expert._register_hooks()

    # 2 streams (bk=2), not divisible by the old hardcoded K0=5.
    fake_output = torch.randn(2, 3, 4, 5)
    inner.dec_block[0](fake_output)  # runs the registered forward hook

    assert expert._dec_features[0].shape == (1, 2, 3, 4, 5)


def test_inner_model_still_finds_one_level_nesting() -> None:
    """The old one-level shape (self._model.model directly an nn.Module)
    must keep working, in case some SSInference build uses it."""
    real_inner = torch.nn.Linear(4, 4)

    class FakeSSInference:
        def __init__(self, inner: torch.nn.Module) -> None:
            self.model = inner

    expert = SRCorrNetExpert(device="cpu")
    expert._model = FakeSSInference(real_inner)  # type: ignore[assignment]

    assert expert._inner_model() is real_inner
