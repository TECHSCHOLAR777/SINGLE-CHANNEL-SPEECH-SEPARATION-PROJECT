"""Phase P0 test for Patch B: E(0) forward hook on model.encoder (task P0-B3).

Gate M0 requires this file to pass alongside attractor_test.py.

Two test classes:
  TestE0Shape, always runs; mock confirms "e0" key is present and hook fires.
  TestE0HookLive: requires sr_corrnet + checkpoint; confirms exact shape
                   (1, T, 65, 128) with T = ceil(L / 64) for 8 kHz audio.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest
import torch

from coralsep.models.srcorrnet import SRCorrNetWrapper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _random_wav(duration_s: float = 4.0, sr: int = 8000) -> torch.Tensor:
    n = int(duration_s * sr)
    wav = torch.randn(1, n)
    return wav / (wav.std() + 1e-8)


def _build_wrapper_with_mock(e0_shape: tuple[int, ...] = (1, 63, 65, 128)) -> SRCorrNetWrapper:
    """Wrapper backed by a mock that fires the encoder hook."""
    fake_module = types.ModuleType("sr_corrnet")
    mock_infer = MagicMock()
    mock_base_nn = MagicMock()

    mock_infer.process_waveform.return_value = {"waveforms": [torch.zeros(8000)]}
    mock_infer.engine.model = mock_base_nn
    mock_base_nn.dec_block = [MagicMock() for _ in range(4)]
    mock_base_nn.spk_split.forward = MagicMock(return_value=None)
    mock_base_nn.forward = MagicMock(return_value=(None, None, None))

    # encoder: real Module so register_forward_hook works
    class _FakeEncoder(torch.nn.Module):
        def forward(self, *a, **kw):  # type: ignore[override]
            return torch.zeros(*e0_shape)

    real_encoder = _FakeEncoder()
    mock_base_nn.encoder = real_encoder

    fake_module.SSInference = MagicMock(return_value=mock_infer)  # type: ignore[attr-defined]

    wrapper = SRCorrNetWrapper(device="cpu", hf_model_id="shinuh/fake")
    wrapper._inference = mock_infer
    wrapper._apply_patches()  # registers hooks

    # Manually fire the encoder hook to populate _e0_cache
    real_encoder(torch.zeros(1, 2, 65, 4))

    return wrapper


# ---------------------------------------------------------------------------
# Shape tests, always run
# ---------------------------------------------------------------------------


class TestE0Shape:
    def test_e0_key_present(self) -> None:
        wrapper = _build_wrapper_with_mock()
        assert "e0" in wrapper._e0_cache, "e0 key missing, Patch B hook not firing"

    def test_e0_has_four_dims(self) -> None:
        wrapper = _build_wrapper_with_mock()
        e0 = wrapper._e0_cache.get("e0")
        assert e0 is not None
        assert e0.dim() == 4, f"e0 has {e0.dim()} dims, expected 4 (B, T, F, C)"

    def test_e0_freq_bins_65(self) -> None:
        wrapper = _build_wrapper_with_mock(e0_shape=(1, 63, 65, 128))
        e0 = wrapper._e0_cache["e0"]
        assert e0.shape[2] == 65, f"freq bins = {e0.shape[2]}, expected 65"

    def test_e0_channel_dim_128(self) -> None:
        wrapper = _build_wrapper_with_mock(e0_shape=(1, 63, 65, 128))
        e0 = wrapper._e0_cache["e0"]
        assert e0.shape[3] == 128, f"channel dim = {e0.shape[3]}, expected 128"


# ---------------------------------------------------------------------------
# Live tests: requires checkpoint
# ---------------------------------------------------------------------------

import importlib.util as _ilu  # noqa: E402

_SR_CORRNET_AVAILABLE = _ilu.find_spec("sr_corrnet") is not None


@pytest.mark.skipif(
    not _SR_CORRNET_AVAILABLE,
    reason="sr_corrnet not installed, skipping live E(0) hook test",
)
class TestE0HookLive:
    """Confirm exact E(0) shape on real checkpoint (BLUEPRINT §15.2)."""

    @pytest.fixture(scope="class")
    def wrapper(self, hub_network_errors: tuple[type[Exception], ...]) -> SRCorrNetWrapper:
        w = SRCorrNetWrapper(device="cpu")
        try:
            w.load()
        except hub_network_errors as exc:
            pytest.skip(f"could not reach the model hub to load the checkpoint: {exc}")
        return w

    def test_e0_not_none_after_forward(self, wrapper: SRCorrNetWrapper) -> None:
        result = wrapper.forward(_random_wav(duration_s=4.0))
        assert result["e0"] is not None, "e0 is None, Patch B hook not firing"

    def test_e0_shape_batch_1(self, wrapper: SRCorrNetWrapper) -> None:
        result = wrapper.forward(_random_wav(duration_s=4.0))
        assert result["e0"].shape[0] == 1

    def test_e0_shape_freq_65(self, wrapper: SRCorrNetWrapper) -> None:
        result = wrapper.forward(_random_wav(duration_s=4.0))
        assert result["e0"].shape[2] == 65, f"freq bins = {result['e0'].shape[2]}"

    def test_e0_shape_channel_128(self, wrapper: SRCorrNetWrapper) -> None:
        result = wrapper.forward(_random_wav(duration_s=4.0))
        assert result["e0"].shape[3] == 128, f"channel dim = {result['e0'].shape[3]}"

    def test_e0_time_matches_stft_frames(self, wrapper: SRCorrNetWrapper) -> None:
        duration_s = 4.0
        sr = 8000
        hop = 64
        L = int(duration_s * sr)
        # STFT frame count depends on center padding; approx L // hop
        result = wrapper.forward(_random_wav(duration_s=duration_s))
        T = result["e0"].shape[1]
        expected_approx = L // hop
        assert abs(T - expected_approx) <= 4, f"E(0) time frames T={T}, expected ~{expected_approx}"

    def test_dec_stages_have_4_entries(self, wrapper: SRCorrNetWrapper) -> None:
        result = wrapper.forward(_random_wav(duration_s=2.0))
        assert (
            len(result["dec_stages"]) == 4
        ), f"dec_stages has {len(result['dec_stages'])} entries, expected 4"
