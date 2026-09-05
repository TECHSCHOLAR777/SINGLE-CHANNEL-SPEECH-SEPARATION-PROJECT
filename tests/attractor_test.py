"""BLOCKING Phase P0 gate test (BLUEPRINT §15.10, task P0-B6).

Gate M0 requires this file to pass before any Phase P1 work starts.

Two test classes:
  TestPkShape, always runs; uses a mock to confirm the wrapper API contract
                without needing the checkpoint.
  TestPkCountAccuracy: requires sr_corrnet installed AND checkpoint downloaded;
                        skipped otherwise. Confirms wrapper loads and returns
                        p_k with the correct shape on real audio fixtures.

To run the full gate test after downloading the checkpoint:
    pytest tests/attractor_test.py -v
"""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import pytest
import torch

from coralsep.models.srcorrnet import SRCorrNetWrapper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _random_wav(duration_s: float = 4.0, sr: int = 8000) -> torch.Tensor:
    """Random noise at 8 kHz, shape (1, L), std-normalized."""
    n = int(duration_s * sr)
    wav = torch.randn(1, n)
    return wav / (wav.std() + 1e-8)


def _make_fake_pres(n_active: int) -> dict:
    """Fake pres dict with n_active slots set above 0.5 (slots 1..n_active)."""
    probs = torch.zeros(1, 7)
    for i in range(1, n_active + 1):
        probs[0, i] = 0.9
    return {"probs": probs, "logits": torch.zeros(1, 7)}


def _build_wrapper_with_mock(n_active: int) -> SRCorrNetWrapper:
    """Wrapper backed by a mock that seeds _pres_cache via patched forward."""
    pres = _make_fake_pres(n_active)
    mock_infer = MagicMock()
    mock_base_nn = MagicMock()

    mock_infer.engine.model = mock_base_nn
    mock_base_nn.encoder = MagicMock()
    mock_base_nn.dec_block = [MagicMock() for _ in range(4)]
    mock_base_nn.spk_split.forward = MagicMock(return_value=None)
    mock_base_nn.forward = MagicMock(return_value=(None, None, pres))

    wrapper = SRCorrNetWrapper(device="cpu", hf_model_id="shinuh/fake")
    wrapper._inference = mock_infer
    wrapper._apply_patches()  # Patch A wraps base_nn.forward to cache pres

    # Fire the patched forward once to seed _pres_cache before test assertions
    mock_infer.engine.model.forward()

    # process_waveform: side-effect fires patched forward to keep cache current
    waveforms = [torch.zeros(8000) for _ in range(n_active)]

    def _fake_process(wav, n_spks=None):
        mock_infer.engine.model.forward()
        return {"waveforms": waveforms}

    mock_infer.process_waveform.side_effect = _fake_process

    return wrapper


# ---------------------------------------------------------------------------
# Shape tests, always run, no checkpoint needed
# ---------------------------------------------------------------------------


class TestPkShape:
    """Confirm wrapper API contract using mocks."""

    def test_wrapper_returns_pk_key(self) -> None:
        wrapper = _build_wrapper_with_mock(n_active=2)
        result = wrapper.forward(_random_wav())
        assert "p_k" in result, "forward() must return p_k key"

    def test_pk_shape_is_1_7(self) -> None:
        wrapper = _build_wrapper_with_mock(n_active=3)
        result = wrapper.forward(_random_wav())
        p_k = result["p_k"]
        assert p_k is not None, "p_k is None, Patch A not applied"
        assert p_k.shape == (1, 7), f"p_k shape {p_k.shape} != (1, 7)"

    def test_n_active_matches_fake_pres(self) -> None:
        for n_true in [2, 3, 4, 5]:
            wrapper = _build_wrapper_with_mock(n_active=n_true)
            result = wrapper.forward(_random_wav())
            assert result["n_active"] == n_true, f"n_active={result['n_active']} != {n_true}"

    def test_result_has_e0_and_dec_stages_keys(self) -> None:
        wrapper = _build_wrapper_with_mock(n_active=2)
        result = wrapper.forward(_random_wav())
        assert "e0" in result
        assert "dec_stages" in result

    def test_prob_thres_stored(self) -> None:
        wrapper = SRCorrNetWrapper(prob_thres=0.7)
        assert wrapper.prob_thres == 0.7

    def test_default_hf_model_is_var_2_5spk(self) -> None:
        wrapper = SRCorrNetWrapper()
        assert "var-2-5spk" in wrapper.hf_model_id

    def test_is_available_false_without_package(self) -> None:
        wrapper = SRCorrNetWrapper(device="cpu", repo_path=None)
        if importlib.util.find_spec("sr_corrnet") is None:
            assert not wrapper.is_available


# ---------------------------------------------------------------------------
# Count accuracy test: requires checkpoint
# ---------------------------------------------------------------------------

_SR_CORRNET_AVAILABLE = importlib.util.find_spec("sr_corrnet") is not None


@pytest.mark.skipif(
    not _SR_CORRNET_AVAILABLE,
    reason="sr_corrnet not installed, skipping live attractor count test",
)
class TestPkCountAccuracy:
    """Gate M0 accuracy test. Requires checkpoint download (P0-B1).

    Uses 4-second white-noise fixtures. Shape and non-null checks always pass;
    count accuracy on noise is not asserted (requires LibriMix audio).
    Run scripts/corpus_transfer_baseline.py for count accuracy on real data.
    """

    @pytest.fixture(scope="class")
    def wrapper(self) -> SRCorrNetWrapper:
        from conftest import hub_network_errors

        w = SRCorrNetWrapper(device="cpu")
        try:
            w.load()
        except hub_network_errors() as exc:
            pytest.skip(f"could not reach the model hub to load the checkpoint: {exc}")
        return w

    def test_pk_not_none_after_forward(self, wrapper: SRCorrNetWrapper) -> None:
        result = wrapper.forward(_random_wav())
        assert result["p_k"] is not None, "Patch A failed, pres not cached"

    def test_pk_shape(self, wrapper: SRCorrNetWrapper) -> None:
        result = wrapper.forward(_random_wav())
        assert result["p_k"].shape == (1, 7), f"shape={result['p_k'].shape}"

    def test_n_active_in_range(self, wrapper: SRCorrNetWrapper) -> None:
        result = wrapper.forward(_random_wav())
        n = result["n_active"]
        assert n is not None
        assert 0 <= n <= 5, f"n_active={n} outside [0, 5]"
