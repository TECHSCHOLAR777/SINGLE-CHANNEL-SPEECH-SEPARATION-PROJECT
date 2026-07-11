"""Demo smoke tests: mock engine always, Gradio interface when installed."""

import numpy as np
import pytest

from demo.app import MockEngine, run_separation


def test_mock_engine_returns_valid_schema() -> None:
    mix = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    result = MockEngine()(mix, 16000)
    assert result.num_streams == 3
    assert result.expert_used == "mock"
    assert result.streams.shape[1] == 16000


def test_run_separation_payload_shape() -> None:
    mix = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    payload = run_separation((16000, mix), MockEngine())
    badge, *slots_and_diag = payload
    assert "MOCK" in badge
    assert len(slots_and_diag) == 6  # 5 audio slots + diagnostics json
    assert slots_and_diag[0] is not None and slots_and_diag[3] is None


def test_run_separation_handles_no_audio() -> None:
    badge, *_ = run_separation(None, MockEngine())
    assert "Upload" in badge


def test_build_demo_constructs() -> None:
    pytest.importorskip("gradio")
    from demo.app import build_demo

    demo = build_demo(MockEngine())
    assert demo is not None
