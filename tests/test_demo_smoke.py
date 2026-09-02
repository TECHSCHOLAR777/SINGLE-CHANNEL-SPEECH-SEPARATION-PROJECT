"""Demo smoke tests: mock engine always, Gradio interface when installed."""

import numpy as np
import pytest

from coralsep.demo.cli import MockEngine, run_separation


def test_mock_engine_returns_valid_schema() -> None:
    mix = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    result = MockEngine()(mix, 16000)
    assert result.num_streams == 3
    assert result.expert_used == "mock"
    assert result.streams.shape[1] == 16000


def test_run_separation_payload_shape() -> None:
    mix = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    payload = run_separation((16000, mix), MockEngine())
    badge, *rest = payload
    assert "MOCK" in badge
    # 5 audio slots + gate_md + transcript_md + diagnostics json = 8
    assert len(rest) == 8
    assert rest[0] is not None   # first audio slot occupied
    assert rest[3] is None       # 4th slot empty (only 3 mock streams)


def test_run_separation_handles_no_audio() -> None:
    badge, *_ = run_separation(None, MockEngine())
    assert "Upload" in badge


def test_build_demo_constructs() -> None:
    pytest.importorskip("gradio")
    from coralsep.demo.cli import build_demo

    demo = build_demo(MockEngine())
    assert demo is not None
