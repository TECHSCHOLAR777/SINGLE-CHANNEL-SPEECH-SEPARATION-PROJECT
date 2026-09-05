"""Shared pytest fixtures.

The mock expert here replaces `MockCoralSepWrapper`, which lived in
`pipeline/infer.py` until the branch integration replaced that module with the
`CoralSepPipeline` implementation. A test double belongs in the test tree rather
than in the production package, so it was not restored to its old home.
"""

from __future__ import annotations

import numpy as np
import pytest

from coralsep.schemas.separation_result import SeparationResult


class MockExpert:
    """Weight-free stand-in for `SRCorrNetExpert`.

    Implements the only method `CoralSepPipeline` calls on its expert:
    `separate(waveform, sample_rate, n_spks) -> SeparationResult`. Streams are
    produced by band-splitting the mixture, so outputs are deterministic,
    distinct from one another, and sum to roughly the input. That is enough to
    exercise chunking, stitching, counting and band recovery without weights.
    """

    def __init__(self, n_speakers: int = 2, *, attractor_probs: np.ndarray | None = None) -> None:
        if n_speakers < 1:
            raise ValueError("n_speakers must be >= 1")
        self.n_speakers = n_speakers
        self._attractor_probs = attractor_probs
        self.calls: list[dict] = []

    def separate(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        n_spks: int | None = None,
    ) -> SeparationResult:
        self.calls.append({"sample_rate": sample_rate, "n_spks": n_spks})
        k = int(n_spks) if n_spks else self.n_speakers
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)

        # Deterministic, distinguishable streams: successive smoothing passes
        # give each stream a different spectral tilt without any randomness.
        streams = np.empty((k, wav.shape[0]), dtype=np.float32)
        current = wav
        for i in range(k):
            streams[i] = current / (i + 1.0)
            current = np.convolve(
                current, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same"
            )

        probs = self._attractor_probs
        if probs is None:
            probs = np.zeros(5, dtype=np.float32)
            probs[:k] = 0.9

        return SeparationResult(
            streams=streams,
            sample_rate=sample_rate,
            speaker_count=k,
            mixture=wav,
            expert_used="mock",
            attractor_probs=probs,
        )


@pytest.fixture
def mock_expert() -> MockExpert:
    """A two-speaker mock expert."""
    return MockExpert(n_speakers=2)


@pytest.fixture
def make_mock_expert():
    """Factory for tests that need more than one mock expert."""
    return MockExpert


@pytest.fixture
def two_tone_mixture() -> tuple[np.ndarray, int]:
    """Three seconds of two overlapping tones at 8 kHz, as a crude mixture."""
    sr = 8000
    t = np.arange(int(sr * 3.0), dtype=np.float32) / sr
    s1 = 0.3 * np.sin(2 * np.pi * 220 * t)
    s2 = 0.3 * np.sin(2 * np.pi * 440 * t)
    return (s1 + s2).astype(np.float32), sr


@pytest.fixture(scope="session")
def hub_network_errors() -> tuple[type[Exception], ...]:
    """Exception types that mean 'could not reach the model hub', not a code bug.

    Used by tests that load the real, live SR-CorrNet checkpoint (attractor_test.py
    TestPkCountAccuracy, e0_hook_test.py TestE0HookLive): their `sr_corrnet`-installed
    skipif guard does not catch a transient Hub rate limit or outage at load time,
    which previously failed CI outright (I-056) whenever huggingface.co rate-limited
    the shared GitHub Actions runner IP pool. A pytest fixture, not a plain importable
    function: `from conftest import ...` inside a test module is not reliably on
    sys.path across pytest's rootdir/pythonpath configurations (confirmed by a real
    CI failure, `ModuleNotFoundError: No module named 'conftest'`, that a purely local
    check did not catch because the class-level skipif never let the import line run
    locally either). Imported lazily so a huggingface_hub version without one of these
    names still degrades to a smaller, still-useful tuple rather than an ImportError.
    """
    errors: list[type[Exception]] = [ConnectionError, TimeoutError]
    try:
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        errors += [HfHubHTTPError, LocalEntryNotFoundError]
    except ImportError:
        pass
    return tuple(errors)
