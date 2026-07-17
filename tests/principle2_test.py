"""Principle-2 smoke test (BLUEPRINT §4 / §8.4): never worse than base on clean.

Real comparison requires trained adapters + frozen checkpoint. Until then this
file asserts the API contract and skips the numerical gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import pit_si_sdr
from models.srcorrnet import SRCorrNetWrapper
from pipeline.infer import CalmSepEngine, MockCalmSepWrapper


TOLERANCE_DB = 0.1


def test_principle2_api_with_mock():
    """Structural check: base_only and full-path engines both return scores."""
    wav = np.random.randn(8000).astype(np.float32) * 0.05
    refs = np.stack([wav * 0.6, wav * 0.4], axis=0)
    mix = refs.sum(0)

    base = CalmSepEngine(wrapper=MockCalmSepWrapper(2), base_only=True)
    full = CalmSepEngine(wrapper=MockCalmSepWrapper(2), base_only=False, use_adapters=False)
    r_base = base(mix, 8000)
    r_full = full(mix, 8000)
    # Upsample refs for scoring at 16 kHz output
    from models.preprocess import resample_audio

    refs16 = np.stack([resample_audio(r, 8000, 16000) for r in refs], axis=0)
    n = min(r_base.streams.shape[-1], refs16.shape[-1])
    s_base = pit_si_sdr(r_base.streams[:, :n], refs16[:, :n], r_base.mixture[:n] if r_base.mixture is not None else mix)
    s_full = pit_si_sdr(r_full.streams[:, :n], refs16[:, :n], r_full.mixture[:n] if r_full.mixture is not None else mix)
    # Mock engines are not quality-optimized; only ensure finite scores.
    assert np.isfinite(s_base.mean_si_sdri)
    assert np.isfinite(s_full.mean_si_sdri)


@pytest.mark.skipif(
    not SRCorrNetWrapper().is_available,
    reason="frozen checkpoint not installed — run after training for real Principle-2 gate",
)
def test_principle2_never_worse_on_clean_live():
    """LIVE GATE (user runs after Stage 3): full system >= base - 0.1 dB on clean."""
    # Placeholder structure for the real Libri2Mix comparison notebook/script.
    pytest.skip("Populate with 20 clean Libri2Mix segments after Stage-3 training")
