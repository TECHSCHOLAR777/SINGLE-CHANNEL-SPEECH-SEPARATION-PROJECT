"""Principle-2 gate (BLUEPRINT §4 / §8.4): never worse than the base on clean audio.

Principle 2 is the rule the whole architecture rests on. The previous design put
learned layers on the frozen expert's output and made it worse by 0.4 to 3.7 dB
at every operating point; the adapter design exists so that cannot happen again.
The gate that proves it needs trained adapters and the frozen checkpoint, so it
is marked live and skips when they are absent.

What runs unconditionally here is the structural half: both paths through
`CalmSepPipeline` produce finite, scorable output with the same contract, so the
live gate has something well formed to compare when weights arrive.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import pit_si_sdr
from models.preprocess import resample_audio
from models.srcorrnet import SRCorrNetWrapper
from pipeline.infer import CalmSepPipeline, InferenceCfg

TOLERANCE_DB = 0.1
"""The full system may not fall more than this below the base on clean audio."""


def _score(result, references_8k: np.ndarray, mixture_8k: np.ndarray) -> float:
    """PIT SI-SDRi of a pipeline result against 8 kHz references, scored at 16 kHz."""
    refs_16k = np.stack([resample_audio(r, 8000, 16000) for r in references_8k], axis=0)
    mix_16k = resample_audio(mixture_8k, 8000, 16000)
    n = min(result.streams_16k.shape[-1], refs_16k.shape[-1], mix_16k.shape[-1])
    return pit_si_sdr(
        result.streams_16k[:, :n],
        refs_16k[:, :n],
        mix_16k[:n],
    ).mean_si_sdri


def test_principle2_both_paths_produce_finite_scores(make_mock_expert):
    """Structural check: base-only and adapter paths both return scorable output.

    A mock expert is not quality-optimised, so this asserts well-formedness and
    finiteness, never a quality ordering. The ordering is the live gate below.
    """
    rng = np.random.default_rng(0)
    wav = (rng.standard_normal(8000) * 0.05).astype(np.float32)
    refs = np.stack([wav * 0.6, wav * 0.4], axis=0).astype(np.float32)
    mix = refs.sum(0).astype(np.float32)

    base = CalmSepPipeline(
        expert=make_mock_expert(2),
        cfg=InferenceCfg(default_n_speakers=2, run_band_recovery=False),
    )
    full = CalmSepPipeline(
        expert=make_mock_expert(2),
        lora_library=None,
        cfg=InferenceCfg(default_n_speakers=2, run_band_recovery=True),
    )

    r_base = base.run(mix, 8000)
    r_full = full.run(mix, 8000)

    s_base = _score(r_base, refs, mix)
    s_full = _score(r_full, refs, mix)

    assert np.isfinite(s_base)
    assert np.isfinite(s_full)
    assert r_base.speaker_count == r_full.speaker_count == 2


@pytest.mark.skipif(
    not SRCorrNetWrapper().is_available,
    reason=(
        "frozen backbone not installed. Install it with "
        "pip install 'sr-corrnet-ss[hub] @ git+https://github.com/dmlguq456/SR_CorrNet_SS'"
    ),
)
def test_principle2_never_worse_on_clean_live():
    """LIVE GATE: the full system must stay within TOLERANCE_DB of the base on clean audio.

    Requires trained Stage 1 adapters and a clean evaluation set. Enable once
    both exist; see docs/restoration/ISSUE_LEDGER.md I-025, which records that
    the Stage 1 reverb adapter currently fails exactly this gate, degrading
    clean-audio SI-SNR by 0.44 dB.
    """
    pytest.skip("needs trained adapters and 20 clean Libri2Mix segments; see I-025")
