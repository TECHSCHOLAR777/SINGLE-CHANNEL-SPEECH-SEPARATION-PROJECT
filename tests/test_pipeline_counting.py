"""Speaker-count readout in the inference pipeline (BLUEPRINT §7).

Regression coverage for two defects in the no-counter fallback path:
numpy attractor probabilities crashed it, and it counted the wrong slots.
"""

from __future__ import annotations

import numpy as np
import torch

from coralsep.pipeline.infer import CoralSepPipeline, InferenceCfg


def _pipeline() -> CoralSepPipeline:
    """A pipeline with no expert and no counter; only the readout is exercised."""
    return CoralSepPipeline(expert=None, cfg=InferenceCfg())


def _probs(active_speakers: int) -> np.ndarray:
    """A 7-slot attractor vector with `active_speakers` of slots 1..5 above threshold.

    Slot 0 and slot 6 are deliberately set high. They are not speaker slots, so
    a correct readout must ignore them.
    """
    p = np.zeros(7, dtype=np.float32)
    p[0] = 0.99
    p[6] = 0.99
    p[1 : 1 + active_speakers] = 0.9
    return p


def test_fallback_accepts_numpy_attractor_probs():
    """SeparationResult declares attractor_probs as numpy, so the readout must take it."""
    counts = _pipeline()._three_vote_count([_probs(3)], None, [])
    assert counts == 3


def test_fallback_accepts_torch_attractor_probs():
    probs = [torch.from_numpy(_probs(3))]
    assert _pipeline()._three_vote_count(probs, None, []) == 3


def test_fallback_ignores_the_non_speaker_slots():
    """Slots 0 and 6 are high in every fixture; counting them would inflate N by 2."""
    for n in (2, 3, 4, 5):
        assert _pipeline()._three_vote_count([_probs(n)], None, []) == n


def test_fallback_clips_below_two_speakers():
    assert _pipeline()._three_vote_count([_probs(1)], None, []) == 2


def test_fallback_takes_the_majority_across_chunks():
    chunks = [_probs(3), _probs(3), _probs(5)]
    assert _pipeline()._three_vote_count(chunks, None, []) == 3


def test_explicit_speaker_count_overrides_the_readout():
    pipe = CoralSepPipeline(expert=None, cfg=InferenceCfg(default_n_speakers=4))
    assert pipe._three_vote_count([_probs(2)], None, []) == 4
