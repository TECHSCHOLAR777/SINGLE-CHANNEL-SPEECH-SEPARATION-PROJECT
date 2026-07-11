"""Optional M1 acceptance tests with real models and a real Libri3Mix clip.

Skipped in normal CI. Run on a GPU host with:

    RUN_REAL_EXPERTS=1 LIBRIMIX_ROOT=/workspace/Libri3Mix \
    python -m pytest -q -s tests/test_m1_real_experts.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from align.hungarian import xcorr_cost_matrix
from align.integration import run_and_align, run_and_align_long
from data.mixer_stub import discover_librimix_samples
from models.experts.mossformer2 import MossFormer2Expert
from models.experts.tfgridnet import get_expensive_expert

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_EXPERTS") != "1",
    reason="set RUN_REAL_EXPERTS=1 on a host with model weights and Libri3Mix",
)


def _sample():
    root = os.environ.get("LIBRIMIX_ROOT")
    if not root:
        pytest.skip("LIBRIMIX_ROOT is not set")
    samples = discover_librimix_samples(root, subset="test", max_samples=1)
    if not samples:
        pytest.skip(f"no Libri3Mix test samples found under {root}")
    sample = samples[0]
    if sample.mixture.shape[0] <= 4 * sample.sample_rate:
        pytest.skip("the selected Libri3Mix clip is not longer than four seconds")
    return sample


def _cheap_engine() -> MossFormer2Expert:
    """Load only the cheap expert used by the long-form acceptance check."""
    return MossFormer2Expert(
        device=os.getenv("DEVICE", "cuda"),
        compute_embeddings=True,
    )


def _engines():
    device = os.getenv("DEVICE", "cuda")
    cheap = _cheap_engine()
    expensive = get_expensive_expert(
        device=device,
        srcorrnet_repo=os.getenv("SRCORRNET_REPO"),
        srcorrnet_checkpoint=os.getenv("SRCORRNET_CHECKPOINT"),
        tfgridnet_tag=os.getenv("TFGRIDNET_TAG"),
        num_speakers=3,
    )
    return cheap, expensive


def _persistent_assignments(
    tracks: np.ndarray,
    references: np.ndarray,
    sample_rate: int,
    chunk_sec: float,
    overlap_sec: float,
) -> list[tuple[int, ...]]:
    chunk = int(round(chunk_sec * sample_rate))
    hop = int(round((chunk_sec - overlap_sec) * sample_rate))
    assignments: list[tuple[int, ...]] = []
    for start in range(0, tracks.shape[1], hop):
        stop = min(start + chunk, tracks.shape[1])
        if stop - start < sample_rate:
            break
        cost = xcorr_cost_matrix(tracks[:, start:stop], references[:, start:stop])
        rows, cols = linear_sum_assignment(cost)
        mapping = [-1] * tracks.shape[0]
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            mapping[row] = col
        assignments.append(tuple(mapping))
        if stop == tracks.shape[1]:
            break
    return assignments


def test_p1_int1_real_experts_align_on_same_clip() -> None:
    sample = _sample()
    cheap, expensive = _engines()
    clip = sample.mixture[: 4 * sample.sample_rate]
    paired = run_and_align(cheap, expensive, clip, sample.sample_rate)

    assert paired.anchor.num_streams >= 2
    assert paired.aligned.num_streams >= 2
    assert paired.alignment.method == "embedding"
    assert np.isfinite(paired.mean_matched_cost)
    assert paired.mean_matched_cost < 1.0


def test_p1_int2_real_long_clip_has_no_identity_switches() -> None:
    sample = _sample()
    cheap = _cheap_engine()
    output = run_and_align_long(
        cheap,
        sample.mixture,
        sample.sample_rate,
        chunk_sec=4.0,
        overlap_sec=1.0,
        match_threshold=0.35,
        ema=0.7,
    )
    assignments = _persistent_assignments(
        output.result.streams,
        sample.references,
        output.result.sample_rate,
        chunk_sec=4.0,
        overlap_sec=1.0,
    )

    assert output.num_chunks >= 2
    assert output.result.num_streams == sample.references.shape[0]
    assert len(assignments) >= 2
    assert all(reference >= 0 for reference in assignments[0])
    first = assignments[0]
    assert all(mapping == first for mapping in assignments[1:]), assignments
