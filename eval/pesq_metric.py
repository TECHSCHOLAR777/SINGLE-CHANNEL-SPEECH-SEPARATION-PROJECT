"""PESQ wrapper for reference-based perceptual quality (BLUEPRINT §9)."""

from __future__ import annotations

import numpy as np

from models.preprocess import resample_audio

PESQ_SR = 16000


def pesq_score(
    estimate: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Wideband PESQ when sample_rate allows; resamples to 16 kHz.

    Returns NaN if the ``pesq`` package is not installed (never invents scores).
    """
    try:
        from pesq import pesq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pesq package not installed. pip install pesq  (optional extra: calm-sep[eval])"
        ) from exc

    est = np.asarray(estimate, dtype=np.float32).reshape(-1)
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    if sample_rate != PESQ_SR:
        est = resample_audio(est, sample_rate, PESQ_SR)
        ref = resample_audio(ref, sample_rate, PESQ_SR)
    n = min(len(est), len(ref))
    if n < PESQ_SR // 4:
        raise ValueError("audio too short for PESQ")
    return float(pesq(PESQ_SR, ref[:n], est[:n], "wb"))


def pesq_or_none(
    estimate: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
) -> float | None:
    try:
        return pesq_score(estimate, reference, sample_rate)
    except Exception:
        return None
