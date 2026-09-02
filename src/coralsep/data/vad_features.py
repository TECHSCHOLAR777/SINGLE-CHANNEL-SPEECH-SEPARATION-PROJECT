"""
Voiced-frame density for CoRAL-Sep Level-1 condition analysis (BLUEPRINT §5.4).

Primary path: SileroVAD at 8 kHz (pre-trained, not re-trained).
Fallback: voiced-energy-fraction from the STFT when Silero is unavailable.

Used as a proxy for overlap density and as a count-prior cross-check. Validated
on LibriCSS overlap subsets via validate_vad_proxy().
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

CORALSEP_SR: int = 8_000
"""Operating rate for Level-1 VAD features."""

_STFT_WINDOW: int = 128
_STFT_HOP: int = 64

_silero_model = None
_silero_utils = None
_silero_load_failed = False


def _silero_enabled() -> bool:
    """Silero is opt-in via CALMSEP_ENABLE_SILERO to keep CPU smoke tests torch-free."""
    return os.environ.get("CALMSEP_ENABLE_SILERO", "").lower() in ("1", "true", "yes")


def _load_silero():
    """Lazy-load Silero VAD; returns (model, utils) or (None, None)."""
    global _silero_model, _silero_utils, _silero_load_failed
    if _silero_load_failed or not _silero_enabled():
        return None, None
    if _silero_model is not None:
        return _silero_model, _silero_utils
    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _silero_model = model
        _silero_utils = utils
        return model, utils
    except Exception:
        _silero_load_failed = True
        return None, None


def _stft_voiced_energy_fraction(waveform: np.ndarray, sr: int) -> float:
    """
    Fallback voiced-energy fraction from STFT frame energy.

    Frames above the 30th percentile of log-energy are treated as voiced.
    """
    wav = np.asarray(waveform, dtype=np.float64).ravel()
    if wav.size == 0:
        return 0.0

    n_frames = max(1, 1 + (wav.size - _STFT_WINDOW) // _STFT_HOP)
    energies: list[float] = []
    for i in range(n_frames):
        start = i * _STFT_HOP
        frame = wav[start : start + _STFT_WINDOW]
        if frame.size < _STFT_WINDOW:
            frame = np.pad(frame, (0, _STFT_WINDOW - frame.size))
        energies.append(float(np.sum(frame**2) + 1e-10))

    arr = np.asarray(energies)
    if float(arr.max()) < 1e-12:
        return 0.0

    log_e = np.log10(arr)
    if float(log_e.std()) < 1e-6:
        return 0.0

    threshold = float(np.percentile(log_e, 30))
    voiced = log_e >= threshold
    return float(np.mean(voiced))


def _silero_voiced_density(waveform: np.ndarray, sr: int) -> float | None:
    """SileroVAD voiced-frame density; None when Silero cannot load."""
    model, utils = _load_silero()
    if model is None or utils is None:
        return None

    import torch

    (get_speech_timestamps, _, read_audio, *_) = utils

    wav = np.asarray(waveform, dtype=np.float32).ravel()
    if sr != CORALSEP_SR:
        # Silero expects 8 kHz or 16 kHz; resample linearly if needed.
        duration = wav.size / sr
        n_out = int(round(duration * CORALSEP_SR))
        x_old = np.linspace(0.0, duration, num=wav.size, endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
        sr = CORALSEP_SR

    tensor = torch.from_numpy(wav)
    timestamps = get_speech_timestamps(tensor, model, sampling_rate=sr)

    if wav.size == 0:
        return 0.0

    voiced_samples = sum(int(ts["end"]) - int(ts["start"]) for ts in timestamps)
    return float(np.clip(voiced_samples / wav.size, 0.0, 1.0))


def voiced_frame_density(waveform: np.ndarray, sr: int = CORALSEP_SR) -> float:
    """
    Fraction of the signal flagged as voiced (SileroVAD or STFT fallback).

    Args:
        waveform: Mono waveform, shape [T].
        sr: Sample rate in Hz; default 8000 (CoRAL-Sep operating rate).

    Returns:
        Voiced fraction in [0, 1]. Higher values indicate more active speech.
    """
    wav = np.asarray(waveform, dtype=np.float64).ravel()
    if wav.size == 0:
        return 0.0

    silero_val = _silero_voiced_density(wav, sr)
    if silero_val is not None:
        return silero_val

    return _stft_voiced_energy_fraction(wav, sr)


def silero_available() -> bool:
    """True when SileroVAD can be loaded."""
    model, _ = _load_silero()
    return model is not None


VadBackend = Literal["silero", "stft_fallback"]


@dataclass
class VadProxyValidation:
    """Result of validate_vad_proxy on a LibriCSS-style overlap subset."""

    n_items: int
    backend: VadBackend
    mean_density: float
    std_density: float
    min_density: float
    max_density: float
    discriminative: bool
    note: str


def validate_vad_proxy(
    waveforms: list[np.ndarray],
    overlap_labels: list[float] | None = None,
    sr: int = CORALSEP_SR,
    min_spread: float = 0.05,
) -> VadProxyValidation:
    """
    Check whether voiced-frame density separates overlap conditions (LibriCSS note).

    When overlap_labels are provided (fraction of overlap per mixture), the proxy
    is considered discriminative if Pearson correlation magnitude exceeds
    ``min_spread`` or the density range exceeds ``min_spread``.

    Args:
        waveforms: List of mono mixture waveforms.
        overlap_labels: Optional overlap ratios in [0, 1] per waveform.
        sr: Sample rate.
        min_spread: Minimum range or |correlation| to pass.

    Returns:
        VadProxyValidation summary.
    """
    if not waveforms:
        raise ValueError("waveforms must be non-empty")

    densities = [voiced_frame_density(w, sr) for w in waveforms]
    backend: VadBackend = "silero" if silero_available() else "stft_fallback"

    arr = np.asarray(densities, dtype=np.float64)
    discriminative = float(arr.max() - arr.min()) >= min_spread

    note = "SileroVAD active" if backend == "silero" else "STFT energy fallback (Silero unavailable)"

    if overlap_labels is not None and len(overlap_labels) == len(waveforms):
        labels = np.asarray(overlap_labels, dtype=np.float64)
        if labels.std() > 0 and arr.std() > 0:
            corr = float(np.corrcoef(labels, arr)[0, 1])
            discriminative = discriminative or abs(corr) >= min_spread
            note = f"{note}; overlap correlation={corr:.3f}"

    return VadProxyValidation(
        n_items=len(waveforms),
        backend=backend,
        mean_density=float(arr.mean()),
        std_density=float(arr.std()),
        min_density=float(arr.min()),
        max_density=float(arr.max()),
        discriminative=discriminative,
        note=note,
    )

