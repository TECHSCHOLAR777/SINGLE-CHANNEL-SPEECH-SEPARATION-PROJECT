"""
DNSMOS reference-free quality scoring (Dev C / shared eval, BLUEPRINT §2.4).

When ``sig_bak_ovrl.onnx`` is configured and onnxruntime is installed, runs real
P.835 inference: ~9 s segments with hop, polynomial MOS mapping, mean SIG/BAK/OVRL.

When the model file is absent, every call is availability-gated and fails loudly.
When the model is present but onnxruntime is missing, raises a clear install error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DNSMOS_SAMPLE_RATE = 16000
"""DNSMOS P.835 operates on 16 kHz input."""

INPUT_LENGTH_SAMPLES = int(9.01 * DNSMOS_SAMPLE_RATE)
"""Segment length (~9.01 s at 16 kHz), matching Microsoft DNSMOS reference."""

SEGMENT_HOP_SAMPLES = int(4.5 * DNSMOS_SAMPLE_RATE)
"""Hop between segments (~4.5 s overlap)."""

_DOWNLOAD_HINT = (
    "DNSMOS model not configured. Download sig_bak_ovrl.onnx from the "
    "microsoft/DNS-Challenge repository (DNSMOS/DNSMOS directory), then set "
    "eval.dnsmos.model_path in configs/eval.yaml."
)

_ONNX_MISSING_HINT = (
    "DNSMOS model is configured but onnxruntime is not installed. "
    "Install with: pip install onnxruntime"
)

# Polynomial coefficients from Microsoft DNSMOS reference implementation.
_POLY_SIG = np.array([-0.06766283, 1.11546468, 0.04602535])
_POLY_BAK = np.array([-0.08397278, 1.16009577, 0.04077506])
_POLY_OVR = np.array([-0.06766283, 1.11546468, 0.04602535])


def _poly_map(raw: float, coeffs: np.ndarray) -> float:
    """Map raw DNSMOS model output to MOS scale via polynomial."""
    return float(np.polyval(coeffs, raw))


def _segment_audio(waveform: np.ndarray) -> list[np.ndarray]:
    """Split waveform into ~9 s segments with hop."""
    wav = np.asarray(waveform, dtype=np.float32).ravel()
    if wav.size == 0:
        return [np.zeros(INPUT_LENGTH_SAMPLES, dtype=np.float32)]

    if wav.size <= INPUT_LENGTH_SAMPLES:
        padded = np.zeros(INPUT_LENGTH_SAMPLES, dtype=np.float32)
        padded[: wav.size] = wav
        return [padded]

    segments: list[np.ndarray] = []
    start = 0
    while start + INPUT_LENGTH_SAMPLES <= wav.size:
        segments.append(wav[start : start + INPUT_LENGTH_SAMPLES].copy())
        start += SEGMENT_HOP_SAMPLES
        if start + INPUT_LENGTH_SAMPLES > wav.size and start < wav.size:
            # Final partial segment: take tail aligned to end.
            segments.append(wav[-INPUT_LENGTH_SAMPLES:].copy())
            break

    return segments if segments else [wav[:INPUT_LENGTH_SAMPLES]]


class DnsmosScorer:
    """
    Reference-free quality scorer, availability-gated.

    Args:
        model_path: Path to the DNSMOS P.835 ONNX file, or None (unavailable).
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._session = None

    @property
    def is_available(self) -> bool:
        """True when a model file is configured and exists on disk."""
        return self.model_path is not None and self.model_path.exists()

    def _ensure_session(self):
        """Lazy-create ONNX Runtime session."""
        if self._session is not None:
            return self._session
        if not self.is_available:
            raise RuntimeError(_DOWNLOAD_HINT)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(_ONNX_MISSING_HINT) from exc

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        return self._session

    def _run_segment(self, segment: np.ndarray) -> tuple[float, float, float]:
        session = self._ensure_session()
        inp_name = session.get_inputs()[0].name
        audio = segment.astype(np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        outputs = session.run(None, {inp_name: audio})
        raw_sig, raw_bak, raw_ovr = (float(outputs[i].squeeze()) for i in range(3))
        return (
            _poly_map(raw_sig, _POLY_SIG),
            _poly_map(raw_bak, _POLY_BAK),
            _poly_map(raw_ovr, _POLY_OVR),
        )

    def score(self, waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
        """
        Score one waveform; requires the model to be available.

        Args:
            waveform: [T] mono waveform.
            sample_rate: Sample rate in Hz; must be DNSMOS_SAMPLE_RATE
                (callers resample upstream, matching the pipeline convention).

        Returns:
            Dict with keys "sig", "bak", "ovrl" on the 1-5 MOS scale.

        Raises:
            RuntimeError: When the model is not configured or onnxruntime is
                missing, so absent scores can never be mistaken for real ones.
        """
        if not self.is_available:
            raise RuntimeError(_DOWNLOAD_HINT)
        if sample_rate != DNSMOS_SAMPLE_RATE:
            raise ValueError(f"DNSMOS expects {DNSMOS_SAMPLE_RATE} Hz, got {sample_rate}")

        segments = _segment_audio(waveform)
        sigs: list[float] = []
        baks: list[float] = []
        ovrls: list[float] = []
        for seg in segments:
            s, b, o = self._run_segment(seg)
            sigs.append(s)
            baks.append(b)
            ovrls.append(o)

        return {
            "sig": float(np.mean(sigs)),
            "bak": float(np.mean(baks)),
            "ovrl": float(np.mean(ovrls)),
        }

    def score_or_none(self, waveform: np.ndarray, sample_rate: int) -> dict[str, float] | None:
        """Graceful variant for batch eval loops: None when unavailable, never fake."""
        if not self.is_available:
            return None
        return self.score(waveform, sample_rate)
