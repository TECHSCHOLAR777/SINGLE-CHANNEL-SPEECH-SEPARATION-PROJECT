"""
DNSMOS P.835 reference-free quality scorer (Dev C, P0-C4).

Download sig_bak_ovrl.onnx from the Microsoft DNS-Challenge repository:
  https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS

Place the file at a path and configure eval.dnsmos.model_path in configs/devc.yaml,
or pass the path directly to DnsmosScorer(model_path=...).

Inference: segments the waveform into 9.01 s windows (matching the model's
training distribution), runs ONNX inference on each, and averages the scores.
Scores are on the 1-5 MOS scale (higher = better).

SIG = speech quality, BAK = background noise quality, OVRL = overall quality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DNSMOS_SAMPLE_RATE = 16000
"""DNSMOS P.835 requires 16 kHz input."""

_SEGMENT_SAMPLES = int(9.01 * DNSMOS_SAMPLE_RATE)
_HOP_SAMPLES = int(4.0 * DNSMOS_SAMPLE_RATE)
_N_FFT = 321
_HOP_LENGTH = 160
_WIN_LENGTH = 320

_DOWNLOAD_HINT = (
    "DNSMOS model not configured. Download sig_bak_ovrl.onnx from:\n"
    "  https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS\n"
    "Then pass its path to DnsmosScorer(model_path=...) or set eval.dnsmos.model_path\n"
    "in configs/devc.yaml."
)


class DnsmosScorer:
    """
    Reference-free quality scorer using DNSMOS P.835 ONNX model.

    Args:
        model_path: Path to sig_bak_ovrl.onnx, or None (unavailable mode).
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._session: object | None = None

    @property
    def is_available(self) -> bool:
        """True when the model file exists and onnxruntime is installed."""
        if self.model_path is None or not self.model_path.exists():
            return False
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("onnxruntime not installed: pip install onnxruntime") from e
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

    def _compute_features(self, segment: np.ndarray) -> np.ndarray:
        """Log-magnitude STFT features expected by the DNSMOS P.835 model."""
        import torch
        wav = torch.from_numpy(segment.astype(np.float32))
        spec = torch.stft(
            wav,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
            window=torch.hann_window(_WIN_LENGTH),
            return_complex=True,
            center=True,
        )
        mag = spec.abs().clamp(min=1e-10)
        log_mag = torch.log(mag).numpy().astype(np.float32)
        return log_mag.T[np.newaxis, :, :]  # (1, frames, freq_bins)

    def score(self, waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
        """
        Score one waveform with DNSMOS P.835.

        Args:
            waveform: [T] mono float32 at sample_rate.
            sample_rate: Must be DNSMOS_SAMPLE_RATE.

        Returns:
            Dict with keys 'sig', 'bak', 'ovrl' in [1, 5].

        Raises:
            RuntimeError: When model is not configured.
            ValueError: When sample_rate != DNSMOS_SAMPLE_RATE.
        """
        if not self.is_available:
            raise RuntimeError(_DOWNLOAD_HINT)
        if sample_rate != DNSMOS_SAMPLE_RATE:
            raise ValueError(f"DNSMOS requires {DNSMOS_SAMPLE_RATE} Hz input, got {sample_rate} Hz")
        self._load()

        wav = np.asarray(waveform, dtype=np.float32).squeeze()
        if wav.ndim != 1:
            raise ValueError(f"Expected 1-D waveform, got shape {wav.shape}")

        n = len(wav)
        if n < _SEGMENT_SAMPLES:
            wav = np.pad(wav, (0, _SEGMENT_SAMPLES - n))
            n = len(wav)

        sig_scores: list[float] = []
        bak_scores: list[float] = []
        ovrl_scores: list[float] = []

        starts = list(range(0, n - _SEGMENT_SAMPLES + 1, _HOP_SAMPLES)) or [0]
        for start in starts:
            seg = wav[start : start + _SEGMENT_SAMPLES]
            if len(seg) < _SEGMENT_SAMPLES:
                seg = np.pad(seg, (0, _SEGMENT_SAMPLES - len(seg)))

            feats = self._compute_features(seg)
            assert self._session is not None
            input_name = self._session.get_inputs()[0].name  # type: ignore[union-attr]
            outputs = self._session.run(None, {input_name: feats})  # type: ignore[union-attr]

            result = np.array(outputs).flatten()
            if result.shape[0] >= 3:
                sig_scores.append(_mos_map(float(result[0])))
                bak_scores.append(_mos_map(float(result[1])))
                ovrl_scores.append(_mos_map(float(result[2])))
            else:
                v = _mos_map(float(result[0]))
                sig_scores.append(v)
                bak_scores.append(v)
                ovrl_scores.append(v)

        return {
            "sig": float(np.mean(sig_scores)),
            "bak": float(np.mean(bak_scores)),
            "ovrl": float(np.mean(ovrl_scores)),
        }

    def score_or_none(self, waveform: np.ndarray, sample_rate: int) -> dict[str, float] | None:
        """Graceful variant: returns None when model unavailable, never fakes scores."""
        if not self.is_available:
            return None
        try:
            return self.score(waveform, sample_rate)
        except Exception:
            return None


def _mos_map(raw: float) -> float:
    return float(np.clip(raw, 1.0, 5.0))
