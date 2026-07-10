"""
REAL-M blind SI-SNR quality estimator (Dev B, Phase 1).

Scores separated streams against the mixture without ground-truth references.
Drives the cascade gate in Phase 2. Source: SpeechBrain REAL-M-sisnr-estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from schemas.separation_result import SeparationResult

if TYPE_CHECKING:
    from speechbrain.inference.metrics import SNREstimator


@dataclass
class QualityEstimate:
    """Per-stream blind SI-SNR estimates from REAL-M.

    Attributes:
        sisnr_db_per_stream: Estimated SI-SNR in dB for each stream.
        mean_sisnr_db: Mean across streams.
        min_sisnr_db: Minimum across streams (conservative gate signal).
    """

    sisnr_db_per_stream: list[float]
    mean_sisnr_db: float
    min_sisnr_db: float


class REALMQualityEstimator:
    """
    Blind separation quality estimator using SpeechBrain REAL-M.

    Estimates SI-SNR without reference clean sources. Model outputs are in
    dB/10 (range 0–1); this wrapper converts to dB via gettrue_snrrange().
    """

    MODEL_SOURCE = "speechbrain/REAL-M-sisnr-estimator"

    def __init__(self, device: str | torch.device = "cpu", savedir: str | None = None) -> None:
        self.device = torch.device(device)
        self._savedir = savedir or "pretrained_models/REAL-M-sisnr-estimator"
        self._model: SNREstimator | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from speechbrain.inference.metrics import SNREstimator

        self._model = SNREstimator.from_hparams(
            source=self.MODEL_SOURCE,
            savedir=self._savedir,
            run_opts={"device": str(self.device)},
        )
        self._model.eval()

    def estimate(
        self,
        mixture: np.ndarray | torch.Tensor,
        streams: np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
    ) -> QualityEstimate:
        """
        Estimate blind SI-SNR for each separated stream.

        Args:
            mixture: Mono mixture [T].
            streams: Separated streams [K, T].
            sample_rate: Audio sample rate (must match mixture and streams).

        Returns:
            QualityEstimate with per-stream SI-SNR in dB.
        """
        self._load()
        assert self._model is not None

        mix = self._to_mono_tensor(mixture, sample_rate)
        est = self._streams_to_predictions(streams, mix.shape[-1], sample_rate)

        with torch.no_grad():
            raw = self._model.estimate_batch(mix, est)
            if hasattr(self._model, "gettrue_snrrange"):
                sisnr_db = self._model.gettrue_snrrange(raw)
            else:
                sisnr_db = raw * 10.0

        if sisnr_db.ndim == 0:
            per_stream = [float(sisnr_db.item())]
        else:
            flat = sisnr_db.detach().cpu().numpy().reshape(-1)
            per_stream = [float(v) for v in flat]

        return QualityEstimate(
            sisnr_db_per_stream=per_stream,
            mean_sisnr_db=float(np.mean(per_stream)),
            min_sisnr_db=float(np.min(per_stream)),
        )

    def estimate_result(self, result: SeparationResult) -> QualityEstimate:
        """Estimate quality from a SeparationResult (requires mixture attached)."""
        if result.mixture is None:
            raise ValueError("SeparationResult.mixture is required for blind quality estimation")
        return self.estimate(result.mixture, result.streams, result.sample_rate)

    def _to_mono_tensor(self, audio: np.ndarray | torch.Tensor, sample_rate: int) -> torch.Tensor:
        from models.preprocess import PROJECT_SAMPLE_RATE, resample_audio

        if isinstance(audio, np.ndarray):
            wav = resample_audio(audio, sample_rate, PROJECT_SAMPLE_RATE)
            t = torch.from_numpy(wav)
        else:
            t = audio.float().squeeze()
            if sample_rate != PROJECT_SAMPLE_RATE:
                from models.preprocess import resample_audio

                wav = resample_audio(t.numpy(), sample_rate, PROJECT_SAMPLE_RATE)
                t = torch.from_numpy(wav)
        return t.unsqueeze(0).to(self.device)

    def _streams_to_predictions(
        self,
        streams: np.ndarray | torch.Tensor,
        target_len: int,
        sample_rate: int,
    ) -> torch.Tensor:
        """Convert [K, T] streams to SpeechBrain format [B, T, C]."""
        from models.preprocess import PROJECT_SAMPLE_RATE, resample_audio

        if isinstance(streams, np.ndarray):
            arr = streams.astype(np.float32)
        else:
            arr = streams.detach().cpu().numpy().astype(np.float32)

        if sample_rate != PROJECT_SAMPLE_RATE:
            resampled = []
            for i in range(arr.shape[0]):
                resampled.append(resample_audio(arr[i], sample_rate, PROJECT_SAMPLE_RATE))
            arr = np.stack(resampled, axis=0)

        t_len = arr.shape[1]
        if t_len > target_len:
            arr = arr[:, :target_len]
        elif t_len < target_len:
            pad = np.zeros((arr.shape[0], target_len - t_len), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)

        # [K, T] → [1, T, K]
        pred = torch.from_numpy(arr.T).unsqueeze(0).float().to(self.device)
        return pred
