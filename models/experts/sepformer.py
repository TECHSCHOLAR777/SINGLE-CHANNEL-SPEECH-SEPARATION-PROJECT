"""
SepFormer expert wrapper (SpeechBrain pretrained WSJ0-3mix).

Phase 0 control baseline. Frozen weights, inference only.
Source: huggingface.co/speechbrain/sepformer-wsj03mix
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from schemas.separation_result import SeparationResult, StreamMetadata

if TYPE_CHECKING:
    from speechbrain.inference.separation import SepformerSeparation


class SepFormerExpert:
    """Inference wrapper around SpeechBrain SepFormer for 3-speaker separation."""

    SAMPLE_RATE = 16000
    EXPERT_NAME = "sepformer"
    MAX_SPEAKERS = 3

    def __init__(self, device: str | torch.device = "cpu", savedir: str | None = None) -> None:
        self.device = torch.device(device)
        self._model: SepformerSeparation | None = None
        self._savedir = savedir

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from speechbrain.inference.separation import SepformerSeparation

        kwargs: dict = {
            "source": "speechbrain/sepformer-wsj03mix",
            "savedir": self._savedir or "pretrained_models/sepformer-wsj03mix",
            "run_opts": {"device": str(self.device)},
        }
        self._model = SepformerSeparation.from_hparams(**kwargs)
        self._model.eval()

    @property
    def model(self) -> SepformerSeparation:
        self._load_model()
        assert self._model is not None
        return self._model

    def separate(self, mixture: np.ndarray | torch.Tensor, sample_rate: int) -> SeparationResult:
        """
        Separate a mono mixture into up to 3 speaker streams.

        Args:
            mixture: Mono waveform [T] or [1, T].
            sample_rate: Input sample rate (resampled to 16 kHz if needed).

        Returns:
            SeparationResult with streams [K, T], K <= 3.
        """
        if sample_rate != self.SAMPLE_RATE:
            mixture = self._resample(mixture, sample_rate, self.SAMPLE_RATE)
            sample_rate = self.SAMPLE_RATE

        if isinstance(mixture, np.ndarray):
            wav = torch.from_numpy(mixture.astype(np.float32))
        else:
            wav = mixture.float()

        wav = wav.squeeze()
        if wav.ndim != 1:
            raise ValueError(f"Expected mono mixture [T], got shape {tuple(wav.shape)}")

        with torch.no_grad():
            est_sources = self.model.separate_batch(wav.unsqueeze(0).to(self.device))
            # SpeechBrain returns [batch, time, speakers] or [batch, speakers, time]
            est = est_sources.squeeze(0)
            if est.shape[0] == wav.shape[0]:
                est = est.T  # [speakers, time]

        streams = est.detach().cpu().numpy().astype(np.float32)
        k = streams.shape[0]
        metadata = [
            StreamMetadata(expert_source=self.EXPERT_NAME, confidence=1.0) for _ in range(k)
        ]

        mixture_np = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else wav
        return SeparationResult(
            streams=streams,
            sample_rate=sample_rate,
            speaker_count=k,
            metadata=metadata,
            mixture=np.asarray(mixture_np, dtype=np.float32),
            escalated=False,
            expert_used=self.EXPERT_NAME,
        )

    @staticmethod
    def _resample(
        audio: np.ndarray | torch.Tensor,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        import torchaudio.functional as F

        if isinstance(audio, np.ndarray):
            t = torch.from_numpy(audio.astype(np.float32))
        else:
            t = audio.float()
        t = t.squeeze()
        resampled = F.resample(t.unsqueeze(0), orig_sr, target_sr).squeeze(0)
        return resampled.numpy()
