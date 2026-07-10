"""
MossFormer2 expert wrapper (cheap time-domain separator, Dev B Phase 1).

Frozen MossFormer2_SS_16K weights via ClearVoice (ClearerVoice-Studio).
Source: github.com/modelscope/ClearerVoice-Studio / pip package `clearvoice`
RTF ~0.05; fixed output of up to 3 streams.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from models.experts.embeddings import attach_ecapa_embeddings
from models.preprocess import preprocess
from schemas.separation_result import SeparationResult, StreamMetadata

if TYPE_CHECKING:
    pass


class MossFormer2Expert:
    """Inference wrapper for MossFormer2_SS_16K speech separation."""

    SAMPLE_RATE = 16000
    EXPERT_NAME = "mossformer2"
    MAX_SPEAKERS = 3
    DEFAULT_MODEL = "MossFormer2_SS_16K"

    def __init__(
        self,
        device: str | torch.device = "cpu",
        model_name: str = DEFAULT_MODEL,
        compute_embeddings: bool = True,
        embedder_savedir: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name
        self.compute_embeddings = compute_embeddings
        self._embedder_savedir = embedder_savedir
        self._cv: object | None = None

    @staticmethod
    def is_available() -> bool:
        """True when the clearvoice package is installed."""
        try:
            import clearvoice  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_model(self) -> None:
        if self._cv is not None:
            return
        if not self.is_available():
            raise RuntimeError(
                "MossFormer2 requires the `clearvoice` package. "
                "Install with: pip install clearvoice"
            )
        from clearvoice import ClearVoice

        self._cv = ClearVoice(task="speech_separation", model_names=[self.model_name])

    def separate(self, mixture: np.ndarray | torch.Tensor, sample_rate: int) -> SeparationResult:
        """
        Separate a mono mixture into up to 3 speaker streams.

        Args:
            mixture: Mono waveform [T] or [1, T].
            sample_rate: Input sample rate (resampled to 16 kHz internally).

        Returns:
            SeparationResult with streams [K, T], K <= 3, optional ECAPA embeddings.
        """
        self._load_model()
        assert self._cv is not None

        pre = preprocess(mixture, sample_rate)
        wav = pre.waveform
        batch = np.reshape(wav, [1, wav.shape[0]]).astype(np.float32)

        output = self._cv(batch, False)  # type: ignore[operator]
        streams, confidences = self._parse_clearvoice_output(output, wav.shape[0])

        metadata = [
            StreamMetadata(
                expert_source=self.EXPERT_NAME,
                confidence=float(confidences[i]) if i < len(confidences) else 1.0,
                extra={"stream_index": i},
            )
            for i in range(streams.shape[0])
        ]

        result = SeparationResult(
            streams=streams,
            sample_rate=self.SAMPLE_RATE,
            speaker_count=streams.shape[0],
            metadata=metadata,
            mixture=wav,
            escalated=False,
            expert_used=self.EXPERT_NAME,
        )

        if self.compute_embeddings:
            from models.experts.embeddings import ECAPAEmbedder

            embedder = ECAPAEmbedder(device=self.device, savedir=self._embedder_savedir)
            result = attach_ecapa_embeddings(result, embedder=embedder)

        return result

    @staticmethod
    def _parse_clearvoice_output(
        output: np.ndarray | torch.Tensor | tuple,
        time_len: int,
    ) -> tuple[np.ndarray, list[float]]:
        """
        Normalize ClearVoice output to [K, T] numpy.

        ClearVoice returns [spk, batch, length] for speech_separation.
        """
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, torch.Tensor):
            arr = output.detach().cpu().numpy()
        else:
            arr = np.asarray(output, dtype=np.float32)

        if arr.ndim == 3:
            # [spk, batch, length] — take batch index 0
            streams = arr[:, 0, :]
        elif arr.ndim == 2:
            streams = arr
        else:
            raise ValueError(f"Unexpected ClearVoice output shape: {arr.shape}")

        # Trim or pad to input length
        t = streams.shape[1]
        if t > time_len:
            streams = streams[:, :time_len]
        elif t < time_len:
            pad = np.zeros((streams.shape[0], time_len - t), dtype=np.float32)
            streams = np.concatenate([streams, pad], axis=1)

        confidences = [1.0] * streams.shape[0]
        return streams.astype(np.float32), confidences
