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
        target_speakers: int | None = None,
    ) -> None:
        """
        Args:
            target_speakers: Pad the output up to this many streams when the
                checkpoint emits fewer.

                MossFormer2_SS_16K is a TWO-speaker model. On a three-speaker
                mixture it returns K=2, while SR-CorrNet returns K=3, and
                CAMoSEFusion.forward raises `stream shape mismatch` the moment
                the two meet. The cascade therefore cannot train at all on
                Libri3Mix without reconciling K.

                When target_speakers=3 and the model returns 2 streams, the
                missing stream is filled with the RESIDUAL, mixture minus the
                sum of the emitted streams. On a 3-speaker mixture that residual
                is approximately the speaker the model failed to extract, plus
                artefacts, so it is a real signal rather than padding. It is
                marked `synthetic="residual"` with low confidence in
                StreamMetadata so the router, the gate, and the fusion head can
                learn to distrust it, and its energy is exactly the quantity the
                cascade should escalate on.

                A gap larger than one cannot be recovered this way (the residual
                would blend several speakers), so remaining slots are zero-filled
                and marked `synthetic="zero"` with confidence 0.
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.compute_embeddings = compute_embeddings
        self._embedder_savedir = embedder_savedir
        self.target_speakers = target_speakers
        self._cv: object | None = None
        self._embedder: object | None = None

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
        streams, confidences, synthetic = self._pad_to_target(
            streams, confidences, wav, self.target_speakers
        )

        metadata = [
            StreamMetadata(
                expert_source=self.EXPERT_NAME,
                confidence=float(confidences[i]) if i < len(confidences) else 1.0,
                extra={"stream_index": i, "synthetic": synthetic[i]},
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
            if self._embedder is None:
                from models.experts.embeddings import ECAPAEmbedder

                self._embedder = ECAPAEmbedder(device=self.device, savedir=self._embedder_savedir)
            result = attach_ecapa_embeddings(result, embedder=self._embedder)

        return result

    @staticmethod
    def _pad_to_target(
        streams: np.ndarray,
        confidences: list[float],
        mixture: np.ndarray,
        target: int | None,
    ) -> tuple[np.ndarray, list[float], list[str | None]]:
        """
        Reconcile the emitted stream count with the expected speaker count.

        Returns (streams, confidences, synthetic) where synthetic[i] is None for
        a real model output, "residual" for the mixture-minus-sum stream, and
        "zero" for a hard-padded slot.
        """
        k = streams.shape[0]
        synthetic: list[str | None] = [None] * k
        if target is None or k >= target:
            return streams, confidences, synthetic

        extra = np.zeros((target - k, streams.shape[1]), dtype=np.float32)
        # Slot 0 of the padding carries the residual: what the model left behind.
        residual = mixture.astype(np.float32) - streams.sum(axis=0)
        extra[0] = residual
        synthetic.append("residual")
        confidences.append(0.0)
        for _ in range(target - k - 1):
            synthetic.append("zero")
            confidences.append(0.0)

        return np.concatenate([streams, extra], axis=0), confidences, synthetic

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
