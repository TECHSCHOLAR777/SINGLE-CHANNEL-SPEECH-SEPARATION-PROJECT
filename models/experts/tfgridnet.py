"""
TF-GridNet fallback expert (expensive time-frequency separator).

Used when SR-CorrNet weights are unavailable (P1-B3). Attempts ESPnet
pretrained TF-GridNet; falls back to SpeechBrain SepFormer as last resort
per MASTER §5.1 fallback plan.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import torch

from models.preprocess import preprocess
from schemas.separation_result import SeparationResult, StreamMetadata


class TFGridNetExpert:
    """TF-GridNet via ESPnet, or SepFormer fallback when ESPnet is not configured."""

    SAMPLE_RATE = 16000
    EXPERT_NAME = "tfgridnet"
    MAX_SPEAKERS = 4

    def __init__(
        self,
        device: str | torch.device = "cpu",
        espnet_model_tag: str | None = None,
        num_speakers: int = 3,
        use_sepformer_fallback: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.espnet_model_tag = espnet_model_tag
        self.num_speakers = num_speakers
        self.use_sepformer_fallback = use_sepformer_fallback
        self._model: object | None = None
        self._backend: str = ""

    @property
    def is_available(self) -> bool:
        """True if ESPnet or SepFormer fallback can be loaded."""
        if self.espnet_model_tag and importlib.util.find_spec("espnet2") is not None:
            return True
        if self.use_sepformer_fallback and importlib.util.find_spec("speechbrain") is not None:
            return True
        return False

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.espnet_model_tag:
            model = self._try_load_espnet()
            if model is not None:
                self._model = model
                self._backend = "espnet"
                return
        if self.use_sepformer_fallback:
            from models.experts.sepformer import SepFormerExpert

            self._model = SepFormerExpert(device=self.device)
            self._backend = "sepformer_fallback"
            return
        raise RuntimeError(
            "TF-GridNet fallback unavailable. Install espnet or enable SepFormer fallback."
        )

    def _try_load_espnet(self) -> object | None:
        """Load ESPnet separation model if espnet2 is installed."""
        try:
            from espnet2.bin.enh_inference import SeparateSpeech

            if self.espnet_model_tag is None:
                return None
            return SeparateSpeech.from_pretrained(model_tag=self.espnet_model_tag, device=str(self.device))
        except (ImportError, OSError, RuntimeError):
            return None

    def separate(self, mixture: np.ndarray | torch.Tensor, sample_rate: int) -> SeparationResult:
        """Separate mixture; delegates to loaded backend."""
        self._load_model()
        assert self._model is not None

        if self._backend == "sepformer_fallback":
            from models.experts.sepformer import SepFormerExpert

            assert isinstance(self._model, SepFormerExpert)
            result = self._model.separate(mixture, sample_rate)
            return SeparationResult(
                streams=result.streams,
                sample_rate=result.sample_rate,
                speaker_count=result.speaker_count,
                metadata=[
                    StreamMetadata(
                        expert_source=self.EXPERT_NAME,
                        confidence=m.confidence,
                        extra={"backend": "sepformer_fallback", **m.extra},
                    )
                    for m in result.metadata
                ],
                mixture=result.mixture,
                escalated=True,
                expert_used=self.EXPERT_NAME,
            )

        pre = preprocess(mixture, sample_rate)
        wav = pre.waveform
        with torch.no_grad():
            output = self._model(wav)  # type: ignore[operator]
        streams, confidences = _parse_separation_output(output, wav.shape[0])
        metadata = [
            StreamMetadata(
                expert_source=self.EXPERT_NAME,
                confidence=float(confidences[i]) if i < len(confidences) else 1.0,
                extra={"backend": "espnet", "attractor_index": i},
            )
            for i in range(streams.shape[0])
        ]
        return SeparationResult(
            streams=streams,
            sample_rate=self.SAMPLE_RATE,
            speaker_count=streams.shape[0],
            metadata=metadata,
            mixture=wav,
            escalated=True,
            expert_used=self.EXPERT_NAME,
        )


def _parse_separation_output(
    output: torch.Tensor | tuple | dict,
    time_len: int,
) -> tuple[np.ndarray, list[float]]:
    """Normalize separation model output to [K, T] numpy."""
    if isinstance(output, dict):
        est = output.get("wav_spk") or output.get("est_sources") or output["wav"]
        conf = output.get("confidence", [])
    elif isinstance(output, tuple):
        est, *rest = output
        conf = rest[0] if rest else []
    else:
        est = output
        conf = []

    if isinstance(est, torch.Tensor):
        est = est.detach().cpu().numpy()
    est = np.asarray(est, dtype=np.float32).squeeze()
    if est.ndim == 3:
        est = est[0]
    if est.ndim == 2 and est.shape[0] != time_len and est.shape[1] == time_len:
        pass
    elif est.ndim == 2 and est.shape[0] == time_len:
        est = est.T

    if isinstance(conf, torch.Tensor):
        confidences = conf.detach().cpu().tolist()
    elif isinstance(conf, (list, tuple)):
        confidences = list(conf)
    else:
        confidences = [1.0] * est.shape[0]

    return est.astype(np.float32), confidences


def get_expensive_expert(
    device: str | torch.device = "cpu",
    srcorrnet_repo: str | Path | None = None,
    srcorrnet_checkpoint: str | Path | None = None,
    tfgridnet_tag: str | None = None,
    num_speakers: int = 3,
) -> "SRCorrNetExpert | TFGridNetExpert":
    """
    Factory: prefer SR-CorrNet; fall back to TF-GridNet when unavailable.

    Returns the first configured expert that passes is_available.
    """
    from models.experts.srcorrnet import SRCorrNetExpert

    sr = SRCorrNetExpert(
        device=device,
        repo_path=srcorrnet_repo,
        checkpoint_path=srcorrnet_checkpoint,
        num_speakers=num_speakers,
    )
    if sr.is_available:
        return sr

    tf = TFGridNetExpert(device=device, espnet_model_tag=tfgridnet_tag, num_speakers=num_speakers)
    if tf.is_available:
        return tf

    raise RuntimeError(
        "No expensive expert available. Configure SR-CorrNet repo/checkpoint "
        "or install espnet for TF-GridNet fallback."
    )
