"""
Speaker embedding utilities for expert wrappers (Dev B, Phase 1).

Attaches ECAPA-TDNN embeddings to SeparationResult streams so Hungarian
alignment (align/hungarian.py) can match experts without waiting for Dev C's
standalone ECAPA wrapper (P1-C1). Uses the same SpeechBrain checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from schemas.separation_result import SeparationResult, StreamMetadata

if TYPE_CHECKING:
    from speechbrain.inference.speaker import EncoderClassifier


class ECAPAEmbedder:
    """Lazy-loaded SpeechBrain ECAPA-TDNN speaker embedder."""

    SAMPLE_RATE = 16000
    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, device: str | torch.device = "cpu", savedir: str | None = None) -> None:
        self.device = torch.device(device)
        self._savedir = savedir or "pretrained_models/spkrec-ecapa-voxceleb"
        self._model: EncoderClassifier | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from speechbrain.inference.speaker import EncoderClassifier

        self._model = EncoderClassifier.from_hparams(
            source=self.MODEL_SOURCE,
            savedir=self._savedir,
            run_opts={"device": str(self.device)},
        )
        self._model.eval()

    def embed_stream(self, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Return L2-normalized ECAPA embedding [D] for one mono stream."""
        self._load()
        assert self._model is not None
        wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).squeeze()
        if sample_rate != self.SAMPLE_RATE:
            import torchaudio.functional as F

            wav = F.resample(wav.unsqueeze(0), sample_rate, self.SAMPLE_RATE).squeeze(0)
        with torch.no_grad():
            emb = self._model.encode_batch(wav.unsqueeze(0).to(self.device))
            vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec)) + 1e-8
        return (vec / norm).astype(np.float32)


def attach_ecapa_embeddings(
    result: SeparationResult,
    embedder: ECAPAEmbedder | None = None,
    device: str | torch.device = "cpu",
) -> SeparationResult:
    """
    Return a copy of result with ECAPA embeddings attached to each stream.

    Existing embeddings are preserved; only missing slots are filled.
    """
    emb = embedder or ECAPAEmbedder(device=device)
    new_metadata: list[StreamMetadata] = []
    for i, meta in enumerate(result.metadata):
        if meta.embedding is not None:
            new_metadata.append(meta)
            continue
        embedding = emb.embed_stream(result.streams[i], result.sample_rate)
        new_metadata.append(
            StreamMetadata(
                expert_source=meta.expert_source,
                confidence=meta.confidence,
                embedding=embedding,
                extra=dict(meta.extra),
            )
        )
    return SeparationResult(
        streams=result.streams,
        sample_rate=result.sample_rate,
        speaker_count=result.speaker_count,
        metadata=new_metadata,
        mixture=result.mixture,
        escalated=result.escalated,
        expert_used=result.expert_used,
    )
