"""
ECAPA-TDNN speaker embedding wrapper (Dev C, P1-C1).

Produces the per-stream speaker embeddings that Hungarian alignment
(align/hungarian.py) and the cross-chunk identity lock (align/chunking.py)
consume. Wraps SpeechBrain `spkrec-ecapa-voxceleb` behind a lazy import so
the module imports cleanly in environments without speechbrain (CI runs the
mocked tests; real inference needs `pip install speechbrain` plus one-time
HuggingFace weight download).

The glue function embed_result() fills StreamMetadata.embedding on a
SeparationResult, which is the P1 alignment interface agreed in
docs/decisions.md: expert wrappers produce streams, this module attaches
embeddings, the aligner consumes them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from coralsep.schemas.separation_result import SeparationResult

ECAPA_SAMPLE_RATE = 16000
"""spkrec-ecapa-voxceleb is a 16 kHz model; inputs at other rates are resampled."""

DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class EcapaEmbedder:
    """
    Speaker embedding extractor over separated streams.

    Args:
        source: HuggingFace model id or local path for the ECAPA checkpoint.
        device: torch device string ("cpu", "cuda", "mps").
        savedir: Optional local cache directory for downloaded weights.
    """

    def __init__(
        self,
        source: str = DEFAULT_SOURCE,
        device: str = "cpu",
        savedir: str | None = None,
    ) -> None:
        self.source = source
        self.device = device
        self.savedir = savedir
        self._model = None  # loaded lazily on first embed call

    def _load(self):
        """Import speechbrain and load the classifier on first use only."""
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            kwargs = {"source": self.source, "run_opts": {"device": self.device}}
            if self.savedir is not None:
                kwargs["savedir"] = self.savedir
            self._model = EncoderClassifier.from_hparams(**kwargs)
        return self._model

    @staticmethod
    def _resample(streams: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resample [K, T] streams to ECAPA_SAMPLE_RATE when needed."""
        if sample_rate == ECAPA_SAMPLE_RATE:
            return streams
        import torch
        import torchaudio.functional as taf

        wavs = torch.from_numpy(np.ascontiguousarray(streams)).float()
        out = taf.resample(wavs, orig_freq=sample_rate, new_freq=ECAPA_SAMPLE_RATE)
        return out.numpy()

    def embed(self, streams: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute one L2-normalized embedding per stream.

        Args:
            streams: [K, T] float waveforms (any count K >= 1).
            sample_rate: Sample rate of the streams in Hz.

        Returns:
            [K, D] float64 numpy array, rows L2-normalized (D = 192 for the
            default checkpoint). Alignment cost is then 1 - dot product.
        """
        import torch

        arr = np.atleast_2d(np.asarray(streams, dtype=np.float32))
        if arr.size == 0:
            raise ValueError("streams is empty; nothing to embed")
        arr = self._resample(arr, sample_rate)

        model = self._load()
        with torch.no_grad():
            wavs = torch.from_numpy(np.ascontiguousarray(arr)).float()
            emb = model.encode_batch(wavs)  # [K, 1, D]
        out = emb.squeeze(1).cpu().numpy().astype(np.float64)
        norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
        return out / norms

    def embed_result(self, result: SeparationResult) -> SeparationResult:
        """
        Return a copy of a SeparationResult with metadata embeddings filled.

        The returned object is new; the input is not mutated. Existing
        embeddings (e.g. attractor vectors from an expert) are preserved,
        only missing ones are computed.
        """
        missing = [i for i, m in enumerate(result.metadata) if m.embedding is None]
        if not missing:
            return result

        embeddings = self.embed(result.streams[missing], result.sample_rate)
        metadata = list(result.metadata)
        for row, i in enumerate(missing):
            metadata[i] = replace(metadata[i], embedding=embeddings[row])

        return SeparationResult(
            streams=result.streams,
            sample_rate=result.sample_rate,
            speaker_count=result.speaker_count,
            metadata=metadata,
            mixture=result.mixture,
            escalated=result.escalated,
            expert_used=result.expert_used,
        )
