"""
Stop-classifier feature extractors (Dev B, Phase 3 / P3-B1).

Computes the four supplementary signals from MASTER §4.5 plus assembly into
the frozen feature vector consumed by ``models/stop_classifier.py``:

1. Residual energy ratio — unexplained mixture energy after accepted stems
   plus the candidate stem.
2. VAD speech probability on the residual — is speech still present?
3. Minimum ECAPA-TDNN cosine distance from the candidate to prior stems.
4. Mixture-consistency reconstruction error — self-grading without reference.

``CountingFeatureExtractor`` orchestrates all four from waveforms. Individual
functions are exposed for unit tests and ablations (e.g. N6 mixture-consistency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

_EPS = 1e-8

FEATURE_NAMES: tuple[str, ...] = (
    "residual_energy_ratio",
    "vad_speech_prob",
    "min_embedding_distance",
    "mixture_consistency_error",
    "attractor_stop_logit",
)
"""Frozen feature order. Index positions are part of the checkpoint contract."""

if TYPE_CHECKING:
    from models.experts.embeddings import ECAPAEmbedder

VADBackend = Literal["energy", "silero"]


def _as_waveform(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("waveform must be non-empty")
    return arr


def _as_stems(stems: np.ndarray, length: int) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(stems, dtype=np.float64))
    if arr.size == 0:
        return np.zeros((0, length), dtype=np.float64)
    if arr.shape[1] != length:
        raise ValueError(f"accepted_stems length {arr.shape[1]} != mixture length {length}")
    return arr


def compute_residual(
    mixture: np.ndarray,
    accepted_stems: np.ndarray,
    candidate_stem: np.ndarray,
) -> np.ndarray:
    """
    Residual waveform after subtracting accepted stems and the candidate.

    Args:
        mixture: [T] original mixture.
        accepted_stems: [K, T] stems accepted so far (K may be 0).
        candidate_stem: [T] newly extracted stem under consideration.

    Returns:
        Residual [T] float64.
    """
    mix = _as_waveform(mixture)
    stems = _as_stems(accepted_stems, mix.shape[0])
    cand = _as_waveform(candidate_stem)
    if cand.shape[0] != mix.shape[0]:
        raise ValueError("candidate_stem length must match mixture")
    explained = stems.sum(axis=0) + cand
    return mix - explained


def residual_energy_ratio(
    mixture: np.ndarray,
    accepted_stems: np.ndarray,
    candidate_stem: np.ndarray,
) -> float:
    """
    Ratio of residual energy to mixture energy (MASTER §4.5 feature 1).

    ``||residual||^2 / ||mixture||^2``
    """
    mix = _as_waveform(mixture)
    residual = compute_residual(mix, accepted_stems, candidate_stem)
    mix_energy = float(np.dot(mix, mix)) + _EPS
    return float(np.dot(residual, residual)) / mix_energy


def mixture_consistency_error(
    mixture: np.ndarray,
    accepted_stems: np.ndarray,
    candidate_stem: np.ndarray,
) -> float:
    """
    Normalized reconstruction error (MASTER §4.5 feature 4 / N6 self-grade).

    ``||residual|| / ||mixture||``
    """
    mix = _as_waveform(mixture)
    residual = compute_residual(mix, accepted_stems, candidate_stem)
    return float(np.linalg.norm(residual)) / (float(np.linalg.norm(mix)) + _EPS)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance ``1 - cos_sim`` for L2-normalized or arbitrary vectors."""
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = (float(np.linalg.norm(va)) + _EPS) * (float(np.linalg.norm(vb)) + _EPS)
    return float(1.0 - np.dot(va, vb) / denom)


def compute_min_embedding_distance(
    candidate_stem: np.ndarray,
    accepted_stems: np.ndarray,
    embedder: ECAPAEmbedder,
    sample_rate: int = 16_000,
) -> float:
    """
    Minimum ECAPA cosine distance from candidate to accepted stems (feature 3).

    Returns 1.0 when no stems are accepted yet (no duplicate to compare against).
    """
    stems = _as_stems(accepted_stems, _as_waveform(candidate_stem).shape[0])
    if stems.shape[0] == 0:
        return 1.0
    cand_emb = embedder.embed_stream(candidate_stem, sample_rate=sample_rate)
    distances = [
        cosine_distance(cand_emb, embedder.embed_stream(stems[i], sample_rate=sample_rate))
        for i in range(stems.shape[0])
    ]
    return float(min(distances))


class VADAdapter:
    """
    Voice-activity probability on a mono waveform in [0, 1].

    Default ``energy`` backend is deterministic and needs no downloads (CI-safe).
    Optional ``silero`` uses torch.hub Silero VAD when available.
    """

    SAMPLE_RATE = 16_000
    FRAME_SAMPLES = 400  # 25 ms at 16 kHz
    HOP_SAMPLES = 160  # 10 ms

    def __init__(self, backend: VADBackend = "energy", device: str = "cpu") -> None:
        self.backend = backend
        self.device = device
        self._silero_model = None

    def _load_silero(self) -> None:
        if self._silero_model is not None:
            return
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._silero_model = (model, utils)

    def _speech_prob_energy(self, waveform: np.ndarray, sample_rate: int) -> float:
        wav = _as_waveform(waveform)
        if sample_rate != self.SAMPLE_RATE:
            import torch
            import torchaudio.functional as F

            t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
            wav = F.resample(t, sample_rate, self.SAMPLE_RATE).squeeze(0).numpy()

        global_rms = float(np.sqrt(np.mean(wav * wav)))
        if global_rms < 1e-6:
            return 0.0

        frame_len = self.FRAME_SAMPLES
        hop = self.HOP_SAMPLES
        if wav.shape[0] < frame_len:
            return float(np.clip(global_rms / 0.02, 0.0, 1.0))

        frames = []
        for start in range(0, wav.shape[0] - frame_len + 1, hop):
            chunk = wav[start : start + frame_len]
            frames.append(float(np.sqrt(np.mean(chunk * chunk))))
        energies = np.asarray(frames, dtype=np.float64)
        peak = float(energies.max())
        noise_floor = float(np.percentile(energies, 10))
        dynamic_range = peak - noise_floor

        # Stationary noise/residual: flat frame energy → use global RMS proxy.
        if dynamic_range < peak * 0.15:
            return float(np.clip(global_rms / 0.02, 0.0, 1.0))

        threshold = noise_floor + 0.35 * dynamic_range
        speech_frames = energies > threshold
        return float(np.clip(speech_frames.mean(), 0.0, 1.0))

    def _speech_prob_silero(self, waveform: np.ndarray, sample_rate: int) -> float:
        import torch

        self._load_silero()
        assert self._silero_model is not None
        model, utils = self._silero_model
        get_speech_timestamps = utils[0]

        wav = _as_waveform(waveform).astype(np.float32)
        if sample_rate != self.SAMPLE_RATE:
            import torchaudio.functional as F

            wav = (
                F.resample(
                    torch.from_numpy(wav).unsqueeze(0),
                    sample_rate,
                    self.SAMPLE_RATE,
                )
                .squeeze(0)
                .numpy()
            )

        tensor = torch.from_numpy(wav)
        timestamps = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=self.SAMPLE_RATE,
            return_seconds=False,
        )
        if not timestamps:
            return 0.0
        speech_samples = sum(int(seg["end"]) - int(seg["start"]) for seg in timestamps)
        return float(np.clip(speech_samples / max(len(wav), 1), 0.0, 1.0))

    def speech_prob(self, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
        """Return speech probability in [0, 1] for the given mono waveform."""
        if self.backend == "silero":
            try:
                return self._speech_prob_silero(waveform, sample_rate)
            except Exception:
                return self._speech_prob_energy(waveform, sample_rate)
        return self._speech_prob_energy(waveform, sample_rate)


@dataclass
class StopFeatureBundle:
    """Named intermediate features before vector assembly."""

    residual_energy_ratio: float
    vad_speech_prob: float
    min_embedding_distance: float
    mixture_consistency_error: float
    attractor_stop_logit: float = 0.0

    def to_vector(self) -> np.ndarray:
        return assemble_stop_features(
            residual_energy_ratio=self.residual_energy_ratio,
            vad_speech_prob=self.vad_speech_prob,
            min_embedding_distance=self.min_embedding_distance,
            mixture_consistency_error=self.mixture_consistency_error,
            attractor_stop_logit=self.attractor_stop_logit,
        )


def assemble_stop_features(
    *,
    residual_energy_ratio: float,
    vad_speech_prob: float,
    min_embedding_distance: float,
    mixture_consistency_error: float,
    attractor_stop_logit: float = 0.0,
) -> np.ndarray:
    """
    Pack features into the frozen ``FEATURE_NAMES`` order.

    Used by the stop-classifier MLP and feature JSONL logging.
    """
    return np.asarray(
        [
            float(residual_energy_ratio),
            float(vad_speech_prob),
            float(min_embedding_distance),
            float(mixture_consistency_error),
            float(attractor_stop_logit),
        ],
        dtype=np.float32,
    )


def compute_stop_features(
    mixture: np.ndarray,
    accepted_stems: np.ndarray,
    candidate_stem: np.ndarray,
    vad_speech_prob: float | None = None,
    min_embedding_distance: float | None = None,
    attractor_stop_logit: float = 0.0,
    *,
    vad: VADAdapter | None = None,
    embedder: ECAPAEmbedder | None = None,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """
    Build the stop-classifier feature vector for one peel decision.

    When ``vad_speech_prob`` and ``min_embedding_distance`` are omitted, they
    are computed from the residual and ECAPA embedder respectively.

    Args:
        mixture: [T] original mixture waveform.
        accepted_stems: [K, T] stems accepted so far (K may be 0).
        candidate_stem: [T] newly extracted stem under consideration.
        vad_speech_prob: Precomputed VAD on residual; computed if None.
        min_embedding_distance: Precomputed ECAPA distance; computed if None.
        attractor_stop_logit: Raw stop logit from SR-CorrNet attractors.
        vad: Optional VAD adapter for residual speech probability.
        embedder: Optional ECAPA embedder for speaker-distance feature.
        sample_rate: Waveform sample rate (Hz).

    Returns:
        Feature vector [len(FEATURE_NAMES)] float32.
    """
    mix = _as_waveform(mixture)
    rer = residual_energy_ratio(mix, accepted_stems, candidate_stem)
    mce = mixture_consistency_error(mix, accepted_stems, candidate_stem)

    if vad_speech_prob is None:
        residual = compute_residual(mix, accepted_stems, candidate_stem)
        adapter = vad or VADAdapter()
        vad_speech_prob = adapter.speech_prob(residual, sample_rate=sample_rate)

    emb_dist = min_embedding_distance
    if emb_dist is None:
        if embedder is None:
            from models.experts.embeddings import ECAPAEmbedder

            embedder = ECAPAEmbedder(device="cpu")
        emb_dist = compute_min_embedding_distance(
            candidate_stem,
            accepted_stems,
            embedder,
            sample_rate=sample_rate,
        )

    return assemble_stop_features(
        residual_energy_ratio=rer,
        vad_speech_prob=float(vad_speech_prob),
        min_embedding_distance=float(emb_dist),
        mixture_consistency_error=mce,
        attractor_stop_logit=attractor_stop_logit,
    )


class CountingFeatureExtractor:
    """
    Production entry point for stop-classifier feature extraction.

    Lazily holds VAD and ECAPA adapters so repeated peel decisions on the
    same mixture reuse loaded models.
    """

    def __init__(
        self,
        vad_backend: VADBackend = "energy",
        device: str = "cpu",
        embedder: ECAPAEmbedder | None = None,
    ) -> None:
        self.vad = VADAdapter(backend=vad_backend, device=device)
        self.device = device
        self._embedder = embedder

    @property
    def embedder(self) -> ECAPAEmbedder:
        if self._embedder is None:
            from models.experts.embeddings import ECAPAEmbedder

            self._embedder = ECAPAEmbedder(device=self.device)
        return self._embedder

    def extract_bundle(
        self,
        mixture: np.ndarray,
        accepted_stems: np.ndarray,
        candidate_stem: np.ndarray,
        attractor_stop_logit: float = 0.0,
        sample_rate: int = 16_000,
    ) -> StopFeatureBundle:
        """Return named features before vector assembly."""
        mix = _as_waveform(mixture)
        residual = compute_residual(mix, accepted_stems, candidate_stem)
        return StopFeatureBundle(
            residual_energy_ratio=residual_energy_ratio(mix, accepted_stems, candidate_stem),
            vad_speech_prob=self.vad.speech_prob(residual, sample_rate=sample_rate),
            min_embedding_distance=compute_min_embedding_distance(
                candidate_stem,
                accepted_stems,
                self.embedder,
                sample_rate=sample_rate,
            ),
            mixture_consistency_error=mixture_consistency_error(mix, accepted_stems, candidate_stem),
            attractor_stop_logit=float(attractor_stop_logit),
        )

    def extract(
        self,
        mixture: np.ndarray,
        accepted_stems: np.ndarray,
        candidate_stem: np.ndarray,
        attractor_stop_logit: float = 0.0,
        sample_rate: int = 16_000,
    ) -> np.ndarray:
        """Return [len(FEATURE_NAMES)] vector for ``StopClassifier.forward``."""
        return self.extract_bundle(
            mixture,
            accepted_stems,
            candidate_stem,
            attractor_stop_logit=attractor_stop_logit,
            sample_rate=sample_rate,
        ).to_vector()
