"""
Full CALM-Sep inference pipeline (Dev C, P4-C3).

Orchestrates:
  1. Dual-branch preprocessing (8 kHz + 16 kHz STFT).
  2. Level-1 condition analysis (DSP features).
  3. Chunking into 2.4 s / 0.8 s-step windows.
  4. Per-chunk SR-CorrNet separation (Pass 1) with LoRA gate routing.
  5. Level-2 condition analysis (E(0) pooled features: T60, count prior).
  6. Three-vote speaker counting.
  7. Pass 2 (optional second pass if count changes).
  8. Overlap-add stitching with ECAPA permutation tie-break.
  9. Band recovery (8 kHz → 16 kHz, guarded by dual-metric).
  10. Quality flag (completeness, OOD).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from models.preprocess import calmsep_preprocess, CalmSepPreprocessedAudio, CALMSEP_SAMPLE_RATE
from pipeline.chunker import Chunker, AudioChunk
from pipeline.stitcher import ChunkStitcher, StitchedOutput
from schemas.separation_result import SeparationResult, StreamMetadata

# Sentinel when optional modules are unavailable.
_MISSING = object()

_N_VALID = {2, 3, 4, 5}


@dataclass
class InferenceCfg:
    """Runtime knobs for the inference pipeline."""

    device: str = "cpu"
    """Torch device string."""

    default_n_speakers: int | None = None
    """Override the three-vote count (for oracle / ablation experiments)."""

    run_band_recovery: bool = True
    """Whether to apply the band recovery head (false → zero-pad to 16 kHz)."""

    run_pass2: bool = True
    """Whether to run a second SR-CorrNet pass when count changes after Level 2."""

    gate_ema_alpha: float = 0.7
    """EMA smoothing factor for gate values across chunks."""

    target_dbfs: float = -26.0


@dataclass
class PipelineResult:
    """Full pipeline output."""

    streams_8k: np.ndarray
    """(K, T_8k) float32 separated streams at 8 kHz."""

    streams_16k: np.ndarray
    """(K, T_16k) float32 streams upsampled/band-recovered to 16 kHz."""

    speaker_count: int
    gate_vector: dict[str, float] = field(default_factory=dict)
    completeness_prob: float = 1.0
    ood_flag: bool = False
    vote_details: dict = field(default_factory=dict)
    mixture_8k: np.ndarray | None = None
    mixture_16k: np.ndarray | None = None


class CalmSepPipeline:
    """
    End-to-end CALM-Sep inference pipeline.

    Lazy-loads optional components (LoRA, gate, band recovery, DNSMOS) so
    the pipeline runs in a minimal environment with just SR-CorrNet.

    Args:
        expert: SRCorrNetExpert instance (required).
        lora_library: LoRALibrary for adapter routing (optional).
        gate_net: GateNetwork (optional).
        level2_analyzer: Level2Analyzer (optional).
        three_vote_counter: ThreeVoteCounter (optional).
        band_recovery_head: BandRecoveryHead (optional).
        confidence_head: StreamConfidenceHead (optional).
        completeness_head: CompletenessHead (optional).
        ood_detector: MahalanobisOOD (optional).
        dnsmos_scorer: DnsmosScorer (optional).
        cfg: InferenceCfg.
    """

    def __init__(
        self,
        expert,
        lora_library=None,
        gate_net=None,
        level2_analyzer=None,
        three_vote_counter=None,
        band_recovery_head=None,
        confidence_head=None,
        completeness_head=None,
        ood_detector=None,
        dnsmos_scorer=None,
        cfg: InferenceCfg | None = None,
    ) -> None:
        self.expert = expert
        self.lora = lora_library
        self.gate = gate_net
        self.level2 = level2_analyzer
        self.counter = three_vote_counter
        self.band_recovery = band_recovery_head
        self.confidence_head = confidence_head
        self.completeness_head = completeness_head
        self.ood_detector = ood_detector
        self.dnsmos = dnsmos_scorer
        self.cfg = cfg or InferenceCfg()

    @torch.no_grad()
    def run(
        self,
        mixture: np.ndarray,
        sample_rate: int,
    ) -> PipelineResult:
        """
        Run the full CALM-Sep pipeline on a mono mixture.

        Args:
            mixture: [T] float32 mono mixture at sample_rate.
            sample_rate: Input sample rate in Hz.

        Returns:
            PipelineResult with 8 kHz and 16 kHz separated streams.
        """
        # 1. Dual-branch preprocessing.
        proc = calmsep_preprocess(mixture, sample_rate, target_dbfs=self.cfg.target_dbfs)

        # 2. Level-1 condition analysis (DSP).
        condition_l1 = _level1_condition(proc)

        # 3. Gate routing from Level-1 features.
        gate_vec = self._compute_gate(condition_l1)

        # 4. Chunk + separate (Pass 1).
        chunker = Chunker(proc.waveform_8k, proc.waveform_16k)
        stitcher = ChunkStitcher(
            n_speakers=self.cfg.default_n_speakers,
            sample_rate=CALMSEP_SAMPLE_RATE,
            use_ecapa=True,
        )

        chunk_results: list[SeparationResult] = []
        e0_list: list[torch.Tensor] = []
        attractor_probs_list: list[torch.Tensor] = []

        for chunk in chunker:
            result = self._separate_chunk(chunk, gate_vec, n_spks=self.cfg.default_n_speakers)
            chunk_results.append(result)
            if result.encoder_e0 is not None:
                e0_list.append(result.encoder_e0)
            if result.attractor_probs is not None:
                attractor_probs_list.append(result.attractor_probs)
            emb = self._ecapa_embeddings(result)
            stitcher.feed_chunk(result.streams, embeddings=emb)

        # 5. Level-2 analysis on pooled E(0).
        count_prior_probs = None
        if e0_list and self.level2 is not None:
            e0_pooled = _pool_e0(e0_list)
            l2_feat = self.level2(e0_pooled)
            count_prior_probs = l2_feat[:, 2:].softmax(dim=-1)  # N∈{2,3,4,5} logits

        # 6. Three-vote counting.
        n_speakers = self._three_vote_count(
            attractor_probs_list,
            count_prior_probs,
            chunk_results,
        )
        n_speakers = max(2, min(5, n_speakers))  # clamp N∈{2,3,4,5}

        # 7. Pass 2 (if count changed and run_pass2 is enabled).
        if (
            self.cfg.run_pass2
            and n_speakers != stitcher.n_speakers
            and stitcher.n_speakers is not None
        ):
            stitcher.reset()
            for chunk in chunker:
                result = self._separate_chunk(chunk, gate_vec, n_spks=n_speakers)
                emb = self._ecapa_embeddings(result)
                stitcher.feed_chunk(result.streams, embeddings=emb)

        # 8. Overlap-add stitching.
        T_8k = len(proc.waveform_8k)
        stitched: StitchedOutput = stitcher.finalize(total_samples=T_8k)

        # 9. Band recovery (8 kHz → 16 kHz).
        T_16k = len(proc.waveform_16k)
        streams_16k = self._band_recover(stitched.waveforms, proc.waveform_16k, T_16k)

        # 10. Quality flags.
        completeness_prob, ood_flag = self._quality_flags(e0_list, stitched.waveforms)

        return PipelineResult(
            streams_8k=stitched.waveforms,
            streams_16k=streams_16k,
            speaker_count=stitched.speaker_count,
            gate_vector=gate_vec,
            completeness_prob=completeness_prob,
            ood_flag=ood_flag,
            mixture_8k=proc.waveform_8k,
            mixture_16k=proc.waveform_16k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _separate_chunk(
        self,
        chunk: AudioChunk,
        gate_vec: dict[str, float],
        n_spks: int | None,
    ) -> SeparationResult:
        """Run SR-CorrNet on one chunk with LoRA gate applied."""
        wav = chunk.waveform_8k
        if self.lora is not None:
            # LoRA gate injection handled inside expert via context manager.
            with self.lora.forward_context(gate_vec):
                return self.expert.separate(wav, sample_rate=CALMSEP_SAMPLE_RATE, n_spks=n_spks)
        return self.expert.separate(wav, sample_rate=CALMSEP_SAMPLE_RATE, n_spks=n_spks)

    def _compute_gate(self, condition: dict) -> dict[str, float]:
        """Level-1 DSP features → gate values via GateNetwork (or oracle 1.0 s)."""
        if self.gate is None:
            return {"reverb": 1.0, "noise": 1.0, "codec": 1.0}

        cond_tensor = _condition_dict_to_tensor(condition).unsqueeze(0)
        gate_dict = self.gate.gate_dict(cond_tensor)
        # GateSmoother EMA is applied inside GateNetwork if configured.
        return {k: float(v) for k, v in gate_dict.items()}

    def _ecapa_embeddings(self, result: SeparationResult) -> np.ndarray | None:
        """Extract ECAPA embeddings from per-stream metadata if present."""
        embs = []
        for m in result.metadata:
            if m.embedding is not None:
                embs.append(m.embedding)
        if len(embs) == result.speaker_count:
            return np.stack(embs, axis=0)
        return None

    def _three_vote_count(
        self,
        probs_list: list[torch.Tensor],
        count_prior_probs: torch.Tensor | None,
        chunk_results: list[SeparationResult],
    ) -> int:
        if self.cfg.default_n_speakers is not None:
            return self.cfg.default_n_speakers

        if self.counter is None:
            # Fallback: majority vote from attractor probabilities.
            if not probs_list:
                return chunk_results[0].speaker_count if chunk_results else 2
            counts = []
            for p in probs_list:
                cnt = int((p.squeeze() > 0.5).sum().clamp(2, 5).item())
                counts.append(cnt)
            return max(set(counts), key=counts.count)

        # Pool attractor probs across chunks.
        probs = torch.stack(probs_list).mean(dim=0)  # (1, 7) mean
        count_prior = count_prior_probs.mean(0).unsqueeze(0) if count_prior_probs is not None else None

        # Residual sweep uses the first chunk as a proxy (full audio too long).
        mix_proxy = chunk_results[0].mixture if chunk_results else None
        sep_proxy = chunk_results[0].streams if chunk_results else None

        result = self.counter.estimate(
            probs=probs,
            count_prior_probs=count_prior,
            mixture=mix_proxy,
            separated=sep_proxy,
        )
        return int(result.get("final_count", 2))

    def _band_recover(
        self,
        streams_8k: np.ndarray,
        mixture_16k: np.ndarray,
        T_16k: int,
    ) -> np.ndarray:
        """Apply band recovery head (guarded) or zero-pad to 16 kHz."""
        if not self.cfg.run_band_recovery or self.band_recovery is None:
            return _zero_pad_to_16k(streams_8k, T_16k)

        from models.band_recovery import apply_band_recovery_guarded
        return apply_band_recovery_guarded(
            self.band_recovery,
            separated_8k=streams_8k,
            mixture_16k=mixture_16k,
            references_8k=None,
            dnsmos_scorer=self.dnsmos,
            device=self.cfg.device,
        )

    def _quality_flags(
        self,
        e0_list: list[torch.Tensor],
        streams: np.ndarray,
    ) -> tuple[float, bool]:
        """Return (completeness_prob, ood_flag)."""
        completeness_prob = 1.0
        ood_flag = False

        if e0_list and self.completeness_head is not None:
            e0_pooled = _pool_e0(e0_list)
            mag = e0_pooled.abs().mean(dim=(-2, -1))  # (1, T)
            cp = self.completeness_head(mag.mean(dim=-1, keepdim=True))
            completeness_prob = float(cp.squeeze().sigmoid().item())

        if e0_list and self.ood_detector is not None:
            e0_pooled = _pool_e0(e0_list)
            feat_np = e0_pooled.squeeze().mean(0).cpu().numpy()  # (T, F, D) → (F*D,)
            feat_np = feat_np.reshape(-1)
            ood_flag = not self.ood_detector.discount(feat_np)

        return completeness_prob, ood_flag


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def _level1_condition(proc: CalmSepPreprocessedAudio) -> dict:
    """Compute Level-1 DSP features from the 8 kHz preprocessed audio."""
    try:
        from models.condition import level1_features
        return level1_features(proc.waveform_8k)
    except Exception:
        return {
            "snr_est_db": 0.0,
            "codec_bw_ratio": 1.0,
            "voiced_density": 0.5,
            "total_energy_db": -26.0,
        }


def _condition_dict_to_tensor(cond: dict) -> torch.Tensor:
    """Level-1 dict → (4,) float32 tensor in canonical order."""
    keys = ["snr_est_db", "codec_bw_ratio", "voiced_density", "total_energy_db"]
    vals = [float(cond.get(k, 0.0)) for k in keys]
    return torch.tensor(vals, dtype=torch.float32)


def _pool_e0(e0_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate and mean-pool E(0) tensors from multiple chunks.

    Each tensor is (1, T, F, D). Returns (1, T_pooled, F, D) — all frames
    concatenated along T then mean-pooled to the median chunk length.
    """
    # Stack along T axis.
    chunks = [e.squeeze(0) for e in e0_list]  # each (T, F, D)
    cat = torch.cat(chunks, dim=0)              # (T_total, F, D)
    return cat.unsqueeze(0)                     # (1, T_total, F, D)


def _zero_pad_to_16k(streams_8k: np.ndarray, T_16k: int) -> np.ndarray:
    """
    Zero-pad 8 kHz streams to 16 kHz by interleaving with zeros.

    This is the worst-case fallback when band recovery is disabled.
    """
    K, T_8k = streams_8k.shape
    out = np.zeros((K, T_16k), dtype=np.float32)
    # Every other 16 kHz sample corresponds to 8 kHz.
    t = min(T_8k, T_16k // 2)
    out[:, ::2][:, :t] = streams_8k[:, :t]
    return out
