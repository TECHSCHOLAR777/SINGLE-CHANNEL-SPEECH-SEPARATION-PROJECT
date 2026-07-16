"""CALM-Sep per-chunk inference order (BLUEPRINT §6.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.band_recovery import BandRecoveryHead, apply_band_recovery, zero_pad_8k_to_16k
from models.condition import ConditionAnalyzer, ConditionVector
from models.confidence import ConfidenceSubsystem
from models.counting import CountingSubsystem, residual_energy_fraction
from models.gate import GateMLP
from models.lora import LoRALibrary, register_lora
from models.preprocess import CALMSEP_SR, OUTPUT_SR, preprocess_calmsep
from pipeline.chunker import chunk_audio
from pipeline.stitcher import CalmSepStitcher, upsample_streams_8k_to_16k
from schemas.separation_result import SeparationResult, StreamMetadata


class CalmSepEngine:
    """End-to-end CALM-Sep engine with config off-switches for every mechanism."""

    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: str | Path | None = None,
        *,
        use_adapters: bool = True,
        use_gate: bool = True,
        use_residual_sweep: bool = True,
        use_band_recovery: bool = True,
        base_only: bool = False,
        prob_thres: float = 0.5,
        wrapper: Any | None = None,
    ) -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.use_adapters = use_adapters and not base_only
        self.use_gate = use_gate and not base_only
        self.use_residual_sweep = use_residual_sweep
        self.use_band_recovery = use_band_recovery
        self.base_only = base_only
        self.prob_thres = prob_thres

        self.wrapper = wrapper
        self.lora: LoRALibrary | None = None
        self.condition = ConditionAnalyzer()
        self.gate = GateMLP(enabled=self.use_gate)
        self.counting = CountingSubsystem(
            prob_thres=prob_thres, enabled_sweep=use_residual_sweep
        )
        self.confidence = ConfidenceSubsystem()
        self.band_head = BandRecoveryHead()
        self.band_head.enabled = use_band_recovery
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if self.wrapper is None:
            from models.srcorrnet import SRCorrNetWrapper

            self.wrapper = SRCorrNetWrapper(
                device=self.device,
                checkpoint_path=self.checkpoint_path,
                prob_thres=self.prob_thres,
            )
        if not self.wrapper.is_available:
            raise RuntimeError(
                "SR-CorrNet checkpoint unavailable. Install sr_corrnet and download "
                "the frozen checkpoint, or construct CalmSepEngine(wrapper=...) with a mock."
            )
        self.wrapper.load()
        if self.use_adapters:
            self.lora = register_lora(self.wrapper.base_nn)
        self.condition.to(self.device)
        self.gate.to(self.device)
        self.band_head.to(self.device)
        self._loaded = True

    def _apply_gates(self, gates: dict[str, float]) -> None:
        if self.lora is not None:
            self.lora.set_gates(gates)

    def process_chunk(
        self,
        wav_8k: np.ndarray,
        wav_16k: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run §6.2 order on one chunk. Returns internals + 8 kHz streams."""
        self.load()
        assert self.wrapper is not None

        # 2. Level-1 DSP
        level1 = self.condition.forward_level1(wav_8k, CALMSEP_SR)

        # 3. Pass 1: frozen forward → E(0), p_k (gates at 0 for identity)
        if self.lora is not None:
            self.lora.set_gates({n: 0.0 for n in self.lora.adapter_names})
        wav_t = torch.from_numpy(wav_8k).float()
        pass1 = self.wrapper.forward(wav_t, n_spks=None)
        e0 = pass1.get("e0")
        if e0 is None:
            # Synthetic fallback shape for mock wrappers.
            e0 = torch.zeros(1, 8, 65, 128)

        # 4. Level-2
        level2 = self.condition.forward_level2(e0.to(self.device), level1)
        cond = ConditionVector(
            snr_db=level1["snr_db"],
            t60_s=level2["t60_s"],
            codec_class=level1["codec_class"],
            codec_class_idx=level1["codec_class_idx"],
            codec_bitrate_bps=level1["codec_bitrate_bps"],
            voiced_density=level1["voiced_density"],
            count_prior=level2["count_prior"],
        )

        # 5. Gate
        gates_raw = self.gate(cond, apply_ema=True)
        gates = self.gate.as_adapter_scalars(gates_raw)  # type: ignore[arg-type]
        if not self.use_gate or self.base_only:
            gates = {n: 0.0 for n in ("reverb", "noise", "codec")}

        # 6. Pass 2 with adapters
        self._apply_gates(gates)
        pass2 = self.wrapper.forward(wav_t, n_spks=None)
        streams_list = pass2.get("waveforms") or []
        if streams_list:
            streams_8k = np.stack(
                [np.asarray(w, dtype=np.float32).reshape(-1) for w in streams_list], axis=0
            )
        else:
            streams_8k = np.zeros((2, len(wav_8k)), dtype=np.float32)

        p_k = pass2.get("p_k")
        if p_k is None:
            p_k = torch.zeros(1, 7)
            p_k[0, 1:3] = 0.9

        # 7. Counting (+ optional residual sweep)
        def _resid(n: int) -> float:
            # Force known count forward for residual probe when possible.
            try:
                forced = self.wrapper.forward(wav_t, n_spks=n)
                forced_streams = forced.get("waveforms") or []
                if not forced_streams:
                    return 1.0
                arr = np.stack([np.asarray(w).reshape(-1) for w in forced_streams], axis=0)
                return residual_energy_fraction(wav_8k, arr)
            except Exception:
                return 1.0

        decision = self.counting.decide(
            p_k,
            cond.count_prior,
            residual_fn=_resid if self.use_residual_sweep else None,
        )
        # If count differs from current streams, re-run with forced n.
        if decision.n_hat != streams_8k.shape[0]:
            try:
                forced = self.wrapper.forward(wav_t, n_spks=decision.n_hat)
                fl = forced.get("waveforms") or []
                if fl:
                    streams_8k = np.stack(
                        [np.asarray(w, dtype=np.float32).reshape(-1) for w in fl], axis=0
                    )
                    p_k = forced.get("p_k", p_k)
            except Exception:
                pass

        # Truncate/pad streams to match n_hat for consistency.
        n_hat = int(decision.n_hat)
        if streams_8k.shape[0] > n_hat:
            streams_8k = streams_8k[:n_hat]
        elif streams_8k.shape[0] < n_hat:
            pad = np.zeros((n_hat - streams_8k.shape[0], streams_8k.shape[1]), dtype=np.float32)
            streams_8k = np.concatenate([streams_8k, pad], axis=0)

        mix_16 = wav_16k if wav_16k is not None else zero_pad_8k_to_16k(wav_8k)

        # 8. Band recovery
        if self.use_band_recovery:
            br = apply_band_recovery(streams_8k, mix_16, self.band_head)
            streams_16k = br.waveforms_16k
            br_applied = br.applied
        else:
            streams_16k = upsample_streams_8k_to_16k(streams_8k)
            br_applied = [False] * streams_16k.shape[0]

        # 9. Confidence + completeness
        pk_np = p_k.detach().cpu().numpy() if isinstance(p_k, torch.Tensor) else np.asarray(p_k)
        conf = self.confidence(
            p_k=pk_np,
            streams=streams_8k,
            mixture=wav_8k,
            dec_stages=pass2.get("dec_stages"),
            condition_vec=cond.to_tensor().numpy(),
            sample_rate=CALMSEP_SR,
        )

        return {
            "streams_8k": streams_8k,
            "streams_16k": streams_16k,
            "p_k": pk_np,
            "n_hat": n_hat,
            "count_posterior": decision.posterior,
            "gates": gates,
            "condition": cond.to_dict(),
            "completeness": conf.completeness,
            "ood_flag": conf.ood_flag,
            "per_stream_confidence": conf.per_stream,
            "band_recovery_applied": br_applied,
            "sweep_triggered": decision.sweep_triggered,
        }

    def __call__(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        """Full recording: chunk → process → stitch → SeparationResult at 16 kHz."""
        prep = preprocess_calmsep(mixture, sample_rate)
        chunks = chunk_audio(mixture, sample_rate)
        stitcher = CalmSepStitcher(sample_rate=OUTPUT_SR)
        last: dict[str, Any] = {}

        if len(chunks) == 1:
            last = self.process_chunk(chunks[0].wav_8k, chunks[0].wav_16k)
            streams = last["streams_16k"]
        else:
            for ch in chunks:
                last = self.process_chunk(ch.wav_8k, ch.wav_16k)
                stitcher.add_chunk(last["streams_16k"])
            stitched = stitcher.finalize()
            streams = stitched.streams_16k
            if streams.shape[0] == 0:
                streams = last.get("streams_16k", np.zeros((2, 1), dtype=np.float32))

        k = int(streams.shape[0])
        confs = last.get("per_stream_confidence") or [1.0] * k
        if len(confs) < k:
            confs = (confs + [0.5] * k)[:k]
        metadata = [
            StreamMetadata(expert_source="calm-sep", confidence=float(confs[i]))
            for i in range(k)
        ]
        return SeparationResult(
            streams=streams.astype(np.float32),
            sample_rate=OUTPUT_SR,
            speaker_count=k,
            metadata=metadata,
            mixture=prep.wav_16k,
            expert_used="calm-sep",
            p_k=last.get("p_k"),
            gate_vector=last.get("gates"),
            completeness=last.get("completeness"),
            ood_flag=bool(last.get("ood_flag", False)),
            condition_estimates=last.get("condition"),
            count_posterior=last.get("count_posterior"),
        )


class MockCalmSepWrapper:
    """Weight-free wrapper for CI / demo when checkpoint is absent."""

    is_available = True

    def __init__(self, n_speakers: int = 2) -> None:
        self.n_speakers = n_speakers
        self.base_nn = torch.nn.Linear(4, 4)  # dummy for LoRA tests
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def forward(self, wav: torch.Tensor, n_spks: int | None = None) -> dict[str, Any]:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        n = int(n_spks) if n_spks is not None else self.n_speakers
        n = int(np.clip(n, 2, 5))
        L = wav.shape[-1]
        waves = []
        for i in range(n):
            # Simple band-split mock.
            spec = torch.fft.rfft(wav[0])
            freqs = torch.fft.rfftfreq(L, d=1.0 / CALMSEP_SR)
            mask = (freqs >= i * 1000) & (freqs < (i + 1) * 1500 + 500)
            out = torch.fft.irfft(spec * mask.float(), n=L)
            waves.append(out.detach())
        p_k = torch.zeros(1, 7)
        p_k[0, 1 : 1 + n] = 0.9
        e0 = torch.randn(1, max(L // 64, 1), 65, 128) * 0.01
        return {
            "waveforms": waves,
            "p_k": p_k,
            "n_active": n,
            "e0": e0,
            "dec_stages": {0: e0.unsqueeze(1).expand(1, n, -1, 65, 128)[0]},
        }
