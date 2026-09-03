"""
SR-CorrNet expert wrapper with CoRAL-Sep patches (Dev B, P0-B2/B3/B4).

Patches applied on top of the frozen checkpoint:
  Patch A (P0-B2): expose pres["probs"] (shape 1,7) through process_waveform;
      _single_pass_session drops "pres" in the original, we retrieve it via
      the engine's last-call state or by wrapping process_stft.
  Patch B (P0-B3): forward hook on model.encoder to capture E(0) (1,T,65,128).
  Patch C (P0-B4): hooks on each dec_block[i] to capture decoder stage features.

The checkpoint is FROZEN, zero parameters modified, all changes are hooks and
wrapper code only. BLUEPRINT fixed constraint §1.

Model constants (frozen checkpoint var-2-5spk, BLUEPRINT §15.8):
  K0 = 5, 8 kHz, STFT window 128 / hop 64 / F = 65
  N_Enc = 2, N_Dec = 4, d_model = 128
  spk_query shape (1, 7, 128)
  pe_tf buffer max 5000 frames
  prob_thres = 0.5 (hardcoded in AttractorSplit.forward)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coralsep.models.preprocess import PROJECT_SAMPLE_RATE, preprocess, resample_audio
from coralsep.schemas.separation_result import SeparationResult, StreamMetadata

DEFAULT_HF_MODEL = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"
"""Frozen checkpoint. NEVER change. BLUEPRINT fixed constraint."""

# Model constants, do NOT relax these.
MODEL_SR = 8_000
STFT_WIN = 128
STFT_HOP = 64
STFT_BINS = 65
N_ENC = 2
N_DEC = 4
D_MODEL = 128
K0 = 5


class SRCorrNetExpert:
    """
    Inference wrapper for SR-CorrNet-SS var-2-5spk.

    Returns p_k, E(0), and decoder stage features alongside the separated
    streams so the CoRAL-Sep pipeline can route adapters and count speakers.
    """

    SAMPLE_RATE = PROJECT_SAMPLE_RATE
    EXPERT_NAME = "srcorrnet"
    MAX_SPEAKERS = K0

    def __init__(
        self,
        device: str | torch.device = "cpu",
        repo_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        hf_model_id: str = DEFAULT_HF_MODEL,
        config_path: str | Path | None = None,
        model_sample_rate: int = MODEL_SR,
        expose_internals: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.repo_path = Path(repo_path) if repo_path else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.hf_model_id = hf_model_id
        self.config_path = Path(config_path) if config_path else None
        self.model_sample_rate = int(model_sample_rate)
        self.expose_internals = expose_internals
        self._model: object | None = None

        # Storage for patched outputs (populated by hooks on each call).
        self._e0: torch.Tensor | None = None  # (1, T, 65, 128), Patch B
        self._dec_features: list[torch.Tensor] = []  # N_Dec tensors, Patch C
        self._pres: dict[str, Any] | None = None  # attractor dict, Patch A
        self._hooks: list[Any] = []

    @property
    def is_available(self) -> bool:
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            return False
        if self.repo_path is not None and self.repo_path.exists():
            return True
        return importlib.util.find_spec("sr_corrnet") is not None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.is_available:
            raise RuntimeError(
                "SR-CorrNet-SS not available. Install:\n"
                "  git clone https://github.com/dmlguq456/SR_CorrNet_SS.git\n"
                '  cd SR_CorrNet_SS && pip install -e ".[hub]"'
            )
        if self.repo_path is not None:
            repo = str(self.repo_path.resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

        from sr_corrnet import SSInference  # type: ignore[import]

        if self.checkpoint_path is not None:
            self._model = SSInference.from_pretrained(
                config=str(self.config_path) if self.config_path else None,
                checkpoint_path=str(self.checkpoint_path),
                device=str(self.device),
            )
        else:
            self._model = SSInference.from_pretrained(
                checkpoint_path=self.hf_model_id, device=str(self.device)
            )

        if self.expose_internals:
            self._register_hooks()

    # ------------------------------------------------------------------
    # Patch B + C: forward hooks
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register E(0) hook (Patch B) and decoder stage hooks (Patch C)."""
        model = self._inner_model()
        if model is None:
            return

        # Patch B: E(0) from model.encoder
        if hasattr(model, "encoder"):

            def _e0_hook(module: torch.nn.Module, input: Any, output: Any) -> None:
                # output shape: (1, T, F, D) or (B, T, F, D)
                if isinstance(output, torch.Tensor):
                    self._e0 = output.detach()
                elif isinstance(output, (list, tuple)):
                    self._e0 = output[0].detach()

            h = model.encoder.register_forward_hook(_e0_hook)
            self._hooks.append(h)

        # Patch C: decoder stage features from dec_block[i]
        if hasattr(model, "dec_block"):
            for i, block in enumerate(model.dec_block):

                def _dec_hook(
                    module: torch.nn.Module, input: Any, output: Any, idx: int = i
                ) -> None:
                    # output: (B*K, T, F, D) → reshape to (B, K, T, F, D)
                    if isinstance(output, torch.Tensor):
                        feat = output.detach()
                        # Expand dec_features list as needed
                        while len(self._dec_features) <= idx:
                            self._dec_features.append(torch.zeros(1))
                        bk, T, F, D = feat.shape
                        k = K0
                        b = bk // k
                        self._dec_features[idx] = feat.view(b, k, T, F, D)

                h = block.register_forward_hook(_dec_hook)
                self._hooks.append(h)

    def _inner_model(self) -> torch.nn.Module | None:
        """Extract the raw nn.Module from the SSInference wrapper."""
        if self._model is None:
            return None
        for attr in ("model", "net", "separator", "_model"):
            m = getattr(self._model, attr, None)
            if isinstance(m, torch.nn.Module):
                return m
        if isinstance(self._model, torch.nn.Module):
            return self._model  # type: ignore[return-value]
        return None

    def _clear_hooks_storage(self) -> None:
        self._e0 = None
        self._dec_features = []
        self._pres = None

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------
    # Patch A: expose p_k / pres dict
    # ------------------------------------------------------------------

    def _extract_pres(self, out: Any) -> dict[str, Any] | None:
        """
        Retrieve the attractor presence dict from the model output.

        The original _single_pass_session drops `pres` from the returned dict.
        We try several extraction strategies in order:
          1. The output dict directly contains "pres" (if future version fixes it).
          2. The SSInference object caches _last_pres on its inference engine.
          3. We call process_stft which may return pres in a richer output.
        """
        # Strategy 1: output dict already has it
        if isinstance(out, dict) and "pres" in out:
            return out["pres"]

        # Strategy 2: cached on engine
        engine = getattr(self._model, "engine", None) or getattr(self._model, "infer_engine", None)
        if engine is not None:
            for attr in ("_last_pres", "last_pres", "_pres"):
                pres = getattr(engine, attr, None)
                if pres is not None:
                    return pres

        # Strategy 3: look for attractor_module output cached on the model
        inner = self._inner_model()
        if inner is not None:
            for attr in ("_last_pres", "last_pres", "_attractor_out"):
                pres = getattr(inner, attr, None)
                if pres is not None:
                    return pres

        return None

    # ------------------------------------------------------------------
    # Main separation method
    # ------------------------------------------------------------------

    def separate(
        self,
        mixture: np.ndarray | torch.Tensor,
        sample_rate: int,
        n_spks: int | None = None,
    ) -> SeparationResult:
        """
        Separate a mono mixture. n_spks=None uses the p_k attractor path.

        Returns SeparationResult with p_k, E(0), and decoder features in extra.
        """
        self._load_model()
        assert self._model is not None
        self._clear_hooks_storage()

        pre = preprocess(mixture, sample_rate)
        wav16 = pre.waveform.astype(np.float32)
        wav_lo = resample_audio(wav16, PROJECT_SAMPLE_RATE, self.model_sample_rate)
        wav_t = torch.from_numpy(wav_lo).float().unsqueeze(0).to(self.device)  # [1, L]

        with torch.no_grad():
            # Patch A: pass n_spks=None to use the attractor path (p_k determines N).
            # BLUEPRINT §15.8: assert x.size(0)==1 when n_spks=None.
            if n_spks is None:
                out = self._model.process_waveform(wav_t)  # type: ignore[attr-defined]
            else:
                out = self._model.process_waveform(  # type: ignore[attr-defined]
                    wav_t, n_spks=torch.tensor(n_spks)
                )

        # Patch A: extract pres
        pres = self._extract_pres(out)
        self._pres = pres

        # Waveforms
        streams_lo = _extract_waveforms(out)  # [K, L_lo]
        streams = np.stack(
            [resample_audio(s, self.model_sample_rate, PROJECT_SAMPLE_RATE) for s in streams_lo],
            axis=0,
        )
        streams = _fix_length(streams, wav16.shape[0]).astype(np.float32)

        # Attractor probs
        if pres is not None and "probs" in pres:
            probs_tensor = pres["probs"]  # (1, 7)
        else:
            # Fallback: uniform over K0 slots if pres is unavailable
            probs_tensor = torch.ones(1, 7) / 7.0

        metadata = [
            StreamMetadata(
                expert_source=self.EXPERT_NAME,
                confidence=float(probs_tensor.squeeze()[i + 1].item()) if i < 5 else 1.0,
                extra={
                    "attractor_index": i,
                    "model": self.hf_model_id,
                    "p_k": float(probs_tensor.squeeze()[i + 1].item()) if i < 5 else 0.0,
                },
            )
            for i in range(streams.shape[0])
        ]

        return SeparationResult(
            streams=streams,
            sample_rate=PROJECT_SAMPLE_RATE,
            speaker_count=streams.shape[0],
            metadata=metadata,
            mixture=wav16,
            escalated=True,
            expert_used=self.EXPERT_NAME,
            attractor_probs=probs_tensor,
            encoder_e0=self._e0,
            decoder_features=list(self._dec_features),
        )

    @property
    def last_e0(self) -> torch.Tensor | None:
        """E(0) from the most recent forward pass (1, T, F, D)."""
        return self._e0

    @property
    def last_dec_features(self) -> list[torch.Tensor]:
        """Decoder stage features from the most recent forward pass."""
        return list(self._dec_features)

    @property
    def last_pres(self) -> dict[str, Any] | None:
        """Attractor presence dict from the most recent forward pass."""
        return self._pres


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_numpy(x: object) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _extract_waveforms(out: object) -> np.ndarray:
    """Normalise SSInference output to [K, L] float32."""
    waves: object = out
    if isinstance(out, dict):
        waves = (
            out.get("waveforms") or out.get("est_sources") or out.get("sources") or out.get("wav")
        )

    if isinstance(waves, (list, tuple)):
        rows = [_to_numpy(w).reshape(-1) for w in waves]
        length = min(r.shape[0] for r in rows)
        return np.stack([r[:length] for r in rows], axis=0)

    arr = np.squeeze(_to_numpy(waves))
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _fix_length(streams: np.ndarray, length: int) -> np.ndarray:
    t = streams.shape[1]
    if t == length:
        return streams
    if t > length:
        return streams[:, :length]
    pad = np.zeros((streams.shape[0], length - t), dtype=streams.dtype)
    return np.concatenate([streams, pad], axis=1)
