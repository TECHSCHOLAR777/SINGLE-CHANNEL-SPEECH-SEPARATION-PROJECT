"""Frozen SR-CorrNet var-2-5 backbone wrapper for CoRAL-Sep (BLUEPRINT §3, §15.1).

Applies the three patches on load:
  A: model.forward wrapped to cache pres after every pass — exposes p_k
  B: forward hook on model.encoder captures E(0), shape (1, T, 65, 128)
  C: forward hooks on each dec_block[i] capture decoder-stage inputs

Hard constraints (never change):
  - Batch size for unknown-speaker inference: 1  (assert in AttractorSplit.forward)
  - Sample rate: 8000 Hz (locked by checkpoint STFT params)
  - STFT window/hop/bins: 128 / 64 / 65
  - spk_query shape: (1, 7, 128) — slots 1-5 are speaker existence slots

Usage::

    wrapper = SRCorrNetWrapper(device="cpu")
    result = wrapper.forward(wav_8k)   # wav_8k: (1, L) at 8 kHz
    p_k     = result["p_k"]            # (1, 7) — attractor probs
    e0      = result["e0"]             # (1, T, 65, 128)
    stages  = result["dec_stages"]     # {0..3: (K, T, 65, 128)}
    n_hat   = result["n_active"]       # int in {2..5}
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

DEFAULT_HF_MODEL = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"


class SRCorrNetWrapper:
    """Frozen SR-CorrNet var-2-5 with CoRAL-Sep patch set.

    Never fine-tuned. Operates at 8 kHz only.
    """

    SAMPLE_RATE = 8000
    MAX_SPEAKERS = 5

    def __init__(
        self,
        device: str | torch.device = "cpu",
        repo_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        hf_model_id: str = DEFAULT_HF_MODEL,
        config_path: str | Path | None = None,
        prob_thres: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.repo_path = Path(repo_path) if repo_path else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.hf_model_id = hf_model_id
        self.config_path = Path(config_path) if config_path else None
        self.prob_thres = prob_thres
        self._inference: Any = None
        self._pres_cache: dict[str, Any] = {}
        self._e0_cache: dict[str, torch.Tensor] = {}
        self._dec_cache: dict[int, torch.Tensor] = {}

    @property
    def is_available(self) -> bool:
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            return False
        if self.repo_path is not None and self.repo_path.exists():
            return True
        if "sr_corrnet" in sys.modules:
            return True
        try:
            return importlib.util.find_spec("sr_corrnet") is not None
        except (ValueError, ModuleNotFoundError):
            return False

    def load(self) -> None:
        """Load the checkpoint and apply patches A, B, C. Idempotent."""
        if self._inference is not None:
            return
        if not self.is_available:
            raise RuntimeError(
                "SR-CorrNet-SS not installed. Run:\n"
                "  git clone https://github.com/dmlguq456/SR_CorrNet_SS.git\n"
                '  cd SR_CorrNet_SS && pip install -e ".[hub]"'
            )
        if self.repo_path is not None:
            repo = str(self.repo_path.resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

        from sr_corrnet import SSInference

        if self.checkpoint_path is not None:
            infer = SSInference.from_pretrained(
                config=str(self.config_path) if self.config_path else None,
                checkpoint_path=str(self.checkpoint_path),
                device=str(self.device),
            )
        else:
            infer = SSInference.from_pretrained(
                checkpoint_path=self.hf_model_id,
                device=str(self.device),
            )

        self._inference = infer
        self._apply_patches()

    def _apply_patches(self) -> None:
        base_nn = self._inference.engine.model
        self._patch_b_e0(base_nn)
        self._patch_c_dec(base_nn)
        self._patch_a_pres(base_nn)
        self._patch_prob_thres(base_nn)

    def _patch_a_pres(self, base_nn: Any) -> None:
        """Patch A (BLUEPRINT §15.1): wrap model.forward to cache pres every pass."""
        pres_cache = self._pres_cache
        original = base_nn.forward

        @functools.wraps(original)
        def _fwd(*args: Any, **kwargs: Any) -> Any:
            out, out_aux, pres = original(*args, **kwargs)
            if pres is not None:
                pres_cache["pres"] = pres
            return out, out_aux, pres

        base_nn.forward = _fwd

    def _patch_b_e0(self, base_nn: Any) -> None:
        """Patch B (BLUEPRINT §15.1): hook model.encoder to capture E(0)."""
        cache = self._e0_cache

        def _hook(module: Any, inp: Any, out: Any) -> None:
            cache["e0"] = out  # (1, T, 65, 128)

        base_nn.encoder.register_forward_hook(_hook)

    def _patch_c_dec(self, base_nn: Any) -> None:
        """Patch C (BLUEPRINT §15.1): hook each dec_block[i] to capture stage inputs."""
        cache = self._dec_cache

        for i, blk in enumerate(base_nn.dec_block):
            def _make_hook(idx: int):
                def _hook(module: Any, inp: Any, out: Any) -> None:
                    cache[idx] = inp[0]  # (B*K, T, F, 128); B=1 for unknown-spk inference
                return _hook

            blk.register_forward_hook(_make_hook(i))

    def _patch_prob_thres(self, base_nn: Any) -> None:
        """Expose prob_thres on AttractorSplit.forward (BLUEPRINT §15.4)."""
        original = base_nn.spk_split.forward
        base_nn.spk_split.forward = functools.partial(original, prob_thres=self.prob_thres)

    @property
    def base_nn(self) -> Any:
        """Raw nn.Module (Model instance). Triggers load on first access."""
        self.load()
        return self._inference.engine.model

    def forward(
        self,
        wav: torch.Tensor,
        n_spks: int | None = None,
    ) -> dict[str, Any]:
        """Run the frozen backbone and return all CoRAL-Sep internal signals.

        Args:
            wav: float32 tensor, shape (1, L) or (L,), at 8 kHz, std-normalized.
            n_spks: None for attractor-inferred count (batch_size must be 1);
                    int to force a known count (bypasses p_k).

        Returns dict with:
            waveforms: list of K (L,) float32 tensors at 8 kHz
            p_k: (1, 7) float tensor — attractor probs; active speaker slots 1-5
            n_active: int — number of slots above prob_thres (= N_hat when n_spks=None)
            e0: (1, T, 65, 128) float tensor — encoder output before enc_block
            dec_stages: dict mapping stage index to (K, T, 65, 128) float tensor
        """
        self.load()

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # (1, L)

        n_spks_arg = torch.tensor(n_spks) if n_spks is not None else None

        with torch.no_grad():
            result = self._inference.process_waveform(wav, n_spks=n_spks_arg)

        raw_pres = self._pres_cache.get("pres")
        p_k = raw_pres["probs"] if raw_pres is not None else None

        n_active: int | None = None
        if p_k is not None:
            n_active = int((p_k[0, 1:6] > self.prob_thres).sum().item())

        return {
            "waveforms": result.get("waveforms", []),
            "p_k": p_k,
            "n_active": n_active,
            "e0": self._e0_cache.get("e0"),
            "dec_stages": dict(self._dec_cache),
        }
