"""
SR-CorrNet expert wrapper (expensive time-frequency separator).

Phase 0 baseline and Phase 1 integration target.
Source: github.com/dmlguq456/SR_CorrNet

SR-CorrNet is not pip-installable; clone the repository and set
`srcorrnet_repo` in configs/baseline.yaml. If unavailable, the baseline
runner skips this expert and reports SepFormer-only results.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from models.preprocess import preprocess
from schemas.separation_result import SeparationResult, StreamMetadata


class SRCorrNetExpert:
    """Inference wrapper for SR-CorrNet-B[2-5] with attractor-based counting."""

    SAMPLE_RATE = 16000
    EXPERT_NAME = "srcorrnet"
    MAX_SPEAKERS = 5

    def __init__(
        self,
        device: str | torch.device = "cpu",
        repo_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        num_speakers: int = 3,
    ) -> None:
        self.device = torch.device(device)
        self.repo_path = Path(repo_path) if repo_path else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.num_speakers = num_speakers
        self._model: torch.nn.Module | None = None
        self._available = False

    @property
    def is_available(self) -> bool:
        """True if the SR-CorrNet repository and checkpoint are configured."""
        if self.repo_path is None or not self.repo_path.exists():
            return False
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            return False
        return True

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.is_available:
            raise RuntimeError(
                "SR-CorrNet is not configured. Clone github.com/dmlguq456/SR_CorrNet "
                "and set srcorrnet_repo / srcorrnet_checkpoint in configs/baseline.yaml."
            )

        assert self.repo_path is not None
        repo = self.repo_path.resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        # SR-CorrNet layout varies by release; try common entry points.
        model = self._try_load_srcorrnet(repo)
        if model is None:
            raise ImportError(
                f"Could not load SR-CorrNet from {repo}. "
                "Ensure the repo is cloned and dependencies installed."
            )

        model.to(self.device)
        model.eval()
        self._model = model
        self._available = True

    def _try_load_srcorrnet(self, repo: Path) -> torch.nn.Module | None:
        """Attempt to import and instantiate SR-CorrNet from the cloned repo."""
        candidates = [
            ("models.SR_CorrNet", "SR_CorrNet"),
            ("model", "SR_CorrNet"),
            ("src.models", "SR_CorrNet"),
        ]
        for module_name, class_name in candidates:
            spec_path = repo / f"{module_name.replace('.', '/')}.py"
            if not spec_path.exists() and module_name.count(".") > 0:
                continue
            try:
                mod = importlib.import_module(module_name)
                cls = getattr(mod, class_name, None)
                if cls is None:
                    continue
                if self.checkpoint_path:
                    return self._load_checkpoint(cls, self.checkpoint_path)
                return cls()
            except (ImportError, AttributeError, OSError):
                continue
        return None

    @staticmethod
    def _load_checkpoint(model_cls: type, ckpt_path: Path) -> torch.nn.Module:
        model = model_cls()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model

    def separate(self, mixture: np.ndarray | torch.Tensor, sample_rate: int) -> SeparationResult:
        """
        Separate mixture via SR-CorrNet.

        Returns SeparationResult with K streams, attractor confidence in metadata.extra.
        """
        self._load_model()
        assert self._model is not None

        pre = preprocess(mixture, sample_rate)
        wav = pre.waveform
        wav_t = torch.from_numpy(wav).to(self.device)

        with torch.no_grad():
            output = self._model(wav_t.unsqueeze(0))
            streams, confidences, attractors, stop_logit = self._parse_output(output, wav.shape[0])

        mixture_np = wav
        metadata = [
            StreamMetadata(
                expert_source=self.EXPERT_NAME,
                confidence=float(confidences[i]) if i < len(confidences) else 1.0,
                extra={
                    "attractor_index": i,
                    "attractor": attractors[i] if i < len(attractors) else None,
                    "stop_logit": stop_logit,
                },
            )
            for i in range(streams.shape[0])
        ]

        return SeparationResult(
            streams=streams,
            sample_rate=sample_rate,
            speaker_count=streams.shape[0],
            metadata=metadata,
            mixture=mixture_np.astype(np.float32),
            escalated=True,
            expert_used=self.EXPERT_NAME,
        )

    def _parse_output(
        self, output: torch.Tensor | tuple | dict, time_len: int
    ) -> tuple[np.ndarray, list[float], list[np.ndarray | None], float | None]:
        """
        Normalize SR-CorrNet forward output to [K, T] numpy, confidences,
        per-stream attractor vectors, and optional TDA stop logit.
        """
        attractors: list[np.ndarray | None] = []
        stop_logit: float | None = None

        if isinstance(output, dict):
            est = output.get("est_sources")
            if est is None:
                est = output.get("sources")
            if est is None:
                est = output["wav"]
            conf = output.get("confidence", output.get("confidences", []))
            raw_attractors = output.get("attractors")
            if raw_attractors is None:
                raw_attractors = output.get("attractor_vectors")
            stop = output.get("stop_logit")
            if stop is None:
                stop = output.get("stop_token")
        elif isinstance(output, tuple):
            est = output[0]
            conf = output[1] if len(output) > 1 else []
            raw_attractors = output[2] if len(output) > 2 else None
            stop = output[3] if len(output) > 3 else None
        else:
            est = output
            conf = []
            raw_attractors = None
            stop = None

        if isinstance(raw_attractors, torch.Tensor):
            att_arr = raw_attractors.detach().cpu().numpy()
            if att_arr.ndim == 2:
                attractors = [att_arr[i].astype(np.float32) for i in range(att_arr.shape[0])]
            elif att_arr.ndim == 1:
                attractors = [att_arr.astype(np.float32)]
        elif isinstance(raw_attractors, np.ndarray):
            if raw_attractors.ndim == 2:
                attractors = [raw_attractors[i].astype(np.float32) for i in range(raw_attractors.shape[0])]
        elif isinstance(raw_attractors, (list, tuple)):
            attractors = [
                np.asarray(a, dtype=np.float32) if a is not None else None for a in raw_attractors
            ]

        if stop is not None:
            stop_logit = float(stop.item() if isinstance(stop, torch.Tensor) else stop)

        est = est.squeeze(0)
        if est.ndim == 2 and est.shape[0] != time_len and est.shape[1] == time_len:
            pass  # [K, T]
        elif est.ndim == 2 and est.shape[0] == time_len:
            est = est.T

        streams = est.detach().cpu().numpy().astype(np.float32)
        if isinstance(conf, torch.Tensor):
            confidences = conf.detach().cpu().tolist()
        elif isinstance(conf, (list, tuple)):
            confidences = list(conf)
        else:
            confidences = [1.0] * streams.shape[0]

        while len(attractors) < streams.shape[0]:
            attractors.append(None)

        return streams, confidences, attractors, stop_logit


class SepFormerExpertResample:
    """Shared resample helper (avoids circular import with sepformer module)."""

    @staticmethod
    def resample(audio: np.ndarray | torch.Tensor, orig_sr: int, target_sr: int) -> np.ndarray:
        import torchaudio.functional as F

        if isinstance(audio, np.ndarray):
            t = torch.from_numpy(audio.astype(np.float32))
        else:
            t = audio.float()
        t = t.squeeze()
        return F.resample(t.unsqueeze(0), orig_sr, target_sr).squeeze(0).numpy()
