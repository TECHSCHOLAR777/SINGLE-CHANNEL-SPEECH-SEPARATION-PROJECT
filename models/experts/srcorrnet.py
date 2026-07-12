"""
SR-CorrNet expensive expert wrapper (time-frequency separator).

Uses the official SR_CorrNet_SS package (github.com/dmlguq456/SR_CorrNet_SS),
whose ``SSInference`` API loads pretrained checkpoints straight from the
Hugging Face Hub (``shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk`` covers 2-3
speakers, ``...-var-2-5spk`` covers 2-5).

Install on the training box::

    git clone https://github.com/dmlguq456/SR_CorrNet_SS.git
    cd SR_CorrNet_SS && pip install -e ".[hub]"

The published models run at **8 kHz** (n_fft 128, hop 64). This wrapper keeps
the rest of the project at 16 kHz: it resamples the mixture down to 8 kHz for
inference and resamples the separated streams back up to 16 kHz so they line up
with the MossFormer2 cheap expert and the clean references.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from models.preprocess import PROJECT_SAMPLE_RATE, preprocess, resample_audio
from schemas.separation_result import SeparationResult, StreamMetadata

DEFAULT_HF_MODEL = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk"


class SRCorrNetExpert:
    """Inference wrapper for SR-CorrNet-SS via the SSInference API."""

    SAMPLE_RATE = PROJECT_SAMPLE_RATE  # project-facing rate (16 kHz)
    EXPERT_NAME = "srcorrnet"
    MAX_SPEAKERS = 5

    def __init__(
        self,
        device: str | torch.device = "cpu",
        repo_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        num_speakers: int = 3,
        hf_model_id: str = DEFAULT_HF_MODEL,
        config_path: str | Path | None = None,
        model_sample_rate: int = 8000,
    ) -> None:
        self.device = torch.device(device)
        self.repo_path = Path(repo_path) if repo_path else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.num_speakers = num_speakers
        self.hf_model_id = hf_model_id
        self.config_path = Path(config_path) if config_path else None
        self.model_sample_rate = int(model_sample_rate)
        self._model: object | None = None

    @property
    def is_available(self) -> bool:
        """
        True when SR-CorrNet-SS can be loaded.

        A configured-but-missing local checkpoint is treated as unavailable. A
        cloned repo path or an importable ``sr_corrnet`` package is enough — the
        checkpoint itself is pulled from the HF Hub by default.
        """
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
                "SR-CorrNet-SS is not available. Install it with:\n"
                "  git clone https://github.com/dmlguq456/SR_CorrNet_SS.git\n"
                '  cd SR_CorrNet_SS && pip install -e ".[hub]"\n'
                "or pass repo_path to the cloned checkout."
            )
        if self.repo_path is not None:
            repo = str(self.repo_path.resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

        from sr_corrnet import SSInference

        # ``checkpoint_path`` accepts a local file/dir path OR an HF Hub repo id;
        # ``config`` is only for a *local* config name/path and must not receive
        # the Hub id — passing it there makes from_pretrained look for a local
        # "SS/<id>.yaml" and fail with "Config not found" for every sample.
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

    def separate(self, mixture: np.ndarray | torch.Tensor, sample_rate: int) -> SeparationResult:
        """
        Separate a mono mixture into ``num_speakers`` streams.

        Returns a SeparationResult with streams [K, T] at the project 16 kHz
        rate (resampled from the model's 8 kHz output).
        """
        self._load_model()
        assert self._model is not None

        pre = preprocess(mixture, sample_rate)
        wav16 = pre.waveform.astype(np.float32)
        wav_lo = resample_audio(wav16, PROJECT_SAMPLE_RATE, self.model_sample_rate)
        wav_t = torch.from_numpy(wav_lo).float().unsqueeze(0).to(self.device)  # [1, L]

        with torch.no_grad():
            out = self._model.process_waveform(  # type: ignore[attr-defined]
                wav_t, n_spks=torch.tensor(self.num_speakers)
            )

        streams_lo = _extract_waveforms(out)  # [K, L_lo]
        streams = np.stack(
            [resample_audio(s, self.model_sample_rate, PROJECT_SAMPLE_RATE) for s in streams_lo],
            axis=0,
        )
        streams = _fix_length(streams, wav16.shape[0]).astype(np.float32)

        metadata = [
            StreamMetadata(
                expert_source=self.EXPERT_NAME,
                confidence=1.0,
                extra={"attractor_index": i, "model": self.hf_model_id},
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
        )


def _to_numpy(x: object) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _extract_waveforms(out: object) -> np.ndarray:
    """
    Normalize an SSInference output to [K, L] float32.

    ``process_waveform`` returns a dict with ``waveforms`` (a list of 1-D
    tensors, one per speaker); older/other entry points may return a tensor or a
    bare list. All are handled.
    """
    waves: object = out
    if isinstance(out, dict):
        waves = out.get("waveforms")
        if waves is None:
            waves = out.get("est_sources") or out.get("sources") or out.get("wav")

    if isinstance(waves, (list, tuple)):
        rows = [_to_numpy(w).reshape(-1) for w in waves]
        length = min(r.shape[0] for r in rows)
        return np.stack([r[:length] for r in rows], axis=0)

    arr = np.squeeze(_to_numpy(waves))
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _fix_length(streams: np.ndarray, length: int) -> np.ndarray:
    """Crop or zero-pad [K, L] streams to exactly ``length`` samples."""
    t = streams.shape[1]
    if t == length:
        return streams
    if t > length:
        return streams[:, :length]
    pad = np.zeros((streams.shape[0], length - t), dtype=streams.dtype)
    return np.concatenate([streams, pad], axis=1)
