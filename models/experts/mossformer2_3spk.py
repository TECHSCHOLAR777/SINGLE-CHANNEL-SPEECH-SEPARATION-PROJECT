"""
MossFormer2 3-speaker expert wrapper (WSJ0-3mix checkpoint).

The public ``clearvoice`` package only ships the 2-speaker ``MossFormer2_SS_16K``
model. A genuine 3-speaker MossFormer2 checkpoint exists separately on the Hub —
``alibabasglab/mossformer2-wsj0mix-3spk`` (``masknet_numspks: 3``, 8 kHz) — as
SpeechBrain-style component saves (``encoder.ckpt`` / ``masknet.ckpt`` /
``decoder.ckpt``) plus a ``config.json``.

This wrapper reuses the MossFormer architecture that ships *inside* clearvoice
(``clearvoice.models.mossformer2_ss.mossformer2.MossFormer`` — the same Encoder /
MossFormer_MaskNet / Decoder stack), instantiates it with ``num_spks=3`` and the
checkpoint's config, and loads the three component state dicts. The published
model runs at **8 kHz**, so — like SR-CorrNet — the wrapper resamples the 16 kHz
mixture down for inference and the separated streams back up to 16 kHz.

Install / weights: `pip install clearvoice huggingface_hub`; the checkpoint
downloads from the Hub on first use.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from models.preprocess import PROJECT_SAMPLE_RATE, preprocess, resample_audio
from schemas.separation_result import SeparationResult, StreamMetadata

DEFAULT_HF_MODEL = "alibabasglab/mossformer2-wsj0mix-3spk"

# Architecture params from the checkpoint's config.json (WSJ0-3mix).
_ARCH = {
    "in_channels": 512,  # encoder_out_nchannels
    "out_channels": 512,
    "num_blocks": 24,  # intra_numlayers
    "kernel_size": 16,  # encoder_kernel_size
    "norm": "ln",  # masknet_norm
    "num_spks": 3,  # masknet_numspks
    "skip_around_intra": True,  # masknet_extraskipconnection
    "use_global_pos_enc": True,
    "max_length": 20000,
}


def _import_mossformer():
    """Import the MossFormer class from clearvoice, trying the known paths."""
    last: Exception | None = None
    for mod in (
        "clearvoice.models.mossformer2_ss.mossformer2",
        "clearvoice.clearvoice.models.mossformer2_ss.mossformer2",
    ):
        try:
            m = __import__(mod, fromlist=["MossFormer"])
            return m.MossFormer
        except Exception as exc:  # noqa: BLE001 - report the last failure
            last = exc
    raise RuntimeError(
        "Could not import MossFormer from clearvoice. Install with "
        "`pip install clearvoice`. Last error: " + repr(last)
    )


class MossFormer2ThreeSpkExpert:
    """Inference wrapper for the 3-speaker MossFormer2 (WSJ0-3mix)."""

    SAMPLE_RATE = PROJECT_SAMPLE_RATE  # project-facing 16 kHz
    EXPERT_NAME = "mossformer2_3spk"
    MAX_SPEAKERS = 3

    def __init__(
        self,
        device: str | torch.device = "cpu",
        hf_model_id: str = DEFAULT_HF_MODEL,
        model_sample_rate: int = 8000,
        local_dir: str | Path | None = None,
        compute_embeddings: bool = True,
        embedder_savedir: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.hf_model_id = hf_model_id
        self.model_sample_rate = int(model_sample_rate)
        self.local_dir = Path(local_dir) if local_dir else None
        self.compute_embeddings = compute_embeddings
        self._embedder_savedir = embedder_savedir
        self._model: torch.nn.Module | None = None
        self._embedder: object | None = None

    @property
    def is_available(self) -> bool:
        """True when clearvoice (the architecture) and huggingface_hub are present."""
        return (
            importlib.util.find_spec("clearvoice") is not None
            and importlib.util.find_spec("huggingface_hub") is not None
        )

    def _ckpt(self, filename: str) -> str:
        if self.local_dir is not None:
            return str(self.local_dir / filename)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=self.hf_model_id, filename=filename)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.is_available:
            raise RuntimeError(
                "MossFormer2-3spk needs clearvoice + huggingface_hub. "
                "Install: pip install clearvoice huggingface_hub"
            )
        mossformer_cls = _import_mossformer()
        model = mossformer_cls(**_ARCH)

        # The checkpoint stores SpeechBrain-style component state dicts that map
        # onto MossFormer's enc / mask_net / dec submodules.
        pairs = [
            ("encoder.ckpt", model.enc),
            ("masknet.ckpt", model.mask_net),
            ("decoder.ckpt", model.dec),
        ]
        try:
            for filename, submodule in pairs:
                state = torch.load(self._ckpt(filename), map_location="cpu")
                result = submodule.load_state_dict(_clean_state(state), strict=False)
                if result.missing_keys:
                    print(f"[mossformer2_3spk] {filename}: {len(result.missing_keys)} missing keys")
                if result.unexpected_keys:
                    print(f"[mossformer2_3spk] {filename}: {len(result.unexpected_keys)} unexpected keys")
        except Exception as exc:  # noqa: BLE001 - fall back to the bundled full state dict
            try:
                full = torch.load(self._ckpt("pytorch_model.bin"), map_location="cpu")
                # pytorch_model.bin stores flat keys with submodule prefixes;
                # split by prefix and load each submodule with strict=False.
                submodule_map = {"encoder.": model.enc, "masknet.": model.mask_net, "decoder.": model.dec}
                sub_states: dict[str, dict] = {"encoder.": {}, "masknet.": {}, "decoder.": {}}
                remainder: dict = {}
                for k, v in _clean_state(full).items():
                    placed = False
                    for prefix in sub_states:
                        if k.startswith(prefix):
                            sub_states[prefix][k[len(prefix) :]] = v
                            placed = True
                            break
                    if not placed:
                        remainder[k] = v
                if any(sub_states.values()):
                    for prefix, sub_state in sub_states.items():
                        if sub_state:
                            submodule_map[prefix].load_state_dict(sub_state, strict=False)
                else:
                    model.load_state_dict(_clean_state(full), strict=False)
            except Exception as exc2:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to load MossFormer2-3spk weights from component ckpts "
                    f"({exc!r}) and from pytorch_model.bin ({exc2!r}). "
                    "The checkpoint's key layout may differ — inspect torch.load keys."
                ) from exc2

        model.eval()
        self._model = model.to(self.device)

    def separate(self, mixture: np.ndarray | torch.Tensor, sample_rate: int) -> SeparationResult:
        """Separate a mono mixture into 3 streams [3, T] at the project 16 kHz rate."""
        self._load_model()
        assert self._model is not None

        pre = preprocess(mixture, sample_rate)
        wav16 = pre.waveform.astype(np.float32)
        wav_lo = resample_audio(wav16, PROJECT_SAMPLE_RATE, self.model_sample_rate)
        wav_t = torch.from_numpy(wav_lo).float().unsqueeze(0).to(self.device)  # [1, T]

        with torch.no_grad():
            out = self._model(wav_t)  # list of num_spks tensors, each [1, T] @ 8 kHz

        streams_lo = np.stack(
            [o.squeeze().detach().cpu().numpy().astype(np.float32) for o in out], axis=0
        )
        streams = np.stack(
            [resample_audio(s, self.model_sample_rate, PROJECT_SAMPLE_RATE) for s in streams_lo],
            axis=0,
        )
        streams = _fix_length(streams, wav16.shape[0]).astype(np.float32)

        metadata = [
            StreamMetadata(
                expert_source=self.EXPERT_NAME,
                confidence=1.0,
                extra={"stream_index": i, "model": self.hf_model_id},
            )
            for i in range(streams.shape[0])
        ]
        result = SeparationResult(
            streams=streams,
            sample_rate=PROJECT_SAMPLE_RATE,
            speaker_count=streams.shape[0],
            metadata=metadata,
            mixture=wav16,
            escalated=False,
            expert_used=self.EXPERT_NAME,
        )

        if self.compute_embeddings:
            from models.experts.embeddings import ECAPAEmbedder, attach_ecapa_embeddings

            if self._embedder is None:
                self._embedder = ECAPAEmbedder(device=self.device, savedir=self._embedder_savedir)
            result = attach_ecapa_embeddings(result, embedder=self._embedder)

        return result


def _clean_state(state: dict) -> dict:
    """Strip a leading 'model.'/'module.' prefix if the checkpoint carries one."""
    if not isinstance(state, dict):
        return state
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    out = {}
    for k, v in state.items():
        for pre in ("module.", "model."):
            if k.startswith(pre):
                k = k[len(pre) :]
        out[k] = v
    return out


def _fix_length(streams: np.ndarray, length: int) -> np.ndarray:
    t = streams.shape[1]
    if t == length:
        return streams
    if t > length:
        return streams[:, :length]
    pad = np.zeros((streams.shape[0], length - t), dtype=streams.dtype)
    return np.concatenate([streams, pad], axis=1)
