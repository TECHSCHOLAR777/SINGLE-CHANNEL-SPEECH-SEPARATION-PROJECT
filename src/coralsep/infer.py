"""
CoRAL-Sep end-to-end inference.

Usage (Python):
    infer = CALMSepInference("/path/to/checkpoints")
    streams = infer.separate(mixture_16k_array, sr=16000)
    # streams: list of (T_16k,) float32 numpy arrays, one per detected speaker

Usage (CLI):
    python infer.py --checkpoint-dir checkpoints/ --input mix.wav --output-dir out/

Checkpoint directory layout expected:
    stage4_joint/best_joint.pt, gate + analyzer + LoRA adapter weights
    stage4c/calibration.pt, gate temperature scalar
    stage4b_band/best_band.pt, BandRecoveryHead (optional; falls back to 8→16 kHz zero-pad)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------------------

_SR_8K = 8_000
_SR_16K = 16_000
_STFT_8K_WIN = 128
_STFT_8K_HOP = 64
_STFT_16K_WIN = 512
_STFT_16K_HOP = 128
_HIGH_BAND_START = 128  # 4 kHz in 16 kHz STFT with n_fft=512
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav
    try:
        import torchaudio

        t = torch.from_numpy(wav).float().unsqueeze(0)
        return torchaudio.functional.resample(t, src_sr, dst_sr).squeeze(0).numpy()
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src_sr, dst_sr)
        return resample_poly(wav, dst_sr // g, src_sr // g).astype(np.float32)


def _load_joint_ckpt(path: Path, gate_net, analyzer, inner, lib) -> None:
    """Load gate, analyzer, and LoRA adapter weights from best_joint.pt."""
    from coralsep.models.lora import LoRALinear

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    gate_net.load_state_dict(ckpt["gate"])
    analyzer.load_state_dict(ckpt["analyzer"])

    adapter_state: dict[str, torch.Tensor] = ckpt.get("adapter_state", {})
    loaded = 0
    for mod_name, mod in inner.named_modules():
        if not isinstance(mod, LoRALinear):
            continue
        for adapter_name, branch in mod.branches.items():
            for param_name, param in branch.named_parameters():
                key = f"{mod_name}.branches.{adapter_name}.{param_name}"
                if key in adapter_state:
                    param.data.copy_(adapter_state[key].to(param.device))
                    loaded += 1
    log.info("Loaded joint checkpoint: gate + analyzer + %d adapter tensors", loaded)


def _stft_mag(wav_t: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Compute STFT magnitude. wav_t: (..., T) → (..., F, T_f)."""
    window = torch.hann_window(n_fft, device=wav_t.device)
    return torch.stft(
        wav_t,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
    ).abs()


def _stft_complex(wav_t: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=wav_t.device)
    return torch.stft(
        wav_t,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
    )


def _istft(stft: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=stft.device)
    return torch.istft(
        stft,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
    )


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------


class CALMSepInference:
    """
    End-to-end CoRAL-Sep inference.

    Args:
        checkpoint_dir: Directory containing sub-dirs stage4_joint/, stage4c/,
                        and optionally stage4b_band/.
        hf_model: HuggingFace model ID for SR-CorrNet (or local path).
        device: 'cpu' or 'cuda'.
        prob_threshold: Speaker existence threshold for attractor split (default 0.5).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        hf_model: str = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk",
        device: str = "cpu",
        prob_threshold: float = 0.5,
    ) -> None:
        self.ckpt_dir = Path(checkpoint_dir)
        self.hf_model = hf_model
        self.device = torch.device(device)
        self.prob_threshold = prob_threshold
        self._loaded = False

    # ── Lazy load ─────────────────────────────────────────────────────────

    def load(self) -> CALMSepInference:
        if self._loaded:
            return self
        self._load_all()
        self._loaded = True
        return self

    def _load_all(self) -> None:
        from coralsep.models.condition import Level2Analyzer, level1_tensor
        from coralsep.models.gate import GateNetwork
        from coralsep.models.lora import ADAPTER_NAMES, LoRALibrary
        from coralsep.train.stage1_single import _get_inner_module, _load_model

        dev = self.device

        # ── SR-CorrNet (frozen base) ──────────────────────────────────────
        log.info("Loading SR-CorrNet …")
        self._ss_model = _load_model(self.hf_model, dev)
        self._inner = _get_inner_module(self._ss_model)

        # ── LoRA library ──────────────────────────────────────────────────
        self._lib = LoRALibrary(self._inner)
        self._lib.freeze_base()

        # ── Gate + analyzer ───────────────────────────────────────────────
        self._analyzer = Level2Analyzer().to(dev)
        self._gate_net = GateNetwork().to(dev)

        # ── Load best_joint.pt ────────────────────────────────────────────
        joint_ckpt = self.ckpt_dir / "stage4_joint" / "best_joint.pt"
        if joint_ckpt.exists():
            _load_joint_ckpt(joint_ckpt, self._gate_net, self._analyzer, self._inner, self._lib)
        else:
            log.warning("best_joint.pt not found at %s, using random gate", joint_ckpt)

        self._inner.to(dev).eval()
        self._gate_net.eval()
        self._analyzer.eval()

        # ── Temperature scalar from Stage 4c ─────────────────────────────
        self._temperature = 1.0
        calib_path = self.ckpt_dir / "stage4c" / "calibration.pt"
        if calib_path.exists():
            c = torch.load(str(calib_path), map_location="cpu", weights_only=False)
            self._temperature = float(c["temperature"].item())
            log.info("Gate temperature T=%.4f", self._temperature)
        else:
            log.warning("calibration.pt not found, using T=1.0")

        # ── BandRecoveryHead from Stage 4b (optional) ─────────────────────
        from coralsep.models.band_recovery import BandRecoveryHead

        self._band_head: BandRecoveryHead | None = None
        band_path = self.ckpt_dir / "stage4b_band" / "best_band.pt"
        if band_path.exists():
            self._band_head = BandRecoveryHead().to(dev)
            self._band_head.load_state_dict(
                torch.load(str(band_path), map_location=dev, weights_only=False)
            )
            self._band_head.eval()
            log.info("BandRecoveryHead loaded from %s", band_path)
        else:
            log.info("best_band.pt not found, 16 kHz output will be zero-padded")

        # Store reference to level1_tensor for inference loop
        self._level1_tensor = level1_tensor
        self._ADAPTER_NAMES = ADAPTER_NAMES
        log.info("CoRAL-Sep ready on %s", dev)

    # ── Gate (calibrated) ─────────────────────────────────────────────────

    def _calibrated_gate(self, cond: torch.Tensor) -> torch.Tensor:
        """Run gate and apply temperature scaling. Returns (3,) in [0,1]."""
        with torch.no_grad():
            gate_prob = self._gate_net(cond).squeeze(0).float().clamp(1e-6, 1 - 1e-6)
        if self._temperature != 1.0:
            logit = torch.log(gate_prob / (1.0 - gate_prob))
            gate_prob = torch.sigmoid(logit / self._temperature)
        return gate_prob

    # ── Separate ──────────────────────────────────────────────────────────

    def separate(
        self,
        mixture: np.ndarray,
        sr: int = _SR_16K,
        n_spks: int | None = None,
    ) -> list[np.ndarray]:
        """
        Separate speakers from a mono mixture.

        Args:
            mixture: (T,) float32 mono waveform.
            sr: Sample rate of mixture (any rate; will be resampled internally).
            n_spks: Force speaker count (None = auto-detect via attractor).

        Returns:
            List of (T_16k,) float32 waveforms at 16 kHz, one per speaker.
        """
        self.load()
        dev = self.device

        # ── Resample to 16 kHz for band recovery, 8 kHz for SR-CorrNet ───
        mix_16k = _resample(mixture.astype(np.float32), sr, _SR_16K)
        mix_8k = _resample(mixture.astype(np.float32), sr, _SR_8K)

        # Normalise 8 kHz input
        rms = float(np.sqrt(np.maximum(np.mean(mix_8k**2), _EPS)))
        mix_8k = mix_8k / rms

        wav_8k_t = torch.from_numpy(mix_8k).float().to(dev).unsqueeze(0)  # (1, T8)

        # ── Level-1 condition features ─────────────────────────────────────
        l1_feat = self._level1_tensor(wav_8k_t.squeeze(0)).to(dev)  # (4,)
        l2_zeros = torch.zeros(6, device=dev)
        cond = torch.cat([l1_feat, l2_zeros]).unsqueeze(0)  # (1, 10)

        # ── Calibrated gate → LoRA adapter mix ────────────────────────────
        gate = self._calibrated_gate(cond)  # (3,)
        self._lib.set_gates({self._ADAPTER_NAMES[i]: gate[i].item() for i in range(3)})
        self._lib.inject_gates()

        # ── SR-CorrNet forward (no grad) ───────────────────────────────────
        n_spks_t = torch.tensor(n_spks) if n_spks is not None else None
        with torch.no_grad():
            result = self._ss_model.process_waveform(wav_8k_t, n_spks=n_spks_t)

        waves_8k: list[np.ndarray] = []
        for w in result.get("waveforms", []):
            arr = (
                w.squeeze().cpu().numpy()
                if isinstance(w, torch.Tensor)
                else np.asarray(w).squeeze()
            )
            waves_8k.append(arr.astype(np.float32) * rms)  # undo normalisation

        # Clear gates
        self._lib.set_gates({n: 0.0 for n in self._ADAPTER_NAMES})
        self._lib.inject_gates()

        if not waves_8k:
            log.warning("SR-CorrNet returned no separated streams")
            return [mix_16k]

        # ── Band recovery: 8 kHz → 16 kHz ─────────────────────────────────
        return self._band_recover(waves_8k, mix_16k)

    # ── Band recovery ─────────────────────────────────────────────────────

    def _band_recover(
        self,
        waves_8k: list[np.ndarray],
        mix_16k: np.ndarray,
    ) -> list[np.ndarray]:
        """Apply BandRecoveryHead or fall back to zero-padded upsampling."""
        from coralsep.models.band_recovery import stft_to_waveform

        T_16k = len(mix_16k)

        if self._band_head is None:
            return [_zero_pad_to_16k(w, T_16k) for w in waves_8k]

        dev = self.device
        K = len(waves_8k)

        with torch.no_grad():
            # 8 kHz complex STFT of separated streams: (K, 65, T_f8)
            sep_t = torch.stack([torch.from_numpy(w).float() for w in waves_8k]).to(dev)
            stft_sep = _stft_complex(sep_t, _STFT_8K_WIN, _STFT_8K_HOP)  # (K, 65, T_f8)

            # 16 kHz complex STFT of original mixture: (K, 257, T_f16)
            mix_t = torch.from_numpy(mix_16k).float().to(dev)
            stft_mix = _stft_complex(mix_t, _STFT_16K_WIN, _STFT_16K_HOP)  # (257, T_f16)
            stft_mix_k = stft_mix.unsqueeze(0).expand(K, -1, -1).clone()  # (K, 257, T_f16)

            # predict_highband_stft returns (K, 257, T) complex STFT
            out_stft = self._band_head.predict_highband_stft(stft_sep, stft_mix_k)

            # iSTFT → (K, L)
            waves_out = stft_to_waveform(out_stft, n_fft=_STFT_16K_WIN, hop=_STFT_16K_HOP)

        # Trim / pad to original mixture length
        results: list[np.ndarray] = []
        for k in range(K):
            wav = waves_out[k].cpu().numpy()
            L = wav.shape[0]
            if L >= T_16k:
                results.append(wav[:T_16k])
            else:
                results.append(np.pad(wav, (0, T_16k - L)))
        return results


def _zero_pad_to_16k(wav_8k: np.ndarray, T_16k: int) -> np.ndarray:
    """Upsample 8 kHz waveform to 16 kHz by zero-interleaving (no high-band)."""
    out = np.zeros(T_16k, dtype=np.float32)
    n = min(len(wav_8k), T_16k // 2)
    out[: n * 2 : 2] = wav_8k[:n]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoRAL-Sep: condition-aware speech separation")
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Root dir with stage4_joint/, stage4c/, stage4b_band/ sub-dirs",
    )
    p.add_argument("--input", required=True, help="Input mixture WAV/FLAC")
    p.add_argument("--output-dir", required=True, help="Directory for separated streams")
    p.add_argument("--hf-model", default="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--n-spks", type=int, default=None, help="Force speaker count (default: auto-detect)"
    )
    p.add_argument("--sr", type=int, default=16000, help="Output sample rate (default 16000)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    import soundfile as sf

    mixture, sr_in = sf.read(args.input, dtype="float32", always_2d=True)
    mixture = mixture[:, 0]  # mono

    infer = CALMSepInference(
        checkpoint_dir=args.checkpoint_dir,
        hf_model=args.hf_model,
        device=args.device,
    )
    streams = infer.separate(mixture, sr=sr_in, n_spks=args.n_spks)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    for i, wav in enumerate(streams):
        out_path = out_dir / f"{stem}_spk{i+1}.wav"
        sf.write(str(out_path), wav, args.sr)
        log.info("Written: %s", out_path)


if __name__ == "__main__":
    main()
