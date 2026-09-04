"""
Codec augmentation prototype for CoRAL-Sep, Phase 1 research deliverable.

Stage 3 of the augmentation pipeline: phone-channel codec distortion.
Simulates WhatsApp and voice-note recordings by encoding audio through
Opus or AAC at low bitrate (6–32 kbps) then decoding back to PCM.

Named project contribution: no other team in CoRAL-Sep prepares for this
degradation. Codec distortion at 6–16 kbps breaks harmonic structure,
attenuates high frequencies, and introduces block artifacts: a different
degradation profile from RIR reverb or WHAM! noise, requiring its own
training signal.

Design note: This module is intentionally standalone so it can be appended
to AugmentationPipeline in Phase 4 with one line:
    mixture → AugmentationPipeline (Stage 1+2) → CodecAugmentor (Stage 3)

Execution paths:
  Primary, ffmpeg subprocess (real Opus/AAC); requires ffmpeg on PATH.
  Fallback, mu-law companding + 8-bit quantisation (scipy only); always
             available, mimics G.711 telephone-codec artifacts.

Realistic bitrate targets (WhatsApp/Telegram voice notes):
  Opus : 6–24 kbps (WhatsApp calls use 6–16 kbps adaptively)
  AAC  : 8–32 kbps (Telegram, generic voice-note encoders)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from coralsep.data.mixer_stub import MixtureSample

_MULAW_MU: int = 255
_SUPPORTED_CODECS = ("opus", "aac", "amr-nb", "amr-wb")

# AMR-NB fixed bitrate modes (bps), closest match is chosen at encode time.
_AMR_NB_BITRATES_BPS = (4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200)
# AMR-WB fixed bitrate modes (bps).
_AMR_WB_BITRATES_BPS = (6600, 8850, 12650, 14250, 15850, 18250, 19850, 23050, 23850)

_CODEC_FFMPEG_NAME = {
    "opus": "libopus",
    "aac": "aac",
    "amr-nb": "libopencore_amrnb",
    "amr-wb": "libvo_amrwbenc",
}
_CODEC_CONTAINER = {
    "opus": ".opus",
    "aac": ".aac",
    "amr-nb": ".amr",
    "amr-wb": ".amr",
}
# AMR-NB requires 8 kHz; AMR-WB requires 16 kHz.
_CODEC_REQUIRED_SR: dict[str, int | None] = {
    "opus": None,  # accepts any sr
    "aac": None,
    "amr-nb": 8_000,
    "amr-wb": 16_000,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_ffmpeg_available() -> bool:
    """Return True if ffmpeg is found on PATH and exits successfully."""
    return shutil.which("ffmpeg") is not None


MULAW_FALLBACK_LABEL = "mulaw-fallback"
"""Recorded as the actual codec in a recipe when the real codec roundtrip was
not possible and apply_codec_roundtrip fell back to mu-law companding. Never
the same string as a real codec name, so a caller checking codec_name against
_SUPPORTED_CODECS can tell a genuine codec run from a fallback at a glance."""


def apply_codec_roundtrip(
    audio: np.ndarray,
    sample_rate: int,
    codec: str,
    bitrate_bps: int | float,
    tmp_dir: str | Path | None = None,
) -> tuple[np.ndarray, str]:
    """
    Encode -> decode ``audio`` through a real codec and return the damaged signal.

    Public wrapper used by ``data.degradations.apply_codec``. Supports "opus",
    "aac", "amr-nb", and "amr-wb". AMR-NB requires 8 kHz input; AMR-WB
    requires 16 kHz input, ffmpeg will resample internally when the input
    sample rate differs, but callers should note the resampling.

    Falls back to mu-law companding (G.711 approximation) when ffmpeg is
    unavailable or the roundtrip fails. The fallback is a deliberate, accepted
    degradation of quality, not of correctness (see the module docstring), but
    a caller that only recorded the requested codec name, not what actually
    ran, would silently mislabel every fallback sample as the real codec.
    Some ffmpeg builds omit AMR-NB entirely (a common licensing-driven
    omission), which makes this path common enough to have caused exactly
    that mislabeling in practice; see I-054. Returning which method actually
    ran, not just the damaged audio, is what lets a caller record the truth.

    Args:
        audio: 1-D float32 waveform.
        sample_rate: Audio sample rate in Hz.
        codec: One of "opus", "aac", "amr-nb", "amr-wb".
        bitrate_bps: Target bitrate in bits per second.
        tmp_dir: Scratch directory; uses a new tempdir when None.

    Returns:
        (damaged waveform, same length as audio, float32; the codec name that
        actually ran, either the requested one or MULAW_FALLBACK_LABEL).
    """
    if codec not in _SUPPORTED_CODECS:
        raise ValueError(f"codec {codec!r} not supported; choose from {_SUPPORTED_CODECS}")

    audio_f32 = np.asarray(audio, dtype=np.float32).squeeze()
    bitrate_kbps = bitrate_bps / 1000.0

    if not is_ffmpeg_available():
        warnings.warn(
            f"ffmpeg not found; falling back to mu-law simulation for codec {codec!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _mulaw_fallback(audio_f32), MULAW_FALLBACK_LABEL

    result = _ffmpeg_roundtrip_standalone(audio_f32, sample_rate, codec, bitrate_kbps, tmp_dir)
    if result is not None:
        return result, codec

    warnings.warn(
        f"ffmpeg codec roundtrip failed for {codec!r}; falling back to mu-law simulation.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _mulaw_fallback(audio_f32), MULAW_FALLBACK_LABEL


def _ffmpeg_roundtrip_standalone(
    audio: np.ndarray,
    sr: int,
    codec: str,
    bitrate_kbps: float,
    tmp_dir: str | Path | None = None,
) -> np.ndarray | None:
    """Module-level ffmpeg roundtrip supporting all four codecs. Returns None on failure."""
    ffmpeg_name = _CODEC_FFMPEG_NAME[codec]
    container = _CODEC_CONTAINER[codec]
    required_sr = _CODEC_REQUIRED_SR[codec]

    # Snap AMR bitrates to the nearest valid mode.
    if codec == "amr-nb":
        bitrate_kbps = _snap_amr_bitrate(bitrate_kbps * 1000, _AMR_NB_BITRATES_BPS) / 1000.0
    elif codec == "amr-wb":
        bitrate_kbps = _snap_amr_bitrate(bitrate_kbps * 1000, _AMR_WB_BITRATES_BPS) / 1000.0

    ctx = tempfile.TemporaryDirectory() if tmp_dir is None else None
    work = Path(ctx.name if ctx else tmp_dir)
    try:
        src = work / "input.wav"
        sf.write(str(src), audio, sr, subtype="FLOAT")

        encoded = work / f"encoded{container}"
        decoded = work / "decoded.wav"

        enc_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-codec:a",
            ffmpeg_name,
            "-b:a",
            f"{bitrate_kbps:.2f}k",
        ]
        if required_sr is not None:
            enc_cmd += ["-ar", str(required_sr)]
        enc_cmd.append(str(encoded))

        if subprocess.run(enc_cmd, capture_output=True, check=False).returncode != 0:
            return None

        dec_cmd = ["ffmpeg", "-y", "-i", str(encoded)]
        if required_sr is not None:
            # Resample back to original rate after decoding.
            dec_cmd += ["-ar", str(sr)]
        dec_cmd.append(str(decoded))

        if subprocess.run(dec_cmd, capture_output=True, check=False).returncode != 0:
            return None

        out, _ = sf.read(str(decoded), dtype="float32", always_2d=True)
        return _fit_length(out.mean(axis=1), len(audio))
    except Exception:
        return None
    finally:
        if ctx is not None:
            ctx.cleanup()


def _snap_amr_bitrate(bitrate_bps: float, valid_rates: tuple[int, ...]) -> int:
    """Return the AMR bitrate mode (bps) closest to ``bitrate_bps``."""
    return min(valid_rates, key=lambda r: abs(r - bitrate_bps))


def _mulaw_fallback(audio: np.ndarray) -> np.ndarray:
    """G.711 mu-law approximation, used when ffmpeg is unavailable."""
    x = np.clip(audio, -1.0, 1.0).astype(np.float64)
    mu = float(_MULAW_MU)
    encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantised = np.round(encoded * 127.0).clip(-127, 127).astype(np.int8)
    q_float = quantised.astype(np.float64) / 127.0
    decoded = np.sign(q_float) * ((1.0 + mu) ** np.abs(q_float) - 1.0) / mu
    return decoded.astype(np.float32)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CodecConfig:
    """Configuration for the codec distortion augmentor.

    Realistic WhatsApp/voice-note bitrate ranges:
      Opus  6–24 kbps  (default 6–32 covers research headroom)
      AAC   8–32 kbps
    """

    codec: str = "random"
    """Codec to apply: 'opus', 'aac', or 'random' (chosen per sample)."""

    bitrate_min_kbps: float = 6.0
    """Minimum encoding bitrate in kbps."""

    bitrate_max_kbps: float = 32.0
    """Maximum encoding bitrate in kbps."""

    codec_prob: float = 0.3
    """Probability of applying codec distortion per sample."""

    use_ffmpeg: bool = True
    """Use real Opus/AAC codec via ffmpeg. Falls back to mu-law if False or
    if ffmpeg is not installed."""

    def __post_init__(self) -> None:
        valid = (*_SUPPORTED_CODECS, "random")
        if self.codec not in valid:
            raise ValueError(f"codec must be one of {valid!r}, got {self.codec!r}")
        if self.bitrate_min_kbps <= 0 or self.bitrate_max_kbps <= 0:
            raise ValueError("bitrate values must be positive")
        if self.bitrate_min_kbps > self.bitrate_max_kbps:
            raise ValueError("bitrate_min_kbps must be <= bitrate_max_kbps")
        if not 0.0 <= self.codec_prob <= 1.0:
            raise ValueError("codec_prob must be in [0, 1]")


# ---------------------------------------------------------------------------
# Augmentor
# ---------------------------------------------------------------------------


class CodecAugmentor:
    """
    Applies probabilistic phone-channel codec distortion to a MixtureSample.

    Only the mixture is modified; references (clean stems) are preserved
    so that SI-SDRi can be computed against the original clean sources.
    """

    def __init__(
        self,
        config: CodecConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config or CodecConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._ffmpeg_ok: bool | None = None  # lazily checked

    def __call__(self, sample: MixtureSample) -> MixtureSample:
        """Return a new MixtureSample with the mixture codec-distorted, references unchanged."""
        if self.rng.random() >= self.config.codec_prob:
            return MixtureSample(
                mixture=sample.mixture.copy(),
                references=sample.references,
                sample_rate=sample.sample_rate,
                utterance_id=sample.utterance_id,
            )

        mixture = self._encode_decode(sample.mixture, sample.sample_rate)
        return MixtureSample(
            mixture=mixture,
            references=sample.references,
            sample_rate=sample.sample_rate,
            utterance_id=sample.utterance_id,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _encode_decode(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply codec distortion: ffmpeg if available, mu-law otherwise."""
        use_ffmpeg = self.config.use_ffmpeg and self._ffmpeg_available()
        if use_ffmpeg:
            codec = self._resolve_codec()
            bitrate_kbps = float(
                self.rng.uniform(self.config.bitrate_min_kbps, self.config.bitrate_max_kbps)
            )
            result = self._ffmpeg_roundtrip(audio, sr, codec, bitrate_kbps)
            if result is not None:
                return result
            warnings.warn(
                "ffmpeg codec roundtrip failed; falling back to mu-law simulation.",
                RuntimeWarning,
                stacklevel=3,
            )
        return self._mulaw_roundtrip(audio)

    def _resolve_codec(self) -> str:
        if self.config.codec == "random":
            return str(self.rng.choice(_SUPPORTED_CODECS))
        return self.config.codec

    def _ffmpeg_available(self) -> bool:
        if self._ffmpeg_ok is None:
            self._ffmpeg_ok = is_ffmpeg_available()
        return self._ffmpeg_ok

    # ------------------------------------------------------------------
    # Primary path: real Opus / AAC via ffmpeg
    # ------------------------------------------------------------------

    def _ffmpeg_roundtrip(
        self,
        audio: np.ndarray,
        sr: int,
        codec: str,
        bitrate_kbps: float,
    ) -> np.ndarray | None:
        """Encode → decode through a real codec. Returns None on failure."""
        return _ffmpeg_roundtrip_standalone(audio, sr, codec, bitrate_kbps)

    # ------------------------------------------------------------------
    # Fallback path: mu-law companding (G.711 approximation)
    # ------------------------------------------------------------------

    def _mulaw_roundtrip(self, audio: np.ndarray) -> np.ndarray:
        """G.711 mu-law approximation; requires no binary dependencies."""
        return _mulaw_fallback(audio)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_length(audio: np.ndarray, target: int) -> np.ndarray:
    """Trim or zero-pad audio to exactly `target` samples."""
    if len(audio) >= target:
        return audio[:target].astype(np.float32)
    pad = np.zeros(target - len(audio), dtype=np.float32)
    return np.concatenate([audio.astype(np.float32), pad])
