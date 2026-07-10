"""
Codec augmentation prototype for CA-MoSE — Phase 1 research deliverable.

Stage 3 of the augmentation pipeline: phone-channel codec distortion.
Simulates WhatsApp and voice-note recordings by encoding audio through
Opus or AAC at low bitrate (6–32 kbps) then decoding back to PCM.

Named project contribution: no other team in CA-MoSE prepares for this
degradation. Codec distortion at 6–16 kbps breaks harmonic structure,
attenuates high frequencies, and introduces block artifacts — a different
degradation profile from RIR reverb or WHAM! noise, requiring its own
training signal.

Design note: This module is intentionally standalone so it can be appended
to AugmentationPipeline in Phase 4 with one line:
    mixture → AugmentationPipeline (Stage 1+2) → CodecAugmentor (Stage 3)

Execution paths:
  Primary  — ffmpeg subprocess (real Opus/AAC); requires ffmpeg on PATH.
  Fallback — mu-law companding + 8-bit quantisation (scipy only); always
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

from data.mixer_stub import MixtureSample

_MULAW_MU: int = 255
_SUPPORTED_CODECS = ("opus", "aac")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_ffmpeg_available() -> bool:
    """Return True if ffmpeg is found on PATH and exits successfully."""
    return shutil.which("ffmpeg") is not None


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
        if self.codec not in (*_SUPPORTED_CODECS, "random"):
            raise ValueError(
                f"codec must be one of {(*_SUPPORTED_CODECS, 'random')!r}, got {self.codec!r}"
            )
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
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                src = tmp / "input.wav"
                sf.write(str(src), audio, sr, subtype="FLOAT")

                if codec == "opus":
                    encoded = tmp / "encoded.opus"
                    audio_codec = "libopus"
                else:
                    encoded = tmp / "encoded.aac"
                    audio_codec = "aac"

                decoded = tmp / "decoded.wav"

                enc_result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(src),
                        "-codec:a", audio_codec,
                        "-b:a", f"{bitrate_kbps:.0f}k",
                        str(encoded),
                    ],
                    capture_output=True,
                    check=False,
                )
                if enc_result.returncode != 0:
                    return None

                dec_result = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(encoded), str(decoded)],
                    capture_output=True,
                    check=False,
                )
                if dec_result.returncode != 0:
                    return None

                out, _ = sf.read(str(decoded), dtype="float32", always_2d=True)
                out_mono = out.mean(axis=1)
                return _fit_length(out_mono, len(audio))

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Fallback path: mu-law companding (G.711 approximation)
    # ------------------------------------------------------------------

    def _mulaw_roundtrip(self, audio: np.ndarray) -> np.ndarray:
        """
        Simulate codec degradation via mu-law companding and 8-bit quantisation.

        This approximates G.711 telephone-codec artifacts (not true Opus/AAC)
        but requires no binary dependencies and always succeeds.
        """
        x = np.clip(audio, -1.0, 1.0).astype(np.float64)
        mu = float(_MULAW_MU)

        # Encode: compress dynamic range
        encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

        # Quantise to 8-bit signed integer (256 levels)
        quantised = np.round(encoded * 127.0).clip(-127, 127).astype(np.int8)

        # Decode: expand back
        q_float = quantised.astype(np.float64) / 127.0
        decoded = np.sign(q_float) * ((1.0 + mu) ** np.abs(q_float) - 1.0) / mu

        return decoded.astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_length(audio: np.ndarray, target: int) -> np.ndarray:
    """Trim or zero-pad audio to exactly `target` samples."""
    if len(audio) >= target:
        return audio[:target].astype(np.float32)
    pad = np.zeros(target - len(audio), dtype=np.float32)
    return np.concatenate([audio.astype(np.float32), pad])
