"""
Augmentation pipeline for CoRAL-Sep training data.

Two-stage probabilistic augmentation applied to MixtureSample objects:
  Stage 1 — RIR reverb: convolve mixture with a simulated room impulse response
  Stage 2 — WHAM! noise: add ambient noise at a random SNR

Ground-truth references are NEVER modified; only the mixture is augmented.
SI-SDRi is always computed against the original clean stems.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from coralsep.data.mixer_stub import MixtureSample


@dataclass
class AugmentationConfig:
    """Configuration for the two-stage augmentation pipeline."""

    # Stage 1 — RIR reverb
    rir_prob: float = 0.5
    room_dim_min: tuple[float, float, float] = (3.0, 3.0, 2.0)
    room_dim_max: tuple[float, float, float] = (8.0, 8.0, 4.0)
    absorption_min: float = 0.1
    absorption_max: float = 0.6
    max_rir_order: int = 17

    # Stage 2 — WHAM! noise
    noise_prob: float = 0.5
    snr_min_db: float = 5.0
    snr_max_db: float = 20.0
    wham_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.wham_dir is not None:
            self.wham_dir = Path(self.wham_dir)


class AugmentationPipeline:
    """
    Applies probabilistic RIR reverb and WHAM! noise to a MixtureSample.

    The mixture field is augmented; references (clean stems) are preserved
    unchanged so that SI-SDRi can be computed against the original sources.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config or AugmentationConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._noise_files: list[Path] | None = None

    def __call__(self, sample: MixtureSample) -> MixtureSample:
        """Return a new MixtureSample with the mixture augmented, references unchanged."""
        mixture = sample.mixture.copy()
        sr = sample.sample_rate

        if self.rng.random() < self.config.rir_prob:
            mixture = self._apply_rir(mixture, sr)

        if self.config.wham_dir is not None and self.rng.random() < self.config.noise_prob:
            mixture = self._apply_noise(mixture, sr)

        return MixtureSample(
            mixture=mixture,
            references=sample.references,
            sample_rate=sr,
            utterance_id=sample.utterance_id,
        )

    # ------------------------------------------------------------------
    # Stage 1: RIR reverb
    # ------------------------------------------------------------------

    def _apply_rir(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Convolve audio with a random shoebox room impulse response."""
        try:
            import pyroomacoustics as pra  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pyroomacoustics is required for RIR augmentation. "
                "Install it with: pip install pyroomacoustics"
            ) from exc

        rir = self._generate_rir(pra, sr)
        convolved = fftconvolve(audio, rir)
        return convolved[: len(audio)].astype(np.float32)

    def _generate_rir(self, pra: object, sr: int) -> np.ndarray:
        """Generate a random RIR using pyroomacoustics ShoeBox."""
        cfg = self.config
        dims = np.array([
            self.rng.uniform(cfg.room_dim_min[i], cfg.room_dim_max[i]) for i in range(3)
        ])
        absorption = float(self.rng.uniform(cfg.absorption_min, cfg.absorption_max))

        margin = 0.5
        source_pos = np.array([
            self.rng.uniform(margin, dims[0] - margin),
            self.rng.uniform(margin, dims[1] - margin),
            self.rng.uniform(margin, dims[2] - margin),
        ])
        mic_pos = np.array([
            self.rng.uniform(margin, dims[0] - margin),
            self.rng.uniform(margin, dims[1] - margin),
            self.rng.uniform(margin, dims[2] - margin),
        ])

        materials = pra.Material(absorption)
        room = pra.ShoeBox(
            dims,
            fs=sr,
            materials=materials,
            max_order=cfg.max_rir_order,
        )
        room.add_source(source_pos)
        room.add_microphone(mic_pos.reshape(3, 1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            room.simulate()

        rir = room.rir[0][0].astype(np.float32)
        return rir

    # ------------------------------------------------------------------
    # Stage 2: WHAM! noise
    # ------------------------------------------------------------------

    def _apply_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add a WHAM! noise clip to audio at a random SNR."""
        noise = self._load_noise_clip(len(audio), sr)

        signal_rms = float(np.sqrt(np.mean(audio ** 2)))
        noise_rms = float(np.sqrt(np.mean(noise ** 2)))

        if signal_rms < 1e-8 or noise_rms < 1e-8:
            return audio

        snr_db = float(self.rng.uniform(self.config.snr_min_db, self.config.snr_max_db))
        noise_gain = signal_rms / (10.0 ** (snr_db / 20.0) * noise_rms)

        return (audio + noise_gain * noise).astype(np.float32)

    def _load_noise_clip(self, length: int, sr: int) -> np.ndarray:
        """Load a random noise clip from wham_dir, resized to `length` samples."""
        import soundfile as sf  # already a project dependency

        if self._noise_files is None:
            wham_dir = self.config.wham_dir
            assert wham_dir is not None
            self._noise_files = sorted(wham_dir.rglob("*.wav"))
            if not self._noise_files:
                raise FileNotFoundError(
                    f"No WAV files found under wham_dir={wham_dir}. "
                    "Provide a valid WHAM! dataset directory."
                )

        idx = int(self.rng.integers(0, len(self._noise_files)))
        noise_path = self._noise_files[idx]

        raw, file_sr = sf.read(str(noise_path), dtype="float32", always_2d=True)
        noise = raw.mean(axis=1)  # mono

        if file_sr != sr:
            noise = _resample(noise, file_sr, sr)

        noise = _fit_to_length(noise, length, self.rng)
        return noise.astype(np.float32)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fit_to_length(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Tile or crop audio to exactly `length` samples."""
    if len(audio) == 0:
        return np.zeros(length, dtype=np.float32)

    if len(audio) < length:
        repeats = int(np.ceil(length / len(audio)))
        audio = np.tile(audio, repeats)

    if len(audio) > length:
        start = int(rng.integers(0, len(audio) - length + 1))
        audio = audio[start : start + length]

    return audio


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Naive linear-interpolation resample (avoids resampy/librosa dependency)."""
    if src_sr == dst_sr:
        return audio
    dst_len = int(len(audio) * dst_sr / src_sr)
    src_idx = np.linspace(0, len(audio) - 1, dst_len)
    return np.interp(src_idx, np.arange(len(audio)), audio).astype(np.float32)
