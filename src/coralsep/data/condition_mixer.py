"""
CoRAL-Sep 8 kHz dynamic mixer with degradation recipe logging (Dev A, P0-A1).

The frozen SR-CorrNet var-2-5 checkpoint operates at 8 kHz (STFT window 128,
hop 64). Every training and evaluation mixture in CoRAL-Sep is therefore mixed
at 8 kHz. This module is the 8 kHz counterpart to data/mixer.py, which stays
at 16 kHz for the legacy 16 kHz path and the band-recovery targets.

The critical difference from data/mixer.py is the recipe log. BLUEPRINT section
5.4 requires that every condition label (SNR, T60, codec family and bitrate,
speaker count) come free from the synthesis recipe rather than from a neural
estimate. This mixer returns a MixtureRecipe alongside every mixture recording
exactly what was applied, so the condition analyzer and gate train against
ground truth that was never estimated.

Degradations are applied by data/degradations.py; this module owns the source
draw, level offsets, summation, and the recipe record.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from coralsep.data.mixer_stub import MixtureSample, _load_wav

CORALSEP_SAMPLE_RATE: int = 8_000
"""Locked by the frozen checkpoint. BLUEPRINT fixed constraints. Never change."""

BAND_RECOVERY_SAMPLE_RATE: int = 16_000
"""Rate for band-recovery targets and DNSMOS scoring only. Never fed to the base."""

_DEFAULT_ALLOWED_N: list[int] = [2, 3, 4, 5]
"""N in {2,3,4,5}. K0=5 in the checkpoint; there is no 6+ speaker regime."""


@dataclass
class MixtureRecipe:
    """
    Ground-truth record of everything applied to one mixture.

    Every field here is a free supervision target: it is known because this
    code chose it, not because a model estimated it. The condition analyzer
    (BLUEPRINT 5.4) and the gate (5.5) train against these values.

    Attributes:
        n_speakers: True speaker count, the primary counting label.
        speaker_ids: Source speaker IDs, in reference-stream order.
        source_files: Source utterance paths, in reference-stream order.
        level_offsets_db: Per-speaker gain applied, in reference-stream order.
        snr_db: Noise SNR in dB, or None when no noise was added.
        noise_file: Noise source path, or None.
        t60_s: Reverberation time in seconds, or None when anechoic.
        rir_file: RIR path used, or None.
        codec_name: Codec family applied ("opus", "aac", "amr-nb", "amr-wb"),
            or None when uncompressed.
        codec_bitrate_bps: Codec bitrate in bits/sec, or None.
        seed: RNG seed that produced this mixture, when the mixer was seeded.
        sample_rate: Always CORALSEP_SAMPLE_RATE.
    """

    n_speakers: int
    speaker_ids: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    level_offsets_db: list[float] = field(default_factory=list)
    snr_db: float | None = None
    noise_file: str | None = None
    t60_s: float | None = None
    rir_file: str | None = None
    codec_name: str | None = None
    codec_bitrate_bps: int | None = None
    seed: int | None = None
    sample_rate: int = CORALSEP_SAMPLE_RATE

    def to_dict(self) -> dict[str, Any]:
        """Serialize for manifest writing and condition-label extraction."""
        return asdict(self)

    def condition_vector(self) -> dict[str, float]:
        """
        The supervised condition targets, as the analyzer consumes them.

        Absent conditions map to their neutral value rather than None so the
        vector is always dense: no noise means a high SNR, anechoic means a
        near-zero T60, uncompressed means codec class 0. This is what makes
        the gate's clean-input target (all gates near zero) well defined.

        Returns:
            Dict with keys snr_db, t60_s, codec_class, codec_bitrate_kbps,
            n_speakers. Consumed by models/condition.py.
        """
        return {
            "snr_db": 60.0 if self.snr_db is None else float(self.snr_db),
            "t60_s": 0.0 if self.t60_s is None else float(self.t60_s),
            "codec_class": float(_CODEC_CLASS_INDEX.get(self.codec_name or "none", 0)),
            "codec_bitrate_kbps": (
                0.0 if self.codec_bitrate_bps is None else self.codec_bitrate_bps / 1000.0
            ),
            "n_speakers": float(self.n_speakers),
        }


_CODEC_CLASS_INDEX: dict[str, int] = {
    "none": 0,
    "opus": 1,
    "aac": 2,
    "amr-nb": 3,
    "amr-wb": 4,
}
"""Codec family to class index. Index 0 (none) is the clean/neutral class."""


@dataclass
class CoralSepMixture:
    """
    One 8 kHz mixture with its stems and its ground-truth recipe.

    Attributes:
        sample: The mixture and reference stems (MixtureSample, 8 kHz).
        recipe: What was applied, for supervision and manifests.
    """

    sample: MixtureSample
    recipe: MixtureRecipe

    @property
    def mixture(self) -> np.ndarray:
        return self.sample.mixture

    @property
    def references(self) -> np.ndarray:
        return self.sample.references

    @property
    def n_speakers(self) -> int:
        return self.recipe.n_speakers


def _speaker_id(path: Path) -> str:
    """
    Speaker ID from a LibriSpeech-style filename.

    LibriSpeech names are {speaker}-{chapter}-{utterance}.ext, so the ID is the
    component before the first dash. Non-conforming names fall back to the full
    stem, which keeps speaker isolation conservative: an unparsed name is its
    own speaker rather than silently colliding with another.
    """
    return path.stem.split("-")[0]


class CoralSepMixer:
    """
    Draws N clean 8 kHz utterances and mixes them, logging the recipe.

    Speaker isolation is enforced at construction: files belonging to held-out
    speakers are removed from the training pool entirely, so a dev-clean or
    test-clean speaker can never leak into training (BLUEPRINT 7.5, holdout 1).

    Parameters
    ----------
    source_files:
        Clean single-speaker WAV/FLAC files, already at 8 kHz.
    allowed_n:
        Speaker counts to draw from. Defaults to [2, 3, 4, 5].
    db_min, db_max:
        Per-speaker level offset range in dB, drawn independently per speaker.
    held_out_speaker_ids:
        Speakers reserved for validation and evaluation. Excluded from the
        training pool and reachable only via ``mix(split="heldout")``.
    sample_rate:
        Expected input rate. Defaults to CORALSEP_SAMPLE_RATE (8000). A file at
        any other rate raises rather than being silently resampled, because a
        silent resample is how a 16 kHz file ends up interpreted as 8 kHz.
    rng:
        Seeded generator for reproducible mixes.
    seed:
        Recorded into every recipe for traceability. Pass the same value used
        to build ``rng``.
    """

    def __init__(
        self,
        source_files: Sequence[Path | str],
        allowed_n: list[int] | None = None,
        db_min: float = 0.0,
        db_max: float = 5.0,
        held_out_speaker_ids: set[str] | None = None,
        sample_rate: int = CORALSEP_SAMPLE_RATE,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        if db_min > db_max:
            raise ValueError(f"db_min ({db_min}) must be <= db_max ({db_max})")

        self._allowed_n = list(allowed_n) if allowed_n is not None else list(_DEFAULT_ALLOWED_N)
        if not self._allowed_n:
            raise ValueError("allowed_n must contain at least one value")
        bad = [n for n in self._allowed_n if n not in (2, 3, 4, 5)]
        if bad:
            raise ValueError(
                f"allowed_n contains {bad}; CoRAL-Sep supports N in {{2,3,4,5}} only "
                f"(K0=5 in the frozen checkpoint)"
            )

        self._db_min = float(db_min)
        self._db_max = float(db_max)
        self._sample_rate = int(sample_rate)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._seed = seed

        all_files = [Path(f) for f in source_files]
        held = set(held_out_speaker_ids) if held_out_speaker_ids else set()

        self._heldout_files: list[Path] = [f for f in all_files if _speaker_id(f) in held]
        self._train_files: list[Path] = [f for f in all_files if _speaker_id(f) not in held]

        max_n = max(self._allowed_n)
        if len(self._train_files) < max_n:
            raise ValueError(
                f"training pool has {len(self._train_files)} file(s) but "
                f"max(allowed_n)={max_n}; add sources or reduce allowed_n"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def mix(self, split: str = "train", n: int | None = None) -> CoralSepMixture:
        """
        Produce one clean 8 kHz mixture and its recipe.

        Degradations (reverb, noise, codec) are applied afterwards by
        data/degradations.py, which extends the returned recipe in place. This
        split keeps the source draw independent of the condition sampling, so
        the same mixture can be rendered under several conditions for the
        matched-pair analyses in BLUEPRINT 9.5.

        Args:
            split: "train" draws from the training pool (held-out speakers
                excluded), "heldout" draws only from held-out speakers, any
                other value draws from both.
            n: Speaker count override. Must be in allowed_n. Drawn uniformly
                from allowed_n when None.

        Returns:
            CoralSepMixture with an anechoic, noise-free, uncompressed mixture
            and a recipe recording the sources and level offsets.
        """
        chosen_n = self._resolve_n(n)
        pool = self._select_pool(split)

        if len(pool) < chosen_n:
            raise ValueError(
                f"pool for split={split!r} has {len(pool)} file(s) but n={chosen_n} requested"
            )

        indices = self._rng.choice(len(pool), size=chosen_n, replace=False)
        chosen = [pool[int(i)] for i in indices]

        waveforms: list[np.ndarray] = []
        offsets: list[float] = []
        for path in chosen:
            audio, sr = _load_wav(path)
            if sr != self._sample_rate:
                raise ValueError(
                    f"sample rate mismatch: {path} is {sr} Hz, expected {self._sample_rate} Hz. "
                    f"Resample the corpus with data/prepare_librispeech_8k.py rather than "
                    f"resampling here, so the whole pool stays consistent."
                )
            db = float(self._rng.uniform(self._db_min, self._db_max))
            offsets.append(db)
            waveforms.append((audio * (10.0 ** (db / 20.0))).astype(np.float32))

        refs, mixture = self._pad_and_sum(waveforms)
        uid = f"calmsep_{chosen_n}spk_{uuid.uuid4().hex[:8]}"

        recipe = MixtureRecipe(
            n_speakers=chosen_n,
            speaker_ids=[_speaker_id(p) for p in chosen],
            source_files=[str(p) for p in chosen],
            level_offsets_db=offsets,
            seed=self._seed,
            sample_rate=self._sample_rate,
        )
        sample = MixtureSample(
            mixture=mixture,
            references=refs,
            sample_rate=self._sample_rate,
            utterance_id=uid,
        )
        return CoralSepMixture(sample=sample, recipe=recipe)

    @property
    def train_pool_size(self) -> int:
        """Number of source files eligible for training mixes."""
        return len(self._train_files)

    @property
    def heldout_pool_size(self) -> int:
        """Number of source files reserved for validation and evaluation."""
        return len(self._heldout_files)

    @property
    def train_speakers(self) -> set[str]:
        """Speaker IDs present in the training pool."""
        return {_speaker_id(f) for f in self._train_files}

    @property
    def heldout_speakers(self) -> set[str]:
        """Speaker IDs reserved for validation and evaluation."""
        return {_speaker_id(f) for f in self._heldout_files}

    def assert_speaker_isolation(self) -> None:
        """
        Prove no speaker appears in both pools.

        Called by the preflight check and by tests. A violation here means
        BLUEPRINT holdout 1 is broken and every downstream number is suspect,
        so this raises rather than warns.
        """
        overlap = self.train_speakers & self.heldout_speakers
        if overlap:
            raise ValueError(
                f"speaker isolation violated: {sorted(overlap)} appear in both the "
                f"training and held-out pools"
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_n(self, n: int | None) -> int:
        if n is None:
            return int(self._rng.choice(self._allowed_n))
        if n not in self._allowed_n:
            raise ValueError(f"n={n} is not in allowed_n={self._allowed_n}")
        return n

    def _select_pool(self, split: str) -> list[Path]:
        if split == "heldout":
            return self._heldout_files
        if split == "train":
            return self._train_files
        return self._train_files + self._heldout_files

    @staticmethod
    def _pad_and_sum(waveforms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Zero-pad to the longest stem, stack to [N, T], sum to [T]."""
        max_len = max(w.shape[0] for w in waveforms)
        padded = [np.pad(w, (0, max_len - w.shape[0])).astype(np.float32) for w in waveforms]
        refs = np.stack(padded, axis=0)
        return refs, refs.sum(axis=0)
