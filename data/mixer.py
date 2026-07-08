"""
Dynamic on-the-fly audio mixer for CA-MoSE multi-speaker training.

Produces MixtureSample objects directly from clean single-speaker WAV/FLAC
files, applying random per-speaker volume offsets (dB -> linear) and
zero-padding all stems to the longest waveform before summing.  Speaker
isolation guarantees that speakers designated as test-only are never sampled
in training mixes.

Output type is MixtureSample imported from data.mixer_stub so the baseline
runner, evaluation harness, and future training loop all consume the same
object.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

from data.mixer_stub import MixtureSample, _load_wav

_DEFAULT_ALLOWED_N: list[int] = [2, 3, 4, 5]
_PROJECT_SAMPLE_RATE: int = 16_000


def _speaker_id(path: Path) -> str:
    """
    Extract speaker ID from a LibriSpeech-style filename.

    LibriSpeech filenames follow {speaker_id}-{chapter}-{utterance}.ext, so
    the speaker ID is the component before the first dash.  For files that do
    not follow this convention the full stem is returned as a safe fallback.
    """
    parts = path.stem.split("-")
    return parts[0]


class DynamicMixer:
    """
    Randomly mixes N clean single-speaker utterances into one mixture on the fly.

    Each call to mix() independently draws N speakers (without replacement),
    applies a random per-speaker dB gain, zero-pads all waveforms to equal
    length, and sums them to produce a MixtureSample.  A new, unique mixture
    is returned on every call — there is no fixed dataset.

    Parameters
    ----------
    source_files:
        Paths to clean single-speaker WAV or FLAC files (one utterance per
        file, as in LibriSpeech).
    allowed_n:
        Speaker counts to draw from.  mix() picks one value uniformly at
        random unless n is overridden explicitly.  Defaults to [2, 3, 4, 5].
    db_min, db_max:
        Per-speaker volume offset range in dB.  Each speaker independently
        draws from Uniform[db_min, db_max]; the dB value is converted to a
        linear gain via 10 ** (dB / 20).  Defaults to 0.0 and 5.0.
    train_speaker_ids:
        If provided, only files whose speaker ID is in this set are eligible
        for training mixes.  Files not in the set are silently excluded from
        the training pool.
    test_speaker_ids:
        Speaker IDs reserved for evaluation.  Files belonging to these
        speakers are excluded from the training pool and placed in a separate
        test pool accessible via mix(split='test').
    sample_rate:
        Expected sample rate in Hz.  A ValueError is raised if a loaded file
        has a different rate.  Defaults to 16 000 Hz.
    rng:
        NumPy random generator.  Pass a seeded generator for reproducible
        mixes.  Defaults to numpy.random.default_rng() (non-deterministic).
    """

    def __init__(
        self,
        source_files: Sequence[Path | str],
        allowed_n: list[int] | None = None,
        db_min: float = 0.0,
        db_max: float = 5.0,
        train_speaker_ids: set[str] | None = None,
        test_speaker_ids: set[str] | None = None,
        sample_rate: int = _PROJECT_SAMPLE_RATE,
        rng: np.random.Generator | None = None,
    ) -> None:
        if db_min > db_max:
            raise ValueError(f"db_min ({db_min}) must be <= db_max ({db_max}).")

        self._allowed_n = list(allowed_n) if allowed_n is not None else list(_DEFAULT_ALLOWED_N)
        if not self._allowed_n:
            raise ValueError("allowed_n must contain at least one value.")

        self._db_min = float(db_min)
        self._db_max = float(db_max)
        self._sample_rate = sample_rate
        self._rng = rng if rng is not None else np.random.default_rng()

        all_files = [Path(f) for f in source_files]

        # Partition into train and test pools based on speaker IDs.
        if test_speaker_ids is not None:
            self._test_files: list[Path] = [
                f for f in all_files if _speaker_id(f) in test_speaker_ids
            ]
            train_candidates = [
                f for f in all_files if _speaker_id(f) not in test_speaker_ids
            ]
        else:
            self._test_files = []
            train_candidates = list(all_files)

        if train_speaker_ids is not None:
            train_id_set = set(train_speaker_ids)
            self._train_files: list[Path] = [
                f for f in train_candidates if _speaker_id(f) in train_id_set
            ]
        else:
            self._train_files = train_candidates

        self._check_train_pool()

    # ── Public API ────────────────────────────────────────────────────────────

    def mix(self, split: str = "train", n: int | None = None) -> MixtureSample:
        """
        Produce one random mixture.

        Parameters
        ----------
        split:
            ``'train'`` draws from the training pool (test speakers excluded).
            ``'test'`` draws from the test pool (test speakers only).
            Any other value draws from the union of both pools.
        n:
            Override the speaker count for this mix.  Must be a member of
            ``allowed_n``.  If None, a value is drawn uniformly from
            ``allowed_n``.

        Returns
        -------
        MixtureSample
            mixture    : float32 ndarray, shape [T] — sum of gain-scaled stems
            references : float32 ndarray, shape [N, T] — gain-scaled stems,
                         zero-padded to the length of the longest stem
            sample_rate: always ``self._sample_rate``
            utterance_id: unique string for this mix
        """
        chosen_n = self._resolve_n(n)
        pool = self._select_pool(split)

        if len(pool) < chosen_n:
            raise ValueError(
                f"Pool for split='{split}' has {len(pool)} file(s) but "
                f"n={chosen_n} speakers were requested."
            )

        indices = self._rng.choice(len(pool), size=chosen_n, replace=False)
        chosen_files = [pool[int(i)] for i in indices]

        waveforms: list[np.ndarray] = []
        for path in chosen_files:
            audio, sr = _load_wav(path)
            if sr != self._sample_rate:
                raise ValueError(
                    f"Sample rate mismatch: '{path}' is {sr} Hz, "
                    f"expected {self._sample_rate} Hz."
                )
            db = float(self._rng.uniform(self._db_min, self._db_max))
            gain = 10.0 ** (db / 20.0)
            waveforms.append((audio * gain).astype(np.float32))

        refs, mixture = self._pad_and_sum(waveforms)
        uid = f"dyn_{chosen_n}spk_{uuid.uuid4().hex[:8]}"

        return MixtureSample(
            mixture=mixture,
            references=refs,
            sample_rate=self._sample_rate,
            utterance_id=uid,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_n(self, n: int | None) -> int:
        if n is None:
            return int(self._rng.choice(self._allowed_n))
        if n not in self._allowed_n:
            raise ValueError(f"n={n} is not in allowed_n={self._allowed_n}.")
        return n

    def _select_pool(self, split: str) -> list[Path]:
        if split == "test":
            return self._test_files
        if split == "train":
            return self._train_files
        return self._train_files + self._test_files

    @staticmethod
    def _pad_and_sum(
        waveforms: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Zero-pad all waveforms to max length, stack into [N, T], sum to [T]."""
        max_len = max(w.shape[0] for w in waveforms)
        padded = [
            np.pad(w, (0, max_len - w.shape[0])).astype(np.float32) for w in waveforms
        ]
        refs = np.stack(padded, axis=0)  # [N, T]
        mixture = refs.sum(axis=0)  # [T]
        return refs, mixture

    def _check_train_pool(self) -> None:
        max_n = max(self._allowed_n)
        if len(self._train_files) < max_n:
            raise ValueError(
                f"Training pool has {len(self._train_files)} file(s) but "
                f"max(allowed_n)={max_n}. Add more source files or reduce allowed_n."
            )
