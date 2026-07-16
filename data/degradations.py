"""
Degradation application with wet-reference policy (Dev A, P0-A7 and P1-A1/A2).

Applies reverberation, noise, and codec damage to a clean 8 kHz mixture and
extends its MixtureRecipe with the ground-truth labels. This is the module that
turns a clean CalmSepMixture into a training or evaluation example under a
named condition.

The reference policy is the subtle part. BLUEPRINT 7.6: for reverberant data
the target is the **wet source** (the speaker convolved with the RIR, truncated
at n_peak + 512 samples), not the dry source. The system is asked to separate
speakers, not to dereverberate them. Scoring against dry sources would conflate
two tasks and make reverberant SI-SDRi uninterpretable: a perfect separator
that leaves reverb intact would score badly, which is the wrong incentive.

The truncation offset (512 samples at 8 kHz = 64 ms) keeps the direct path and
the early reflections that arrive with it, and discards the late tail. Early
reflections are perceptually fused with the direct sound and carry the speaker's
timbre; the late tail is what a dereverberator would remove. Keeping the early
part in the target is what makes "separate but do not dereverberate" precise.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import signal

from data.calmsep_mixer import CALMSEP_SAMPLE_RATE, CalmSepMixture, MixtureRecipe
from data.mixer_stub import MixtureSample
from data.rir_bank import RirBank, RirRecord, sample_t60

WET_REFERENCE_OFFSET_SAMPLES: int = 512
"""
BLUEPRINT 7.6: wet references are truncated at n_peak + 512. At 8 kHz this is
64 ms of early reflections retained past the direct path. Matches the source
paper's single-channel n_offset.
"""

SNR_MIN_DB: float = -6.0
SNR_MAX_DB: float = 10.0
"""BLUEPRINT 5.3: adapter_noise trains on SNR uniform -6 to +10 dB."""

SEVERE_SNR_DB: float = -4.0
"""
BLUEPRINT 7.5 holdout 3: SNR below -4 dB is a severity holdout, kept to 10% of
noise training samples and probed in evaluation.
"""

SEVERE_FRACTION: float = 0.10
"""Fraction of training draws allowed into the severe (SNR < -4 dB) band."""

EPS: float = 1e-10
"""Energy guard. Matches eval/metrics.py EPS scale."""


def sample_snr(
    rng: np.random.Generator,
    allow_severe: bool = True,
    severe_fraction: float = SEVERE_FRACTION,
) -> float:
    """
    Draw a training SNR, honouring the severity holdout.

    Mirrors data.rir_bank.sample_t60 for the noise axis. BLUEPRINT 7.5
    holdout 3 keeps SNR below -4 dB rare in training so evaluation at low SNR
    measures extrapolation.

    Args:
        rng: Seeded generator.
        allow_severe: When False, never draws below SEVERE_SNR_DB.
        severe_fraction: Probability of drawing from the severe band.

    Returns:
        SNR in dB, within [SNR_MIN_DB, SNR_MAX_DB].
    """
    if not allow_severe:
        return float(rng.uniform(SEVERE_SNR_DB, SNR_MAX_DB))
    if rng.random() < severe_fraction:
        return float(rng.uniform(SNR_MIN_DB, SEVERE_SNR_DB))
    return float(rng.uniform(SEVERE_SNR_DB, SNR_MAX_DB))


def make_wet_reference(
    dry: np.ndarray,
    rir: np.ndarray,
    n_peak: int,
    offset: int = WET_REFERENCE_OFFSET_SAMPLES,
    target_length: int | None = None,
) -> np.ndarray:
    """
    Convolve a dry source with a truncated RIR to make the wet reference.

    The RIR is cut at n_peak + offset before convolution, so the reference
    contains the direct path and early reflections but not the late tail. See
    the module docstring for why this is the correct target.

    Args:
        dry: Clean source waveform [T].
        rir: Full room impulse response [R].
        n_peak: Index of the direct-path peak in rir (from RirRecord.n_peak).
        offset: Samples kept past the peak. Defaults to 512 (64 ms at 8 kHz).
        target_length: Crop or pad the result to exactly this many samples.
            Defaults to len(dry), which keeps the reference time-aligned with
            the dry source so SI-SDR is meaningful.

    Returns:
        Wet reference [target_length] float32.
    """
    d = np.asarray(dry, dtype=np.float32).squeeze()
    h = np.asarray(rir, dtype=np.float32).squeeze()
    if d.ndim != 1:
        raise ValueError(f"dry must be 1-D, got shape {d.shape}")
    if h.ndim != 1:
        raise ValueError(f"rir must be 1-D, got shape {h.shape}")

    cut = min(n_peak + offset, h.shape[0])
    if cut <= 0:
        raise ValueError(f"truncation index {cut} is not positive (n_peak={n_peak})")
    h_early = h[:cut]

    wet = signal.fftconvolve(d, h_early, mode="full")
    length = int(target_length) if target_length is not None else d.shape[0]
    return _fit_length(wet, length).astype(np.float32)


def make_wet_mixture(
    dry: np.ndarray,
    rir: np.ndarray,
    target_length: int | None = None,
) -> np.ndarray:
    """
    Convolve a dry source with the **full** RIR to make the observed signal.

    The mixture the system hears carries the complete reverberation including
    the late tail. Only the *reference* is truncated. Using the truncated RIR
    for both would mean the system never sees the tail it must be robust to.

    Args:
        dry: Clean source waveform [T].
        rir: Full room impulse response [R].
        target_length: Output length. Defaults to len(dry).

    Returns:
        Reverberant observation [target_length] float32.
    """
    d = np.asarray(dry, dtype=np.float32).squeeze()
    h = np.asarray(rir, dtype=np.float32).squeeze()
    wet = signal.fftconvolve(d, h, mode="full")
    length = int(target_length) if target_length is not None else d.shape[0]
    return _fit_length(wet, length).astype(np.float32)


def apply_reverb(
    mixture: CalmSepMixture,
    rir_bank: RirBank,
    rng: np.random.Generator,
    t60_s: float | None = None,
    allow_severe: bool = True,
    record: RirRecord | None = None,
) -> CalmSepMixture:
    """
    Reverberate every stem with one shared room, and rebuild the mixture.

    All speakers share the same RIR because they are in the same room. Drawing
    a separate RIR per speaker would model an acoustically impossible scene and
    would give the model a spurious per-speaker cue to separate on.

    The returned mixture's references are **wet** (truncated RIR) and its
    observation is **fully reverberant** (full RIR). The recipe records the
    achieved T60 and the RIR path.

    Args:
        mixture: A clean CalmSepMixture.
        rir_bank: Loaded simulated RIR bank.
        rng: Seeded generator.
        t60_s: Target T60. Drawn via sample_t60 when None.
        allow_severe: Passed to sample_t60 for the severity holdout.
        record: Use this exact RIR instead of drawing one. Set by the fixed
            evaluation generator so an eval cell pins its rooms.

    Returns:
        A new CalmSepMixture. The input is not modified.
    """
    if record is None:
        target = t60_s if t60_s is not None else sample_t60(rng, allow_severe=allow_severe)
        record = rir_bank.sample(target)
    rir = rir_bank.load(record)

    refs = mixture.references
    length = refs.shape[1]

    wet_refs = np.stack(
        [make_wet_reference(refs[i], rir, record.n_peak, target_length=length) for i in range(refs.shape[0])],
        axis=0,
    )
    wet_obs = np.stack(
        [make_wet_mixture(refs[i], rir, target_length=length) for i in range(refs.shape[0])],
        axis=0,
    )
    observed = wet_obs.sum(axis=0).astype(np.float32)

    recipe = replace(
        mixture.recipe,
        t60_s=record.t60_achieved_s,
        rir_file=record.path,
    )
    sample = MixtureSample(
        mixture=observed,
        references=wet_refs,
        sample_rate=mixture.sample.sample_rate,
        utterance_id=mixture.sample.utterance_id,
    )
    return CalmSepMixture(sample=sample, recipe=recipe)


def apply_noise(
    mixture: CalmSepMixture,
    noise: np.ndarray,
    rng: np.random.Generator,
    snr_db: float | None = None,
    allow_severe: bool = True,
    noise_file: str | None = None,
) -> CalmSepMixture:
    """
    Add background noise at a drawn SNR, leaving the references untouched.

    The references do not change: noise is not a speaker, so removing it is
    part of the task, not part of the target. The SNR is computed against the
    speech mixture's energy so the label means what it says.

    Args:
        mixture: A CalmSepMixture, clean or already reverberated.
        noise: Noise waveform [T_n] at the mixture's rate. Looped or cropped to
            the mixture's length.
        rng: Seeded generator, used for the SNR draw and the noise offset.
        snr_db: Target SNR. Drawn via sample_snr when None.
        allow_severe: Passed to sample_snr for the severity holdout.
        noise_file: Path recorded into the recipe for traceability.

    Returns:
        A new CalmSepMixture with noise added to the observation.
    """
    target_snr = snr_db if snr_db is not None else sample_snr(rng, allow_severe=allow_severe)

    speech = mixture.mixture
    length = speech.shape[0]
    noise_fit = _loop_or_crop(np.asarray(noise, dtype=np.float32).squeeze(), length, rng)

    speech_power = float(np.mean(speech**2))
    noise_power = float(np.mean(noise_fit**2))
    if noise_power < EPS:
        raise ValueError("noise segment is silent; cannot scale it to a target SNR")
    if speech_power < EPS:
        raise ValueError("speech mixture is silent; SNR is undefined")

    # scale so that 10*log10(speech_power / (noise_power * scale^2)) == target_snr
    scale = float(np.sqrt(speech_power / (noise_power * 10.0 ** (target_snr / 10.0))))
    noisy = (speech + noise_fit * scale).astype(np.float32)

    recipe = replace(mixture.recipe, snr_db=target_snr, noise_file=noise_file)
    sample = MixtureSample(
        mixture=noisy,
        references=mixture.references,
        sample_rate=mixture.sample.sample_rate,
        utterance_id=mixture.sample.utterance_id,
    )
    return CalmSepMixture(sample=sample, recipe=recipe)


def apply_codec(
    mixture: CalmSepMixture,
    codec_name: str,
    bitrate_bps: int,
    tmp_dir: str | Path | None = None,
) -> CalmSepMixture:
    """
    Round-trip the observation through a lossy codec, leaving references clean.

    Like noise, codec damage is something the system must undo, so it is applied
    to the observation only. data/codec_augmentation.py owns the ffmpeg calls;
    this wrapper adapts them to the CalmSepMixture type and records the labels.

    Args:
        mixture: A CalmSepMixture at 8 kHz.
        codec_name: One of "opus", "aac", "amr-nb", "amr-wb".
        bitrate_bps: Target bitrate in bits per second.
        tmp_dir: Scratch directory for the encode/decode round trip.

    Returns:
        A new CalmSepMixture with a codec-damaged observation.
    """
    from data.codec_augmentation import apply_codec_roundtrip

    damaged = apply_codec_roundtrip(
        audio=mixture.mixture,
        sample_rate=mixture.sample.sample_rate,
        codec=codec_name,
        bitrate_bps=bitrate_bps,
        tmp_dir=tmp_dir,
    )
    damaged = _fit_length(damaged, mixture.mixture.shape[0]).astype(np.float32)

    recipe = replace(mixture.recipe, codec_name=codec_name, codec_bitrate_bps=int(bitrate_bps))
    sample = MixtureSample(
        mixture=damaged,
        references=mixture.references,
        sample_rate=mixture.sample.sample_rate,
        utterance_id=mixture.sample.utterance_id,
    )
    return CalmSepMixture(sample=sample, recipe=recipe)


def _fit_length(x: np.ndarray, length: int) -> np.ndarray:
    """Crop or zero-pad a 1-D signal to exactly `length` samples."""
    t = x.shape[0]
    if t == length:
        return x
    if t > length:
        return x[:length]
    return np.pad(x, (0, length - t))


def _loop_or_crop(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Fit a noise clip to `length` by looping it or cropping a random window.

    A random offset is used rather than always starting at sample 0, so a long
    noise file contributes many distinct segments across an epoch instead of
    the same opening every time.
    """
    n = noise.shape[0]
    if n == 0:
        raise ValueError("noise clip is empty")
    if n < length:
        reps = int(np.ceil(length / n))
        return np.tile(noise, reps)[:length]
    start = int(rng.integers(0, n - length + 1))
    return noise[start : start + length]


def describe_condition(recipe: MixtureRecipe) -> str:
    """
    Name the condition cell a recipe belongs to.

    Used to key the evaluation matrix (BLUEPRINT 9.4) and to check the
    condition-combination holdout (7.5, holdout 2).

    Returns:
        One of "clean", "reverb", "noise", "codec", "reverb+noise",
        "reverb+codec", "noise+codec", "all-three".
    """
    parts: list[str] = []
    if recipe.t60_s is not None and recipe.t60_s > 0.0:
        parts.append("reverb")
    if recipe.snr_db is not None:
        parts.append("noise")
    if recipe.codec_name is not None:
        parts.append("codec")

    if not parts:
        return "clean"
    if len(parts) == 3:
        return "all-three"
    return "+".join(parts)


HELD_OUT_COMBINATIONS: frozenset[str] = frozenset({"reverb+codec", "noise+codec"})
"""
BLUEPRINT 7.5 holdout 2: these combinations never appear in gate or joint
training and exist only in the evaluation matrix, so compositional
generalisation is measured rather than assumed. assert_not_held_out enforces it.
"""


def assert_not_held_out(recipe: MixtureRecipe) -> None:
    """
    Raise if a recipe belongs to a held-out combination cell.

    Called by the gate and joint-polish training data pipelines. A held-out
    combination reaching training silently invalidates the compositional
    generalisation claim in BLUEPRINT 9.5 analysis 3, so this raises.
    """
    cell = describe_condition(recipe)
    if cell in HELD_OUT_COMBINATIONS:
        raise ValueError(
            f"recipe is in held-out combination cell {cell!r}; it must not enter gate or "
            f"joint training (BLUEPRINT 7.5 holdout 2). Held out: {sorted(HELD_OUT_COMBINATIONS)}"
        )
