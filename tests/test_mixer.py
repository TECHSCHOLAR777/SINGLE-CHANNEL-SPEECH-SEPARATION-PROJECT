"""
Tests for data/mixer.py, DynamicMixer.

All tests use synthetic WAV files written to tmp_path with soundfile.
No GPU, no internet, and no real LibriSpeech data required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from coralsep.data.mixer import DynamicMixer, _speaker_id
from coralsep.data.mixer_stub import MixtureSample

# ── Fixtures ──────────────────────────────────────────────────────────────────

SR = 16_000


def _write_speakers(
    tmp_path: Path,
    n: int,
    n_samples: int = SR,
) -> list[Path]:
    """
    Write n WAV files with LibriSpeech-style names.

    File i gets filename "{i:04d}-001-0000.wav" and contains a constant float
    value of float(i + 1) so that (with gain=1.0) references[k][0] == i+1,
    making it easy to identify which speaker was chosen without inspecting
    private state.

    subtype='FLOAT' preserves exact float32 values; the default PCM_16 would
    quantize values >= 1.0 to 32767/32768, collapsing all speakers to the same
    clipped value and making identity tests unreliable.
    """
    paths: list[Path] = []
    for i in range(n):
        audio = np.full(n_samples, float(i + 1), dtype=np.float32)
        path = tmp_path / f"{i:04d}-001-0000.wav"
        sf.write(str(path), audio, SR, subtype="FLOAT")
        paths.append(path)
    return paths


def _write_speaker_varied_length(tmp_path: Path, lengths: list[int]) -> list[Path]:
    """Write WAV files with distinct lengths; audio value equals the index+1."""
    paths: list[Path] = []
    for i, length in enumerate(lengths):
        audio = np.full(length, float(i + 1), dtype=np.float32)
        path = tmp_path / f"{i:04d}-001-0000.wav"
        sf.write(str(path), audio, SR, subtype="FLOAT")
        paths.append(path)
    return paths


# ── _speaker_id ───────────────────────────────────────────────────────────────


def test_speaker_id_librispeech_style() -> None:
    assert _speaker_id(Path("84-121123-0000.flac")) == "84"


def test_speaker_id_zero_padded() -> None:
    assert _speaker_id(Path("0003-001-0000.wav")) == "0003"


def test_speaker_id_no_dash_fallback() -> None:
    assert _speaker_id(Path("singlename.wav")) == "singlename"


# ── Construction validation ───────────────────────────────────────────────────


def test_construction_raises_if_train_pool_too_small(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 3)
    with pytest.raises(ValueError, match="Training pool"):
        DynamicMixer(files, allowed_n=[4])


def test_construction_raises_if_db_min_exceeds_db_max(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 4)
    with pytest.raises(ValueError, match="db_min"):
        DynamicMixer(files, allowed_n=[2], db_min=5.0, db_max=2.0)


def test_construction_raises_if_allowed_n_empty(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 4)
    with pytest.raises(ValueError, match="allowed_n"):
        DynamicMixer(files, allowed_n=[])


# ── Return type and shape ─────────────────────────────────────────────────────


def test_mix_returns_mixture_sample(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 4)
    mixer = DynamicMixer(files, allowed_n=[2])
    sample = mixer.mix()
    assert isinstance(sample, MixtureSample)


def test_mix_sample_rate_matches_constructor(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 4)
    mixer = DynamicMixer(files, allowed_n=[2], sample_rate=SR)
    sample = mixer.mix()
    assert sample.sample_rate == SR


def test_references_shape_equals_n_for_each_allowed_n(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 5)
    for n in [2, 3, 4, 5]:
        mixer = DynamicMixer(files, allowed_n=[n])
        sample = mixer.mix()
        assert (
            sample.references.shape[0] == n
        ), f"expected {n} references, got {sample.references.shape[0]}"


def test_mixture_is_1d_float32(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 4)
    mixer = DynamicMixer(files, allowed_n=[3])
    sample = mixer.mix()
    assert sample.mixture.ndim == 1
    assert sample.mixture.dtype == np.float32


def test_references_are_2d_float32(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 4)
    mixer = DynamicMixer(files, allowed_n=[3])
    sample = mixer.mix()
    assert sample.references.ndim == 2
    assert sample.references.dtype == np.float32


# ── Core arithmetic: mixture == sum of scaled stems ───────────────────────────


def test_mixture_equals_sum_of_references_zero_gain(tmp_path: Path) -> None:
    """With db_min=db_max=0, gain=1.0, mixture must exactly equal sum of references."""
    files = _write_speakers(tmp_path, 4)
    mixer = DynamicMixer(files, allowed_n=[3], db_min=0.0, db_max=0.0)
    sample = mixer.mix()
    expected = sample.references.sum(axis=0)
    assert np.abs(sample.mixture - expected).max() < 1e-5


def test_mixture_equals_sum_of_references_nonzero_gain(tmp_path: Path) -> None:
    """With an arbitrary fixed gain the invariant mixture == sum(references) still holds."""
    files = _write_speakers(tmp_path, 4)
    mixer = DynamicMixer(files, allowed_n=[3], db_min=3.0, db_max=3.0)
    sample = mixer.mix()
    expected = sample.references.sum(axis=0)
    assert np.abs(sample.mixture - expected).max() < 1e-5


def test_mixture_equals_sum_of_references_random_gain(tmp_path: Path) -> None:
    """Over many mixes with random gains the invariant must hold."""
    files = _write_speakers(tmp_path, 5)
    mixer = DynamicMixer(files, allowed_n=[4], db_min=0.0, db_max=5.0)
    for _ in range(20):
        sample = mixer.mix()
        expected = sample.references.sum(axis=0)
        assert np.abs(sample.mixture - expected).max() < 1e-5


# ── No speaker repeated in one mix ───────────────────────────────────────────


def test_no_speaker_file_repeated_in_one_mix(tmp_path: Path) -> None:
    """
    Each reference comes from a distinct file.

    Files contain unique constant values; with db_min=db_max=0 (gain=1.0) the
    first sample of each reference equals the file's constant, so all N values
    must be distinct.
    """
    files = _write_speakers(tmp_path, 5)
    mixer = DynamicMixer(files, allowed_n=[4], db_min=0.0, db_max=0.0)
    for _ in range(30):
        sample = mixer.mix()
        first_samples = [float(ref[0]) for ref in sample.references]
        assert len(first_samples) == len(
            set(first_samples)
        ), f"Speaker repeated in one mix: values={first_samples}"


# ── Volume offsets within requested dB range ─────────────────────────────────


def test_volume_offsets_within_db_range(tmp_path: Path) -> None:
    """
    With all-ones audio the first sample of each reference equals the applied
    linear gain, allowing direct verification against the dB bounds.
    """
    n_files = 5
    paths: list[Path] = []
    for i in range(n_files):
        audio = np.ones(SR, dtype=np.float32)
        path = tmp_path / f"{i:04d}-001-0000.wav"
        sf.write(str(path), audio, SR, subtype="FLOAT")
        paths.append(path)

    db_min, db_max = 1.0, 4.0
    gain_min = 10.0 ** (db_min / 20.0)
    gain_max = 10.0 ** (db_max / 20.0)

    mixer = DynamicMixer(paths, allowed_n=[3], db_min=db_min, db_max=db_max)
    for _ in range(50):
        sample = mixer.mix()
        for ref in sample.references:
            gain = float(ref[0])  # original audio is 1.0, so ref[0] == gain
            assert (
                gain_min - 1e-6 <= gain <= gain_max + 1e-6
            ), f"gain {gain:.6f} outside [{gain_min:.6f}, {gain_max:.6f}]"


def test_zero_db_offset_leaves_amplitude_unchanged(tmp_path: Path) -> None:
    """db_min=db_max=0 => gain exactly 1.0 => references equal original audio."""
    files = _write_speakers(tmp_path, 3, n_samples=SR)
    mixer = DynamicMixer(files, allowed_n=[2], db_min=0.0, db_max=0.0)
    sample = mixer.mix()
    for ref in sample.references:
        # Each file contains a constant; that constant should be unchanged
        assert np.allclose(ref, ref[0], atol=1e-6)  # all samples equal (constant audio)


# ── Test speaker never appears in training mix ────────────────────────────────


def test_test_speaker_never_in_train_mix(tmp_path: Path) -> None:
    """
    Speakers 3 and 4 are designated as test-only.  Running mix(split='train')
    many times must never produce a reference whose value matches a test speaker.

    File i contains constant float(i+1); with db_min=db_max=0, reference values
    directly identify the source file / speaker.  Test speakers have values 4.0
    and 5.0.
    """
    files = _write_speakers(tmp_path, 5)
    test_speaker_ids = {"0003", "0004"}
    mixer = DynamicMixer(
        files,
        allowed_n=[2],
        db_min=0.0,
        db_max=0.0,
        test_speaker_ids=test_speaker_ids,
    )

    test_values = {4.0, 5.0}  # file 3 → value 4.0, file 4 → value 5.0
    for _ in range(100):
        sample = mixer.mix(split="train")
        for ref in sample.references:
            assert (
                float(ref[0]) not in test_values
            ), f"Test speaker (value={ref[0]}) leaked into training mix."


def test_test_split_draws_only_test_speakers(tmp_path: Path) -> None:
    """mix(split='test') must draw exclusively from the test speaker pool."""
    files = _write_speakers(tmp_path, 5)
    test_speaker_ids = {"0003", "0004"}
    mixer = DynamicMixer(
        files,
        allowed_n=[2],
        db_min=0.0,
        db_max=0.0,
        test_speaker_ids=test_speaker_ids,
    )

    train_values = {1.0, 2.0, 3.0}  # files 0–2
    for _ in range(50):
        sample = mixer.mix(split="test")
        for ref in sample.references:
            assert (
                float(ref[0]) not in train_values
            ), f"Train speaker (value={ref[0]}) appeared in test mix."


def test_train_speaker_ids_further_restricts_pool(tmp_path: Path) -> None:
    """When train_speaker_ids is provided only those speakers appear in train mixes."""
    files = _write_speakers(tmp_path, 5)
    # Only speakers 0 and 1 are allowed in training
    mixer = DynamicMixer(
        files,
        allowed_n=[2],
        db_min=0.0,
        db_max=0.0,
        train_speaker_ids={"0000", "0001"},
    )

    allowed_values = {1.0, 2.0}
    for _ in range(50):
        sample = mixer.mix(split="train")
        for ref in sample.references:
            assert (
                float(ref[0]) in allowed_values
            ), f"Unexpected speaker value {ref[0]} in restricted train mix."


# ── Zero-padding of mismatched lengths ───────────────────────────────────────


def test_mismatched_lengths_zero_padded_to_longest(tmp_path: Path) -> None:
    """All stems are padded to the longest; mixture length equals max stem length."""
    lengths = [8000, 12000, 16000]
    files = _write_speaker_varied_length(tmp_path, lengths)
    mixer = DynamicMixer(files, allowed_n=[3])
    sample = mixer.mix()

    assert sample.mixture.shape[0] == max(lengths)
    assert sample.references.shape == (3, max(lengths))


def test_padding_is_zeros(tmp_path: Path) -> None:
    """Shorter stems must be padded with zeros, not repeated or wrapped audio."""
    short_audio = np.ones(4000, dtype=np.float32)
    long_audio = np.ones(8000, dtype=np.float32)

    short_path = tmp_path / "0000-001-0000.wav"
    long_path = tmp_path / "0001-001-0000.wav"
    sf.write(str(short_path), short_audio, SR, subtype="FLOAT")
    sf.write(str(long_path), long_audio, SR, subtype="FLOAT")

    mixer = DynamicMixer(
        [short_path, long_path],
        allowed_n=[2],
        db_min=0.0,
        db_max=0.0,
    )
    sample = mixer.mix()

    # Find the reference that came from the short file (length 4000, padded to 8000)
    short_ref = None
    for ref in sample.references:
        # Short file has value 1.0; padded region should be 0.0
        if np.all(ref[:4000] == pytest.approx(1.0)) and np.all(ref[4000:] == 0.0):
            short_ref = ref
            break

    assert short_ref is not None, "Could not find the short (padded) reference in the mix."


# ── Unique utterance IDs ──────────────────────────────────────────────────────


def test_utterance_ids_are_unique(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 5)
    mixer = DynamicMixer(files, allowed_n=[2])
    ids = [mixer.mix().utterance_id for _ in range(100)]
    assert len(ids) == len(set(ids)), "Duplicate utterance IDs detected."


def test_utterance_id_contains_speaker_count(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 4)
    for n in [2, 3, 4]:
        mixer = DynamicMixer(files, allowed_n=[n])
        uid = mixer.mix().utterance_id
        assert f"{n}spk" in uid, f"Expected '{n}spk' in utterance_id '{uid}'."


# ── Explicit n override ───────────────────────────────────────────────────────


def test_explicit_n_overrides_allowed_n(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 5)
    mixer = DynamicMixer(files, allowed_n=[2, 3, 4, 5])
    for n in [2, 3, 4, 5]:
        sample = mixer.mix(n=n)
        assert sample.references.shape[0] == n


def test_explicit_n_not_in_allowed_n_raises(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 5)
    mixer = DynamicMixer(files, allowed_n=[2, 3])
    with pytest.raises(ValueError, match="allowed_n"):
        mixer.mix(n=5)


# ── Reproducibility with seeded rng ──────────────────────────────────────────


def test_seeded_rng_produces_identical_mixes(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 5)
    seed = 42
    mixer_a = DynamicMixer(files, allowed_n=[3], rng=np.random.default_rng(seed))
    mixer_b = DynamicMixer(files, allowed_n=[3], rng=np.random.default_rng(seed))
    sample_a = mixer_a.mix()
    sample_b = mixer_b.mix()
    np.testing.assert_array_equal(sample_a.mixture, sample_b.mixture)
    np.testing.assert_array_equal(sample_a.references, sample_b.references)


# ── Overlap wiring (P0-A6 integration) ────────────────────────────────────────


def test_default_mix_is_full_overlap_unchanged(tmp_path: Path) -> None:
    # No overlap args -> current behaviour: all stems start at 0, equal length.
    files = _write_speakers(tmp_path, 3)
    mixer = DynamicMixer(files, allowed_n=[3], rng=np.random.default_rng(0))
    sample = mixer.mix(n=3)
    assert sample.references.shape == (3, SR)
    np.testing.assert_allclose(sample.mixture, sample.references.sum(axis=0), atol=1e-5)


def test_explicit_zero_overlap_is_sequential(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 3)
    mixer = DynamicMixer(files, allowed_n=[3], rng=np.random.default_rng(0))
    sample = mixer.mix(n=3, overlap_ratio=0.0)
    # Laid end to end: total length == sum of stem lengths.
    assert sample.references.shape[1] == 3 * SR
    active = (np.abs(sample.references) > 1e-6).sum(axis=0)
    assert active.max() <= 1  # never two speakers at once


def test_explicit_partial_overlap_length(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 2)
    mixer = DynamicMixer(files, allowed_n=[2], rng=np.random.default_rng(0))
    sample = mixer.mix(n=2, overlap_ratio=0.5)
    # Second stem starts at (1 - 0.5) * SR -> total length 1.5 * SR.
    assert sample.references.shape[1] == int(1.5 * SR)
    # Mixture is still exactly the sum of the (offset) references.
    np.testing.assert_allclose(sample.mixture, sample.references.sum(axis=0), atol=1e-5)


def test_scheduler_drives_overlap_via_progress(tmp_path: Path) -> None:
    from coralsep.data.overlap_scheduler import OverlapScheduler

    files = _write_speakers(tmp_path, 3)
    sched = OverlapScheduler()  # 100% -> 40% -> 20%
    mixer = DynamicMixer(
        files, allowed_n=[3], rng=np.random.default_rng(0), overlap_scheduler=sched
    )
    # Early progress -> full overlap (length SR); late progress -> sparse (longer).
    early = mixer.mix(n=3, progress=0.0)
    late = mixer.mix(n=3, progress=0.9)
    assert early.references.shape[1] == SR
    assert late.references.shape[1] > SR


def test_explicit_ratio_overrides_scheduler(tmp_path: Path) -> None:
    from coralsep.data.overlap_scheduler import OverlapScheduler

    files = _write_speakers(tmp_path, 2)
    sched = OverlapScheduler(phases=[(0.0, 1.0)])  # scheduler would say full overlap
    mixer = DynamicMixer(
        files, allowed_n=[2], rng=np.random.default_rng(0), overlap_scheduler=sched
    )
    sample = mixer.mix(n=2, overlap_ratio=0.0, progress=0.0)
    # Explicit 0.0 wins over the scheduler's 1.0 -> sequential, length 2*SR.
    assert sample.references.shape[1] == 2 * SR


def test_no_scheduler_ignores_progress(tmp_path: Path) -> None:
    # progress without a scheduler must not change the default full-overlap mix.
    files = _write_speakers(tmp_path, 2)
    mixer = DynamicMixer(files, allowed_n=[2], rng=np.random.default_rng(0))
    sample = mixer.mix(n=2, progress=0.9)
    assert sample.references.shape[1] == SR


def test_overlap_mix_is_reproducible(tmp_path: Path) -> None:
    files = _write_speakers(tmp_path, 4)
    a = DynamicMixer(files, allowed_n=[3], rng=np.random.default_rng(11)).mix(
        n=3, overlap_ratio=0.4
    )
    b = DynamicMixer(files, allowed_n=[3], rng=np.random.default_rng(11)).mix(
        n=3, overlap_ratio=0.4
    )
    np.testing.assert_array_equal(a.mixture, b.mixture)
    np.testing.assert_array_equal(a.references, b.references)
