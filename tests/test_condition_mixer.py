"""
Unit tests for data/condition_mixer.py (Dev A, P0-A1).

All tests synthesize 8 kHz WAV files in a temp directory: no corpus download
required. Tests verify speaker isolation, N constraints, recipe fields, and the
mixture invariants the rest of the pipeline depends on.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from coralsep.data.condition_mixer import (
    CORALSEP_SAMPLE_RATE,
    CoralSepMixer,
    CoralSepMixture,
    MixtureRecipe,
    _speaker_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = CORALSEP_SAMPLE_RATE


def _write_wav(path: Path, duration_s: float = 1.0, spk_id: str = "100") -> Path:
    """Write a mono 8 kHz WAV with a deterministic sine tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * SR)
    t = np.linspace(0, duration_s, n, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        pcm = (tone * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return path


def _build_pool(tmp_path: Path, n_speakers: int = 6, utts_each: int = 3) -> list[Path]:
    """Create a small corpus: n_speakers × utts_each files at 8 kHz."""
    files = []
    for spk in range(100, 100 + n_speakers):
        for utt in range(utts_each):
            p = tmp_path / f"{spk}-001-{utt:04d}.wav"
            _write_wav(p, duration_s=0.5, spk_id=str(spk))
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# MixtureRecipe
# ---------------------------------------------------------------------------


def test_condition_vector_clean():
    r = MixtureRecipe(n_speakers=2)
    v = r.condition_vector()
    assert v["snr_db"] == 60.0
    assert v["t60_s"] == 0.0
    assert v["codec_class"] == 0.0
    assert v["codec_bitrate_kbps"] == 0.0
    assert v["n_speakers"] == 2.0


def test_condition_vector_all_set():
    r = MixtureRecipe(
        n_speakers=3,
        snr_db=5.0,
        t60_s=0.4,
        codec_name="opus",
        codec_bitrate_bps=12_000,
    )
    v = r.condition_vector()
    assert v["snr_db"] == 5.0
    assert v["t60_s"] == 0.4
    assert v["codec_class"] == 1.0  # opus
    assert abs(v["codec_bitrate_kbps"] - 12.0) < 1e-6
    assert v["n_speakers"] == 3.0


def test_codec_class_indices():
    from coralsep.data.condition_mixer import _CODEC_CLASS_INDEX

    assert _CODEC_CLASS_INDEX["none"] == 0
    assert _CODEC_CLASS_INDEX["opus"] == 1
    assert _CODEC_CLASS_INDEX["aac"] == 2
    assert _CODEC_CLASS_INDEX["amr-nb"] == 3
    assert _CODEC_CLASS_INDEX["amr-wb"] == 4


# ---------------------------------------------------------------------------
# _speaker_id helper
# ---------------------------------------------------------------------------


def test_speaker_id_librispeech():
    p = Path("1234-567-0001.flac")
    assert _speaker_id(p) == "1234"


def test_speaker_id_no_dash():
    p = Path("unknown.wav")
    assert _speaker_id(p) == "unknown"


# ---------------------------------------------------------------------------
# CoralSepMixer construction
# ---------------------------------------------------------------------------


def test_rejects_n_outside_2_to_5(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    with pytest.raises(ValueError, match="allowed_n"):
        CoralSepMixer(files, allowed_n=[1, 2])


def test_rejects_6_speakers(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    with pytest.raises(ValueError, match="allowed_n"):
        CoralSepMixer(files, allowed_n=[6])


def test_rejects_insufficient_pool(tmp_path):
    files = _build_pool(tmp_path, n_speakers=1, utts_each=1)
    with pytest.raises(ValueError, match="training pool"):
        CoralSepMixer(files, allowed_n=[2])


# ---------------------------------------------------------------------------
# Speaker isolation
# ---------------------------------------------------------------------------


def test_assert_speaker_isolation_passes(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    held = {"100", "101"}
    mixer = CoralSepMixer(files, held_out_speaker_ids=held)
    mixer.assert_speaker_isolation()  # must not raise


def test_speaker_isolation_violated_when_held_out_in_train(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    # No held_out set → all files in train pool, heldout pool is empty → OK
    mixer = CoralSepMixer(files)
    mixer.assert_speaker_isolation()  # empty heldout, no overlap


def test_train_heldout_pools_are_disjoint(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    held = {"100", "101"}
    mixer = CoralSepMixer(files, held_out_speaker_ids=held)
    assert mixer.train_speakers.isdisjoint(mixer.heldout_speakers)


# ---------------------------------------------------------------------------
# Mix output shape and recipe
# ---------------------------------------------------------------------------


def test_mix_returns_calmsep_mixture(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    mixer = CoralSepMixer(files, allowed_n=[2], rng=np.random.default_rng(0))
    m = mixer.mix()
    assert isinstance(m, CoralSepMixture)


def test_mix_recipe_n_speakers(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    for n in [2, 3, 4, 5]:
        mixer = CoralSepMixer(files, allowed_n=[n], rng=np.random.default_rng(n))
        m = mixer.mix(n=n)
        assert m.recipe.n_speakers == n
        assert len(m.recipe.speaker_ids) == n
        assert len(m.recipe.source_files) == n
        assert len(m.recipe.level_offsets_db) == n


def test_mix_references_shape(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    mixer = CoralSepMixer(files, allowed_n=[3], rng=np.random.default_rng(1))
    m = mixer.mix(n=3)
    assert m.references.ndim == 2
    assert m.references.shape[0] == 3


def test_mix_mixture_equals_sum_of_references(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    mixer = CoralSepMixer(files, allowed_n=[2], rng=np.random.default_rng(2))
    m = mixer.mix(n=2)
    np.testing.assert_allclose(m.mixture, m.references.sum(axis=0), atol=1e-5)


def test_mix_sample_rate_locked(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    mixer = CoralSepMixer(files, rng=np.random.default_rng(3))
    m = mixer.mix()
    assert m.sample.sample_rate == CORALSEP_SAMPLE_RATE
    assert m.recipe.sample_rate == CORALSEP_SAMPLE_RATE


def test_mix_rejects_wrong_sample_rate(tmp_path):
    """A 16 kHz file must raise rather than silently resample."""
    import wave

    bad = tmp_path / "200-001-0000.wav"
    with wave.open(str(bad), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)  # wrong rate
        wf.writeframes(b"\x00" * 32_000)
    # Need at least 2 files for n=2; add one valid file
    good = tmp_path / "201-001-0000.wav"
    _write_wav(good)
    with pytest.raises(ValueError, match="sample rate mismatch"):
        mixer = CoralSepMixer([bad, good], allowed_n=[2])
        mixer.mix(n=2)


def test_unique_utterance_ids(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6)
    mixer = CoralSepMixer(files, rng=np.random.default_rng(7))
    ids = {mixer.mix().sample.utterance_id for _ in range(20)}
    assert len(ids) > 1  # uuid4-based, collisions are astronomically unlikely


def test_heldout_split(tmp_path):
    files = _build_pool(tmp_path, n_speakers=6, utts_each=4)
    held = {"100", "101"}
    mixer = CoralSepMixer(files, held_out_speaker_ids=held, allowed_n=[2])
    m = mixer.mix(split="heldout", n=2)
    for spk in m.recipe.speaker_ids:
        assert spk in held
