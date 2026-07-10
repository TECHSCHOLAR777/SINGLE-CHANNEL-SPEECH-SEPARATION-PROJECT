"""
Tests for data/make_reverb_eval.py.

The augmentor is injected (identity or a simple transform) so pyroomacoustics is
never required.  Real WAV I/O uses soundfile on tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data.make_reverb_eval import build_reverb_noisy_eval, verify_layout
from data.mixer_stub import MixtureSample, discover_librimix_samples

SR = 16000


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_libri3mix(root: Path, uids=("u1", "u2"), n_src: int = 3, length: int = SR) -> Path:
    """Create a tiny Libri3Mix test layout with FLOAT wavs."""
    base = root / "wav16k" / "max" / "test"
    rng = np.random.default_rng(0)
    for uid in uids:
        refs = rng.standard_normal((n_src, length)).astype(np.float32)
        mix = refs.sum(axis=0)
        (base / "mix_both").mkdir(parents=True, exist_ok=True)
        sf.write(str(base / "mix_both" / f"{uid}.wav"), mix, SR, subtype="FLOAT")
        for i in range(n_src):
            (base / f"s{i + 1}").mkdir(parents=True, exist_ok=True)
            sf.write(str(base / f"s{i + 1}" / f"{uid}.wav"), refs[i], SR, subtype="FLOAT")
    return root


def _identity(sample: MixtureSample) -> MixtureSample:
    return sample


def _add_one_to_mix(sample: MixtureSample) -> MixtureSample:
    """Fake augmentor: perturbs the mixture, leaves references clean."""
    return MixtureSample(
        mixture=sample.mixture + 1.0,
        references=sample.references,
        sample_rate=sample.sample_rate,
        utterance_id=sample.utterance_id,
    )


# ── build_reverb_noisy_eval ───────────────────────────────────────────────────


def test_build_writes_full_layout(tmp_path: Path) -> None:
    src = _make_libri3mix(tmp_path / "Libri3Mix")
    out = tmp_path / "ReverbNoisy"

    build_reverb_noisy_eval(src, out, augmentor=_identity)

    base = out / "wav16k" / "max" / "test"
    for stream in ("mix_both", "s1", "s2", "s3"):
        assert list((base / stream).glob("*.wav")), f"missing wavs in {stream}"


def test_build_output_is_loadable_by_discover(tmp_path: Path) -> None:
    src = _make_libri3mix(tmp_path / "Libri3Mix")
    out = tmp_path / "ReverbNoisy"

    build_reverb_noisy_eval(src, out, augmentor=_identity)
    samples = discover_librimix_samples(out, subset="test")

    assert len(samples) == 2
    assert samples[0].references.shape[0] == 3


def test_build_references_stay_clean(tmp_path: Path) -> None:
    src = _make_libri3mix(tmp_path / "Libri3Mix")
    out = tmp_path / "ReverbNoisy"

    clean = {s.utterance_id: s.references.copy() for s in discover_librimix_samples(src)}
    build_reverb_noisy_eval(src, out, augmentor=_add_one_to_mix)

    for s in discover_librimix_samples(out):
        np.testing.assert_allclose(s.references, clean[s.utterance_id], atol=1e-6)


def test_build_mixture_is_augmented(tmp_path: Path) -> None:
    src = _make_libri3mix(tmp_path / "Libri3Mix")
    out = tmp_path / "ReverbNoisy"

    orig = {s.utterance_id: s.mixture.copy() for s in discover_librimix_samples(src)}
    build_reverb_noisy_eval(src, out, augmentor=_add_one_to_mix)

    for s in discover_librimix_samples(out):
        assert not np.allclose(s.mixture, orig[s.utterance_id])


def test_build_skips_if_output_exists(tmp_path: Path) -> None:
    src = _make_libri3mix(tmp_path / "Libri3Mix")
    out = tmp_path / "ReverbNoisy"
    existing = out / "wav16k" / "max" / "test" / "mix_both"
    existing.mkdir(parents=True)
    (existing / "already.wav").write_bytes(b"RIFF")

    calls = []

    def _counting(sample: MixtureSample) -> MixtureSample:
        calls.append(sample.utterance_id)
        return sample

    build_reverb_noisy_eval(src, out, augmentor=_counting)
    assert calls == []  # skipped, augmentor never invoked


def test_build_raises_when_no_source_samples(tmp_path: Path) -> None:
    src = tmp_path / "Libri3Mix"
    (src / "wav16k" / "max" / "test" / "mix_both").mkdir(parents=True)  # empty
    out = tmp_path / "ReverbNoisy"

    with pytest.raises(RuntimeError, match="No Libri3Mix samples"):
        build_reverb_noisy_eval(src, out, augmentor=_identity)


def test_build_respects_max_samples(tmp_path: Path) -> None:
    src = _make_libri3mix(tmp_path / "Libri3Mix", uids=("u1", "u2", "u3"))
    out = tmp_path / "ReverbNoisy"

    build_reverb_noisy_eval(src, out, max_samples=1, augmentor=_identity)
    assert len(discover_librimix_samples(out, subset="test")) == 1


# ── verify_layout ─────────────────────────────────────────────────────────────


def test_verify_layout_passes(tmp_path: Path) -> None:
    src = _make_libri3mix(tmp_path / "Libri3Mix")
    out = tmp_path / "ReverbNoisy"
    build_reverb_noisy_eval(src, out, augmentor=_identity)
    verify_layout(out)  # must not raise


def test_verify_layout_raises_when_missing(tmp_path: Path) -> None:
    out = tmp_path / "ReverbNoisy"
    (out / "wav16k" / "max" / "test" / "mix_both").mkdir(parents=True)  # no s1/s2

    with pytest.raises(RuntimeError) as exc:
        verify_layout(out)
    assert "s1" in str(exc.value)
