"""
Tests for data/prepare_vctk.py (P0-A4).

Network/zip calls are mocked.  Resample + rename is exercised for real against
tiny 48 kHz FLACs written to tmp_path (no download needed).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from coralsep.data.prepare_vctk import (
    VCTK_URL,
    _parse_vctk_name,
    build_pool,
    discover_vctk_files,
    download_vctk,
    resample_to_16k_mono,
    vctk_speaker_id,
    verify_pool,
)

SRC_SR = 48000


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _write_flac(path: Path, seconds: float = 0.1, sr: int = SRC_SR, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(int(seconds * sr)) * 0.1).astype(np.float32)
    sf.write(str(path), data, sr)  # .flac extension -> FLAC/PCM_16


def _make_vctk_speech_dir(
    root: Path, speakers=("p225", "p226", "p227"), mics=("mic1", "mic2")
) -> Path:
    speech = root / "wav48_silence_trimmed"
    for i, spk in enumerate(speakers):
        for utt in ("001", "002"):
            for mic in mics:
                _write_flac(speech / spk / f"{spk}_{utt}_{mic}.flac", seed=i)
    return speech


# ── name parsing / speaker id ─────────────────────────────────────────────────


def test_parse_vctk_name_keeps_matching_mic() -> None:
    assert _parse_vctk_name(Path("p225_001_mic1.flac"), "mic1") == ("p225", "001")


def test_parse_vctk_name_rejects_other_mic() -> None:
    assert _parse_vctk_name(Path("p225_001_mic2.flac"), "mic1") is None


def test_parse_vctk_name_without_mic_suffix() -> None:
    assert _parse_vctk_name(Path("p300_042.flac"), "mic1") == ("p300", "042")


def test_speaker_id_from_pool_name() -> None:
    assert vctk_speaker_id(Path("p225-001.wav")) == "p225"


# ── resampling ────────────────────────────────────────────────────────────────


def test_resample_changes_length_48k_to_16k() -> None:
    audio = np.zeros(SRC_SR, dtype=np.float32)  # 1 second
    out = resample_to_16k_mono(audio, SRC_SR)
    assert abs(len(out) - 16000) <= 1
    assert out.dtype == np.float32


def test_resample_downmixes_stereo() -> None:
    stereo = np.ones((1000, 2), dtype=np.float32)
    out = resample_to_16k_mono(stereo, 16000)  # same sr -> just downmix
    assert out.ndim == 1
    assert len(out) == 1000


# ── build_pool ────────────────────────────────────────────────────────────────


def test_build_pool_resamples_and_renames(tmp_path: Path) -> None:
    speech = _make_vctk_speech_dir(tmp_path / "vctk")
    pool = tmp_path / "pool"
    build_pool(speech, pool, mic="mic1")

    wavs = discover_vctk_files(pool)
    names = {w.name for w in wavs}
    # 3 speakers x 2 utts, mic1 only, LibriSpeech-style names.
    assert names == {
        "p225-001.wav",
        "p225-002.wav",
        "p226-001.wav",
        "p226-002.wav",
        "p227-001.wav",
        "p227-002.wav",
    }
    # output is 16 kHz
    _, sr = sf.read(str(wavs[0]))
    assert sr == 16000


def test_build_pool_only_keeps_selected_mic(tmp_path: Path) -> None:
    speech = _make_vctk_speech_dir(tmp_path / "vctk")
    pool = tmp_path / "pool"
    build_pool(speech, pool, mic="mic1")
    # mic2 files must not leak in (still 6, not 12)
    assert len(discover_vctk_files(pool)) == 6


def test_build_pool_max_speakers(tmp_path: Path) -> None:
    speech = _make_vctk_speech_dir(tmp_path / "vctk")
    pool = tmp_path / "pool"
    build_pool(speech, pool, mic="mic1", max_speakers=2)
    speakers = {vctk_speaker_id(w) for w in discover_vctk_files(pool)}
    assert speakers == {"p225", "p226"}


def test_build_pool_skips_if_populated(tmp_path: Path) -> None:
    speech = _make_vctk_speech_dir(tmp_path / "vctk")
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "existing-001.wav").write_bytes(b"RIFF")
    build_pool(speech, pool, mic="mic1")  # should skip
    assert discover_vctk_files(pool) == [pool / "existing-001.wav"]


def test_build_pool_no_matching_mic_raises(tmp_path: Path) -> None:
    speech = _make_vctk_speech_dir(tmp_path / "vctk", mics=("mic2",))
    with pytest.raises(RuntimeError, match="No VCTK FLACs"):
        build_pool(speech, tmp_path / "pool", mic="mic1")


# ── produced pool is mixer-compatible ─────────────────────────────────────────


def test_pool_is_consumable_by_dynamic_mixer(tmp_path: Path) -> None:
    speech = _make_vctk_speech_dir(tmp_path / "vctk")
    pool = tmp_path / "pool"
    build_pool(speech, pool, mic="mic1")

    from coralsep.data.mixer import DynamicMixer

    files = discover_vctk_files(pool)
    mixer = DynamicMixer(files, allowed_n=[2], test_speaker_ids={"p227"})
    sample = mixer.mix(split="train", n=2)
    assert sample.references.shape[0] == 2
    assert sample.sample_rate == 16000


# ── verify_pool ───────────────────────────────────────────────────────────────


def test_verify_pool_passes(tmp_path: Path) -> None:
    speech = _make_vctk_speech_dir(tmp_path / "vctk")
    pool = tmp_path / "pool"
    build_pool(speech, pool, mic="mic1")
    verify_pool(pool, min_speakers=3)  # must not raise


def test_verify_pool_raises_on_empty(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    with pytest.raises(RuntimeError, match="no WAV files"):
        verify_pool(pool)


def test_verify_pool_raises_too_few_speakers(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "p225-001.wav").write_bytes(b"RIFF")
    with pytest.raises(RuntimeError, match="need >="):
        verify_pool(pool, min_speakers=2)


# ── download_vctk ─────────────────────────────────────────────────────────────


def test_download_skips_when_speech_dir_present(tmp_path: Path) -> None:
    out = tmp_path / "datasets"
    _make_vctk_speech_dir(out)  # already extracted

    with patch("coralsep.data.prepare_vctk.urlretrieve") as mock_dl:
        result = download_vctk(out)
        mock_dl.assert_not_called()
    assert result.name == "wav48_silence_trimmed"


def test_download_calls_urlretrieve_and_extracts(tmp_path: Path) -> None:
    out = tmp_path / "datasets"

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(b"")

    def fake_zipfile(path):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.extractall = lambda dest: _make_vctk_speech_dir(Path(dest))
        return cm

    with (
        patch("coralsep.data.prepare_vctk.urlretrieve", side_effect=fake_urlretrieve) as mock_dl,
        patch("coralsep.data.prepare_vctk.zipfile.ZipFile", side_effect=fake_zipfile),
    ):
        result = download_vctk(out)

    assert mock_dl.call_args.args[0] == VCTK_URL
    assert result.name == "wav48_silence_trimmed"


def test_download_raises_if_speech_dir_absent_after_extract(tmp_path: Path) -> None:
    out = tmp_path / "datasets"

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(b"")

    def fake_zipfile(path):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.extractall = lambda dest: None  # nothing produced
        return cm

    with (
        patch("coralsep.data.prepare_vctk.urlretrieve", side_effect=fake_urlretrieve),
        patch("coralsep.data.prepare_vctk.zipfile.ZipFile", side_effect=fake_zipfile),
    ):
        with pytest.raises(RuntimeError, match="Could not find"):
            download_vctk(out)
