"""
Tests for data/prepare_wham.py.

Network and zipfile calls are mocked so the suite runs offline.  Filesystem
operations use pytest's tmp_path fixture.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coralsep.data.prepare.wham import (
    WHAM_NOISE_URL,
    download_wham_noise,
    verify_layout,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_noise_layout(wham_noise_dir: Path, splits=("tr", "cv", "tt")) -> None:
    """Create a minimal extracted wham_noise/{tr,cv,tt}/ layout with one wav each."""
    for split in splits:
        split_dir = wham_noise_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "000.wav").write_bytes(b"RIFF")


# ── download_wham_noise ───────────────────────────────────────────────────────


def test_download_skips_existing(tmp_path: Path) -> None:
    out = tmp_path / "datasets"
    extracted = out / "wham_noise"
    _make_noise_layout(extracted)

    with patch("coralsep.data.prepare.wham.urlretrieve") as mock_dl:
        result = download_wham_noise(out)
        mock_dl.assert_not_called()
    assert result == extracted


def test_download_calls_urlretrieve_when_missing(tmp_path: Path) -> None:
    out = tmp_path / "datasets"

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(b"")

    def fake_zipfile(path):
        cm = MagicMock()

        def fake_extractall(dest):
            _make_noise_layout(Path(dest) / "wham_noise")

        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.extractall = fake_extractall
        return cm

    with (
        patch("coralsep.data.prepare.wham.urlretrieve", side_effect=fake_urlretrieve) as mock_dl,
        patch("coralsep.data.prepare.wham.zipfile.ZipFile", side_effect=fake_zipfile),
    ):
        result = download_wham_noise(out)

    mock_dl.assert_called_once()
    # urlretrieve is called with the canonical URL by default
    assert mock_dl.call_args.args[0] == WHAM_NOISE_URL
    assert result == out / "wham_noise"


def test_download_uses_custom_url(tmp_path: Path) -> None:
    out = tmp_path / "datasets"
    custom = "https://mirror.example/wham_noise.zip"

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(b"")

    def fake_zipfile(path):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.extractall = lambda dest: _make_noise_layout(Path(dest) / "wham_noise")
        return cm

    with (
        patch("coralsep.data.prepare.wham.urlretrieve", side_effect=fake_urlretrieve) as mock_dl,
        patch("coralsep.data.prepare.wham.zipfile.ZipFile", side_effect=fake_zipfile),
    ):
        download_wham_noise(out, url=custom)

    assert mock_dl.call_args.args[0] == custom


def test_download_skips_download_when_archive_present(tmp_path: Path) -> None:
    out = tmp_path / "datasets"
    out.mkdir()
    (out / "wham_noise.zip").write_bytes(b"PK")  # archive already downloaded

    def fake_zipfile(path):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.extractall = lambda dest: _make_noise_layout(Path(dest) / "wham_noise")
        return cm

    with (
        patch("coralsep.data.prepare.wham.urlretrieve") as mock_dl,
        patch("coralsep.data.prepare.wham.zipfile.ZipFile", side_effect=fake_zipfile),
    ):
        result = download_wham_noise(out)
        mock_dl.assert_not_called()  # archive present → no re-download
    assert result == out / "wham_noise"


def test_download_raises_if_extraction_wrong_layout(tmp_path: Path) -> None:
    out = tmp_path / "datasets"

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(b"")

    def fake_zipfile(path):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.extractall = lambda dest: None  # produces nothing
        return cm

    with (
        patch("coralsep.data.prepare.wham.urlretrieve", side_effect=fake_urlretrieve),
        patch("coralsep.data.prepare.wham.zipfile.ZipFile", side_effect=fake_zipfile),
    ):
        with pytest.raises(RuntimeError, match="did not produce expected directory"):
            download_wham_noise(out)


# ── verify_layout ─────────────────────────────────────────────────────────────


def test_verify_layout_passes(tmp_path: Path) -> None:
    wham = tmp_path / "wham_noise"
    _make_noise_layout(wham)
    verify_layout(wham)  # must not raise


def test_verify_layout_raises_when_split_missing(tmp_path: Path) -> None:
    wham = tmp_path / "wham_noise"
    _make_noise_layout(wham, splits=("tr", "cv"))  # tt missing

    with pytest.raises(RuntimeError) as exc:
        verify_layout(wham)
    assert "tt" in str(exc.value)


def test_verify_layout_raises_when_split_empty(tmp_path: Path) -> None:
    wham = tmp_path / "wham_noise"
    _make_noise_layout(wham, splits=("tr", "cv"))
    (wham / "tt").mkdir()  # exists but no wav files

    with pytest.raises(RuntimeError, match="no .wav files"):
        verify_layout(wham)


def test_verify_layout_custom_splits(tmp_path: Path) -> None:
    wham = tmp_path / "wham_noise"
    _make_noise_layout(wham, splits=("tt",))
    verify_layout(wham, splits=["tt"])  # only check the test split
