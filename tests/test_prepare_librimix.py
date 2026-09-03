"""
Tests for data/prepare_librimix.py.

All network, subprocess, and tarfile calls are mocked so the suite runs
without internet access, git, or a real LibriSpeech download.  Filesystem
operations use pytest's tmp_path fixture with real directories.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coralsep.data.prepare_librimix import (
    REQUIRED_SUBSETS,
    STREAM_DIRS,
    _ensure_train_alias,
    _find_generation_script,
    _make_filtered_metadata,
    clone_librimix,
    download_librispeech,
    generate_librimix,
    verify_layout,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_librimix_repo(root: Path) -> Path:
    """Create a minimal LibriMix repo skeleton with metadata CSVs."""
    repo = root / "LibriMix"
    (repo / "metadata" / "Libri3Mix").mkdir(parents=True)
    (repo / "README.md").write_text("LibriMix")
    for name in [
        "mixture_dev_mix_both.csv",
        "mixture_test_mix_both.csv",
        "mixture_train-100_mix_both.csv",
        "mixture_train-360_mix_both.csv",
    ]:
        (repo / "metadata" / "Libri3Mix" / name).write_text("dummy")
    return repo


def _make_wav_layout(data_root: Path, subsets: list[str] | None = None) -> None:
    """Create the wav16k/max/{subset}/{stream_dir}/ skeleton under data_root."""
    if subsets is None:
        subsets = REQUIRED_SUBSETS
    wav_root = data_root / "wav16k" / "max"
    for subset in subsets:
        for stream_dir in STREAM_DIRS:
            (wav_root / subset / stream_dir).mkdir(parents=True, exist_ok=True)


# ── _find_generation_script ───────────────────────────────────────────────────


def test_find_script_at_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "LibriMix"
    script = repo / "create_librimix_from_metadata.py"
    script.parent.mkdir(parents=True)
    script.write_text("")
    assert _find_generation_script(repo) == script


def test_find_script_in_scripts_subdir(tmp_path: Path) -> None:
    repo = tmp_path / "LibriMix"
    script = repo / "scripts" / "create_librimix_from_metadata.py"
    script.parent.mkdir(parents=True)
    script.write_text("")
    assert _find_generation_script(repo) == script


def test_find_script_missing_raises(tmp_path: Path) -> None:
    repo = tmp_path / "LibriMix"
    repo.mkdir()
    with pytest.raises(FileNotFoundError, match="create_librimix_from_metadata.py"):
        _find_generation_script(repo)


# ── _make_filtered_metadata ───────────────────────────────────────────────────


def test_filtered_metadata_dev_test_only(tmp_path: Path) -> None:
    repo = _make_librimix_repo(tmp_path)
    work_dir = tmp_path / "work"

    meta_root = _make_filtered_metadata(repo, work_dir, include_train=False)

    libri3_dir = meta_root / "Libri3Mix"
    assert (libri3_dir / "mixture_dev_mix_both.csv").exists()
    assert (libri3_dir / "mixture_test_mix_both.csv").exists()
    assert not (libri3_dir / "mixture_train-360_mix_both.csv").exists()


def test_filtered_metadata_includes_train_when_requested(tmp_path: Path) -> None:
    """Default train split is train-100: the one LibriSpeech split we download.

    Regression guard. This previously defaulted to train-360, whose sources live
    in LibriSpeech train-clean-360, which download_librispeech never fetches. Any
    --include-train run therefore died inside the generator on missing audio.
    """
    repo = _make_librimix_repo(tmp_path)
    work_dir = tmp_path / "work"

    meta_root = _make_filtered_metadata(repo, work_dir, include_train=True)

    libri3_dir = meta_root / "Libri3Mix"
    assert (libri3_dir / "mixture_train-100_mix_both.csv").exists()
    assert not (libri3_dir / "mixture_train-360_mix_both.csv").exists()


def test_filtered_metadata_can_opt_into_train_360(tmp_path: Path) -> None:
    """train-360 stays available for anyone willing to pull the extra 23 GB."""
    repo = _make_librimix_repo(tmp_path)
    work_dir = tmp_path / "work"

    meta_root = _make_filtered_metadata(repo, work_dir, include_train=True, train_split="train-360")

    libri3_dir = meta_root / "Libri3Mix"
    assert (libri3_dir / "mixture_train-360_mix_both.csv").exists()
    assert not (libri3_dir / "mixture_train-100_mix_both.csv").exists()


def test_filtered_metadata_idempotent(tmp_path: Path) -> None:
    """Running twice must not raise."""
    repo = _make_librimix_repo(tmp_path)
    work_dir = tmp_path / "work"
    _make_filtered_metadata(repo, work_dir, include_train=False)
    _make_filtered_metadata(repo, work_dir, include_train=False)


def test_filtered_metadata_missing_csv_raises(tmp_path: Path) -> None:
    repo = tmp_path / "LibriMix"
    (repo / "metadata" / "Libri3Mix").mkdir(parents=True)
    (repo / "README.md").write_text("")
    # dev CSV present, test CSV absent
    (repo / "metadata" / "Libri3Mix" / "mixture_dev_mix_both.csv").write_text("")

    with pytest.raises(FileNotFoundError, match="mixture_test_mix_both.csv"):
        _make_filtered_metadata(repo, tmp_path / "work", include_train=False)


# ── download_librispeech ──────────────────────────────────────────────────────


def test_download_skips_existing_split(tmp_path: Path) -> None:
    """No HTTP call if the extracted directory already exists and is non-empty."""
    ls_dir = tmp_path / "librispeech"
    for split in ["train-clean-100", "dev-clean", "test-clean"]:
        extracted = ls_dir / "LibriSpeech" / split
        extracted.mkdir(parents=True)
        (extracted / "dummy.flac").write_text("")  # non-empty

    with patch("coralsep.data.prepare_librimix.urllib.request.urlretrieve") as mock_dl:
        download_librispeech(ls_dir)
        mock_dl.assert_not_called()


def test_download_calls_urlretrieve_for_missing_split(tmp_path: Path) -> None:
    """urlretrieve is called once per split that is absent."""
    ls_dir = tmp_path / "librispeech"

    # Fake tarball that extracts to the right location
    def fake_urlretrieve(url: str, filename: str, reporthook=None) -> None:
        Path(filename).write_bytes(b"")

    def fake_tarfile_open(path: str, mode: str):
        cm = MagicMock()

        # Simulate extractall creating the expected directory
        def fake_extractall(dest, filter=None):
            for split in ["train-clean-100", "dev-clean", "test-clean"]:
                (Path(dest) / "LibriSpeech" / split).mkdir(parents=True, exist_ok=True)

        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.extractall = fake_extractall
        return cm

    with (
        patch(
            "coralsep.data.prepare_librimix.urllib.request.urlretrieve",
            side_effect=fake_urlretrieve,
        ),
        patch("coralsep.data.prepare_librimix.tarfile.open", side_effect=fake_tarfile_open),
    ):
        download_librispeech(ls_dir)

    # All three splits should now have been downloaded
    for split in ["train-clean-100", "dev-clean", "test-clean"]:
        assert (ls_dir / "LibriSpeech" / split).exists()


def test_download_skips_tarball_download_if_tarball_exists(tmp_path: Path) -> None:
    """If the .tar.gz already exists, urlretrieve is not called for that split."""
    ls_dir = tmp_path / "librispeech"
    ls_dir.mkdir()

    # Pre-place one tarball
    (ls_dir / "test-clean.tar.gz").write_bytes(b"")

    def fake_urlretrieve(url: str, filename: str, reporthook=None) -> None:
        Path(filename).write_bytes(b"")

    def fake_tarfile_open(path: str, mode: str):
        cm = MagicMock()

        def fake_extractall(dest, filter=None):
            for split in ["train-clean-100", "dev-clean", "test-clean"]:
                (Path(dest) / "LibriSpeech" / split).mkdir(parents=True, exist_ok=True)

        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.extractall = fake_extractall
        return cm

    with (
        patch(
            "coralsep.data.prepare_librimix.urllib.request.urlretrieve",
            side_effect=fake_urlretrieve,
        ) as mock_dl,
        patch("coralsep.data.prepare_librimix.tarfile.open", side_effect=fake_tarfile_open),
    ):
        download_librispeech(ls_dir)

    # urlretrieve should NOT have been called for test-clean (tarball already present)
    downloaded_urls = [c.args[0] for c in mock_dl.call_args_list]
    assert all("test-clean" not in url for url in downloaded_urls)


# ── clone_librimix ────────────────────────────────────────────────────────────


def test_clone_skips_existing_repo(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    repo = tools_dir / "LibriMix"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("already cloned")

    with patch("coralsep.data.prepare_librimix.subprocess.run") as mock_run:
        result = clone_librimix(tools_dir)
        mock_run.assert_not_called()

    assert result == repo


def test_clone_calls_git_clone(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"

    def fake_run(cmd, check):
        # Simulate the clone creating the directory
        repo = tools_dir / "LibriMix"
        repo.mkdir(parents=True, exist_ok=True)
        (repo / "README.md").write_text("cloned")

    with patch("coralsep.data.prepare_librimix.subprocess.run", side_effect=fake_run) as mock_run:
        result = clone_librimix(tools_dir)

    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[0] == "git"
    assert args[1] == "clone"
    assert "--depth" in args
    assert "https://github.com/JorisCos/LibriMix" in args
    assert result == tools_dir / "LibriMix"


# ── generate_librimix ─────────────────────────────────────────────────────────


def test_generate_skips_if_output_exists(tmp_path: Path) -> None:
    """No subprocess call when test/mix_both already contains a WAV file."""
    output_dir = tmp_path / "output"
    test_mix = output_dir / "Libri3Mix" / "wav16k" / "max" / "test" / "mix_both"
    test_mix.mkdir(parents=True)
    (test_mix / "sample.wav").write_bytes(b"RIFF")

    repo = _make_librimix_repo(tmp_path / "repo")
    (repo / "create_librimix_from_metadata.py").write_text("")

    with patch("coralsep.data.prepare_librimix.subprocess.run") as mock_run:
        generate_librimix(repo, tmp_path / "ls", output_dir)
        mock_run.assert_not_called()


def test_generate_calls_script_with_correct_args(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    ls_dir = tmp_path / "librispeech"
    (ls_dir / "LibriSpeech").mkdir(parents=True)

    repo = _make_librimix_repo(tmp_path / "repo")
    script = repo / "create_librimix_from_metadata.py"
    script.write_text("")

    with patch("coralsep.data.prepare_librimix.subprocess.run") as mock_run:
        generate_librimix(repo, ls_dir, output_dir, include_train=False)

    mock_run.assert_called_once()
    cmd: list[str] = mock_run.call_args.args[0]

    assert cmd[0] == sys.executable
    assert str(script) == cmd[1]

    # Verify key arguments are present with correct values
    arg_map: dict[str, str] = {}
    for i in range(2, len(cmd) - 1, 2):
        arg_map[cmd[i]] = cmd[i + 1]

    assert arg_map["--n_src"] == "3"
    assert arg_map["--freqs"] == "16000"
    assert arg_map["--modes"] == "max"
    assert arg_map["--types"] == "mix_both"
    assert "--librispeech_path" in arg_map
    assert "--metadata_path" in arg_map
    assert "--librimix_path" in arg_map


def test_generate_does_not_include_train_csv_by_default(tmp_path: Path) -> None:
    """The filtered metadata passed to the script must not contain the train-360 CSV."""
    output_dir = tmp_path / "output"
    ls_dir = tmp_path / "librispeech"
    (ls_dir / "LibriSpeech").mkdir(parents=True)

    repo = _make_librimix_repo(tmp_path / "repo")
    (repo / "create_librimix_from_metadata.py").write_text("")

    captured_meta_path: list[Path] = []

    def capture_run(cmd, check):
        arg_map: dict[str, str] = {}
        for i in range(2, len(cmd) - 1, 2):
            arg_map[cmd[i]] = cmd[i + 1]
        captured_meta_path.append(Path(arg_map["--metadata_path"]))

    with patch("coralsep.data.prepare_librimix.subprocess.run", side_effect=capture_run):
        generate_librimix(repo, ls_dir, output_dir, include_train=False)

    meta_dir = captured_meta_path[0] / "Libri3Mix"
    assert (meta_dir / "mixture_dev_mix_both.csv").exists()
    assert (meta_dir / "mixture_test_mix_both.csv").exists()
    assert not (meta_dir / "mixture_train-360_mix_both.csv").exists()


# ── _ensure_train_alias / _create_directory_alias ────────────────────────────


def test_ensure_train_alias_creates_alias_for_train360(tmp_path: Path) -> None:
    wav_root = tmp_path / "wav16k" / "max"
    train360 = wav_root / "train-360"
    train360.mkdir(parents=True)

    with patch("coralsep.data.prepare_librimix._create_directory_alias") as mock_alias:
        _ensure_train_alias(wav_root)
        mock_alias.assert_called_once_with(train360, wav_root / "train")


def test_ensure_train_alias_no_op_when_train_exists(tmp_path: Path) -> None:
    wav_root = tmp_path / "wav16k" / "max"
    (wav_root / "train").mkdir(parents=True)

    with patch("coralsep.data.prepare_librimix._create_directory_alias") as mock_alias:
        _ensure_train_alias(wav_root)
        mock_alias.assert_not_called()


def test_ensure_train_alias_no_op_when_no_train_data(tmp_path: Path) -> None:
    """No error when no train split exists at all (M0 baseline only needs test)."""
    wav_root = tmp_path / "wav16k" / "max"
    wav_root.mkdir(parents=True)

    with patch("coralsep.data.prepare_librimix._create_directory_alias") as mock_alias:
        _ensure_train_alias(wav_root)
        mock_alias.assert_not_called()


# ── verify_layout ─────────────────────────────────────────────────────────────


def test_verify_layout_passes_with_required_dirs(tmp_path: Path) -> None:
    data_root = tmp_path / "Libri3Mix"
    _make_wav_layout(data_root, subsets=["dev", "test"])
    verify_layout(data_root)  # must not raise


def test_verify_layout_also_checks_train_when_present(tmp_path: Path) -> None:
    data_root = tmp_path / "Libri3Mix"
    _make_wav_layout(data_root, subsets=["train", "dev", "test"])
    verify_layout(data_root)  # must not raise


def test_verify_layout_raises_when_wav_root_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "Libri3Mix"
    data_root.mkdir()  # wav16k/max/ does NOT exist

    with pytest.raises(RuntimeError, match="wav16k/max/"):
        verify_layout(data_root)


def test_verify_layout_raises_listing_missing_dirs(tmp_path: Path) -> None:
    data_root = tmp_path / "Libri3Mix"
    # Create dev layout but omit test entirely
    _make_wav_layout(data_root, subsets=["dev"])

    with pytest.raises(RuntimeError) as exc_info:
        verify_layout(data_root)

    msg = str(exc_info.value)
    assert "test" in msg
    # dev dirs present so they should not appear in the error
    assert "dev" not in msg


def test_verify_layout_raises_listing_missing_stream_dirs(tmp_path: Path) -> None:
    data_root = tmp_path / "Libri3Mix"
    wav_root = data_root / "wav16k" / "max"

    # Create test subset but omit s3/
    for stream_dir in ["mix_both", "s1", "s2"]:
        (wav_root / "test" / stream_dir).mkdir(parents=True)
    (wav_root / "dev" / "mix_both").mkdir(parents=True)
    for stream_dir in STREAM_DIRS:
        (wav_root / "dev" / stream_dir).mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError) as exc_info:
        verify_layout(data_root)

    assert "s3" in str(exc_info.value)


def test_verify_layout_error_message_includes_rerun_hint(tmp_path: Path) -> None:
    data_root = tmp_path / "Libri3Mix"
    (data_root / "wav16k" / "max").mkdir(parents=True)
    # No subsets at all

    with pytest.raises(RuntimeError, match="prepare_librimix"):
        verify_layout(data_root)
