"""
Tests for data/prepare_sparselibrimix.py.

Network, subprocess, and tarfile calls are mocked so the suite runs offline.
Filesystem operations use pytest's tmp_path fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coralsep.data.prepare.sparselibrimix import (
    _expected_stream_dirs,
    _find_make_mixtures_script,
    _metadata_json,
    _parse_n_src,
    clone_sparselibrimix,
    download_librispeech_test,
    generate_sparselibrimix,
    verify_layout,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_repo(
    root: Path,
    n_src_values=(2, 3),
    ratios=("0", "0.2", "0.4", "0.6", "0.8", "1"),
) -> Path:
    """Create a minimal SparseLibriMix repo skeleton with metadata + script."""
    repo = root / "SparseLibriMix"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "make_mixtures.py").write_text("")
    (repo / "requirements.txt").write_text("")
    for n in n_src_values:
        for ratio in ratios:
            meta_dir = repo / "metadata" / f"sparse_{n}_{ratio}"
            meta_dir.mkdir(parents=True)
            (meta_dir / "metadata.json").write_text("{}")
    return repo


def _make_layout(
    data_root: Path, n_src: int, ratio: str, freq: int, with_noise: bool = False
) -> None:
    base = data_root / f"sparse_{n_src}_{ratio}" / f"wav{freq}"
    for stream in _expected_stream_dirs(n_src, with_noise):
        (base / stream).mkdir(parents=True, exist_ok=True)


# ── _find_make_mixtures_script ────────────────────────────────────────────────


def test_find_script_in_scripts_subdir(tmp_path: Path) -> None:
    repo = tmp_path / "SparseLibriMix"
    script = repo / "scripts" / "make_mixtures.py"
    script.parent.mkdir(parents=True)
    script.write_text("")
    assert _find_make_mixtures_script(repo) == script


def test_find_script_at_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "SparseLibriMix"
    repo.mkdir()
    script = repo / "make_mixtures.py"
    script.write_text("")
    assert _find_make_mixtures_script(repo) == script


def test_find_script_missing_raises(tmp_path: Path) -> None:
    repo = tmp_path / "SparseLibriMix"
    repo.mkdir()
    with pytest.raises(FileNotFoundError, match="make_mixtures.py"):
        _find_make_mixtures_script(repo)


# ── _metadata_json ────────────────────────────────────────────────────────────


def test_metadata_json_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    assert _metadata_json(repo, 3, "0.2") == repo / "metadata" / "sparse_3_0.2" / "metadata.json"


# ── download_librispeech_test ─────────────────────────────────────────────────


def test_download_skips_existing(tmp_path: Path) -> None:
    ls_dir = tmp_path / "ls"
    extracted = ls_dir / "LibriSpeech" / "test-clean"
    extracted.mkdir(parents=True)
    (extracted / "dummy.flac").write_text("")

    with patch("coralsep.data.prepare.sparselibrimix.urllib.request.urlretrieve") as mock_dl:
        result = download_librispeech_test(ls_dir)
        mock_dl.assert_not_called()
    assert result == extracted


def test_download_calls_urlretrieve_when_missing(tmp_path: Path) -> None:
    ls_dir = tmp_path / "ls"

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(b"")

    def fake_tarfile_open(path, mode):
        cm = MagicMock()

        def fake_extractall(dest, filter=None):
            (Path(dest) / "LibriSpeech" / "test-clean").mkdir(parents=True, exist_ok=True)

        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.extractall = fake_extractall
        return cm

    with (
        patch(
            "coralsep.data.prepare.sparselibrimix.urllib.request.urlretrieve",
            side_effect=fake_urlretrieve,
        ) as mock_dl,
        patch("coralsep.data.prepare.sparselibrimix.tarfile.open", side_effect=fake_tarfile_open),
    ):
        result = download_librispeech_test(ls_dir)

    mock_dl.assert_called_once()
    assert result == ls_dir / "LibriSpeech" / "test-clean"


# ── clone_sparselibrimix ──────────────────────────────────────────────────────


def test_clone_skips_existing(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    repo = tools / "SparseLibriMix"
    (repo / "metadata").mkdir(parents=True)

    with patch("coralsep.data.prepare.sparselibrimix.subprocess.run") as mock_run:
        result = clone_sparselibrimix(tools)
        mock_run.assert_not_called()
    assert result == repo


def test_clone_calls_git_clone(tmp_path: Path) -> None:
    tools = tmp_path / "tools"

    def fake_run(cmd, check):
        repo = tools / "SparseLibriMix"
        (repo / "metadata").mkdir(parents=True, exist_ok=True)

    with patch(
        "coralsep.data.prepare.sparselibrimix.subprocess.run", side_effect=fake_run
    ) as mock_run:
        result = clone_sparselibrimix(tools)

    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[0] == "git"
    assert args[1] == "clone"
    assert "--depth" in args
    assert "https://github.com/popcornell/SparseLibriMix" in args
    assert result == tools / "SparseLibriMix"


# ── generate_sparselibrimix ───────────────────────────────────────────────────


def test_generate_skips_if_output_exists(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo", n_src_values=(2,), ratios=("0.2",))
    data_root = tmp_path / "SparseLibriMix"
    mix_clean = data_root / "sparse_2_0.2" / "wav16000" / "mix_clean"
    mix_clean.mkdir(parents=True)
    (mix_clean / "x.wav").write_bytes(b"RIFF")

    with patch("coralsep.data.prepare.sparselibrimix.subprocess.run") as mock_run:
        generate_sparselibrimix(repo, tmp_path / "ls", data_root, n_src_values=[2], ratios=["0.2"])
        mock_run.assert_not_called()


def test_generate_calls_script_with_positional_args(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo", n_src_values=(2,), ratios=("0.2",))
    data_root = tmp_path / "SparseLibriMix"
    ls = tmp_path / "ls" / "LibriSpeech" / "test-clean"
    ls.mkdir(parents=True)

    with patch("coralsep.data.prepare.sparselibrimix.subprocess.run") as mock_run:
        generate_sparselibrimix(repo, ls, data_root, n_src_values=[2], ratios=["0.2"], freq=16000)

    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == str(repo / "scripts" / "make_mixtures.py")
    assert cmd[2] == str(repo / "metadata" / "sparse_2_0.2" / "metadata.json")
    assert cmd[3] == str(ls)
    assert cmd[4] == str(data_root / "sparse_2_0.2" / "wav16000")
    assert "--rate" in cmd
    assert cmd[cmd.index("--rate") + 1] == "16000"
    assert "--noise_dir" not in cmd  # no wham dir passed


def test_generate_adds_noise_dir_when_wham_given(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo", n_src_values=(3,), ratios=("0.4",))
    data_root = tmp_path / "SparseLibriMix"
    ls = tmp_path / "ls"
    wham = tmp_path / "wham" / "tt"
    wham.mkdir(parents=True)

    with patch("coralsep.data.prepare.sparselibrimix.subprocess.run") as mock_run:
        generate_sparselibrimix(
            repo, ls, data_root, n_src_values=[3], ratios=["0.4"], wham_noise_dir=wham
        )

    cmd = mock_run.call_args.args[0]
    assert "--noise_dir" in cmd
    assert cmd[cmd.index("--noise_dir") + 1] == str(wham)


def test_generate_missing_metadata_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo" / "SparseLibriMix"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "make_mixtures.py").write_text("")
    # no metadata folders created
    data_root = tmp_path / "SparseLibriMix"

    with patch("coralsep.data.prepare.sparselibrimix.subprocess.run"):
        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            generate_sparselibrimix(
                repo, tmp_path / "ls", data_root, n_src_values=[2], ratios=["0"]
            )


def test_generate_loops_all_configs(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo", n_src_values=(2, 3), ratios=("0", "1"))
    data_root = tmp_path / "SparseLibriMix"
    ls = tmp_path / "ls"

    with patch("coralsep.data.prepare.sparselibrimix.subprocess.run") as mock_run:
        generate_sparselibrimix(repo, ls, data_root, n_src_values=[2, 3], ratios=["0", "1"])

    # 2 speaker counts x 2 ratios = 4 generation calls
    assert mock_run.call_count == 4


# ── verify_layout ─────────────────────────────────────────────────────────────


def test_verify_layout_passes(tmp_path: Path) -> None:
    data_root = tmp_path / "SparseLibriMix"
    _make_layout(data_root, 2, "0.2", 16000)
    verify_layout(data_root, n_src_values=[2], ratios=["0.2"])  # must not raise


def test_verify_layout_raises_when_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "SparseLibriMix"
    # create s1 but not s2/mix_clean
    (data_root / "sparse_2_0.2" / "wav16000" / "s1").mkdir(parents=True)

    with pytest.raises(RuntimeError) as exc:
        verify_layout(data_root, n_src_values=[2], ratios=["0.2"])
    assert "mix_clean" in str(exc.value)
    assert "s2" in str(exc.value)


def test_verify_layout_expect_noise_requires_mix_noisy(tmp_path: Path) -> None:
    data_root = tmp_path / "SparseLibriMix"
    _make_layout(data_root, 2, "0.2", 16000, with_noise=False)  # clean only

    with pytest.raises(RuntimeError, match="mix_noisy"):
        verify_layout(data_root, n_src_values=[2], ratios=["0.2"], expect_noise=True)


def test_verify_layout_expect_noise_passes_when_present(tmp_path: Path) -> None:
    data_root = tmp_path / "SparseLibriMix"
    _make_layout(data_root, 3, "0.6", 16000, with_noise=True)
    verify_layout(data_root, n_src_values=[3], ratios=["0.6"], expect_noise=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def test_expected_stream_dirs_clean() -> None:
    assert _expected_stream_dirs(2, False) == ["mix_clean", "s1", "s2"]
    assert _expected_stream_dirs(3, False) == ["mix_clean", "s1", "s2", "s3"]


def test_expected_stream_dirs_with_noise() -> None:
    dirs = _expected_stream_dirs(2, True)
    assert "mix_noisy" in dirs and "noise" in dirs


def test_parse_n_src() -> None:
    assert _parse_n_src("2,3") == [2, 3]
    assert _parse_n_src("3") == [3]
    assert _parse_n_src(" 2 , 3 ") == [2, 3]
