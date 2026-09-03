"""
Tests for data/prepare_librimix_highn.py.

Network and subprocess calls are mocked so the suite runs offline.  Filesystem
operations use pytest's tmp_path fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from coralsep.data.prepare.librimix_highn import (
    LIBRIMIX_HIGHN_REPO_URL,
    _find_generation_script,
    _make_filtered_metadata,
    _parse_n_src,
    clone_librimix_highn,
    generate_librimix_highn,
    verify_layout,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_repo(root: Path, n_src_values=(4, 5)) -> Path:
    """Create a minimal fork skeleton: generation script + LibriNMix metadata CSVs."""
    repo = root / "librimix_highn"
    repo.mkdir(parents=True)
    (repo / "create_librimix_from_metadata.py").write_text("")
    for n in n_src_values:
        meta = repo / "metadata" / f"Libri{n}Mix"
        meta.mkdir(parents=True)
        for csv in (
            "mixture_dev_mix_both.csv",
            "mixture_test_mix_both.csv",
            "mixture_train-360_mix_both.csv",
        ):
            (meta / csv).write_text("")
    return repo


def _make_layout(data_root: Path, n_src: int, subsets=("dev", "test")) -> None:
    for subset in subsets:
        base = data_root / "wav16k" / "max" / subset
        for stream in ["mix_both"] + [f"s{i}" for i in range(1, n_src + 1)]:
            (base / stream).mkdir(parents=True, exist_ok=True)


# ── clone_librimix_highn ──────────────────────────────────────────────────────


def test_clone_skips_existing(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    repo = tools / "librimix_highn"
    (repo / "metadata").mkdir(parents=True)

    with patch("coralsep.data.prepare.librimix_highn.subprocess.run") as mock_run:
        result = clone_librimix_highn(tools)
        mock_run.assert_not_called()
    assert result == repo


def test_clone_calls_git_clone(tmp_path: Path) -> None:
    tools = tmp_path / "tools"

    def fake_run(cmd, check):
        repo = tools / "librimix_highn"
        (repo / "metadata").mkdir(parents=True, exist_ok=True)

    with patch(
        "coralsep.data.prepare.librimix_highn.subprocess.run", side_effect=fake_run
    ) as mock_run:
        result = clone_librimix_highn(tools)

    args = mock_run.call_args.args[0]
    assert args[0] == "git"
    assert args[1] == "clone"
    assert "--depth" in args
    assert LIBRIMIX_HIGHN_REPO_URL in args
    assert result == tools / "librimix_highn"


# ── _find_generation_script ───────────────────────────────────────────────────


def test_find_script_at_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "librimix_highn"
    repo.mkdir()
    script = repo / "create_librimix_from_metadata.py"
    script.write_text("")
    assert _find_generation_script(repo) == script


def test_find_script_missing_raises(tmp_path: Path) -> None:
    repo = tmp_path / "librimix_highn"
    repo.mkdir()
    with pytest.raises(FileNotFoundError, match="create_librimix_from_metadata.py"):
        _find_generation_script(repo)


# ── _make_filtered_metadata ───────────────────────────────────────────────────


def test_filtered_metadata_excludes_train_by_default(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo", n_src_values=(4,))
    meta_root = _make_filtered_metadata(repo, tmp_path / "work", 4, include_train=False)

    dst = meta_root / "Libri4Mix"
    assert (dst / "mixture_dev_mix_both.csv").exists()
    assert (dst / "mixture_test_mix_both.csv").exists()
    assert not (dst / "mixture_train-360_mix_both.csv").exists()


def test_filtered_metadata_includes_train_when_requested(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo", n_src_values=(5,))
    meta_root = _make_filtered_metadata(repo, tmp_path / "work", 5, include_train=True)

    dst = meta_root / "Libri5Mix"
    assert (dst / "mixture_train-360_mix_both.csv").exists()


def test_filtered_metadata_missing_csv_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo" / "librimix_highn"
    (repo / "metadata" / "Libri4Mix").mkdir(parents=True)  # empty, no CSVs

    with pytest.raises(FileNotFoundError, match="Expected metadata CSV not found"):
        _make_filtered_metadata(repo, tmp_path / "work", 4, include_train=False)


# ── generate_librimix_highn ───────────────────────────────────────────────────


def test_generate_rejects_unsupported_n_src(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo", n_src_values=(4,))
    with pytest.raises(ValueError, match="n_src must be one of"):
        generate_librimix_highn(repo, tmp_path / "ls", tmp_path / "out", 3)


def test_generate_skips_if_output_exists(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo", n_src_values=(4,))
    output_dir = tmp_path / "out"
    test_mix = output_dir / "Libri4Mix" / "wav16k" / "max" / "test" / "mix_both"
    test_mix.mkdir(parents=True)
    (test_mix / "x.wav").write_bytes(b"RIFF")

    with patch("coralsep.data.prepare.librimix_highn.subprocess.run") as mock_run:
        generate_librimix_highn(repo, tmp_path / "ls", output_dir, 4)
        mock_run.assert_not_called()


def test_generate_calls_script_with_expected_args(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo", n_src_values=(4,))
    output_dir = tmp_path / "out"
    ls_dir = tmp_path / "ls"
    (ls_dir / "LibriSpeech").mkdir(parents=True)

    with patch("coralsep.data.prepare.librimix_highn.subprocess.run") as mock_run:
        generate_librimix_highn(repo, ls_dir, output_dir, 4)

    cmd = mock_run.call_args.args[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == str(repo / "create_librimix_from_metadata.py")
    assert "--n_src" in cmd and cmd[cmd.index("--n_src") + 1] == "4"
    assert "--freqs" in cmd and cmd[cmd.index("--freqs") + 1] == "16000"
    assert "--modes" in cmd and cmd[cmd.index("--modes") + 1] == "max"
    assert "--types" in cmd and cmd[cmd.index("--types") + 1] == "mix_both"
    # librispeech path resolves to the LibriSpeech subdir when present
    assert cmd[cmd.index("--librispeech_path") + 1] == str(ls_dir / "LibriSpeech")


# ── verify_layout ─────────────────────────────────────────────────────────────


def test_verify_layout_passes(tmp_path: Path) -> None:
    data_root = tmp_path / "Libri5Mix"
    _make_layout(data_root, 5)
    verify_layout(data_root, 5)  # must not raise


def test_verify_layout_raises_when_stream_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "Libri4Mix"
    _make_layout(data_root, 3)  # only s1..s3 created, s4 missing

    with pytest.raises(RuntimeError) as exc:
        verify_layout(data_root, 4)
    assert "s4" in str(exc.value)


def test_verify_layout_raises_when_wav_root_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "Libri4Mix"
    data_root.mkdir()
    with pytest.raises(RuntimeError, match="wav16k/max/ not found"):
        verify_layout(data_root, 4)


# ── _parse_n_src ──────────────────────────────────────────────────────────────


def test_parse_n_src() -> None:
    assert _parse_n_src("4,5") == [4, 5]
    assert _parse_n_src("4") == [4]
    assert _parse_n_src(" 4 , 5 ") == [4, 5]
