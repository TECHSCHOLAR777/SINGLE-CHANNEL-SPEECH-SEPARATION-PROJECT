"""
Tests for data/prepare_whamr.py.

Subprocess calls are mocked so the suite runs offline.  Filesystem operations
use pytest's tmp_path fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from coralsep.data.prepare_whamr import (
    WHAMR_SCRIPT_NAME,
    find_whamr_script,
    generate_whamr,
    verify_layout,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_scripts_dir(root: Path, nested: bool = False) -> Path:
    scripts = root / "whamr_scripts"
    target = scripts / "scripts" if nested else scripts
    target.mkdir(parents=True)
    (target / WHAMR_SCRIPT_NAME).write_text("")
    return scripts


def _make_whamr_layout(whamr_root: Path, freq: int = 16000) -> None:
    base = whamr_root / f"wav{freq // 1000}k" / "max" / "tt"
    for stream in ("mix_both", "s1", "s2"):
        base.joinpath(stream).mkdir(parents=True)
        (base / stream / "0.wav").write_bytes(b"RIFF")


# ── find_whamr_script ─────────────────────────────────────────────────────────


def test_find_script_at_root(tmp_path: Path) -> None:
    scripts = _make_scripts_dir(tmp_path)
    assert find_whamr_script(scripts) == scripts / WHAMR_SCRIPT_NAME


def test_find_script_nested(tmp_path: Path) -> None:
    scripts = _make_scripts_dir(tmp_path, nested=True)
    assert find_whamr_script(scripts) == scripts / "scripts" / WHAMR_SCRIPT_NAME


def test_find_script_missing_raises(tmp_path: Path) -> None:
    scripts = tmp_path / "whamr_scripts"
    scripts.mkdir()
    with pytest.raises(FileNotFoundError, match=WHAMR_SCRIPT_NAME):
        find_whamr_script(scripts)


# ── generate_whamr ────────────────────────────────────────────────────────────


def test_generate_skips_if_output_exists(tmp_path: Path) -> None:
    scripts = _make_scripts_dir(tmp_path)
    output_dir = tmp_path / "out"
    _make_whamr_layout(output_dir / "WHAMR")

    with patch("coralsep.data.prepare_whamr.subprocess.run") as mock_run:
        generate_whamr(scripts, tmp_path / "wsj0", tmp_path / "noise", output_dir)
        mock_run.assert_not_called()


def test_generate_calls_script_with_expected_args(tmp_path: Path) -> None:
    scripts = _make_scripts_dir(tmp_path)
    output_dir = tmp_path / "out"
    wsj0 = tmp_path / "wsj0"
    noise = tmp_path / "wham_noise"

    with patch("coralsep.data.prepare_whamr.subprocess.run") as mock_run:
        result = generate_whamr(scripts, wsj0, noise, output_dir)

    cmd = mock_run.call_args.args[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == str(scripts / WHAMR_SCRIPT_NAME)
    assert cmd[cmd.index("--wsj0-root") + 1] == str(wsj0)
    assert cmd[cmd.index("--wham-noise-root") + 1] == str(noise)
    assert cmd[cmd.index("--output-dir") + 1] == str(output_dir / "WHAMR")
    assert result == output_dir / "WHAMR"


# ── verify_layout ─────────────────────────────────────────────────────────────


def test_verify_layout_passes(tmp_path: Path) -> None:
    whamr_root = tmp_path / "WHAMR"
    _make_whamr_layout(whamr_root)
    verify_layout(whamr_root)  # must not raise


def test_verify_layout_raises_when_missing(tmp_path: Path) -> None:
    whamr_root = tmp_path / "WHAMR"
    base = whamr_root / "wav16k" / "max" / "tt"
    base.joinpath("mix_both").mkdir(parents=True)  # s1/s2 missing

    with pytest.raises(RuntimeError) as exc:
        verify_layout(whamr_root)
    assert "s1" in str(exc.value)


# ── CLI gating (deferred paths) ───────────────────────────────────────────────


def _run_main(argv: list[str]):
    with patch.object(sys, "argv", ["prepare_whamr.py", *argv]):
        with patch("coralsep.data.prepare_whamr.subprocess.run") as mock_run:
            from coralsep.data.prepare_whamr import main

            main()
            return mock_run


def test_main_defers_without_wsj0(tmp_path: Path, capsys) -> None:
    mock_run = _run_main(["--output-dir", str(tmp_path / "out")])
    mock_run.assert_not_called()
    assert "deferred" in capsys.readouterr().out.lower()


def test_main_defers_without_scripts(tmp_path: Path, capsys) -> None:
    mock_run = _run_main(
        ["--output-dir", str(tmp_path / "out"), "--wsj0-dir", str(tmp_path / "wsj0")]
    )
    mock_run.assert_not_called()
    assert "deferred" in capsys.readouterr().out.lower()
