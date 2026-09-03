"""
Unit tests for utils/hashing.py.

Tests cover config-hash stability, file-hash correctness, manifest write+verify,
and short_hash truncation. No network access required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coralsep.utils.hashing import (
    hash_bytes,
    hash_config,
    hash_file,
    hash_manifest,
    short_hash,
    verify_manifest,
    write_manifest,
)

# ---------------------------------------------------------------------------
# hash_bytes
# ---------------------------------------------------------------------------


def test_hash_bytes_known_vector():
    # SHA-256 of empty string is well-known.
    digest = hash_bytes(b"")
    assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_bytes_returns_lowercase_hex():
    d = hash_bytes(b"hello")
    assert d == d.lower()
    assert all(c in "0123456789abcdef" for c in d)
    assert len(d) == 64


# ---------------------------------------------------------------------------
# hash_config
# ---------------------------------------------------------------------------


def test_hash_config_key_order_invariant():
    a = hash_config({"z": 1, "a": 2})
    b = hash_config({"a": 2, "z": 1})
    assert a == b


def test_hash_config_value_sensitive():
    a = hash_config({"x": 1})
    b = hash_config({"x": 2})
    assert a != b


def test_hash_config_nested():
    a = hash_config({"lr": 1e-4, "model": {"d": 128, "heads": 8}})
    b = hash_config({"model": {"heads": 8, "d": 128}, "lr": 1e-4})
    assert a == b


def test_hash_config_path_serializes():
    # Path objects are not JSON-serializable by default; hash_config must handle them.
    d = hash_config({"path": Path("/tmp/foo")})
    assert len(d) == 64


# ---------------------------------------------------------------------------
# hash_file
# ---------------------------------------------------------------------------


def test_hash_file_correctness(tmp_path):
    f = tmp_path / "data.bin"
    payload = b"hello world"
    f.write_bytes(payload)
    assert hash_file(f) == hash_bytes(payload)


def test_hash_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "nonexistent.bin")


def test_hash_file_directory_raises(tmp_path):
    with pytest.raises(IsADirectoryError):
        hash_file(tmp_path)


def test_hash_file_large_file(tmp_path):
    """Streaming: a file bigger than the 1 MiB chunk should hash correctly."""
    f = tmp_path / "big.bin"
    payload = b"x" * (2 * 1024 * 1024 + 7)  # 2 MiB + 7 bytes
    f.write_bytes(payload)
    assert hash_file(f) == hash_bytes(payload)


# ---------------------------------------------------------------------------
# hash_manifest
# ---------------------------------------------------------------------------


def test_hash_manifest_covers_names(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello")
    f2.write_text("hello")
    # Same content, different names → different digest
    h1 = hash_manifest([f1], root=tmp_path)
    h2 = hash_manifest([f2], root=tmp_path)
    assert h1 != h2


def test_hash_manifest_covers_contents(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("v1")
    h1 = hash_manifest([f], root=tmp_path)
    f.write_text("v2")
    h2 = hash_manifest([f], root=tmp_path)
    assert h1 != h2


def test_hash_manifest_order_invariant(tmp_path):
    files = []
    for name in ("a.txt", "b.txt", "c.txt"):
        p = tmp_path / name
        p.write_text(name)
        files.append(p)
    h1 = hash_manifest(files, root=tmp_path)
    h2 = hash_manifest(list(reversed(files)), root=tmp_path)
    assert h1 == h2


# ---------------------------------------------------------------------------
# write_manifest / verify_manifest
# ---------------------------------------------------------------------------


def test_write_then_verify_clean(tmp_path):
    f = tmp_path / "eval" / "item.wav"
    f.parent.mkdir()
    f.write_bytes(b"\x00" * 100)
    manifest_path = tmp_path / "manifest.json"
    write_manifest([f], manifest_path, root=tmp_path / "eval")
    problems = verify_manifest(manifest_path, root=tmp_path / "eval")
    assert problems == []


def test_verify_detects_modification(tmp_path):
    f = tmp_path / "item.bin"
    f.write_bytes(b"original")
    manifest_path = tmp_path / "manifest.json"
    write_manifest([f], manifest_path)
    f.write_bytes(b"tampered")
    problems = verify_manifest(manifest_path)
    assert len(problems) == 1
    assert "modified" in problems[0]


def test_verify_detects_missing_file(tmp_path):
    f = tmp_path / "gone.bin"
    f.write_bytes(b"data")
    manifest_path = tmp_path / "manifest.json"
    write_manifest([f], manifest_path)
    f.unlink()
    problems = verify_manifest(manifest_path)
    assert len(problems) == 1
    assert "missing" in problems[0]


def test_write_manifest_extra_fields(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hi")
    manifest_path = tmp_path / "m.json"
    write_manifest([f], manifest_path, extra={"seed": 42, "split": "test"})
    data = json.loads(manifest_path.read_text())
    assert data["seed"] == 42
    assert data["split"] == "test"
    assert "set_hash" in data
    assert "n_files" in data


# ---------------------------------------------------------------------------
# short_hash
# ---------------------------------------------------------------------------


def test_short_hash_length():
    d = "a" * 64
    assert len(short_hash(d, length=12)) == 12
    assert len(short_hash(d, length=8)) == 8


def test_short_hash_default_length():
    d = "b" * 64
    assert len(short_hash(d)) == 12


def test_short_hash_zero_raises():
    with pytest.raises(ValueError):
        short_hash("abc", length=0)
