"""
Content-addressed hashing for configs, artifacts, and manifests (cross-cutting).

BLUEPRINT section 13 requires that every checkpoint and result file record the
SHA-256 of the config that produced it, and that frozen artifacts (base
checkpoint, adapter weights, calibration files, evaluation sets) are
content-addressed. Results that cannot be traced to their settings are deleted.

This module is the single implementation of that rule. Config hashing is
canonical: dict key order never changes the hash, so two configs that differ
only in YAML key order produce the same digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK_BYTES = 1 << 20
"""Read files in 1 MiB chunks so multi-GB checkpoints never load into RAM."""

SHORT_HASH_LEN = 12
"""Characters kept by short_hash. 12 hex chars = 48 bits, ample for run labels."""


def hash_bytes(payload: bytes) -> str:
    """SHA-256 of a byte string, as a lowercase hex digest."""
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: str | Path) -> str:
    """
    SHA-256 of a file's contents, streamed in chunks.

    Args:
        path: File to hash. Must exist and be a regular file.

    Returns:
        Lowercase hex digest.

    Raises:
        FileNotFoundError: When path does not exist.
        IsADirectoryError: When path is a directory (use hash_manifest instead).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"cannot hash missing file: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{p} is a directory; use hash_manifest for directories")

    digest = hashlib.sha256()
    with p.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_config(config: dict[str, Any]) -> str:
    """
    SHA-256 of a config dict, invariant to key order.

    The dict is serialized to JSON with sorted keys and no insignificant
    whitespace, so reordering YAML keys never changes the digest while
    changing any value always does.

    Args:
        config: Any JSON-serializable mapping. Non-serializable values (Path,
            numpy scalars) are stringified rather than raising, so a config
            carrying a Path still hashes deterministically.

    Returns:
        Lowercase hex digest.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hash_bytes(canonical.encode("utf-8"))


def hash_manifest(paths: list[str | Path], root: str | Path | None = None) -> str:
    """
    SHA-256 over a set of files: their relative names and their contents.

    Used to fingerprint an evaluation set (BLUEPRINT section 7.4: generated
    once, seeded, hashed, never regenerated). Both the file list and the file
    bodies are covered, so adding, removing, renaming, or editing any member
    changes the digest.

    Args:
        paths: Files to include. Order does not matter; they are sorted.
        root: If given, names are recorded relative to this directory so the
            digest is stable across machines with different absolute paths.

    Returns:
        Lowercase hex digest over the whole set.
    """
    root_path = Path(root).resolve() if root is not None else None
    entries: list[tuple[str, str]] = []

    for raw in paths:
        p = Path(raw)
        name = str(p.resolve().relative_to(root_path)) if root_path else str(p)
        entries.append((name, hash_file(p)))

    entries.sort()
    payload = "\n".join(f"{name}:{digest}" for name, digest in entries)
    return hash_bytes(payload.encode("utf-8"))


def short_hash(digest: str, length: int = SHORT_HASH_LEN) -> str:
    """Truncate a hex digest for use in run labels and directory names."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    return digest[:length]


def write_manifest(
    paths: list[str | Path],
    manifest_path: str | Path,
    root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Write a JSON manifest recording every file, its hash, and the set hash.

    This is the artifact that makes an evaluation set reproducible: it pins
    exactly which files were in the set and what they contained at generation
    time. Committing it means a later regeneration can be proven identical or
    proven different.

    Args:
        paths: Files in the set.
        manifest_path: Where to write the JSON manifest.
        root: Directory that member names are recorded relative to.
        extra: Extra fields to record alongside (seed, generation date,
            recipe name). Merged into the manifest at the top level.

    Returns:
        The manifest dict that was written.
    """
    root_path = Path(root).resolve() if root is not None else None

    files: list[dict[str, str]] = []
    for raw in sorted(Path(p) for p in paths):
        name = str(raw.resolve().relative_to(root_path)) if root_path else str(raw)
        files.append({"path": name, "sha256": hash_file(raw)})

    manifest: dict[str, Any] = {
        "set_hash": hash_manifest([f["path"] for f in files] if root_path is None else paths, root),
        "n_files": len(files),
        "files": files,
    }
    if extra:
        manifest.update(extra)

    out = Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify_manifest(manifest_path: str | Path, root: str | Path | None = None) -> list[str]:
    """
    Re-hash every file in a manifest and report what changed.

    Args:
        manifest_path: A manifest written by write_manifest.
        root: Directory the recorded relative names resolve against. Defaults
            to the manifest's own parent directory.

    Returns:
        List of human-readable mismatch descriptions. Empty means the set on
        disk is byte-identical to the set that was recorded.
    """
    mpath = Path(manifest_path)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    base = Path(root) if root is not None else mpath.parent

    problems: list[str] = []
    for entry in manifest.get("files", []):
        target = base / entry["path"]
        if not target.exists():
            problems.append(f"missing: {entry['path']}")
            continue
        actual = hash_file(target)
        if actual != entry["sha256"]:
            problems.append(
                f"modified: {entry['path']} (recorded {entry['sha256'][:12]}, "
                f"found {actual[:12]})"
            )
    return problems
