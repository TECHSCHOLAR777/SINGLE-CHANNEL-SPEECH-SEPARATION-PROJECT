"""Tests for fixed evaluation manifest generation and hashing."""

from __future__ import annotations

import json
from pathlib import Path

from data.synthesis.fixed_eval import (
    build_eval_manifest,
    generate_all_manifests,
    load_manifest,
    manifest_hash_path,
)
from utils.hashing import hash_file


def test_build_tiny_manifest_and_hash(tmp_path: Path) -> None:
    path = build_eval_manifest(
        tier="clean_2_3",
        n_speakers=2,
        n_items=3,
        seed=99,
        out_dir=tmp_path,
    )
    assert path.exists()
    hash_path = manifest_hash_path(tmp_path, "clean_2_3", 2)
    assert hash_path.exists()

    digest = hash_file(path)
    hash_line = hash_path.read_text(encoding="utf-8").strip()
    assert hash_line.startswith(digest)
    assert path.name in hash_line

    meta, items = load_manifest(path)
    assert meta["tier"] == "clean_2_3"
    assert meta["n_speakers"] == 2
    assert meta["n_items"] == 3
    assert len(items) == 3
    assert items[0]["item_id"].startswith("clean_2_3_n2_")


def test_manifest_deterministic(tmp_path: Path) -> None:
    p1 = build_eval_manifest("codec_only", 2, 2, seed=7, out_dir=tmp_path / "a")
    p2 = build_eval_manifest("codec_only", 2, 2, seed=7, out_dir=tmp_path / "b")
    assert hash_file(p1) == hash_file(p2)


def test_holdout_flags_present(tmp_path: Path) -> None:
    path = build_eval_manifest("reverb_codec_holdout", 2, 1, seed=1, out_dir=tmp_path)
    _, items = load_manifest(path)
    assert items[0]["gate_holdout"] is True
    assert items[0]["recipe"]["conditions"]["degradation"] == "reverb+codec"


def test_generate_all_manifests_writes_matrix_index(tmp_path: Path) -> None:
    paths = generate_all_manifests(tmp_path, seed=123)
    assert len(paths) >= 10
    index = tmp_path / "matrix_index.json"
    assert index.exists()
    doc = json.loads(index.read_text(encoding="utf-8"))
    assert doc["seed"] == 123
    assert doc["n_manifests"] == len(paths)

    holdout = [m for m in doc["manifests"] if m["gate_holdout"]]
    tier_ids = {m["tier"] for m in holdout}
    assert "reverb_codec_holdout" in tier_ids
    assert "noise_codec_holdout" in tier_ids


def test_manifest_rows_are_valid_jsonl(tmp_path: Path) -> None:
    path = build_eval_manifest("reverb_noisy_primary", 2, 1, seed=0, out_dir=tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # meta + 1 item
    for line in lines:
        json.loads(line)
