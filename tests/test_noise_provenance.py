"""Tests for I-044: WHAM split provenance must be recorded and enforced.

LibriMix's official test mixtures are built from WHAM noise, so training the
noise adapter or the gate on any split but tr (or on an unfiltered stage of
the whole corpus) risks the training data overlapping acoustically with the
exact clips the headline results are later scored against. Before this
ticket, nothing recorded which split a staged noise directory came from and
nothing checked it before training.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from coralsep.data.prepare.noise_staging import check_noise_provenance, stage_source


def _write_noise_tree(root, split_dirs: list[str]) -> None:
    """Create root/<split>/clip.wav for each split, one clip per split."""
    for split in split_dirs:
        d = root / split
        d.mkdir(parents=True)
        sf.write(str(d / "clip.wav"), np.random.randn(1600).astype(np.float32), 16000)


def test_stage_source_filters_to_the_requested_split(tmp_path):
    src = tmp_path / "wham_src"
    _write_noise_tree(src, ["tr", "tt", "cv"])
    dst = tmp_path / "staged"
    dst.mkdir()

    entries = stage_source(src, dst, "wham", required_split="tr")

    assert len(entries) == 1
    assert entries[0]["split"] == "tr"
    assert "tr" in entries[0]["src_path"]


def test_stage_source_records_unfiltered_when_no_split_given(tmp_path):
    src = tmp_path / "wham_src"
    _write_noise_tree(src, ["tr"])
    dst = tmp_path / "staged"
    dst.mkdir()

    entries = stage_source(src, dst, "wham")

    assert entries[0]["split"] == "unfiltered"


def test_check_noise_provenance_passes_for_a_clean_tr_manifest(tmp_path):
    manifest = {
        "clips": [
            {"source": "wham", "clip_name": "a", "split": "tr"},
            {"source": "wham", "clip_name": "b", "split": "tr"},
            {"source": "dns4", "clip_name": "c", "split": "unfiltered"},
        ]
    }
    (tmp_path / "noise_manifest.json").write_text(json.dumps(manifest))

    check_noise_provenance(tmp_path)  # must not raise


def test_check_noise_provenance_rejects_the_wrong_split(tmp_path):
    manifest = {"clips": [{"source": "wham", "clip_name": "a", "split": "tt"}]}
    (tmp_path / "noise_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="tt"):
        check_noise_provenance(tmp_path)


def test_check_noise_provenance_rejects_unfiltered(tmp_path):
    """An unfiltered stage is exactly the unsafe case this guard exists for,
    not an acceptable default."""
    manifest = {"clips": [{"source": "wham", "clip_name": "a", "split": "unfiltered"}]}
    (tmp_path / "noise_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="unfiltered"):
        check_noise_provenance(tmp_path)


def test_check_noise_provenance_rejects_a_manifest_predating_the_split_field(tmp_path):
    """An old manifest with no split field at all has unknown provenance,
    which this guard must treat as unsafe, not as a pass."""
    manifest = {"clips": [{"source": "wham", "clip_name": "a"}]}
    (tmp_path / "noise_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="MISSING"):
        check_noise_provenance(tmp_path)


def test_check_noise_provenance_requires_a_manifest_to_exist(tmp_path):
    with pytest.raises(RuntimeError, match="noise_manifest.json"):
        check_noise_provenance(tmp_path)
