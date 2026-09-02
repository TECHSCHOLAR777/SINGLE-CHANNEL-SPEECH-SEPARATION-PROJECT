"""Kaggle dataset slicing: no import-time side effects, and manifest fidelity."""

from __future__ import annotations

import json
import random

import pytest
import soundfile as sf
import numpy as np

from scripts.slice_for_kaggle import (
    parse_args,
    rewrite_manifest,
    slice_noise,
    slice_rirs,
    slice_speech,
)


def _wav(path, seconds=0.05, sr=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(sr * seconds), dtype=np.float32), sr)


@pytest.fixture
def corpus(tmp_path):
    """A miniature stand-in for data/calmsep-8k."""
    root = tmp_path / "src"
    speech = root / "librispeech-8k" / "train-clean-100"
    for speaker in ("1001", "1002", "1003"):
        for i in range(4):
            _wav(speech / speaker / f"{speaker}-{i}.wav")

    (root / "librispeech-8k" / "manifest_8k.json").write_text(
        json.dumps(
            [
                {"split": "train-clean-100", "speaker_ids": ["1001", "1002", "1003"], "file_count": 12},
                {"split": "train-clean-360", "speaker_ids": ["2001"], "file_count": 99},
                {"split": "dev-clean", "speaker_ids": ["3001"], "file_count": 5},
                {"split": "test-clean", "speaker_ids": ["4001"], "file_count": 5},
            ]
        ),
        encoding="utf-8",
    )

    for i in range(6):
        _wav(root / "rirs" / f"rir_{i}.wav")
    (root / "rirs" / "bank.json").write_text("{}", encoding="utf-8")

    for i in range(5):
        _wav(root / "noise" / "wham" / f"noise_{i}.wav")
    (root / "noise" / "noise_manifest.json").write_text("{}", encoding="utf-8")
    return root


def test_importing_the_module_has_no_side_effects():
    """It used to start copying files at import time and then crash."""
    import importlib

    module = importlib.import_module("scripts.slice_for_kaggle")
    importlib.reload(module)  # would raise or copy if work happened at import


def test_slice_speech_caps_speakers_and_utterances(corpus, tmp_path):
    out = tmp_path / "out"
    speakers, copied = slice_speech(corpus, out, random.Random(0), n_speakers=2, max_utterances=3)
    assert len(speakers) == 2
    assert speakers == sorted(speakers)
    assert copied == 6
    assert len(list((out / "librispeech-8k" / "train-clean-100").rglob("*.wav"))) == 6


def test_slice_speech_is_deterministic_for_a_seed(corpus, tmp_path):
    a, _ = slice_speech(corpus, tmp_path / "a", random.Random(7), 2, 2)
    b, _ = slice_speech(corpus, tmp_path / "b", random.Random(7), 2, 2)
    assert a == b


def test_manifest_preserves_the_holdout_splits(corpus, tmp_path):
    """Held-out logic keys on speaker IDs, so dev and test entries must not change."""
    out = tmp_path / "out"
    speakers, copied = slice_speech(corpus, out, random.Random(0), 2, 3)
    path = rewrite_manifest(corpus, out, speakers, copied)
    entries = {e["split"]: e for e in json.loads(path.read_text(encoding="utf-8"))}

    assert entries["train-clean-100"]["speaker_ids"] == speakers
    assert entries["train-clean-100"]["file_count"] == copied
    assert entries["train-clean-360"]["file_count"] == 0
    assert entries["dev-clean"]["speaker_ids"] == ["3001"]
    assert entries["test-clean"]["speaker_ids"] == ["4001"]


def test_rirs_are_kept_whole_by_default(corpus, tmp_path):
    out = tmp_path / "out"
    assert slice_rirs(corpus, out, random.Random(0), None) == 6
    assert (out / "rirs" / "bank.json").exists()


def test_rirs_can_be_sampled(corpus, tmp_path):
    assert slice_rirs(corpus, tmp_path / "out", random.Random(0), 3) == 3


def test_noise_sampling_is_capped_by_availability(corpus, tmp_path):
    out = tmp_path / "out"
    assert slice_noise(corpus, out, random.Random(0), 100) == 5
    assert (out / "noise" / "noise_manifest.json").exists()


def test_missing_corpus_reports_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="train-clean-100"):
        slice_speech(tmp_path / "absent", tmp_path / "out", random.Random(0), 1, 1)


def test_defaults_do_not_write_outside_the_project():
    """The previous version hard-coded /tmp, which does not exist on Windows."""
    args = parse_args([])
    assert not str(args.out_dir).startswith("/tmp")
