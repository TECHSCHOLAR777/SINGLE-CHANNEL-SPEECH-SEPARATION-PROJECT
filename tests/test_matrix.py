"""Tests for eval/matrix.py::run_eval_matrix_npz.

matrix.py had zero test coverage before this file. Scoped to the new
run_eval_matrix_npz loader only (I-002/I-023/I-026's evidence gap), not a
retroactive rewrite of the older JSONL-based run_eval_matrix.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from coralsep.eval.matrix import run_eval_matrix_npz


class FakePipelineResult:
    def __init__(self, streams_8k, speaker_count, gate_vector=None):
        self.streams_8k = streams_8k
        self.streams_16k = streams_8k
        self.speaker_count = speaker_count
        self.completeness_prob = 1.0
        self.ood_flag = False
        self.gate_vector = gate_vector or {}


class FakePipeline:
    """Returns the true references as the 'separated' streams (perfect separation)."""

    def __init__(self):
        self.calls: list[tuple[np.ndarray, int]] = []
        self._next_refs: np.ndarray | None = None

    def run(self, mix_wav, sr):
        self.calls.append((mix_wav, sr))
        refs = self._next_refs
        return FakePipelineResult(streams_8k=refs, speaker_count=refs.shape[0])


def _write_sample(root, condition, n, idx, n_speakers=2, sr=8000, dur_s=1.0):
    cell_dir = root / condition / f"n{n}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    length = int(sr * dur_s)
    rng = np.random.default_rng(idx)
    refs = rng.standard_normal((n_speakers, length)).astype(np.float32)
    mixture = refs.sum(axis=0).astype(np.float32)
    recipe = json.dumps({"n_speakers": n_speakers, "sample_rate": sr})
    condition_vector = json.dumps({"n_speakers": float(n_speakers)})
    path = cell_dir / f"mix_{idx:04d}.npz"
    np.savez(
        path,
        mixture=mixture,
        references=refs,
        recipe=np.array([recipe]),
        condition_vector=np.array([condition_vector]),
    )
    return f"{condition}/n{n}/mix_{idx:04d}.npz", refs


@pytest.fixture
def fixed_eval_dir(tmp_path):
    files = []
    rel, refs0 = _write_sample(tmp_path, "clean", 2, 0)
    files.append({"path": rel, "sha256": "x"})
    rel, refs1 = _write_sample(tmp_path, "clean", 2, 1)
    files.append({"path": rel, "sha256": "x"})
    rel, refs2 = _write_sample(tmp_path, "reverb", 3, 0, n_speakers=3)
    files.append({"path": rel, "sha256": "x"})

    manifest = {
        "conditions": ["clean", "reverb"],
        "files": files,
        "held_out_conditions": [],
        "n_files": len(files),
        "n_per_cell": 1,
        "n_values": [2, 3],
        "sample_rate": 8000,
        "seed": 42,
        "set_hash": "test",
        "total_files": len(files),
    }
    (tmp_path / "eval_manifest.json").write_text(json.dumps(manifest))
    return tmp_path, {
        "clean/n2/mix_0000.npz": refs0,
        "clean/n2/mix_0001.npz": refs1,
        "reverb/n3/mix_0000.npz": refs2,
    }


def test_reads_condition_and_n_true_from_the_path(fixed_eval_dir):
    eval_dir, refs_by_path = fixed_eval_dir
    pipeline = FakePipeline()

    def run(mix_wav, sr):
        # Perfect separation: return whichever reference set matches this call's length.
        for refs in refs_by_path.values():
            if refs.sum(axis=0).shape == mix_wav.shape and np.allclose(
                refs.sum(axis=0), mix_wav, atol=1e-4
            ):
                return FakePipelineResult(streams_8k=refs, speaker_count=refs.shape[0])
        raise AssertionError("no matching reference set found")

    pipeline.run = run
    matrix = run_eval_matrix_npz(pipeline, eval_dir)

    assert len(matrix.records) == 3
    by_id = {r.mixture_id: r for r in matrix.records}
    assert by_id["mix_0000"].n_true in (2, 3)
    conditions = {r.condition for r in matrix.records}
    assert conditions == {"clean", "reverb"}
    n_trues = sorted(r.n_true for r in matrix.records)
    assert n_trues == [2, 2, 3]


def test_perfect_separation_gives_a_large_positive_si_sdri(fixed_eval_dir):
    eval_dir, refs_by_path = fixed_eval_dir
    pipeline = FakePipeline()

    def run(mix_wav, sr):
        for refs in refs_by_path.values():
            if refs.sum(axis=0).shape == mix_wav.shape and np.allclose(
                refs.sum(axis=0), mix_wav, atol=1e-4
            ):
                return FakePipelineResult(streams_8k=refs, speaker_count=refs.shape[0])
        raise AssertionError("no matching reference set found")

    pipeline.run = run
    matrix = run_eval_matrix_npz(pipeline, eval_dir)

    for record in matrix.records:
        assert record.si_sdri > 20.0, f"expected near-perfect SI-SDRi, got {record.si_sdri}"
        assert record.n_hat == record.n_true


def test_max_per_bucket_limits_samples_per_condition_and_n(fixed_eval_dir):
    eval_dir, refs_by_path = fixed_eval_dir
    pipeline = FakePipeline()

    def run(mix_wav, sr):
        for refs in refs_by_path.values():
            if refs.shape[1] == mix_wav.shape[0]:
                return FakePipelineResult(streams_8k=refs, speaker_count=refs.shape[0])
        raise AssertionError("no matching reference set found")

    pipeline.run = run
    matrix = run_eval_matrix_npz(pipeline, eval_dir, max_per_bucket=1)

    assert len(matrix.records) == 2  # one clean/n2, one reverb/n3


def test_a_pipeline_error_on_one_sample_does_not_abort_the_run(fixed_eval_dir, capsys):
    eval_dir, refs_by_path = fixed_eval_dir
    calls = {"n": 0}

    class FailFirstPipeline:
        def run(self, mix_wav, sr):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            for refs in refs_by_path.values():
                if refs.shape[1] == mix_wav.shape[0]:
                    return FakePipelineResult(streams_8k=refs, speaker_count=refs.shape[0])
            raise AssertionError("no matching reference set found")

    matrix = run_eval_matrix_npz(FailFirstPipeline(), eval_dir)
    assert len(matrix.records) == 2
    assert "pipeline error" in capsys.readouterr().out
