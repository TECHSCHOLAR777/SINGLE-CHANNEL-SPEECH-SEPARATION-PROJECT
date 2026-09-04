"""Tests for I-002 and I-026: no oracle count by default, and per-sample scores retained.

Both _run_baseline and _run_calmsep now accept n_spks=None and, in that case,
call process_waveform without a count argument, matching the model's own
attractor-path behaviour documented in models/experts/srcorrnet.py's Patch A
(separate(n_spks=None) uses p_k to decide). _score_split defaults to that path
(oracle_count=False) and records each model's own count accuracy against the
true count read from the split name, instead of never measuring it at all.
It also now retains a per-sample record (I-026), so a bootstrap confidence
interval (eval/stats.py::bootstrap_ci) can be computed without a rerun, once
there are enough samples to resample meaningfully.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from coralsep.eval.run_eval import _run_baseline, _score_split


class _FakeModel:
    """Stands in for SSInference. Returns a fixed number of streams and
    records whether n_spks was passed, so tests can assert on call shape
    without a real backbone."""

    def __init__(self, n_streams: int) -> None:
        self.n_streams = n_streams
        self.calls: list[dict] = []

    def process_waveform(self, wav, n_spks=None):
        self.calls.append({"n_spks": n_spks})
        t = wav.shape[-1]
        return {"waveforms": [torch.zeros(t) for _ in range(self.n_streams)]}


def test_run_baseline_oracle_mode_passes_n_spks():
    model = _FakeModel(n_streams=3)
    mix = np.zeros(8000, dtype=np.float32)
    est = _run_baseline(model, mix, n_spks=3, device=torch.device("cpu"))
    assert est.shape[0] == 3
    assert model.calls[0]["n_spks"] is not None


def test_run_baseline_non_oracle_mode_does_not_pass_n_spks():
    model = _FakeModel(n_streams=2)
    mix = np.zeros(8000, dtype=np.float32)
    est = _run_baseline(model, mix, n_spks=None, device=torch.device("cpu"))
    assert est.shape[0] == 2
    assert model.calls[0]["n_spks"] is None


def test_score_split_default_is_non_oracle_and_records_count_accuracy(tmp_path, monkeypatch):
    # Build a fake LibriMix-shaped directory with one 2-speaker mixture.
    base = tmp_path / "Libri2Mix" / "wav8k" / "min" / "test"
    (base / "mix_both").mkdir(parents=True)
    (base / "s1").mkdir(parents=True)
    (base / "s2").mkdir(parents=True)

    import soundfile as sf

    wav = np.sin(2 * np.pi * 220 * np.arange(4000) / 8000).astype(np.float32)
    sf.write(str(base / "mix_both" / "u1.wav"), wav, 8000)
    sf.write(str(base / "s1" / "u1.wav"), wav, 8000)
    sf.write(str(base / "s2" / "u1.wav"), wav, 8000)

    # Baseline model under-counts (guesses 1); coralsep model gets it right (2).
    ss_base = _FakeModel(n_streams=1)
    ss_calm = _FakeModel(n_streams=2)

    from coralsep.models.condition import level1_tensor

    result = _score_split(
        split="Libri2Mix",
        librimix_root=tmp_path,
        n=1,
        ss_base=ss_base,
        ss_calm=ss_calm,
        inner=MagicMock(),
        lib=MagicMock(),
        gate_net=MagicMock(return_value=torch.tensor([[0.5, 0.5, 0.5]])),
        temperature=1.0,
        l1fn=level1_tensor,
        ADAPTER_NAMES=["reverb", "noise", "codec"],
        device=torch.device("cpu"),
    )

    assert result["oracle_count"] is False
    # Baseline saw n_spks=None on every call: the model's own guess, not the answer.
    assert all(c["n_spks"] is None for c in ss_base.calls)
    assert all(c["n_spks"] is None for c in ss_calm.calls)
    # Baseline guessed 1 against a true count of 2: 0% count accuracy.
    assert result["baseline"]["count_accuracy"] == 0.0
    # CoRAL-Sep guessed 2 against a true count of 2: 100% count accuracy.
    assert result["coralsep"]["count_accuracy"] == 1.0
    # I-026: per-sample records exist, one per mixture actually scored.
    assert len(result["baseline"]["per_sample"]) == 1
    assert result["baseline"]["per_sample"][0]["uid"] == "u1"
    assert result["baseline"]["per_sample"][0]["n_hat"] == 1
    assert len(result["coralsep"]["per_sample"]) == 1
    # Below the resample floor (n=1 < 8), no CI is computed rather than a
    # meaningless one.
    assert result["baseline"]["si_sdri_ci"] is None


def test_score_split_oracle_count_flag_still_available(tmp_path):
    base = tmp_path / "Libri2Mix" / "wav8k" / "min" / "test"
    (base / "mix_both").mkdir(parents=True)
    (base / "s1").mkdir(parents=True)
    (base / "s2").mkdir(parents=True)

    import soundfile as sf

    wav = np.sin(2 * np.pi * 220 * np.arange(4000) / 8000).astype(np.float32)
    sf.write(str(base / "mix_both" / "u1.wav"), wav, 8000)
    sf.write(str(base / "s1" / "u1.wav"), wav, 8000)
    sf.write(str(base / "s2" / "u1.wav"), wav, 8000)

    ss_base = _FakeModel(n_streams=2)
    ss_calm = _FakeModel(n_streams=2)

    from coralsep.models.condition import level1_tensor

    result = _score_split(
        split="Libri2Mix",
        librimix_root=tmp_path,
        n=1,
        ss_base=ss_base,
        ss_calm=ss_calm,
        inner=MagicMock(),
        lib=MagicMock(),
        gate_net=MagicMock(return_value=torch.tensor([[0.5, 0.5, 0.5]])),
        temperature=1.0,
        l1fn=level1_tensor,
        ADAPTER_NAMES=["reverb", "noise", "codec"],
        device=torch.device("cpu"),
        oracle_count=True,
    )

    assert result["oracle_count"] is True
    assert all(c["n_spks"] is not None for c in ss_base.calls)
    assert all(c["n_spks"] is not None for c in ss_calm.calls)


def test_score_split_computes_ci_at_the_resample_floor(tmp_path):
    base = tmp_path / "Libri2Mix" / "wav8k" / "min" / "test"
    (base / "mix_both").mkdir(parents=True)
    (base / "s1").mkdir(parents=True)
    (base / "s2").mkdir(parents=True)

    import soundfile as sf

    for i in range(8):
        wav = np.sin(2 * np.pi * 220 * np.arange(4000) / 8000).astype(np.float32)
        sf.write(str(base / "mix_both" / f"u{i}.wav"), wav, 8000)
        sf.write(str(base / "s1" / f"u{i}.wav"), wav, 8000)
        sf.write(str(base / "s2" / f"u{i}.wav"), wav, 8000)

    ss_base = _FakeModel(n_streams=2)
    ss_calm = _FakeModel(n_streams=2)

    from coralsep.models.condition import level1_tensor

    result = _score_split(
        split="Libri2Mix",
        librimix_root=tmp_path,
        n=8,
        ss_base=ss_base,
        ss_calm=ss_calm,
        inner=MagicMock(),
        lib=MagicMock(),
        gate_net=MagicMock(return_value=torch.tensor([[0.5, 0.5, 0.5]])),
        temperature=1.0,
        l1fn=level1_tensor,
        ADAPTER_NAMES=["reverb", "noise", "codec"],
        device=torch.device("cpu"),
    )

    assert result["n_samples"] == 8
    ci = result["baseline"]["si_sdri_ci"]
    assert ci is not None
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
