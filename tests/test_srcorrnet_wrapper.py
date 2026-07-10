"""Tests for SR-CorrNet expert availability checks and output parsing."""

from pathlib import Path

import torch

from models.experts.srcorrnet import SRCorrNetExpert


def test_srcorrnet_not_available_without_repo() -> None:
    expert = SRCorrNetExpert(device="cpu", repo_path=None)
    assert not expert.is_available


def test_srcorrnet_not_available_missing_checkpoint(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    expert = SRCorrNetExpert(
        device="cpu",
        repo_path=repo,
        checkpoint_path=tmp_path / "missing.pt",
    )
    assert not expert.is_available


def test_srcorrnet_available_with_repo_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    expert = SRCorrNetExpert(device="cpu", repo_path=repo, checkpoint_path=None)
    assert expert.is_available


def test_srcorrnet_parse_output_with_attractors() -> None:
    t = 4000
    k = 3
    est = torch.randn(k, t)
    conf = torch.tensor([0.9, 0.85, 0.8])
    attractors = torch.randn(k, 16)
    stop = torch.tensor(-2.5)

    output = {
        "est_sources": est.unsqueeze(0),
        "confidence": conf,
        "attractors": attractors,
        "stop_logit": stop,
    }
    expert = SRCorrNetExpert(device="cpu")
    streams, confidences, att, stop_logit = expert._parse_output(output, t)

    assert streams.shape == (k, t)
    assert len(confidences) == k
    assert len(att) == k
    assert att[0] is not None and att[0].shape == (16,)
    assert stop_logit == -2.5
