"""Tests for SR-CorrNet expert availability checks."""

from pathlib import Path

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
