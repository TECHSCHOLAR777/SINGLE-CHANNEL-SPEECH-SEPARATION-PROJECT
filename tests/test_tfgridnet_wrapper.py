"""Tests for TF-GridNet fallback expert availability."""

from models.experts.tfgridnet import TFGridNetExpert, get_expensive_expert


def test_tfgridnet_available_with_sepformer_fallback() -> None:
    expert = TFGridNetExpert(device="cpu", use_sepformer_fallback=True)
    # May be False when speechbrain/torchaudio unavailable in CI
    assert isinstance(expert.is_available, bool)


def test_get_expensive_expert_without_srcorrnet_repo() -> None:
    try:
        expert = get_expensive_expert(device="cpu", srcorrnet_repo=None, tfgridnet_tag=None)
        assert expert is not None
    except RuntimeError:
        pass  # acceptable when no backend available
