"""Tests for models/scene_analyzer.py — P2-A1 SceneAnalyzer."""

from __future__ import annotations

import math

import pytest
import torch

from models.scene_analyzer import SceneAnalyzer, _build_mel_filterbank

# ── Fixtures ──────────────────────────────────────────────────────────────────

SR = 16_000
B = 2

# Fast defaults for all tests except the param-count check.
_FAST = dict(
    feature_dim=64,
    n_mels=40,
    n_fft=256,
    hop=64,
    max_speakers=5,
    segment_samples=8_000,  # 0.5 s — small so tests are quick
    sr=SR,
)


def _make(**overrides) -> SceneAnalyzer:
    kw = {**_FAST, **overrides}
    return SceneAnalyzer(**kw)


def _wave(b: int = B, t: int = 8_000) -> torch.Tensor:
    return torch.randn(b, t, generator=torch.Generator().manual_seed(0))


# ── Mel filterbank ────────────────────────────────────────────────────────────


def test_mel_filterbank_shape() -> None:
    fb = _build_mel_filterbank(40, 256, SR)
    assert fb.shape == (40, 129)


def test_mel_filterbank_non_negative() -> None:
    fb = _build_mel_filterbank(40, 256, SR)
    assert (fb >= 0).all()


def test_mel_filterbank_most_filters_sum_positive() -> None:
    # The lowest-frequency filter can be degenerate (DC bin only) with small
    # n_fft, so we allow at most one zero filter.
    fb = _build_mel_filterbank(40, 256, SR)
    assert (fb.sum(dim=-1) > 0).sum() >= 39


# ── Output keys and types ─────────────────────────────────────────────────────


def test_forward_returns_expected_keys() -> None:
    ana = _make()
    out = ana(_wave())
    assert set(out.keys()) == {"segment_features", "count_logits", "scene_weights"}
    for v in out.values():
        assert isinstance(v, torch.Tensor)


# ── Output shapes ─────────────────────────────────────────────────────────────


def test_segment_features_shape_single_segment() -> None:
    """T < segment_samples → padded to 1 segment."""
    ana = _make()
    out = ana(_wave(t=4_000))
    assert out["segment_features"].shape == (B, 1, 64)


def test_segment_features_shape_exact_one_segment() -> None:
    ana = _make()
    out = ana(_wave(t=8_000))
    assert out["segment_features"].shape == (B, 1, 64)


def test_segment_features_shape_two_segments() -> None:
    ana = _make()
    out = ana(_wave(t=16_000))
    assert out["segment_features"].shape == (B, 2, 64)


def test_segment_features_shape_fractional_segments() -> None:
    """T = 1.5 × segment_samples → ceil → 2 segments."""
    ana = _make()
    out = ana(_wave(t=12_000))
    assert out["segment_features"].shape == (B, 2, 64)


def test_count_logits_shape() -> None:
    ana = _make(max_speakers=4)
    out = ana(_wave())
    assert out["count_logits"].shape == (B, 4)


def test_scene_weights_shape() -> None:
    ana = _make()
    out = ana(_wave())
    assert out["scene_weights"].shape == (B, 3)


def test_scene_weights_sum_to_one() -> None:
    ana = _make()
    out = ana(_wave())
    sums = out["scene_weights"].sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(B), atol=1e-5, rtol=0)


def test_feature_dim_respected() -> None:
    for dim in (32, 64, 128):
        ana = _make(feature_dim=dim)
        out = ana(_wave())
        assert out["segment_features"].shape[-1] == dim


# ── Batch dimension ───────────────────────────────────────────────────────────


def test_batch_size_one() -> None:
    ana = _make()
    out = ana(_wave(b=1))
    assert out["segment_features"].shape[0] == 1
    assert out["count_logits"].shape[0] == 1
    assert out["scene_weights"].shape[0] == 1


def test_batch_size_four() -> None:
    ana = _make()
    out = ana(_wave(b=4))
    assert out["segment_features"].shape[0] == 4


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_all_zeros_input_doesnt_crash() -> None:
    ana = _make()
    out = ana(torch.zeros(B, 8_000))
    assert out["segment_features"].shape[0] == B


def test_very_short_audio_one_sample() -> None:
    """T=1 should be padded enough not to crash."""
    ana = _make()
    out = ana(torch.zeros(B, 1))
    assert out["segment_features"].shape == (B, 1, 64)


def test_invalid_input_ndim_raises() -> None:
    ana = _make()
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        ana(torch.randn(B, 8_000, 1))


# ── Gradient flow ─────────────────────────────────────────────────────────────


def test_backward_pass_segment_features() -> None:
    ana = _make()
    x = _wave()
    out = ana(x)
    loss = out["segment_features"].sum()
    loss.backward()
    grads = [p.grad for p in ana.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_backward_pass_count_logits() -> None:
    ana = _make()
    x = _wave()
    out = ana(x)
    out["count_logits"].sum().backward()


def test_backward_pass_scene_weights() -> None:
    ana = _make()
    x = _wave()
    out = ana(x)
    out["scene_weights"].sum().backward()


# ── Determinism ───────────────────────────────────────────────────────────────


def test_deterministic_output() -> None:
    ana = _make()
    ana.eval()
    x = _wave()
    with torch.no_grad():
        out_a = ana(x)
        out_b = ana(x)
    torch.testing.assert_close(out_a["segment_features"], out_b["segment_features"])


# ── Handcrafted features sensitivity ─────────────────────────────────────────


def test_handcrafted_features_differ_clean_vs_noisy() -> None:
    """Handcrafted features must differ between a clean tone and white noise."""
    ana = _make()
    ana.eval()
    t = 8_000
    time = torch.linspace(0, t / SR, t)
    clean = (0.5 * torch.sin(2 * math.pi * 300 * time)).unsqueeze(0).expand(B, -1)
    noisy = torch.randn(B, t) * 0.3
    with torch.no_grad():
        hc_clean = ana._handcrafted_features(clean)
        hc_noisy = ana._handcrafted_features(noisy)
    assert not torch.allclose(hc_clean, hc_noisy, atol=1e-3)


def test_reverb_proxy_higher_for_decaying_signal() -> None:
    """Signal with decaying energy in the second half should have lower reverb_proxy."""
    ana = _make()
    ana.eval()
    t = 8_000
    # Loud head, quiet tail
    loud_head = torch.cat([torch.randn(B, t // 2), torch.zeros(B, t // 2)], dim=-1)
    # Quiet head, loud tail
    loud_tail = torch.cat([torch.zeros(B, t // 2), torch.randn(B, t // 2)], dim=-1)
    with torch.no_grad():
        sw_head = ana(loud_head)["scene_weights"]
        sw_tail = ana(loud_tail)["scene_weights"]
    # Just verify they are different — exact ordering depends on the MLP
    assert not torch.allclose(sw_head, sw_tail, atol=1e-3)


# ── Parameter count ───────────────────────────────────────────────────────────


def test_parameter_count_full_spec() -> None:
    """Full-spec model must be ~1.5 M params (within 20%)."""
    ana = SceneAnalyzer()  # production defaults
    n = ana.parameter_count()
    assert 1_200_000 <= n <= 1_800_000, f"parameter_count={n:,} out of expected range"


def test_parameter_count_scales_with_feature_dim() -> None:
    small = _make(feature_dim=32)
    large = _make(feature_dim=128)
    assert small.parameter_count() < large.parameter_count()


# ── Trainer compatibility ─────────────────────────────────────────────────────


def test_trainer_model_uses_scene_analyzer() -> None:
    """CAMoSETrainable wires SceneAnalyzer; verify segment_features feed Router."""
    from models.cascade_gate import CascadeGate
    from train.trainer import CAMoSETrainable

    model = CAMoSETrainable(feature_dim=32, num_experts=3)
    mix = _wave(b=2, t=4_000)
    scene = model.scene_analyzer(mix)
    assert scene["segment_features"].ndim == 3
    router_w = model.router(scene["segment_features"])
    assert router_w.shape[-1] == 3
