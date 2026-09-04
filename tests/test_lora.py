"""CPU smoke tests for parallel-branch LoRA library (BLUEPRINT §5.3)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from coralsep.models.lora import (
    ADAPTER_NAMES,
    LoRALibrary,
    LoRALinear,
    _target_paths,
    lora_summary,
    olora_penalty,
)


def test_adapter_names_match_blueprint():
    assert set(ADAPTER_NAMES) == {"reverb", "noise", "codec"}
    assert len(ADAPTER_NAMES) == 3


def test_lora_linear_identity_at_zero_gate():
    base = nn.Linear(8, 16, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.randn_like(base.weight))
    wrap = LoRALinear(base, list(ADAPTER_NAMES), rank=2)
    x = torch.randn(3, 8)
    y0 = wrap(x, gates={n: 0.0 for n in ADAPTER_NAMES})
    y_base = nn.functional.linear(x, wrap.weight)
    assert torch.allclose(y0, y_base, atol=1e-5)


def test_lora_linear_nonzero_gate_changes_output():
    base = nn.Linear(8, 16, bias=False)
    wrap = LoRALinear(base, ["reverb"], rank=2)
    with torch.no_grad():
        wrap.branches["reverb"].B.fill_(1.0)
    x = torch.randn(2, 8)
    y0 = wrap(x, gates={"reverb": 0.0})
    y1 = wrap(x, gates={"reverb": 1.0})
    assert not torch.allclose(y0, y1)


def test_lora_linear_adapter_parameters_trainable():
    base = nn.Linear(8, 8, bias=False)
    wrap = LoRALinear(base, list(ADAPTER_NAMES), rank=4)
    # Base weight should be a buffer (not a parameter)
    assert not any(True for _ in wrap.parameters() if _ is wrap.weight)
    # Adapter params should exist and be trainable
    params = wrap.adapter_parameters("reverb")
    assert len(params) == 2  # A and B
    assert all(p.requires_grad for p in params)


def test_olora_penalty_non_negative():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

    m = M()
    LoRALibrary(m, adapter_names=("reverb", "noise"))
    pen = olora_penalty(m)
    assert pen.ndim == 0
    assert pen.item() >= 0.0


def test_forward_context_sets_active_adapter_gate():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

    m = M()
    lib = LoRALibrary(m, adapter_names=list(ADAPTER_NAMES))
    with lib.forward_context("reverb", co_activate=True):
        gates = lib.gate_dict()
    assert gates["reverb"] == 1.0
    assert 0.0 <= gates["noise"] <= 0.2
    assert 0.0 <= gates["codec"] <= 0.2


def test_lora_summary_returns_adapter_counts_from_wrapped_linear():
    base = nn.Linear(8, 8)
    wrap = LoRALinear(base, list(ADAPTER_NAMES), rank=4)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = wrap

    m = M()
    counts = lora_summary(m)
    assert set(counts.keys()) == set(ADAPTER_NAMES)
    assert all(v > 0 for v in counts.values())


# ---------------------------------------------------------------------------
# Regression: gate-vector misuse must fail loudly, not silently randomise gates
# ---------------------------------------------------------------------------


def _tiny_library() -> LoRALibrary:
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

    return LoRALibrary(M(), adapter_names=list(ADAPTER_NAMES))


def test_set_adapter_rejects_a_gate_mapping():
    """A gate dict passed where an adapter name belongs used to corrupt silently.

    No adapter name compares equal to a dict, so every adapter fell through to
    the co-activation branch and received a random gate, discarding the routed
    vector entirely. It must raise instead.
    """
    lib = _tiny_library()
    with pytest.raises(TypeError, match="set_gates"):
        lib.set_adapter({"reverb": 0.9, "noise": 0.1, "codec": 0.0})


def test_set_adapter_rejects_an_unknown_name():
    lib = _tiny_library()
    with pytest.raises(KeyError):
        lib.set_adapter("bandwidth")


def test_forward_context_without_a_name_preserves_the_routed_gates():
    lib = _tiny_library()
    routed = {"reverb": 0.9, "noise": 0.25, "codec": 0.0}
    lib.set_gates(routed)
    with lib.forward_context():
        assert lib.gate_dict() == routed


def test_set_gates_rejects_a_non_mapping():
    lib = _tiny_library()
    with pytest.raises(TypeError):
        lib.set_gates("reverb")


def test_set_gates_rejects_an_unknown_adapter():
    lib = _tiny_library()
    with pytest.raises(KeyError):
        lib.set_gates({"reverb": 1.0, "bandwidth": 0.5})


def test_injected_gates_reach_every_wrapped_linear():
    base = nn.Linear(4, 4)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = LoRALinear(base, list(ADAPTER_NAMES), rank=2)

    m = M()
    lib = LoRALibrary(m, adapter_names=list(ADAPTER_NAMES))
    routed = {"reverb": 0.9, "noise": 0.25, "codec": 0.0}
    lib.set_gates(routed)
    with lib.forward_context():
        assert m.lin._injected_gates == routed
    assert m.lin._injected_gates == {}


# ---------------------------------------------------------------------------
# Rank override, for the I-025 rank ablation
# ---------------------------------------------------------------------------


def test_target_paths_default_matches_blueprint_schedule():
    paths = _target_paths(nn.Module())
    ranks = {rank for _, rank in paths}
    # Attention targets use rank 8, filter heads use rank 4, per BLUEPRINT
    # 5.3. Both must appear, since a broken override could collapse them
    # to a single value without any test noticing.
    assert 8 in ranks
    assert 4 in ranks


def test_target_paths_attn_rank_override_applies_uniformly():
    paths = _target_paths(nn.Module(), attn_rank=32, filter_rank=4)
    attn_paths = [r for p, r in paths if "filter_estim" not in p]
    filter_paths = [r for p, r in paths if "filter_estim" in p]
    assert attn_paths and all(r == 32 for r in attn_paths)
    assert filter_paths and all(r == 4 for r in filter_paths)


def test_lora_library_threads_rank_override_into_attach():
    """
    Regression for the I-025 rank ablation: LoRALibrary must actually use
    the rank it was given, not the BLUEPRINT default, when it attaches.
    """
    base = nn.Linear(128, 384)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_block = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "freq_block": nn.ModuleDict(
                                {
                                    "block": nn.ModuleDict(
                                        {
                                            "sa": nn.ModuleDict(
                                                {"block": nn.ModuleDict({"qkv": base})}
                                            )
                                        }
                                    )
                                }
                            )
                        }
                    )
                    for _ in range(2)
                ]
            )

    m = M()
    lib = LoRALibrary(m, adapter_names=list(ADAPTER_NAMES), attn_rank=32)
    assert lib.n_attached >= 1
    wrapped = m.enc_block[0]["freq_block"]["block"]["sa"]["block"]["qkv"]
    assert isinstance(wrapped, LoRALinear)
    assert wrapped.branches["reverb"].rank == 32
