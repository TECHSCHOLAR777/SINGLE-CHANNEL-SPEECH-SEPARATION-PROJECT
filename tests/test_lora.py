"""CPU smoke tests for parallel-branch LoRA library (BLUEPRINT §5.3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.lora import (
    ADAPTER_NAMES,
    LoRALayer,
    LoRALinear,
    LoRALibrary,
    _target_paths,
    olora_penalty,
    lora_summary,
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
