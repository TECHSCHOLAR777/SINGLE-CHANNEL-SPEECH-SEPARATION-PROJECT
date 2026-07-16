"""CPU smoke tests for parallel-branch LoRA (P1-B1)."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.lora import (
    PRIMARY_LORA_TARGETS,
    LoRALinear,
    coactivation_sampler,
    count_lora_parameters,
    freeze_base,
    orthogonal_penalty,
    register_lora,
)


def test_primary_target_count():
    assert len(PRIMARY_LORA_TARGETS) == 17


def test_lora_linear_identity_at_zero_gate():
    base = nn.Linear(8, 16, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.randn_like(base.weight))
    wrap = LoRALinear(base, ["reverb", "noise"], rank=2, alpha=2.0)
    x = torch.randn(3, 8)
    wrap.set_gates({"reverb": 0.0, "noise": 0.0})
    y0 = wrap(x)
    y_base = torch.nn.functional.linear(x, base.weight, base.bias)
    assert torch.allclose(y0, y_base, atol=1e-6)


def test_lora_linear_nonzero_gate_changes_output():
    base = nn.Linear(8, 16, bias=False)
    wrap = LoRALinear(base, ["reverb"], rank=2, alpha=2.0)
    # Make B non-zero so gate matters.
    with torch.no_grad():
        wrap.lora_B["reverb"].weight.copy_(torch.ones_like(wrap.lora_B["reverb"].weight))
    x = torch.randn(2, 8)
    wrap.set_gate("reverb", 0.0)
    y0 = wrap(x)
    wrap.set_gate("reverb", 1.0)
    y1 = wrap(x)
    assert not torch.allclose(y0, y1)


def test_register_on_dummy_module():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_block = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "freq_block": nn.Sequential(),
                        }
                    )
                ]
            )
            # Minimal path that register can skip missing — just wrap a linear manually.
            self.lin = nn.Linear(4, 4)

    m = Tiny()
    # Direct wrap smoke.
    m.lin = LoRALinear(m.lin, ["noise"], rank=2, alpha=2.0)
    freeze_base(m)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n in trainable)
    assert all("lora_" in n for n in trainable)


def test_coactivation_sampler_range():
    g = coactivation_sampler("noise")
    assert g["noise"] == 1.0
    assert 0.0 <= g["reverb"] <= 0.2
    assert 0.0 <= g["codec"] <= 0.2


def test_orthogonal_penalty_runs():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = LoRALinear(nn.Linear(8, 8), ["reverb", "noise"], 2, 2.0)

    from models.lora import LoRALibrary

    lib = LoRALibrary()
    lib.layers["layer"] = M().layer
    pen = orthogonal_penalty(lib)
    assert pen.ndim == 0
