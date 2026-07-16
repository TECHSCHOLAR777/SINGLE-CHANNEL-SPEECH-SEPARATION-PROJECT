"""Parallel-branch LoRA library for CALM-Sep (BLUEPRINT §5.3).

Composition: y = W0 x + sum_i (g_i * scale * B_i(A_i x))
Corrections are never merged into W0. Identity fallback at g=0 is structural.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_NAMES: tuple[str, ...] = ("reverb", "noise", "codec")

def _default_target_specs(*, include_aggregate: bool = True, include_aux: bool = False) -> list[tuple[str, int]]:
    """Build LoRA target path list.

    BLUEPRINT §5.3 primary count of 17 = all QKV projections in enc/dec/dec_cs
    plus the filter head (4+8+4+1). Aggregate-head and aux-filter attaches are
    optional extensions controlled by flags.
    """
    specs: list[tuple[str, int]] = []
    for i in range(2):
        for axis in ("freq_block", "time_block"):
            specs.append((f"enc_block.{i}.{axis}.block.sa.block.qkv", 8))
            if include_aggregate:
                specs.append((f"enc_block.{i}.{axis}.block.sa.block.aggregate_heads.0", 8))
    for i in range(4):
        for axis in ("freq_block", "time_block"):
            specs.append((f"dec_block.{i}.{axis}.block.sa.block.qkv", 8))
            if include_aggregate:
                specs.append((f"dec_block.{i}.{axis}.block.sa.block.aggregate_heads.0", 8))
    for i in range(4):
        specs.append((f"dec_cs.{i}.block.block.sa.block.qkv", 8))
        if include_aggregate:
            specs.append((f"dec_cs.{i}.block.block.sa.block.aggregate_heads.0", 8))
    specs.append(("filter_estim.mask.net", 4))
    if include_aux:
        for i in range(4):
            specs.append((f"filter_estim_aux.{i}.mask.net", 4))
    return specs


# Primary 17 targets (qkv + filter head only).
PRIMARY_LORA_TARGETS: list[tuple[str, int]] = _default_target_specs(
    include_aggregate=False, include_aux=False
)
assert len(PRIMARY_LORA_TARGETS) == 17, len(PRIMARY_LORA_TARGETS)

# Extended set used when use_primary_only=False.
LORA_TARGET_SPECS: list[tuple[str, int]] = _default_target_specs(
    include_aggregate=True, include_aux=True
)


@dataclass
class LoRAConfig:
    """Per-adapter LoRA configuration."""

    name: str
    rank_attn: int = 8
    rank_filter: int = 4
    alpha_attn: float = 8.0
    alpha_filter: float = 4.0
    enabled: bool = True


class LoRALinear(nn.Module):
    """Drop-in Linear wrapper with parallel LoRA branches per adapter."""

    def __init__(
        self,
        base: nn.Linear,
        adapter_names: Sequence[str],
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(base)}")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.adapter_names = list(adapter_names)

        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        for name in adapter_names:
            a = nn.Linear(self.in_features, rank, bias=False)
            b = nn.Linear(rank, self.out_features, bias=False)
            nn.init.kaiming_uniform_(a.weight, a=5**0.5)
            nn.init.zeros_(b.weight)
            self.lora_A[name] = a
            self.lora_B[name] = b

        # Per-adapter gate scalars (set externally each forward).
        self._gates: dict[str, float] = {n: 0.0 for n in adapter_names}

        # Freeze base weights.
        for p in self.base.parameters():
            p.requires_grad_(False)

    def set_gate(self, adapter: str, value: float) -> None:
        if adapter not in self._gates:
            raise KeyError(f"unknown adapter {adapter}")
        self._gates[adapter] = float(value)

    def set_gates(self, gates: dict[str, float]) -> None:
        for k, v in gates.items():
            if k in self._gates:
                self._gates[k] = float(v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.base.weight, self.base.bias)
        for name in self.adapter_names:
            g = self._gates.get(name, 0.0)
            if abs(g) < 1e-8:
                continue
            delta = self.lora_B[name](self.lora_A[name](x))
            y = y + g * self.scale * delta
        return y

    def lora_parameters(self, adapter: str | None = None) -> list[nn.Parameter]:
        names = [adapter] if adapter else self.adapter_names
        params: list[nn.Parameter] = []
        for n in names:
            params.extend(list(self.lora_A[n].parameters()))
            params.extend(list(self.lora_B[n].parameters()))
        return params


def _resolve_module(root: nn.Module, dotted: str) -> tuple[nn.Module, str, nn.Module]:
    """Return (parent, attr_name, child) for a dotted path.

    Handles ModuleDict keys like ``block.block.sa`` where ``sa`` is a dict key.
    """
    parts = dotted.split(".")
    parent: nn.Module = root
    for i, part in enumerate(parts[:-1]):
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        elif isinstance(parent, nn.ModuleDict) or (
            hasattr(parent, "__contains__") and part in parent  # type: ignore[operator]
        ):
            try:
                parent = parent[part]  # type: ignore[index]
            except Exception:
                parent = getattr(parent, part)
        else:
            parent = getattr(parent, part)
    attr = parts[-1]
    if attr.isdigit():
        child = parent[int(attr)]  # type: ignore[index]
        return parent, attr, child
    if isinstance(parent, nn.ModuleDict) or (
        hasattr(parent, "__contains__") and attr in parent  # type: ignore[operator]
    ):
        try:
            child = parent[attr]  # type: ignore[index]
            return parent, attr, child
        except Exception:
            pass
    child = getattr(parent, attr)
    return parent, attr, child


def _set_module(parent: nn.Module, attr: str, value: nn.Module) -> None:
    if attr.isdigit():
        parent[int(attr)] = value  # type: ignore[index]
        return
    if isinstance(parent, nn.ModuleDict):
        parent[attr] = value
        return
    if hasattr(parent, "__contains__") and attr in parent:  # type: ignore[operator]
        try:
            parent[attr] = value  # type: ignore[index]
            return
        except Exception:
            pass
    setattr(parent, attr, value)


@dataclass
class LoRALibrary:
    """Manages LoRA branches attached to a frozen backbone."""

    adapters: list[LoRAConfig] = field(
        default_factory=lambda: [LoRAConfig(n) for n in ADAPTER_NAMES]
    )
    layers: dict[str, LoRALinear] = field(default_factory=dict)
    use_primary_only: bool = True

    @property
    def adapter_names(self) -> list[str]:
        return [a.name for a in self.adapters if a.enabled]

    def register(self, model: nn.Module, adapter_names: Sequence[str] | None = None) -> int:
        """Wrap target Linear layers in-place. Returns number of layers wrapped."""
        names = list(adapter_names) if adapter_names else self.adapter_names
        specs = PRIMARY_LORA_TARGETS if self.use_primary_only else LORA_TARGET_SPECS
        n_wrapped = 0
        for path, default_rank in specs:
            try:
                parent, attr, child = _resolve_module(model, path)
            except (AttributeError, KeyError, IndexError, TypeError):
                continue
            if isinstance(child, LoRALinear):
                self.layers[path] = child
                n_wrapped += 1
                continue
            if not isinstance(child, nn.Linear):
                continue
            rank = default_rank
            alpha = float(rank)
            wrapped = LoRALinear(child, names, rank=rank, alpha=alpha)
            _set_module(parent, attr, wrapped)
            self.layers[path] = wrapped
            n_wrapped += 1
        return n_wrapped

    def set_gates(self, gates: dict[str, float]) -> None:
        """Broadcast per-adapter scalar gates to every layer."""
        for layer in self.layers.values():
            layer.set_gates(gates)

    def set_per_layer_gates(self, gates: dict[str, dict[str, float]]) -> None:
        """gates[layer_path][adapter] = value."""
        for path, layer_gates in gates.items():
            if path in self.layers:
                self.layers[path].set_gates(layer_gates)

    def parameters(self, adapter: str | None = None) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers.values():
            params.extend(layer.lora_parameters(adapter))
        return params

    def state_dict_adapter(self, adapter: str) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for path, layer in self.layers.items():
            out[f"{path}.lora_A.{adapter}.weight"] = layer.lora_A[adapter].weight.detach().cpu()
            out[f"{path}.lora_B.{adapter}.weight"] = layer.lora_B[adapter].weight.detach().cpu()
        return out

    def load_adapter(self, adapter: str, state: dict[str, torch.Tensor], strict: bool = False) -> None:
        for path, layer in self.layers.items():
            ka = f"{path}.lora_A.{adapter}.weight"
            kb = f"{path}.lora_B.{adapter}.weight"
            if ka in state:
                layer.lora_A[adapter].weight.data.copy_(state[ka].to(layer.lora_A[adapter].weight.device))
            elif strict:
                raise KeyError(ka)
            if kb in state:
                layer.lora_B[adapter].weight.data.copy_(state[kb].to(layer.lora_B[adapter].weight.device))
            elif strict:
                raise KeyError(kb)


def freeze_base(model: nn.Module) -> int:
    """Freeze all non-LoRA parameters. Returns count frozen."""
    n = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        else:
            if param.requires_grad:
                n += 1
            param.requires_grad_(False)
    return n


def coactivation_sampler(
    active_adapter: str,
    adapter_names: Sequence[str] = ADAPTER_NAMES,
    low: float = 0.0,
    high: float = 0.2,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Sample co-activation gates: active=1.0, others ~ U(low, high)."""
    gates: dict[str, float] = {}
    for name in adapter_names:
        if name == active_adapter:
            gates[name] = 1.0
        else:
            u = torch.rand(1, generator=generator).item()
            gates[name] = low + (high - low) * float(u)
    return gates


def orthogonal_penalty(library: LoRALibrary, adapters: Sequence[str] | None = None) -> torch.Tensor:
    """O-LoRA-style penalty: encourage A matrices of different adapters to be orthogonal."""
    names = list(adapters) if adapters else library.adapter_names
    if len(names) < 2:
        return torch.tensor(0.0)
    total = None
    n_terms = 0
    for layer in library.layers.values():
        present = [n for n in names if n in layer.lora_A]
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                wa = layer.lora_A[a].weight
                wb = layer.lora_A[b].weight
                prod = wa @ wb.T
                term = (prod**2).sum()
                total = term if total is None else total + term
                n_terms += 1
    if n_terms == 0 or total is None:
        return torch.tensor(0.0)
    return total / n_terms


def register_lora(
    model: nn.Module,
    adapter_names: Sequence[str] = ADAPTER_NAMES,
    use_primary_only: bool = True,
) -> LoRALibrary:
    """Convenience: create library, register, freeze base."""
    lib = LoRALibrary(
        adapters=[LoRAConfig(n) for n in adapter_names],
        use_primary_only=use_primary_only,
    )
    lib.register(model, adapter_names)
    freeze_base(model)
    return lib


def count_lora_parameters(library: LoRALibrary, adapter: str | None = None) -> int:
    return sum(p.numel() for p in library.parameters(adapter))
