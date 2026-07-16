"""Gate MLP for continuous LoRA mixture (BLUEPRINT §5.5)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from models.condition import ConditionVector
from models.lora import ADAPTER_NAMES, PRIMARY_LORA_TARGETS

GateMode = Literal["per_adapter", "per_layer"]


class GateMLP(nn.Module):
    """Two-hidden-layer perceptron → sigmoid × 1.5 gate values."""

    def __init__(
        self,
        in_dim: int = 9,
        hidden: int = 256,
        n_adapters: int = 3,
        n_layers: int = 17,
        mode: GateMode = "per_layer",
        max_gate: float = 1.5,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.max_gate = max_gate
        self.enabled = enabled
        self.n_adapters = n_adapters
        self.n_layers = n_layers
        out_dim = n_adapters if mode == "per_adapter" else n_adapters * n_layers
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.adapter_names = list(ADAPTER_NAMES)[:n_adapters]
        self.layer_paths = [p for p, _ in PRIMARY_LORA_TARGETS[:n_layers]]
        self._ema: dict[str, float] | None = None
        self.ema_coeff = 0.7
        self.sparsity_weight = 1e-3

    def raw_forward(self, c: torch.Tensor) -> torch.Tensor:
        if c.dim() == 1:
            c = c.unsqueeze(0)
        return torch.sigmoid(self.net(c)) * self.max_gate

    def forward(
        self,
        condition: ConditionVector | torch.Tensor,
        apply_ema: bool = True,
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        """Return gates. per_adapter → {name: g}; per_layer → {path: {name: g}}."""
        if not self.enabled:
            if self.mode == "per_adapter":
                return {n: 0.0 for n in self.adapter_names}
            return {p: {n: 0.0 for n in self.adapter_names} for p in self.layer_paths}

        if isinstance(condition, ConditionVector):
            c = condition.to_tensor()
        else:
            c = condition
        gates_t = self.raw_forward(c)[0]

        if self.mode == "per_adapter":
            raw = {n: float(gates_t[i].item()) for i, n in enumerate(self.adapter_names)}
            if apply_ema:
                raw = self._smooth(raw)
            return raw

        out: dict[str, dict[str, float]] = {}
        flat_idx = 0
        flat_for_ema: dict[str, float] = {}
        for path in self.layer_paths:
            layer_g: dict[str, float] = {}
            for name in self.adapter_names:
                v = float(gates_t[flat_idx].item())
                layer_g[name] = v
                flat_for_ema[f"{path}::{name}"] = v
                flat_idx += 1
            out[path] = layer_g
        if apply_ema:
            smoothed = self._smooth(flat_for_ema)
            for path in self.layer_paths:
                for name in self.adapter_names:
                    out[path][name] = smoothed[f"{path}::{name}"]
        return out

    def _smooth(self, gates: dict[str, float]) -> dict[str, float]:
        if self._ema is None:
            self._ema = dict(gates)
            return dict(gates)
        a = self.ema_coeff
        for k, v in gates.items():
            prev = self._ema.get(k, v)
            self._ema[k] = a * prev + (1.0 - a) * v
        return dict(self._ema)

    def reset_ema(self) -> None:
        self._ema = None

    def sparsity_loss(self, gates_tensor: torch.Tensor) -> torch.Tensor:
        return self.sparsity_weight * gates_tensor.abs().mean()

    def as_adapter_scalars(
        self, gates: dict[str, float] | dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Collapse per-layer gates to per-adapter means for LoRALibrary.set_gates."""
        if not gates:
            return {n: 0.0 for n in self.adapter_names}
        first = next(iter(gates.values()))
        if isinstance(first, dict):
            sums = {n: 0.0 for n in self.adapter_names}
            for layer_g in gates.values():  # type: ignore[union-attr]
                for n in self.adapter_names:
                    sums[n] += float(layer_g[n])  # type: ignore[index]
            n_layers = max(len(gates), 1)
            return {n: sums[n] / n_layers for n in self.adapter_names}
        return {n: float(gates[n]) for n in self.adapter_names}  # type: ignore[index]
