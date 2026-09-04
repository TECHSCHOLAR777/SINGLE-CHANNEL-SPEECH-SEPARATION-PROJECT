"""
Parallel-branch LoRA for CoRAL-Sep signal adapters (Dev B, P1-B1).

Architecture: y = W0 x + sum_i( g_i * B_i(A_i x) )

Three adapters share attachment points on the same frozen base model:
  adapter_reverb, adapter_noise, adapter_codec.
All three use the same rank schedule:
  rank 8  on attention projections (QKV fused Linear(128,384), output Linear(128,128))
  rank 4  on filter head (Linear(128,27))

Co-activation warm-up (BLUEPRINT §5.4): during single-adapter Stage 1 training,
the other two adapters are randomly active with gate drawn from Uniform(0.0, 0.2).
This prevents composition failures when all three run together in Stage 4.

Target modules (37 per adapter from BLUEPRINT §5.3):
  enc_block[0,1]  × {freq,time} × {qkv, agg}   →  8 modules  (rank 8)
  dec_block[0-3]  × {freq,time} × {qkv, agg}   → 16 modules  (rank 8)
  dec_cs[0-3]     ×              {qkv, agg}     →  8 modules  (rank 8)
  filter_estim.mask.net                          →  1 module   (rank 4)
  filter_estim_aux[0-3].mask.net                 →  4 modules  (rank 4)
  ─────────────────────────────────────────────────────────────────────
  Total                                             37 modules
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Core LoRA layer
# ---------------------------------------------------------------------------


class LoRALayer(nn.Module):
    """
    One LoRA branch: B(Ax) where A: in→r, B: r→out.

    The gate g is NOT stored here; it is held by LoRALibrary and injected at
    forward time so the gate can vary between samples (co-activation, Stage 4).

    Initialisation: A ~ N(0, 1/sqrt(r)), B = 0, so the branch contributes
    zero at init and the base's pretrained behaviour is preserved.
    """

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.T) @ self.B.T


# ---------------------------------------------------------------------------
# LoRA-wrapped Linear layer
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """
    Replaces a frozen base Linear with a sum of the base output and N LoRA branches.

    y = W0 x + sum_i( g_i * B_i(A_i x) )

    The base weight W0 is registered as a frozen buffer (not a parameter) after
    the Linear is replaced. `gates` is a 1-D tensor [N_adapters] injected by the
    caller; each scalar scales one adapter's branch.
    """

    def __init__(
        self,
        base: nn.Linear,
        adapter_names: list[str],
        rank: int,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.adapter_names = list(adapter_names)

        # Freeze and store the base weight + bias.
        self.register_buffer("weight", base.weight.data.clone())
        self.bias = nn.Parameter(base.bias.data.clone()) if base.bias is not None else None
        # Prevent base weight from being a param.
        # (It's already a buffer, so no grad by default.)

        # One LoRA branch per adapter.
        self.branches = nn.ModuleDict(
            {name: LoRALayer(base.in_features, base.out_features, rank) for name in adapter_names}
        )

    def forward(self, x: torch.Tensor, gates: dict | None = None) -> torch.Tensor:
        y = nn.functional.linear(x, self.weight, self.bias)
        if gates:
            for name, branch in self.branches.items():
                g = gates.get(name, 0.0)
                # Allow tensor gates (Stage 4 differentiable routing).
                if isinstance(g, torch.Tensor) or g != 0.0:
                    y = y + g * branch(x)
        return y

    def adapter_parameters(self, name: str) -> list[nn.Parameter]:
        return list(self.branches[name].parameters())


# ---------------------------------------------------------------------------
# LoRA library, manages attachment and gate injection
# ---------------------------------------------------------------------------

ADAPTER_NAMES: tuple[str, ...] = ("reverb", "noise", "codec")
"""Canonical adapter names. Order determines gate vector indexing."""

# Rank schedule from BLUEPRINT §5.3
_ATTN_RANK = 8
_FILTER_RANK = 4


def _resolve_module(root: nn.Module, path: str) -> nn.Module | None:
    """Walk a dot-separated attribute path; return None if any step is missing."""
    obj: object = root
    for part in path.split("."):
        if part.startswith("[") and part.endswith("]"):
            # list index access like [0]
            idx = int(part[1:-1])
            try:
                obj = obj[idx]  # type: ignore[index]
            except (IndexError, TypeError):
                return None
        elif part.startswith("'") and part.endswith("'"):
            # ModuleDict key access like ['sa']
            key = part[1:-1]
            try:
                obj = obj[key]  # type: ignore[index]
            except KeyError:
                return None
        else:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
    return obj  # type: ignore[return-value]


def _set_module(root: nn.Module, path: str, new_module: nn.Module) -> bool:
    """Replace the module at `path` with `new_module`. Returns False if path not found."""
    parts = path.rsplit(".", 1)
    if len(parts) == 1:
        parent_path, attr = "", parts[0]
        parent = root
    else:
        parent_path, attr = parts
        parent = _resolve_module(root, parent_path)  # type: ignore[assignment]
        if parent is None:
            return False

    if attr.startswith("[") and attr.endswith("]"):
        idx = int(attr[1:-1])
        try:
            parent[idx] = new_module  # type: ignore[index]
            return True
        except (IndexError, TypeError):
            return False
    setattr(parent, attr, new_module)
    return True


def _target_paths(
    model: nn.Module,
    attn_rank: int = _ATTN_RANK,
    filter_rank: int = _FILTER_RANK,
) -> list[tuple[str, int]]:
    """
    Return (dot-path, rank) for every LoRA target in `model`.

    Paths follow BLUEPRINT §5.3 exactly. Missing paths are skipped gracefully
    so the function works even on partially-initialised models.

    attn_rank and filter_rank default to the BLUEPRINT schedule (8 and 4) and
    exist as parameters, not just module constants, so a rank ablation
    (I-025's "rank 8 may be too small" candidate) can attach a library at a
    different rank without editing this file. Both apply uniformly; the
    original schedule using two different fixed ranks for attention versus
    filter heads is preserved, only their absolute values are overridable.
    """
    paths: list[tuple[str, int]] = []

    # Encoder blocks (N_Enc = 2)
    for i in range(2):
        for branch in ("freq_block", "time_block"):
            base = f"enc_block.{i}.{branch}.block.sa.block"
            paths.append((f"{base}.qkv", attn_rank))
            paths.append((f"{base}.aggregate_heads.0", attn_rank))

    # Decoder blocks (N_Dec = 4)
    for i in range(4):
        for branch in ("freq_block", "time_block"):
            base = f"dec_block.{i}.{branch}.block.sa.block"
            paths.append((f"{base}.qkv", attn_rank))
            paths.append((f"{base}.aggregate_heads.0", attn_rank))

    # Decoder cross-attention blocks (N_Dec = 4, ModuleDict key 'sa')
    for i in range(4):
        base = f"dec_cs.{i}.block.block.sa.block"
        paths.append((f"{base}.qkv", attn_rank))
        paths.append((f"{base}.aggregate_heads.0", attn_rank))

    # Filter estimation heads
    paths.append(("filter_estim.mask.net", filter_rank))
    for i in range(4):
        paths.append((f"filter_estim_aux.{i}.mask.net", filter_rank))

    return paths


class LoRALibrary:
    """
    Attaches LoRA branches to a frozen model and manages per-forward gates.

    Usage
    -----
    lib = LoRALibrary(model, adapter_names=ADAPTER_NAMES)
    lib.freeze_base()

    # Stage 1: train one adapter with co-activation warm-up
    lib.set_adapter("reverb")
    optimizer = torch.optim.Adam(lib.active_parameters(), lr=1e-4)

    # Forward pass (gates are set automatically by set_adapter + co_activation)
    with lib.forward_gates():
        out = model(...)
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_names: Sequence[str] = ADAPTER_NAMES,
        co_activation_range: tuple[float, float] = (0.0, 0.2),
        rng: torch.Generator | None = None,
        attn_rank: int = _ATTN_RANK,
        filter_rank: int = _FILTER_RANK,
    ) -> None:
        self.model = model
        self.adapter_names = list(adapter_names)
        self.co_lo, self.co_hi = co_activation_range
        self.rng = rng
        self.attn_rank = attn_rank
        self.filter_rank = filter_rank

        # Gate values for the current forward pass.
        self._gates: dict[str, float] = {n: 0.0 for n in self.adapter_names}
        self._active_adapter: str | None = None
        self._n_attached = 0

        self._attach()

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------

    def _attach(self) -> None:
        """Replace every target Linear with a LoRALinear in-place."""
        targets = _target_paths(self.model, self.attn_rank, self.filter_rank)
        attached = 0
        for path, rank in targets:
            mod = _resolve_module(self.model, path)
            if mod is None or not isinstance(mod, nn.Linear):
                continue
            lora_lin = LoRALinear(mod, self.adapter_names, rank)
            if _set_module(self.model, path, lora_lin):
                attached += 1
        self._n_attached = attached

    @property
    def n_attached(self) -> int:
        return self._n_attached

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def freeze_base(self) -> None:
        """Freeze all parameters that are NOT LoRA branches."""
        for mod in self.model.modules():
            if isinstance(mod, LoRALinear):
                if mod.bias is not None:
                    mod.bias.requires_grad_(False)
                for branch in mod.branches.values():
                    for p in branch.parameters():
                        p.requires_grad_(True)
            elif isinstance(mod, LoRALayer):
                # Depth-first: LoRALayer is a child of LoRALinear.branches and
                # its params were just set trainable above, so leave them alone.
                pass
            else:
                for p in mod.parameters(recurse=False):
                    p.requires_grad_(False)

    def adapter_parameters(self, name: str) -> list[nn.Parameter]:
        """Return all parameters belonging to adapter `name`."""
        params: list[nn.Parameter] = []
        for mod in self.model.modules():
            if isinstance(mod, LoRALinear) and name in mod.branches:
                params.extend(mod.branches[name].parameters())
        return params

    def active_parameters(self) -> list[nn.Parameter]:
        """Return only the active (requires_grad=True) adapter parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def param_count(self, name: str) -> int:
        return sum(p.numel() for p in self.adapter_parameters(name))

    # ------------------------------------------------------------------
    # Gate control
    # ------------------------------------------------------------------

    def set_adapter(
        self,
        name: str,
        co_activate: bool = True,
    ) -> None:
        """
        Set one adapter as the primary (gate=1.0) for the next forward pass.

        With co_activate=True (Stage 1 warm-up), the other adapters are set to
        a random gate in [co_lo, co_hi] rather than 0.0, so the model learns to
        compose from the first epoch. BLUEPRINT §5.4.

        Raises:
            TypeError: if `name` is not a string. Passing a gate dict here is a
                silent-corruption bug rather than a crash: no adapter name would
                compare equal to the dict, so every adapter would fall through to
                the co-activation branch and receive a random gate. Use
                `set_gates` for a full gate vector.
            KeyError: if `name` is not a known adapter.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"set_adapter expects an adapter name, got {type(name).__name__}. "
                "To apply a full gate vector, call set_gates(gates) instead."
            )
        if name not in self.adapter_names:
            raise KeyError(f"unknown adapter {name!r}; known adapters: {list(self.adapter_names)}")
        self._active_adapter = name
        for n in self.adapter_names:
            if n == name:
                self._gates[n] = 1.0
            elif co_activate:
                g = float(
                    torch.zeros(1).uniform_(self.co_lo, self.co_hi, generator=self.rng).item()
                )
                self._gates[n] = g
            else:
                self._gates[n] = 0.0

    def set_gates(self, gates: dict[str, float]) -> None:
        """
        Directly set gate values (Stage 4 joint training and gated inference).

        Raises:
            TypeError: if `gates` is not a mapping.
            KeyError: if `gates` names an adapter this library does not manage.
        """
        if not isinstance(gates, Mapping):
            raise TypeError(
                f"set_gates expects a mapping of adapter name to gate, got {type(gates).__name__}"
            )
        unknown = set(gates) - set(self.adapter_names)
        if unknown:
            raise KeyError(
                f"unknown adapters {sorted(unknown)}; known adapters: {list(self.adapter_names)}"
            )
        self._gates = dict(gates)
        self._active_adapter = None

    def gate_dict(self) -> dict[str, float]:
        return dict(self._gates)

    def inject_gates(self) -> None:
        """
        Push current gate values into every LoRALinear in the model.

        Must be called before every forward pass. The standard pattern is to
        override the model's forward method; here we patch the LoRALinear's
        forward to close over the gates dict instead.

        Implementation: we store the gates dict as an attribute on LoRALinear so
        its forward() reads them. This avoids re-writing the frozen base's forward.
        """
        for mod in self.model.modules():
            if isinstance(mod, LoRALinear):
                mod._injected_gates = dict(self._gates)

    # ------------------------------------------------------------------
    # Context manager for safe gate injection
    # ------------------------------------------------------------------

    def forward_context(self, name: str | None = None, co_activate: bool = True):
        """Context manager: inject gates before block, clear after."""
        return _GateContext(self, name, co_activate)


class _GateContext:
    def __init__(self, lib: LoRALibrary, name: str | None, co_activate: bool) -> None:
        self.lib = lib
        self.name = name
        self.co_activate = co_activate

    def __enter__(self) -> LoRALibrary:
        if self.name is not None:
            self.lib.set_adapter(self.name, self.co_activate)
        self.lib.inject_gates()
        return self.lib

    def __exit__(self, *_: object) -> None:
        # Clear injected gates to avoid stale values.
        for mod in self.lib.model.modules():
            if isinstance(mod, LoRALinear):
                mod._injected_gates = {}


# ---------------------------------------------------------------------------
# Patch LoRALinear.forward to read injected gates
# ---------------------------------------------------------------------------

_orig_lora_linear_forward = LoRALinear.forward


def _patched_forward(
    self: LoRALinear, x: torch.Tensor, gates: dict[str, float] | None = None
) -> torch.Tensor:
    # Prefer explicitly passed gates; fall back to injected gates.
    g = gates if gates is not None else getattr(self, "_injected_gates", {})
    return _orig_lora_linear_forward(self, x, g)


LoRALinear.forward = _patched_forward  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# O-LoRA orthogonality penalty (BLUEPRINT §7.2 / P1-C2)
# ---------------------------------------------------------------------------


def olora_penalty(model: nn.Module, alpha: float = 1e-3) -> torch.Tensor:
    """
    Penalise A-matrix overlap between adapters on the same layer.

    For each LoRALinear with ≥ 2 adapters, add alpha * ||A_i A_j^T||_F^2
    summed over all pairs (i, j). This encourages each adapter to use a
    different subspace of the input, reducing cross-interference.

    Returns a scalar tensor (0.0 if only one adapter per layer).
    """
    loss: torch.Tensor | None = None
    names = None
    for mod in model.modules():
        if not isinstance(mod, LoRALinear):
            continue
        if names is None:
            names = list(mod.branches.keys())
        As = [mod.branches[n].A for n in names if n in mod.branches]
        for i in range(len(As)):
            for j in range(i + 1, len(As)):
                overlap = As[i] @ As[j].T  # [r, r]
                term = alpha * overlap.pow(2).sum()
                loss = term if loss is None else loss + term
    return loss if loss is not None else torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Param-count summary
# ---------------------------------------------------------------------------


def lora_summary(model: nn.Module, adapter_names: Sequence[str] = ADAPTER_NAMES) -> dict[str, int]:
    """Return {adapter_name: param_count} for every adapter in the model."""
    counts: dict[str, int] = {n: 0 for n in adapter_names}
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            for n in adapter_names:
                if n in mod.branches:
                    counts[n] += sum(p.numel() for p in mod.branches[n].parameters())
    return counts
