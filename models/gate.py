"""
Gate network for CALM-Sep adapter routing (Dev B, P2-B3).

Architecture: MLP with two hidden layers of 256, GELU, sigmoid scaled to 1.5.
Input: concatenation of Level-1 features (4-D) + Level-2 features (6-D) = 10-D.
Output: 3-D gate vector, one scalar per adapter (reverb, noise, codec).

Regularisation:
  L1 sparsity (lambda=1e-3): encourages the gate to select exactly the adapters
  that are needed, not hedging across all three.
  EMA smoothing (alpha=0.7): applied to the gate output during inference to
  prevent per-chunk flickering on long recordings.

Supervision (Stage 3, BLUEPRINT §6.1): the gate is trained jointly with the
Level-2 analyzer on the SAME free labels from MixtureRecipe.condition_vector().
The "oracle gate" is derived directly from the recipe: reverb adapter is on iff
t60_s > 0, noise adapter iff snr_db < 60.0, codec adapter iff codec_class > 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lora import ADAPTER_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_L1_LAMBDA: float = 1e-3
_GATE_SCALE: float = 1.5   # sigmoid * scale, output range [0, 1.5]
_EMA_ALPHA: float = 0.7    # EMA smoothing coefficient for streaming inference


# ---------------------------------------------------------------------------
# Gate MLP
# ---------------------------------------------------------------------------


class GateNetwork(nn.Module):
    """
    Routes the 10-D condition vector to 3 adapter gate values in [0, 1.5].

    Input:  cat(level1_feats [4], level2_feats [6]) = [10]
    Hidden: Linear(10, 256) → GELU → Linear(256, 256) → GELU
    Output: Linear(256, 3) → sigmoid × 1.5

    Attributes:
        adapter_names: Names of the 3 adapters in gate-vector order.
    """

    def __init__(
        self,
        in_features: int = 10,
        hidden: int = 256,
        n_adapters: int = 3,
        gate_scale: float = _GATE_SCALE,
    ) -> None:
        super().__init__()
        self.adapter_names = list(ADAPTER_NAMES)
        self.gate_scale = gate_scale

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_adapters),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: (B, 10) concatenated Level-1 + Level-2 features.
        Returns:
            (B, 3) gate values in [0, gate_scale].
        """
        return torch.sigmoid(self.net(condition)) * self.gate_scale

    def gate_dict(self, condition: torch.Tensor) -> dict[str, float]:
        """
        Compute and return a {adapter_name: gate_value} dict for a single sample.

        Args:
            condition: (1, 10) or (10,) condition vector.
        Returns:
            Dict mapping each adapter name to its gate value.
        """
        g = self.forward(condition.unsqueeze(0) if condition.ndim == 1 else condition)
        return {name: float(g[0, i].item()) for i, name in enumerate(self.adapter_names)}

    def l1_penalty(self, gates: torch.Tensor) -> torch.Tensor:
        """L1 sparsity penalty on the gate vector (BLUEPRINT §6.1)."""
        return _L1_LAMBDA * gates.abs().mean()


# ---------------------------------------------------------------------------
# Oracle gate (for supervised Stage-3 training)
# ---------------------------------------------------------------------------


def oracle_gate(recipe_vectors: list[dict[str, float]], device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Derive the oracle gate from ground-truth condition labels.

    The oracle is a hard binary gate:
      reverb = 1 iff t60_s > 0.0
      noise  = 1 iff snr_db < 60.0  (60 dB ≡ "no noise" in the mixer)
      codec  = 1 iff codec_class > 0

    Args:
        recipe_vectors: List of B dicts from MixtureRecipe.condition_vector().
    Returns:
        (B, 3) float32 tensor with values in {0.0, 1.0}.
    """
    rows: list[list[float]] = []
    for rv in recipe_vectors:
        reverb = float(rv.get("t60_s", 0.0) > 0.0)
        noise = float(rv.get("snr_db", 60.0) < 60.0)
        codec = float(rv.get("codec_class", 0.0) > 0.0)
        rows.append([reverb, noise, codec])
    return torch.tensor(rows, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Gate training loss
# ---------------------------------------------------------------------------


def gate_loss(
    gate_net: GateNetwork,
    condition: torch.Tensor,
    recipe_vectors: list[dict[str, float]],
    sep_loss: torch.Tensor,
) -> torch.Tensor:
    """
    Total gate training loss = sep_loss + BCE supervision + L1 sparsity.

    BLUEPRINT §6.1: the gate is trained jointly with the separation loss.
    The BCE term teaches the gate to reproduce the oracle given the condition.
    The L1 term pushes sparse, clean routing rather than diffuse hedging.

    Args:
        gate_net: GateNetwork module.
        condition: (B, 10) condition feature vector.
        recipe_vectors: Ground-truth recipe labels (B dicts).
        sep_loss: Scalar separation loss from the upstream model.

    Returns:
        Scalar total loss.
    """
    gates = gate_net(condition)  # (B, 3)
    oracle = oracle_gate(recipe_vectors, device=condition.device)

    bce = F.binary_cross_entropy(
        gates / gate_net.gate_scale,  # normalise to [0,1] for BCE
        oracle,
    )
    l1 = gate_net.l1_penalty(gates)
    return sep_loss + bce + l1


# ---------------------------------------------------------------------------
# EMA smoother (streaming inference)
# ---------------------------------------------------------------------------


class GateSmoother:
    """
    Exponential moving average of gate values across consecutive chunks.

    Prevents per-chunk flickering on long recordings. BLUEPRINT §6.3.
    """

    def __init__(self, alpha: float = _EMA_ALPHA) -> None:
        self.alpha = alpha
        self._state: dict[str, float] | None = None

    def smooth(self, gates: dict[str, float]) -> dict[str, float]:
        """Update EMA and return smoothed gate dict."""
        if self._state is None:
            self._state = dict(gates)
            return dict(gates)
        result: dict[str, float] = {}
        for k in gates:
            smoothed = self.alpha * self._state.get(k, gates[k]) + (1.0 - self.alpha) * gates[k]
            result[k] = smoothed
            self._state[k] = smoothed
        return result

    def reset(self) -> None:
        """Reset state at the start of a new recording."""
        self._state = None
