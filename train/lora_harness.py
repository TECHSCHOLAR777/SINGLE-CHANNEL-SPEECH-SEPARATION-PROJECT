"""Freeze-and-attach LoRA training harness (BLUEPRINT §8.2)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lora import (
    ADAPTER_NAMES,
    LoRALibrary,
    coactivation_sampler,
    freeze_base,
    orthogonal_penalty,
    register_lora,
)


def attach_adapters(
    model: nn.Module,
    adapter_names: tuple[str, ...] = ADAPTER_NAMES,
) -> LoRALibrary:
    """Register LoRA on target linears and freeze the backbone."""
    return register_lora(model, adapter_names=adapter_names)


def build_adapter_optimizer(
    library: LoRALibrary,
    adapter: str,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    params = library.parameters(adapter)
    if not params:
        raise RuntimeError(f"no LoRA parameters for adapter={adapter}")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def presence_target(n_speakers: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Binary presence target shape (1, 7): slots 1..N → 1."""
    t = torch.zeros(1, 7, device=device)
    n = int(max(2, min(5, n_speakers)))
    t[0, 1 : 1 + n] = 1.0
    return t


def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Single-stream negative SI-SDR (for fallback when engine losses unavailable)."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    s_energy = (ref**2).sum(dim=-1, keepdim=True) + eps
    scale = dot / s_energy
    target = scale * ref
    noise = est - target
    ratio = (target**2).sum(dim=-1) / ((noise**2).sum(dim=-1) + eps)
    return -10.0 * torch.log10(ratio + eps)


def pit_si_sdr_loss(estimates: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
    """Brute-force PIT over permutations for N≤5.

    Prefers SR-CorrNet engine ``PIT_SISNR_time`` when installed; otherwise
    falls back to a local SI-SDR PIT (CPU/GPU safe).
    """
    try:
        from sr_corrnet.models.SR_CorrNet_SS.loss import PIT_SISNR_time  # type: ignore

        criterion = PIT_SISNR_time(scale_inv=True)
        # Engine expects list of (B, L); ensure batch dim.
        est_b = [e.unsqueeze(0) if e.dim() == 1 else e for e in estimates]
        tgt_b = [t.unsqueeze(0) if t.dim() == 1 else t for t in targets]
        return criterion(est_b, tgt_b)
    except Exception:
        pass

    import itertools

    n = len(targets)
    best = None
    for perm in itertools.permutations(range(n)):
        loss = torch.stack(
            [si_sdr_loss(estimates[i], targets[perm[i]]) for i in range(n)]
        ).mean()
        if best is None or loss < best:
            best = loss
    assert best is not None
    return best


def adapter_training_step(
    *,
    library: LoRALibrary,
    active_adapter: str,
    model_forward: Any,
    mixture_wav: torch.Tensor,
    target_wavs: list[torch.Tensor],
    n_speakers: int,
    optimizer: torch.optim.Optimizer,
    use_coactivation: bool = True,
    o_lora_weight: float = 0.0,
    grad_clip: float = 5.0,
) -> dict[str, float]:
    """One optimization step for a single adapter."""
    if use_coactivation:
        gates = coactivation_sampler(active_adapter, library.adapter_names)
    else:
        gates = {n: (1.0 if n == active_adapter else 0.0) for n in library.adapter_names}
    library.set_gates(gates)

    optimizer.zero_grad(set_to_none=True)
    out = model_forward(mixture_wav, n_spks=None)
    # Expect waveforms list or tensor list
    estimates = out.get("waveforms")
    if estimates is None:
        raise RuntimeError("model_forward must return waveforms")
    est_list = [e if isinstance(e, torch.Tensor) else torch.as_tensor(e) for e in estimates]
    # Align count
    n = min(len(est_list), len(target_wavs), n_speakers)
    loss_main = pit_si_sdr_loss(est_list[:n], target_wavs[:n])

    loss_pres = torch.tensor(0.0, device=loss_main.device)
    pres = out.get("pres") or {}
    logits = None
    if isinstance(pres, dict):
        logits = pres.get("logits")
    p_k = out.get("p_k")
    if logits is not None:
        tgt = presence_target(n_speakers, device=logits.device)
        loss_pres = F.binary_cross_entropy_with_logits(logits, tgt)
    elif p_k is not None and isinstance(p_k, torch.Tensor):
        # Soft proxy when only probs available.
        tgt = presence_target(n_speakers, device=p_k.device)
        loss_pres = F.mse_loss(p_k, tgt)

    loss = loss_main + loss_pres
    if o_lora_weight > 0:
        loss = loss + o_lora_weight * orthogonal_penalty(library)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(library.parameters(active_adapter), grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "loss_main": float(loss_main.detach().cpu()),
        "loss_pres": float(loss_pres.detach().cpu()),
        **{f"gate_{k}": v for k, v in gates.items()},
    }
