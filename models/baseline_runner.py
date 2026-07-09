"""
Phase 0 baseline runner.

Loads pretrained SepFormer and SR-CorrNet on Libri3Mix test clips,
computes SI-SDRi with permutation-invariant matching, and writes a
baseline results table for milestone M0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from data.mixer_stub import MixtureSample, discover_librimix_samples
from models.experts.sepformer import SepFormerExpert
from models.experts.srcorrnet import SRCorrNetExpert
from schemas.separation_result import SeparationResult


@dataclass
class BaselineConfig:
    """Configuration for a baseline evaluation run."""

    data_root: str
    subset: str = "test"
    max_samples: int | None = 50
    device: str = "cuda"
    output_dir: str = "outputs/baseline"
    experts: list[str] = field(default_factory=lambda: ["sepformer", "srcorrnet"])
    srcorrnet_repo: str | None = None
    srcorrnet_checkpoint: str | None = None
    sample_rate: int = 16000

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> BaselineConfig:
        return cls(
            data_root=cfg["data_root"],
            subset=cfg.get("subset", "test"),
            max_samples=cfg.get("max_samples"),
            device=cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            output_dir=cfg.get("output_dir", "outputs/baseline"),
            experts=cfg.get("experts", ["sepformer", "srcorrnet"]),
            srcorrnet_repo=cfg.get("srcorrnet_repo"),
            srcorrnet_checkpoint=cfg.get("srcorrnet_checkpoint"),
            sample_rate=cfg.get("sample_rate", 16000),
        )


@dataclass
class ExpertBaselineResult:
    """Aggregated SI-SDRi for one expert across all test samples."""

    expert: str
    mean_sisdri_db: float
    std_sisdri_db: float
    num_samples: int
    per_sample_sisdri: list[float] = field(default_factory=list)


def compute_sisdri(
    estimates: np.ndarray,
    references: np.ndarray,
    mixture: np.ndarray,
) -> float:
    """
    Compute SI-SDRi in dB with permutation-invariant speaker matching.

    Uses Asteroid PIT when available; falls back to numpy permutation search.
    """
    try:
        return _sisdri_asteroid(estimates, references, mixture)
    except (ImportError, OSError, RuntimeError):
        return _sisdri_numpy(estimates, references, mixture)


def _sisdri_asteroid(
    estimates: np.ndarray,
    references: np.ndarray,
    mixture: np.ndarray,
) -> float:
    from asteroid.losses import PITLossWrapper
    from asteroid.losses.sdr import pairwise_neg_sisdr

    ref_t = torch.from_numpy(references).float().unsqueeze(0)
    est_t = torch.from_numpy(estimates).float().unsqueeze(0)
    mix_t = torch.from_numpy(mixture).float().unsqueeze(0)

    n_ref = ref_t.shape[1]
    n_est = est_t.shape[1]
    if n_est < n_ref:
        pad = torch.zeros(1, n_ref - n_est, est_t.shape[2])
        est_t = torch.cat([est_t, pad], dim=1)
    elif n_est > n_ref:
        est_t = est_t[:, :n_ref, :]

    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    with torch.no_grad():
        sisdr = -loss_func(est_t, ref_t).item()
        mix_rep = mix_t.expand_as(ref_t)
        sisdr_mix = -loss_func(mix_rep, ref_t).item()

    return float(sisdr - sisdr_mix)


def _sisdri_numpy(estimates: np.ndarray, references: np.ndarray, mixture: np.ndarray) -> float:
    """SI-SDRi via exhaustive permutation search (up to 5 speakers)."""
    from itertools import permutations

    n_ref = references.shape[0]
    n_est = estimates.shape[0]
    k = min(n_ref, n_est)

    best_sisdr = -np.inf
    for perm in permutations(range(n_est), k):
        sisdr_vals = [_sisdr_single(estimates[j], references[i]) for i, j in enumerate(perm)]
        best_sisdr = max(best_sisdr, float(np.mean(sisdr_vals)))

    mix_vals = [_sisdr_single(mixture, references[i]) for i in range(k)]
    sisdr_mix = float(np.mean(mix_vals))
    return best_sisdr - sisdr_mix


def _sisdr_single(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR for one source pair."""
    eps = 1e-8
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)
    dot = np.sum(ref * est)
    ref_energy = np.sum(ref**2) + eps
    proj = (dot / ref_energy) * ref
    noise = est - proj
    return float(10 * np.log10((np.sum(proj**2) + eps) / (np.sum(noise**2) + eps)))


def evaluate_expert_on_sample(
    expert_name: str,
    expert: SepFormerExpert | SRCorrNetExpert,
    sample: MixtureSample,
) -> float:
    """Run one expert on one mixture and return SI-SDRi."""
    result: SeparationResult = expert.separate(sample.mixture, sample.sample_rate)
    return compute_sisdri(result.streams, sample.references, sample.mixture)


def run_baseline(config: BaselineConfig) -> dict[str, ExpertBaselineResult]:
    """
    Run baseline evaluation for all configured experts.

    Returns:
        Mapping from expert name to aggregated results.
    """
    samples = discover_librimix_samples(
        config.data_root,
        subset=config.subset,
        max_samples=config.max_samples,
    )
    if not samples:
        raise RuntimeError(f"No samples found under {config.data_root}")

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    experts: dict[str, SepFormerExpert | SRCorrNetExpert] = {}
    if "sepformer" in config.experts:
        experts["sepformer"] = SepFormerExpert(device=device)
    if "srcorrnet" in config.experts:
        sr = SRCorrNetExpert(
            device=device,
            repo_path=config.srcorrnet_repo,
            checkpoint_path=config.srcorrnet_checkpoint,
        )
        if sr.is_available:
            experts["srcorrnet"] = sr
        else:
            print(
                "WARNING: SR-CorrNet not configured — skipping. "
                "Set srcorrnet_repo and srcorrnet_checkpoint in config."
            )

    results: dict[str, ExpertBaselineResult] = {}
    for name, expert in experts.items():
        sisdris: list[float] = []
        for sample in tqdm(samples, desc=f"Baseline [{name}]"):
            sisdris.append(evaluate_expert_on_sample(name, expert, sample))

        arr = np.array(sisdris)
        results[name] = ExpertBaselineResult(
            expert=name,
            mean_sisdri_db=float(arr.mean()),
            std_sisdri_db=float(arr.std()),
            num_samples=len(sisdris),
            per_sample_sisdri=sisdris,
        )

    _write_results(config, results)
    return results


def _write_results(config: BaselineConfig, results: dict[str, ExpertBaselineResult]) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = {
        name: {
            "mean_sisdri_db": r.mean_sisdri_db,
            "std_sisdri_db": r.std_sisdri_db,
            "num_samples": r.num_samples,
        }
        for name, r in results.items()
    }

    json_path = out_dir / "baseline_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    md_path = out_dir / "baseline_results.md"
    lines = [
        "# Phase 0 Baseline Results (Libri3Mix)",
        "",
        f"- Data root: `{config.data_root}`",
        f"- Subset: `{config.subset}`",
        f"- Samples: {results[next(iter(results))].num_samples if results else 0}",
        "",
        "| Expert | Mean SI-SDRi (dB) | Std (dB) | N |",
        "|--------|-------------------|----------|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r.mean_sisdri_db:.2f} | {r.std_sisdri_db:.2f} | {r.num_samples} |"
        )
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nBaseline results written to {json_path} and {md_path}")
    for name, r in results.items():
        print(
            f"  {name}: {r.mean_sisdri_db:.2f} +/- {r.std_sisdri_db:.2f} dB SI-SDRi "
            f"({r.num_samples} samples)"
        )
