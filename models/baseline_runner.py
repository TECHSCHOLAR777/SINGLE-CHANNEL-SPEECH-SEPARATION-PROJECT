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
from eval.metrics import pit_si_sdr
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
    Compute mean SI-SDRi in dB with permutation-invariant speaker matching.

    Delegates to the canonical eval.metrics.pit_si_sdr implementation.
    """
    result = pit_si_sdr(estimates, references, mixture)
    return result.mean_si_sdri


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
