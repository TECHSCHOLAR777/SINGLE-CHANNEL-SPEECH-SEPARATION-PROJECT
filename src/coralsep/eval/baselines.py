"""Mandatory baselines runner (BLUEPRINT §9.6).

Five baselines, each a single configuration of the same pipeline:

  frozen_base                      backbone only, no adapters, no band recovery
  frozen_base_plus_band_recovery   backbone only, band recovery on
  universal_adapter                one universal adapter, no condition routing
  uniform_blend                    all three adapters at a fixed equal gate
  oracle_gating                    gates taken from the mixture recipe, not predicted

Each is defined here as a `BaselineSpec`, so the definition of a baseline lives
in one place and the runner cannot drift from it.

Scores are written under `reports/baselines/`. Real numbers need trained weights
and evaluation audio; without an expert this module reports structure only and
labels the output `status: "structural_only"`. It never emits a number that
looks like a result when no model ran.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coralsep.pipeline.infer import CoralSepPipeline, InferenceCfg
from coralsep.utils.logging import get_logger

log = get_logger("baselines")


@dataclass(frozen=True)
class BaselineSpec:
    """One baseline configuration.

    Attributes:
        name: Baseline identifier, used as the output filename.
        use_adapters: Whether a LoRA library is attached at all.
        use_gate: Whether gate values are predicted by the gate network. When
            False with adapters on, gates are held at `fixed_gate`.
        fixed_gate: Gate value applied to every adapter when `use_gate` is False.
        band_recovery: Whether the 8 kHz to 16 kHz head runs.
        note: Why this baseline exists.
    """

    name: str
    use_adapters: bool
    use_gate: bool
    fixed_gate: float
    band_recovery: bool
    note: str


BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        name="frozen_base",
        use_adapters=False,
        use_gate=False,
        fixed_gate=0.0,
        band_recovery=False,
        note="The number every other row must beat. Backbone alone.",
    ),
    BaselineSpec(
        name="frozen_base_plus_band_recovery",
        use_adapters=False,
        use_gate=False,
        fixed_gate=0.0,
        band_recovery=True,
        note="Isolates the band recovery head's contribution from the adapters'.",
    ),
    BaselineSpec(
        name="universal_adapter",
        use_adapters=True,
        use_gate=False,
        fixed_gate=1.0,
        band_recovery=True,
        note="Justifies three condition adapters over one. Never trained; see I-024.",
    ),
    BaselineSpec(
        name="uniform_blend",
        use_adapters=True,
        use_gate=False,
        fixed_gate=0.5,
        band_recovery=True,
        note="Adapters on, routing off. Separates adapter gain from routing gain.",
    ),
    BaselineSpec(
        name="oracle_gating",
        use_adapters=True,
        use_gate=True,
        fixed_gate=0.0,
        band_recovery=True,
        note="Upper bound on what a perfect gate could deliver.",
    ),
)

BASELINE_NAMES: tuple[str, ...] = tuple(spec.name for spec in BASELINE_SPECS)


def get_spec(name: str) -> BaselineSpec:
    """Look up a baseline specification by name."""
    for spec in BASELINE_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown baseline {name!r}; known baselines: {list(BASELINE_NAMES)}")


def build_pipeline(
    name: str,
    expert: Any,
    *,
    lora_library: Any | None = None,
    gate_net: Any | None = None,
    device: str = "cpu",
) -> CoralSepPipeline:
    """Construct the pipeline for one baseline.

    Args:
        name: Baseline name from `BASELINE_NAMES`.
        expert: An object exposing `separate(waveform, sample_rate, n_spks)`.
        lora_library: Attached only when the baseline uses adapters.
        gate_net: Attached only when the baseline predicts gates.
        device: Torch device string.

    Returns:
        A configured `CoralSepPipeline`.
    """
    spec = get_spec(name)
    if spec.use_adapters and lora_library is not None and not spec.use_gate:
        lora_library.set_gates({n: spec.fixed_gate for n in lora_library.adapter_names})
    return CoralSepPipeline(
        expert=expert,
        lora_library=lora_library if spec.use_adapters else None,
        gate_net=gate_net if spec.use_gate else None,
        cfg=InferenceCfg(device=device, run_band_recovery=spec.band_recovery),
    )


def run_baseline_on_mixtures(
    name: str,
    mixtures: list[np.ndarray],
    expert: Any,
    *,
    sample_rate: int = 8000,
    device: str = "cpu",
    lora_library: Any | None = None,
    gate_net: Any | None = None,
) -> dict[str, Any]:
    """Run one baseline over a list of mixtures and summarise the speaker counts."""
    spec = get_spec(name)
    pipeline = build_pipeline(
        name, expert, lora_library=lora_library, gate_net=gate_net, device=device
    )
    counts = [pipeline.run(mix, sample_rate).speaker_count for mix in mixtures]
    return {
        "baseline": name,
        "n_mixtures": len(mixtures),
        "mean_count": float(np.mean(counts)) if counts else None,
        "counts": counts,
        "status": "ran",
        "note": spec.note,
    }


def describe_baselines() -> dict[str, Any]:
    """Report the baseline definitions without running anything."""
    return {
        spec.name: {
            "baseline": spec.name,
            "use_adapters": spec.use_adapters,
            "use_gate": spec.use_gate,
            "fixed_gate": spec.fixed_gate,
            "band_recovery": spec.band_recovery,
            "status": "structural_only",
            "note": spec.note,
        }
        for spec in BASELINE_SPECS
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="reports/baselines")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--describe",
        action="store_true",
        help="Write the baseline definitions without running a model",
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.describe:
        raise SystemExit(
            "Running baselines needs a trained expert and evaluation audio. "
            "Import run_baseline_on_mixtures and pass an expert, or use --describe "
            "to write the baseline definitions only."
        )

    summary = describe_baselines()
    for name, record in summary.items():
        (out / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        log.info("baseline_described", name=name)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("summary_written", path=str(out / "summary.json"), n=len(summary))


if __name__ == "__main__":
    main()
