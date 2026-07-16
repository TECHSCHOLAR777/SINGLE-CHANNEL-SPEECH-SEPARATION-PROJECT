"""
Evaluation matrix runner (BLUEPRINT §9.4).

Iterates fixed_eval manifests, scores systems on each cell, and produces
aggregated result tables suitable for reports/ and RunLog ingestion.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data.synthesis.fixed_eval import load_manifest
from eval.metrics import PitResult, pit_si_sdr, score_result
from eval.stats import bootstrap_ci, mean_ci
from schemas.separation_result import SeparationResult

ScoreFn = Callable[[dict[str, Any]], SeparationResult]
"""Maps one manifest row to a SeparationResult (system under test)."""

LoadRefsFn = Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray] | None]
"""Maps one manifest row to (references [N,T], mixture [T]) or None when unavailable."""


@dataclass
class CellScore:
    """Scores for one manifest item."""

    item_id: str
    tier: str
    n_speakers: int
    n_estimated: int
    mean_si_sdri: float
    penalized_si_sdri: float
    count_correct: bool
    gate_holdout: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatrixCellSummary:
    """Aggregated scores for one (tier, N) manifest."""

    tier: str
    n_speakers: int
    system: str
    n_items: int
    mean_si_sdri: float
    mean_penalized_si_sdri: float
    count_accuracy: float
    si_sdri_ci_low: float
    si_sdri_ci_high: float
    gate_holdout: bool


def iter_manifest_items(manifest_dir: str | Path) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Yield (meta, item) pairs from every JSONL manifest in a directory.

    Skips matrix_index.json and hash sidecars.
    """
    root = Path(manifest_dir)
    for path in sorted(root.glob("*.jsonl")):
        meta, items = load_manifest(path)
        for item in items:
            yield meta, item


def score_manifest_item(
    item: dict[str, Any],
    score_fn: ScoreFn,
    load_refs: LoadRefsFn | None = None,
    *,
    missing_policy: str = "mixture_fallback",
) -> CellScore | None:
    """
    Score one manifest row through score_fn against ground-truth references.

    Returns None when references are unavailable (e.g. LibriCSS real recordings).
    """
    if not item.get("requires_reference", True):
        return None

    refs_data = load_refs(item) if load_refs is not None else None
    if refs_data is None:
        return None

    references, mixture = refs_data
    result = score_fn(item)
    pit = pit_si_sdr(
        estimates=result.streams,
        references=references,
        mixture=mixture,
        missing_policy=missing_policy,  # type: ignore[arg-type]
    )
    n_est = result.speaker_count if result.speaker_count is not None else pit.n_estimated
    return CellScore(
        item_id=item["item_id"],
        tier=item["tier"],
        n_speakers=item["n_speakers"],
        n_estimated=int(n_est),
        mean_si_sdri=pit.mean_si_sdri,
        penalized_si_sdri=pit.penalized_si_sdri,
        count_correct=int(n_est) == int(item["n_speakers"]),
        gate_holdout=bool(item.get("gate_holdout", False)),
        extra={"n_hallucinated": len(pit.unassigned_estimates)},
    )


def score_matrix(
    manifest_dir: str | Path,
    system: str,
    score_fn: ScoreFn,
    load_refs: LoadRefsFn | None = None,
    *,
    max_items: int | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> list[MatrixCellSummary]:
    """
    Score a system on every fixed_eval manifest and aggregate by (tier, N).

    Args:
        manifest_dir: Directory containing JSONL manifests.
        system: System label for the summary table.
        score_fn: Callable producing SeparationResult per manifest row.
        load_refs: Optional loader for references + mixture WAVs.
        max_items: Cap total items scored (smoke testing).
        n_bootstrap: Bootstrap resamples for confidence intervals.
        seed: RNG seed for bootstrap.

    Returns:
        List of MatrixCellSummary, one per (tier, n_speakers) cell with scores.
    """
    by_cell: dict[tuple[str, int], list[CellScore]] = {}
    n_scored = 0

    for _meta, item in iter_manifest_items(manifest_dir):
        if max_items is not None and n_scored >= max_items:
            break
        cell = score_manifest_item(item, score_fn, load_refs)
        if cell is None:
            continue
        key = (cell.tier, cell.n_speakers)
        by_cell.setdefault(key, []).append(cell)
        n_scored += 1

    summaries: list[MatrixCellSummary] = []
    for (tier, n_spk), scores in sorted(by_cell.items()):
        sdri = np.asarray([s.mean_si_sdri for s in scores], dtype=np.float64)
        penal = np.asarray([s.penalized_si_sdri for s in scores], dtype=np.float64)
        count_acc = float(np.mean([s.count_correct for s in scores]))
        ci = bootstrap_ci(sdri, n_resamples=n_bootstrap, seed=seed)
        summaries.append(
            MatrixCellSummary(
                tier=tier,
                n_speakers=n_spk,
                system=system,
                n_items=len(scores),
                mean_si_sdri=float(sdri.mean()),
                mean_penalized_si_sdri=float(penal.mean()),
                count_accuracy=count_acc,
                si_sdri_ci_low=ci.low,
                si_sdri_ci_high=ci.high,
                gate_holdout=bool(scores[0].gate_holdout),
            )
        )

    return summaries


def summaries_to_table(summaries: Sequence[MatrixCellSummary]) -> list[dict[str, Any]]:
    """Convert summaries to plain dicts for JSON/markdown export."""
    return [asdict(s) for s in summaries]


def write_matrix_results(
    summaries: Sequence[MatrixCellSummary],
    out_path: str | Path,
) -> Path:
    """Write aggregated matrix results as JSON."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"n_cells": len(summaries), "cells": summaries_to_table(summaries)}
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return out


def oracle_score_from_refs(
    references: np.ndarray,
    mixture: np.ndarray,
) -> PitResult:
    """Score oracle (references as estimates) for sanity checks."""
    return pit_si_sdr(references, references, mixture)


def score_result_row(
    result: SeparationResult,
    references: np.ndarray,
) -> PitResult:
    """Thin wrapper around score_result for matrix code paths."""
    return score_result(result, references)
