#!/usr/bin/env python3
"""Run P1-INT1 and P1-INT2 acceptance checks on one real Libri3Mix clip.

Example:

    python scripts/validate_alignment.py \
      --librimix-root /workspace/Libri3Mix \
      --device cuda --output-dir outputs/p1_alignment --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from align.hungarian import xcorr_cost_matrix  # noqa: E402
from align.integration import run_and_align, run_and_align_long  # noqa: E402
from data.mixer_stub import MixtureSample, discover_librimix_samples  # noqa: E402
from models.experts.mossformer2 import MossFormer2Expert  # noqa: E402
from models.experts.tfgridnet import get_expensive_expert  # noqa: E402


def _assignment_trace(
    tracks: np.ndarray,
    references: np.ndarray,
    sample_rate: int,
    chunk_sec: float,
    overlap_sec: float,
) -> list[dict[str, Any]]:
    chunk = int(round(chunk_sec * sample_rate))
    hop = int(round((chunk_sec - overlap_sec) * sample_rate))
    trace: list[dict[str, Any]] = []
    for start in range(0, tracks.shape[1], hop):
        stop = min(start + chunk, tracks.shape[1])
        if stop - start < sample_rate:
            break
        cost = xcorr_cost_matrix(tracks[:, start:stop], references[:, start:stop])
        rows, cols = linear_sum_assignment(cost)
        mapping = [-1] * tracks.shape[0]
        matched_costs: list[float] = []
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            mapping[row] = col
            matched_costs.append(float(cost[row, col]))
        trace.append(
            {
                "start_sec": start / sample_rate,
                "stop_sec": stop / sample_rate,
                "track_to_reference": mapping,
                "mean_xcorr_cost": float(np.mean(matched_costs)),
            }
        )
        if stop == tracks.shape[1]:
            break
    return trace


def _identity_switches(trace: list[dict[str, Any]]) -> int:
    if len(trace) < 2:
        return 0
    baseline = tuple(int(value) for value in trace[0]["track_to_reference"])
    return sum(
        tuple(int(value) for value in row["track_to_reference"]) != baseline for row in trace[1:]
    )


def _write_streams(root: Path, prefix: str, streams: np.ndarray, sample_rate: int) -> None:
    for index, stream in enumerate(streams, start=1):
        sf.write(root / f"{prefix}_s{index}.wav", stream, sample_rate)


def _select_sample(root: str, index: int) -> MixtureSample:
    samples = discover_librimix_samples(root, subset="test", max_samples=index + 1)
    if len(samples) <= index:
        raise RuntimeError(f"requested sample index {index}, only found {len(samples)} samples")
    sample = samples[index]
    if sample.mixture.shape[0] <= 4 * sample.sample_rate:
        duration = sample.mixture.shape[0] / sample.sample_rate
        raise RuntimeError(
            f"sample {sample.utterance_id} is only {duration:.2f}s; "
            "P1-INT2 requires a clip longer than four seconds"
        )
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librimix-root", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/p1_alignment")
    parser.add_argument("--srcorrnet-repo")
    parser.add_argument("--srcorrnet-checkpoint")
    parser.add_argument("--tfgridnet-tag")
    parser.add_argument("--chunk-sec", type=float, default=4.0)
    parser.add_argument("--overlap-sec", type=float, default=1.0)
    parser.add_argument("--match-threshold", type=float, default=0.35)
    parser.add_argument("--ema", type=float, default=0.7)
    parser.add_argument(
        "--skip-pair",
        action="store_true",
        help="Skip P1-INT1 and run only the long-form P1-INT2 check.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless every requested acceptance check passes.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = _select_sample(args.librimix_root, args.sample_index)
    cheap = MossFormer2Expert(device=args.device, compute_embeddings=True)

    report: dict[str, Any] = {
        "utterance_id": sample.utterance_id,
        "duration_sec": sample.mixture.shape[0] / sample.sample_rate,
        "sample_rate": sample.sample_rate,
    }

    if not args.skip_pair:
        expensive = get_expensive_expert(
            device=args.device,
            srcorrnet_repo=args.srcorrnet_repo,
            srcorrnet_checkpoint=args.srcorrnet_checkpoint,
            tfgridnet_tag=args.tfgridnet_tag,
            num_speakers=sample.references.shape[0],
        )
        pair_len = min(sample.mixture.shape[0], int(round(args.chunk_sec * sample.sample_rate)))
        paired = run_and_align(
            cheap,
            expensive,
            sample.mixture[:pair_len],
            sample.sample_rate,
        )
        _write_streams(
            output_dir,
            "p1_int1_anchor",
            paired.anchor.streams,
            paired.anchor.sample_rate,
        )
        _write_streams(
            output_dir,
            "p1_int1_aligned",
            paired.aligned.streams,
            paired.aligned.sample_rate,
        )
        report["p1_int1"] = {
            "anchor_expert": paired.anchor.expert_used,
            "other_expert": paired.aligned.expert_used,
            "anchor_streams": paired.anchor.num_streams,
            "aligned_streams": paired.aligned.num_streams,
            "method": paired.alignment.method,
            "assignment": paired.alignment.assignment,
            "mean_matched_cost": paired.mean_matched_cost,
            "passed": bool(
                paired.alignment.method == "embedding"
                and np.isfinite(paired.mean_matched_cost)
                and paired.mean_matched_cost < 1.0
            ),
        }

    long_output = run_and_align_long(
        cheap,
        sample.mixture,
        sample.sample_rate,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        match_threshold=args.match_threshold,
        ema=args.ema,
    )
    _write_streams(
        output_dir,
        "p1_int2_track",
        long_output.result.streams,
        long_output.result.sample_rate,
    )
    trace = _assignment_trace(
        long_output.result.streams,
        sample.references,
        long_output.result.sample_rate,
        args.chunk_sec,
        args.overlap_sec,
    )
    switches = _identity_switches(trace)
    p1_int2_passed = bool(
        len(trace) >= 2
        and long_output.result.num_streams == sample.references.shape[0]
        and switches == 0
    )
    report["p1_int2"] = {
        "num_chunks": long_output.num_chunks,
        "num_persistent_tracks": long_output.result.num_streams,
        "num_reference_speakers": int(sample.references.shape[0]),
        "chunk_track_ids": [list(ids) for ids in long_output.chunk_track_ids],
        "assignment_trace": trace,
        "identity_switches": switches,
        "passed": p1_int2_passed,
    }

    report_path = output_dir / "alignment_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nArtifacts written to {output_dir}")

    requested = [report["p1_int2"]]
    if "p1_int1" in report:
        requested.append(report["p1_int1"])
    all_passed = all(bool(item["passed"]) for item in requested)
    if args.strict and not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
