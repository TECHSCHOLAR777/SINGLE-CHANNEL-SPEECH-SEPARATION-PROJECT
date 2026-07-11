#!/usr/bin/env python3
"""Run P1-INT1 and P1-INT2 acceptance checks on one real Libri3Mix clip.

Example:

    python scripts/validate_alignment.py \
      --librimix-root /workspace/datasets/Libri3Mix \
      --device cuda --output-dir outputs/p1_alignment --skip-pair --strict

Notes on the P1-INT2 metric (rewritten 2026-07-11):

The previous version scored identity stability by Hungarian-matching every
stitched track against every reference in each window, using waveform
cross-correlation. That is unsound when a track is silent for part of the
timeline: a silent track is an all-zero row, its cost against every reference
is degenerate, and Hungarian then assigns it arbitrarily. The assignment churns
window to window and gets counted as an "identity switch" even though no real
speaker moved. It also compared each window against window 0 rather than
against its predecessor, so one genuine swap early on scored as a switch in
every later window.

This version:
  * marks a track inactive in a window when its RMS sits below
    --silence-floor-db relative to the loudest track in that same window, and
    excludes it from the cost matrix entirely,
  * counts a switch only when a track that is active in two CONSECUTIVE windows
    changes which reference it maps to,
  * reports streams_per_chunk, so an expert emitting fewer streams than there
    are reference speakers is visible immediately rather than showing up
    downstream as mysterious identity churn.
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


def _rms(block: np.ndarray) -> np.ndarray:
    """Per-row RMS of a [K, T] block."""
    return np.sqrt(np.mean(np.square(block.astype(np.float64)), axis=1))


def _active_tracks(window: np.ndarray, floor_db: float) -> np.ndarray:
    """
    Boolean mask of tracks carrying real energy in this window.

    Scale-invariant: a track is active when its RMS is within floor_db of the
    loudest track in the same window. An all-zero track (one that was never
    assigned a segment overlapping this window) is always inactive.
    """
    rms = _rms(window)
    loudest = float(np.max(rms)) if rms.size else 0.0
    if loudest <= 0.0:
        return np.zeros(window.shape[0], dtype=bool)
    floor = loudest * (10.0 ** (floor_db / 20.0))
    return rms > floor


def _assignment_trace(
    tracks: np.ndarray,
    references: np.ndarray,
    sample_rate: int,
    chunk_sec: float,
    overlap_sec: float,
    silence_floor_db: float,
) -> list[dict[str, Any]]:
    chunk = int(round(chunk_sec * sample_rate))
    hop = int(round((chunk_sec - overlap_sec) * sample_rate))
    trace: list[dict[str, Any]] = []

    for start in range(0, tracks.shape[1], hop):
        stop = min(start + chunk, tracks.shape[1])
        if stop - start < sample_rate:
            break

        track_win = tracks[:, start:stop]
        ref_win = references[:, start:stop]
        active = _active_tracks(track_win, silence_floor_db)
        mapping = [-1] * tracks.shape[0]
        matched_costs: list[float] = []

        active_idx = np.flatnonzero(active)
        if active_idx.size:
            cost = xcorr_cost_matrix(track_win[active_idx], ref_win)
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
                mapping[int(active_idx[row])] = col
                matched_costs.append(float(cost[row, col]))

        trace.append(
            {
                "start_sec": start / sample_rate,
                "stop_sec": stop / sample_rate,
                "active_tracks": [bool(flag) for flag in active],
                "track_to_reference": mapping,
                "mean_xcorr_cost": float(np.mean(matched_costs)) if matched_costs else float("nan"),
            }
        )
        if stop == tracks.shape[1]:
            break
    return trace


def _identity_switches(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Count references changing hands between CONSECUTIVE windows.

    Only tracks active (and matched) in both windows of a pair are considered;
    a track going silent and coming back is a track-lifecycle event, not an
    identity switch, and is reported separately.
    """
    if len(trace) < 2:
        return {"identity_switches": 0, "windows_with_switch": 0, "switch_detail": []}

    switches = 0
    windows_with_switch = 0
    detail: list[dict[str, Any]] = []

    for index in range(1, len(trace)):
        prev, cur = trace[index - 1], trace[index]
        window_switched = False
        for tid, (was, now) in enumerate(
            zip(prev["track_to_reference"], cur["track_to_reference"], strict=True)
        ):
            both_matched = was != -1 and now != -1
            both_active = prev["active_tracks"][tid] and cur["active_tracks"][tid]
            if both_matched and both_active and was != now:
                switches += 1
                window_switched = True
                detail.append(
                    {
                        "track": tid,
                        "from_window": index - 1,
                        "to_window": index,
                        "reference_was": was,
                        "reference_now": now,
                    }
                )
        if window_switched:
            windows_with_switch += 1

    return {
        "identity_switches": switches,
        "windows_with_switch": windows_with_switch,
        "switch_detail": detail,
    }


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
        "--max-tracks",
        type=int,
        default=None,
        help="Cap on the persistent track bank. Defaults to the reference "
        "speaker count. Without it, one unstable output slot mints a phantom "
        "track per chunk.",
    )
    parser.add_argument(
        "--silence-floor-db",
        type=float,
        default=-60.0,
        help="A track whose window RMS falls this far below the loudest track "
        "in the same window is treated as inactive and excluded from scoring.",
    )
    parser.add_argument("--skip-pair", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = _select_sample(args.librimix_root, args.sample_index)
    num_references = int(sample.references.shape[0])
    max_tracks = args.max_tracks if args.max_tracks is not None else num_references
    cheap = MossFormer2Expert(device=args.device, compute_embeddings=True)

    report: dict[str, Any] = {
        "utterance_id": sample.utterance_id,
        "duration_sec": sample.mixture.shape[0] / sample.sample_rate,
        "sample_rate": sample.sample_rate,
        "max_tracks": max_tracks,
    }

    if not args.skip_pair:
        expensive = get_expensive_expert(
            device=args.device,
            srcorrnet_repo=args.srcorrnet_repo,
            srcorrnet_checkpoint=args.srcorrnet_checkpoint,
            tfgridnet_tag=args.tfgridnet_tag,
            num_speakers=num_references,
        )
        pair_len = min(sample.mixture.shape[0], int(round(args.chunk_sec * sample.sample_rate)))
        paired = run_and_align(cheap, expensive, sample.mixture[:pair_len], sample.sample_rate)
        _write_streams(output_dir, "p1_int1_anchor", paired.anchor.streams, paired.anchor.sample_rate)
        _write_streams(output_dir, "p1_int1_aligned", paired.aligned.streams, paired.aligned.sample_rate)
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
        max_tracks=max_tracks,
    )
    _write_streams(
        output_dir, "p1_int2_track", long_output.result.streams, long_output.result.sample_rate
    )
    trace = _assignment_trace(
        long_output.result.streams,
        sample.references,
        long_output.result.sample_rate,
        args.chunk_sec,
        args.overlap_sec,
        args.silence_floor_db,
    )
    switch_report = _identity_switches(trace)

    # The single most diagnostic number: how many streams the cheap expert
    # actually emits per chunk. If this is below num_reference_speakers, the
    # expert structurally cannot cover the mixture and no amount of alignment
    # work will close the gate.
    streams_per_chunk = sorted({int(len(ids)) for ids in long_output.chunk_track_ids})
    expert_covers_all_speakers = all(k >= num_references for k in streams_per_chunk)

    # expert_covers_all_speakers is part of the pass criterion on purpose. With
    # the track cap in place, a 2-stream expert on a 3-speaker mixture can still
    # end up with exactly 3 tracks and 0 counted switches, because the cap forces
    # the wandering slot onto an existing track. The numbers would go green while
    # one track holds a single chunk of one speaker and another holds two
    # speakers' content spliced together. A gate that can pass on that is a gate
    # that lies, so stream coverage is checked explicitly.
    p1_int2_passed = bool(
        len(trace) >= 2
        and expert_covers_all_speakers
        and long_output.result.num_streams == num_references
        and switch_report["identity_switches"] == 0
    )
    report["p1_int2"] = {
        "num_chunks": long_output.num_chunks,
        "streams_per_chunk": streams_per_chunk,
        "num_reference_speakers": num_references,
        "expert_covers_all_speakers": expert_covers_all_speakers,
        "num_persistent_tracks": long_output.result.num_streams,
        "chunk_track_ids": [list(ids) for ids in long_output.chunk_track_ids],
        "assignment_trace": trace,
        **switch_report,
        "passed": p1_int2_passed,
    }
    if not expert_covers_all_speakers:
        report["p1_int2"]["blocker"] = (
            f"cheap expert emitted {streams_per_chunk} stream(s) per chunk for "
            f"{num_references} reference speakers; the gate cannot pass until the "
            "expert produces one stream per speaker (expert/config issue, not alignment)"
        )

    report_path = output_dir / "alignment_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nArtifacts written to {output_dir}")

    requested = [report["p1_int2"]]
    if "p1_int1" in report:
        requested.append(report["p1_int1"])
    if args.strict and not all(bool(item["passed"]) for item in requested):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
