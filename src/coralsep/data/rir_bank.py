"""
Simulated RIR bank generation and lookup (Dev A, P0-A3).

BLUEPRINT section 7.2 calls for a cached bank of 10k room impulse responses,
1k per 0.1 s T60 step across 0.2 to 1.0 s, generated with pyroomacoustics
before training begins. Generating RIRs on the fly would make every epoch pay
the image-source cost and would make the T60 label depend on a live simulation
rather than a recorded one.

The T60 label is the free supervision target for the Level-2 reverberation head
(BLUEPRINT 5.4). It is recorded as the *requested* T60, and the *achieved* T60
is measured back from the generated RIR by Schroeder integration: the two can
differ because pyroomacoustics solves for absorption from Sabine's formula,
which is an approximation. Both are stored. The achieved value is the honest
label and is what the head trains against.

Real measured RIRs (BUT ReverbDB, OpenSLR SLR17) are handled separately by
data/prepare_but_reverbdb.py and are evaluation-only: the sim-to-real gap is a
mandatory measurement (BLUEPRINT 7.4), so simulated RIRs must never appear in
that tier.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from coralsep.data.condition_mixer import CORALSEP_SAMPLE_RATE

T60_MIN_S: float = 0.2
T60_MAX_S: float = 1.0
"""BLUEPRINT 5.3: adapter_reverb trains on T60 uniform 0.2 to 1.0 s."""

T60_STEP_S: float = 0.1
"""One bank bucket per 0.1 s of T60, giving 8 buckets across the range."""

DEFAULT_RIRS_PER_BUCKET: int = 1_250
"""8 buckets x 1250 = 10k RIRs, matching the BLUEPRINT 7.2 target."""

SEVERE_T60_S: float = 0.9
"""
BLUEPRINT 7.5 holdout 3: T60 above 0.9 s is a severity holdout, kept to 10% of
reverb training samples and probed in evaluation. sample_t60 enforces this.
"""

SEVERE_FRACTION: float = 0.10
"""Fraction of training draws allowed into the severe (T60 > 0.9 s) band."""

_ROOM_DIM_MIN_M = np.array([3.0, 3.0, 2.4])
_ROOM_DIM_MAX_M = np.array([10.0, 8.0, 4.0])
"""
Room size range in metres. Small office through medium meeting room. The upper
bound is held below concert-hall scale because the evaluation target is speech
in rooms, and because Sabine's formula degrades for very large volumes.
"""

_MIN_WALL_MARGIN_M = 0.5
"""Keep sources and the mic off the walls; image-source models are unreliable at the boundary."""


@dataclass
class RirRecord:
    """
    One generated RIR and the room that produced it.

    Attributes:
        rir_id: Stable identifier, also the filename stem.
        path: Location of the .wav holding the impulse response.
        t60_requested_s: T60 asked of the simulator.
        t60_achieved_s: T60 measured back from the RIR by Schroeder integration.
            This is the honest label; the head trains against it.
        room_dim_m: Room dimensions [x, y, z] in metres.
        source_pos_m: Source position [x, y, z] in metres.
        mic_pos_m: Microphone position [x, y, z] in metres.
        absorption: Sabine absorption coefficient solved for the requested T60.
        max_order: Image-source reflection order used.
        n_peak: Index of the direct-path peak. The wet reference truncates at
            n_peak + 512 (BLUEPRINT 7.6), so it is recorded here rather than
            recomputed at mix time.
        sample_rate: Always CORALSEP_SAMPLE_RATE.
    """

    rir_id: str
    path: str
    t60_requested_s: float
    t60_achieved_s: float
    room_dim_m: list[float]
    source_pos_m: list[float]
    mic_pos_m: list[float]
    absorption: float
    max_order: int
    n_peak: int
    sample_rate: int = CORALSEP_SAMPLE_RATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_t60(rir: np.ndarray, sample_rate: int = CORALSEP_SAMPLE_RATE) -> float:
    """
    Measure T60 from an impulse response by Schroeder backward integration.

    The energy decay curve is integrated backwards from the tail, converted to
    dB, and a line is fitted over the -5 to -35 dB span (the T30 convention),
    then extrapolated to a 60 dB decay. T30 extrapolation is used rather than a
    direct -5 to -65 dB fit because real and simulated tails hit the noise floor
    before -65 dB, which would bias a full-range fit toward short T60.

    Args:
        rir: Impulse response [T].
        sample_rate: Rate of the impulse response in Hz.

    Returns:
        Estimated T60 in seconds. Returns 0.0 for a degenerate (silent or
        single-sample) response rather than raising, so a failed simulation is
        visible as an outlier in the bank rather than crashing generation.
    """
    h = np.asarray(rir, dtype=np.float64).squeeze()
    if h.ndim != 1 or h.size < 2:
        return 0.0

    energy = h**2
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0

    # Schroeder curve: remaining energy from each point to the end.
    decay = np.cumsum(energy[::-1])[::-1]
    decay = decay / decay[0]
    with np.errstate(divide="ignore"):
        decay_db = 10.0 * np.log10(np.maximum(decay, 1e-20))

    start_idx = int(np.argmax(decay_db <= -5.0))
    end_idx = int(np.argmax(decay_db <= -35.0))
    if end_idx <= start_idx:
        return 0.0

    times = np.arange(start_idx, end_idx) / float(sample_rate)
    values = decay_db[start_idx:end_idx]
    if times.size < 2:
        return 0.0

    slope, _ = np.polyfit(times, values, 1)
    if slope >= 0.0:
        return 0.0
    return float(-60.0 / slope)


def find_direct_path_peak(rir: np.ndarray) -> int:
    """
    Index of the direct-path arrival in an impulse response.

    The direct path is the largest-magnitude sample: it travels the shortest
    distance and undergoes no absorption, so it dominates every reflection in
    the rooms this bank covers. BLUEPRINT 7.6 truncates the wet reference at
    this index plus an offset.

    Args:
        rir: Impulse response [T].

    Returns:
        Sample index of the peak.
    """
    h = np.asarray(rir).squeeze()
    if h.ndim != 1 or h.size == 0:
        raise ValueError(f"rir must be a non-empty 1-D array, got shape {h.shape}")
    return int(np.argmax(np.abs(h)))


def t60_buckets(
    t60_min: float = T60_MIN_S,
    t60_max: float = T60_MAX_S,
    step: float = T60_STEP_S,
) -> list[tuple[float, float]]:
    """
    The [low, high) T60 intervals the bank is stratified over.

    Returns:
        List of (low, high) pairs, e.g. [(0.2, 0.3), (0.3, 0.4), ...].
    """
    edges = np.arange(t60_min, t60_max + 1e-9, step)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def sample_t60(
    rng: np.random.Generator,
    allow_severe: bool = True,
    severe_fraction: float = SEVERE_FRACTION,
) -> float:
    """
    Draw a training T60, honouring the severity holdout.

    BLUEPRINT 7.5 holdout 3 keeps T60 above 0.9 s rare in training (10%) so
    that evaluation at high T60 measures extrapolation rather than memorisation.
    This function is the single place that rule is enforced for reverb.

    Args:
        rng: Seeded generator.
        allow_severe: When False, never draws above SEVERE_T60_S. Use for the
            gate-training pool where the severity holdout is strictest.
        severe_fraction: Probability of drawing from the severe band.

    Returns:
        T60 in seconds, within [T60_MIN_S, T60_MAX_S].
    """
    if not allow_severe:
        return float(rng.uniform(T60_MIN_S, SEVERE_T60_S))
    if rng.random() < severe_fraction:
        return float(rng.uniform(SEVERE_T60_S, T60_MAX_S))
    return float(rng.uniform(T60_MIN_S, SEVERE_T60_S))


def _sample_room(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a room, a source position, and a mic position with wall margins."""
    dim = rng.uniform(_ROOM_DIM_MIN_M, _ROOM_DIM_MAX_M)
    lo = np.full(3, _MIN_WALL_MARGIN_M)
    hi = dim - _MIN_WALL_MARGIN_M
    source = rng.uniform(lo, hi)
    mic = rng.uniform(lo, hi)
    # Keep a minimum source-mic separation; co-located source and mic makes the
    # direct path dominate so heavily that the RIR carries no room information.
    for _ in range(10):
        if np.linalg.norm(source - mic) >= 0.5:
            break
        mic = rng.uniform(lo, hi)
    return dim, source, mic


def generate_rir(
    t60_s: float,
    rng: np.random.Generator,
    sample_rate: int = CORALSEP_SAMPLE_RATE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Simulate one RIR at a requested T60 using pyroomacoustics.

    Args:
        t60_s: Requested reverberation time in seconds.
        rng: Seeded generator, so the room draw is reproducible.
        sample_rate: Output rate in Hz.

    Returns:
        (rir, meta) where rir is [T] float32 and meta carries the room
        geometry, the solved absorption, and the achieved T60.

    Raises:
        ImportError: When pyroomacoustics is not installed, with the install
            command, since RIR generation is the one step that cannot be
            faked or deferred.
    """
    try:
        import pyroomacoustics as pra
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "pyroomacoustics is required to generate the RIR bank. Install it with:\n"
            "  pip install pyroomacoustics\n"
            "It is CPU-only and needs no GPU."
        ) from exc

    dim, source, mic = _sample_room(rng)

    # Sabine's formula gives the absorption that yields the requested T60 for
    # this specific room volume. max_order is capped: image-source cost grows
    # cubically and beyond ~40 the added reflections are below the noise floor.
    absorption, max_order = pra.inverse_sabine(t60_s, dim.tolist())
    max_order = int(min(max_order, 40))

    room = pra.ShoeBox(
        dim.tolist(),
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=max_order,
    )
    room.add_source(source.tolist())
    room.add_microphone(mic.reshape(3, 1))
    room.compute_rir()

    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    achieved = measure_t60(rir, sample_rate)

    meta: dict[str, Any] = {
        "t60_requested_s": float(t60_s),
        "t60_achieved_s": float(achieved),
        "room_dim_m": [float(v) for v in dim],
        "source_pos_m": [float(v) for v in source],
        "mic_pos_m": [float(v) for v in mic],
        "absorption": float(absorption),
        "max_order": int(max_order),
        "n_peak": find_direct_path_peak(rir),
    }
    return rir, meta


def build_rir_bank(
    output_dir: str | Path,
    rirs_per_bucket: int = DEFAULT_RIRS_PER_BUCKET,
    sample_rate: int = CORALSEP_SAMPLE_RATE,
    seed: int = 0,
    progress: bool = True,
) -> list[RirRecord]:
    """
    Generate the full stratified RIR bank and write it to disk.

    One bucket per 0.1 s T60 step; within a bucket the requested T60 is drawn
    uniformly so the bank covers the range continuously rather than at 8
    discrete values. Each RIR is written as a .wav and indexed in bank.json.

    This is a one-time, CPU-only, offline step. At the default 1250 per bucket
    it produces 10k RIRs and takes roughly 20-40 minutes on a laptop core.

    Args:
        output_dir: Directory for the .wav files and bank.json.
        rirs_per_bucket: RIRs to generate per 0.1 s T60 bucket.
        sample_rate: Output rate in Hz.
        seed: RNG seed. The bank is fully reproducible from this value.
        progress: Print per-bucket progress.

    Returns:
        The RirRecord list, also written to output_dir/bank.json.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    records: list[RirRecord] = []
    buckets = t60_buckets()

    for b_idx, (low, high) in enumerate(buckets):
        if progress:
            print(f"[rir_bank] bucket {b_idx + 1}/{len(buckets)}: T60 {low:.1f}-{high:.1f} s")
        for i in range(rirs_per_bucket):
            t60 = float(rng.uniform(low, high))
            rir, meta = generate_rir(t60, rng, sample_rate)

            rir_id = f"rir_t60_{low:.1f}_{i:05d}"
            path = out / f"{rir_id}.wav"
            sf.write(path, rir, sample_rate)

            records.append(
                RirRecord(
                    rir_id=rir_id,
                    path=str(path.relative_to(out)),
                    sample_rate=sample_rate,
                    **meta,
                )
            )

    index = {
        "seed": seed,
        "sample_rate": sample_rate,
        "rirs_per_bucket": rirs_per_bucket,
        "n_rirs": len(records),
        "t60_range_s": [T60_MIN_S, T60_MAX_S],
        "records": [r.to_dict() for r in records],
    }
    (out / "bank.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    if progress:
        achieved = np.array([r.t60_achieved_s for r in records])
        requested = np.array([r.t60_requested_s for r in records])
        err = np.abs(achieved - requested)
        print(
            f"[rir_bank] wrote {len(records)} RIRs to {out}\n"
            f"[rir_bank] T60 error vs requested: mean {err.mean():.3f} s, "
            f"p95 {np.percentile(err, 95):.3f} s"
        )
    return records


class RirBank:
    """
    Loads a generated bank and samples RIRs by T60.

    Sampling is by achieved T60, not requested, so a draw for "T60 near 0.5 s"
    returns an RIR that actually decays in 0.5 s.

    Parameters
    ----------
    bank_dir:
        Directory containing bank.json and the .wav files.
    rng:
        Seeded generator for reproducible draws.
    """

    def __init__(self, bank_dir: str | Path, rng: np.random.Generator | None = None) -> None:
        self.bank_dir = Path(bank_dir)
        index_path = self.bank_dir / "bank.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"no bank.json in {self.bank_dir}. Generate the bank first:\n"
                f"  python -m coralsep.data.rir_bank --output {self.bank_dir}"
            )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.records = [RirRecord(**r) for r in index["records"]]
        self.sample_rate = int(index["sample_rate"])
        self._rng = rng if rng is not None else np.random.default_rng()
        self._achieved = np.array([r.t60_achieved_s for r in self.records])

    def __len__(self) -> int:
        return len(self.records)

    def sample(self, t60_s: float | None = None, tolerance_s: float = 0.05) -> RirRecord:
        """
        Draw one RIR, optionally near a target T60.

        Args:
            t60_s: Target reverberation time. When None, draws uniformly from
                the whole bank.
            tolerance_s: Half-width of the acceptance window around t60_s. When
                no RIR falls inside, the nearest one by achieved T60 is
                returned rather than raising, so a sparse bucket degrades the
                label slightly instead of failing the epoch.

        Returns:
            The chosen RirRecord.
        """
        if t60_s is None:
            return self.records[int(self._rng.integers(len(self.records)))]

        within = np.abs(self._achieved - t60_s) <= tolerance_s
        candidates = np.flatnonzero(within)
        if candidates.size == 0:
            return self.records[int(np.argmin(np.abs(self._achieved - t60_s)))]
        return self.records[int(self._rng.choice(candidates))]

    def load(self, record: RirRecord) -> np.ndarray:
        """Read one RIR's samples from disk as float32 [T]."""
        audio, sr = sf.read(self.bank_dir / record.path, dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(f"{record.path} is {sr} Hz, bank declares {self.sample_rate} Hz")
        return np.asarray(audio, dtype=np.float32).squeeze()


def _main() -> None:
    """CLI: python -m coralsep.data.rir_bank --output datasets/rirs --per-bucket 1250"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate the CoRAL-Sep simulated RIR bank.")
    parser.add_argument("--output", default="datasets/rirs", help="Output directory")
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=DEFAULT_RIRS_PER_BUCKET,
        help=f"RIRs per 0.1 s T60 bucket (default {DEFAULT_RIRS_PER_BUCKET}, 8 buckets)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--sample-rate", type=int, default=CORALSEP_SAMPLE_RATE, help="Output sample rate"
    )
    args = parser.parse_args()

    build_rir_bank(
        output_dir=args.output,
        rirs_per_bucket=args.per_bucket,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()
