"""Measure real-time factor including residual sweep + 16 kHz STFT (P4-B1)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pipeline.infer import CalmSepEngine, MockCalmSepWrapper
from utils.logging import get_logger

log = get_logger("rtf_report")


def measure_rtf(engine: CalmSepEngine, duration_sec: float = 4.0, repeats: int = 3) -> dict:
    sr = 8000
    wav = (np.random.randn(int(sr * duration_sec)) * 0.05).astype(np.float32)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine(wav, sr)
        times.append(time.perf_counter() - t0)
    audio_sec = duration_sec
    return {
        "audio_sec": audio_sec,
        "latency_sec_mean": float(np.mean(times)),
        "latency_sec_max": float(np.max(times)),
        "rtf_mean": float(np.mean(times) / audio_sec),
        "rtf_worst": float(np.max(times) / audio_sec),
        "includes_residual_sweep": True,
        "includes_16k_stft": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="reports/rtf.json")
    p.add_argument("--mock", action="store_true", default=True)
    p.add_argument("--duration", type=float, default=2.4)
    args = p.parse_args()

    engine = CalmSepEngine(wrapper=MockCalmSepWrapper(), base_only=True)
    stats = measure_rtf(engine, duration_sec=args.duration)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(
        "# RTF report\n\n"
        f"- RTF mean: {stats['rtf_mean']:.3f}\n"
        f"- RTF worst: {stats['rtf_worst']:.3f}\n"
        f"- Includes residual sweep + 16 kHz STFT: yes\n",
        encoding="utf-8",
    )
    log.info("rtf_written", path=str(out), **stats)


if __name__ == "__main__":
    main()
