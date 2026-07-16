"""CALM-Sep CLI entry point (P4-A1).

Usage:
    python -m scripts.calmsep_infer separate input.wav --out out_dir --mock
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.infer import CalmSepEngine, MockCalmSepWrapper
from utils.hashing import hash_config, hash_file
from utils.logging import get_logger

log = get_logger("calmsep_infer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CALM-Sep separation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("separate", help="Separate a mono mixture")
    s.add_argument("input", type=str)
    s.add_argument("--out", type=str, required=True)
    s.add_argument("--config", type=str, default="configs/base_checkpoint.yaml")
    s.add_argument("--checkpoint", type=str, default=None)
    s.add_argument("--mock", action="store_true")
    s.add_argument("--base-only", action="store_true")
    s.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def write_bundle(out_dir: Path, config_path: Path, report: dict) -> None:
    bundle = out_dir / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copy2(config_path, bundle / config_path.name)
        (bundle / "config.sha256").write_text(hash_file(config_path), encoding="utf-8")
    (bundle / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (bundle / "report.sha256").write_text(
        hash_file(bundle / "report.json"), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.cmd != "separate":
        raise SystemExit(f"unknown command {args.cmd}")

    wav, sr = sf.read(args.input, always_2d=False)
    if getattr(wav, "ndim", 1) > 1:
        wav = np.mean(wav, axis=-1)
    wav = np.asarray(wav, dtype=np.float32)

    if args.mock:
        engine = CalmSepEngine(device=args.device, wrapper=MockCalmSepWrapper(), base_only=True)
    else:
        engine = CalmSepEngine(
            device=args.device,
            checkpoint_path=args.checkpoint,
            base_only=args.base_only,
        )

    result = engine(wav, sr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, stream in enumerate(result.streams):
        sf.write(out / f"spk_{i+1}.wav", stream, result.sample_rate)

    report = result.to_report_dict()
    report["input"] = str(args.input)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_bundle(out, Path(args.config), report)
    log.info("separation_done", out=str(out), n=result.speaker_count)


if __name__ == "__main__":
    main()
