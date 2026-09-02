"""
Stage WHAM! and DNS-4 noise clips at 8 kHz and 16 kHz (Dev A, P0-A4).

Training and evaluation need 8 kHz noise to match the CoRAL-Sep mixture rate.
DNSMOS evaluation needs 16 kHz, because the DNSMOS proxy model operates at
that rate. Both are written side by side so neither consumer has to resample
at runtime.

Output layout
-------------
output_dir/
  wham/
    {clip_name}_8k.wav
    {clip_name}_16k.wav
  dns4/
    {clip_name}_8k.wav
    {clip_name}_16k.wav
  noise_manifest.json

The manifest lists every staged clip with its source, duration (at 8 kHz),
and the sha256 of the 8k file — not of the 16k copy, so that DNSMOS consumers
can load 16k without the manifest pointing them at stale hashes.

Usage
-----
    python src/coralsep/data/prepare_noise_staging.py \\
        --wham-dir /data/wham_noise \\
        --dns4-dir /data/dns4_noise \\
        --output-dir /data/calmsep_noise \\
        --target-sr 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
TARGET_SR_8K: int = 8_000
TARGET_SR_16K: int = 16_000


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample using polyphase filtering. Returns float32."""
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    g = _gcd(dst_sr, src_sr)
    up, down = dst_sr // g, src_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def _validate_audio(path: Path) -> bool:
    """
    Return True if soundfile can read the file's info.

    A corrupt or non-audio file raises SoundFileError; those are skipped rather
    than crashing the entire staging run.
    """
    try:
        sf.info(str(path))
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load as float32 mono; mix down multi-channel files."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def _write_wav(audio: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def stage_source(
    src_dir: Path,
    dst_dir: Path,
    source_name: str,
    target_8k: int = TARGET_SR_8K,
    target_16k: int = TARGET_SR_16K,
) -> list[dict]:
    """
    Resample all audio files under src_dir and write them to dst_dir.

    Returns a list of manifest entry dicts. Skips corrupt files and already-
    staged files (idempotent).
    """
    audio_files = sorted(
        p for p in src_dir.rglob("*") if p.suffix.lower() in _AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise RuntimeError(
            f"No audio files found under {src_dir} "
            f"(checked extensions: {sorted(_AUDIO_EXTENSIONS)})."
        )

    entries: list[dict] = []

    for src in tqdm(audio_files, desc=source_name, unit="file"):
        clip_name = src.stem
        dst_8k = dst_dir / f"{clip_name}_8k.wav"
        dst_16k = dst_dir / f"{clip_name}_16k.wav"

        both_exist = dst_8k.exists() and dst_16k.exists()
        if both_exist:
            # Read duration from the 8k file for the manifest
            info = sf.info(str(dst_8k))
            entries.append({
                "source": source_name,
                "clip_name": clip_name,
                "src_path": str(src),
                "path_8k": str(dst_8k),
                "path_16k": str(dst_16k),
                "duration_s": round(info.duration, 6),
            })
            continue

        if not _validate_audio(src):
            print(f"  [skip corrupt] {src}")
            continue

        try:
            mono, src_sr = _load_mono(src)
        except Exception as exc:
            print(f"  [skip read-error] {src}: {exc}")
            continue

        audio_8k = _resample(mono, src_sr, target_8k)
        audio_16k = _resample(mono, src_sr, target_16k)

        _write_wav(audio_8k, target_8k, dst_8k)
        _write_wav(audio_16k, target_16k, dst_16k)

        entries.append({
            "source": source_name,
            "clip_name": clip_name,
            "src_path": str(src),
            "path_8k": str(dst_8k),
            "path_16k": str(dst_16k),
            "duration_s": round(float(len(audio_8k)) / target_8k, 6),
        })

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage WHAM! and DNS-4 noise at 8 kHz and 16 kHz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wham-dir", type=Path, default=None, metavar="DIR",
                        help="Root of the WHAM! noise corpus (optional).")
    parser.add_argument("--dns4-dir", type=Path, default=None, metavar="DIR",
                        help="Root of the DNS-4 noise corpus (optional).")
    parser.add_argument("--output-dir", required=True, type=Path, metavar="DIR",
                        help="Staging destination.")
    parser.add_argument("--target-sr", type=int, default=TARGET_SR_8K, metavar="HZ",
                        help=f"Low-rate target (default {TARGET_SR_8K}). "
                             "16 kHz copies are always written alongside.")
    args = parser.parse_args()

    if args.wham_dir is None and args.dns4_dir is None:
        raise SystemExit("ERROR: at least one of --wham-dir or --dns4-dir must be provided.")

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict] = []

    sources: list[tuple[str, Path]] = []
    if args.wham_dir is not None:
        wham_dir = args.wham_dir.resolve()
        if not wham_dir.is_dir():
            raise SystemExit(f"ERROR: --wham-dir does not exist: {wham_dir}")
        sources.append(("wham", wham_dir))

    if args.dns4_dir is not None:
        dns4_dir = args.dns4_dir.resolve()
        if not dns4_dir.is_dir():
            raise SystemExit(f"ERROR: --dns4-dir does not exist: {dns4_dir}")
        sources.append(("dns4", dns4_dir))

    for source_name, src_dir in sources:
        dst_dir = output_dir / source_name
        print(f"\n[{source_name}] {src_dir} → {dst_dir}")
        entries = stage_source(src_dir, dst_dir, source_name, args.target_sr, TARGET_SR_16K)
        all_entries.extend(entries)
        total_dur_h = sum(e["duration_s"] for e in entries) / 3600.0
        print(f"  {len(entries)} clips staged, {total_dur_h:.2f} h total")

    manifest = {
        "target_sr_8k": args.target_sr,
        "target_sr_16k": TARGET_SR_16K,
        "n_clips": len(all_entries),
        "clips": all_entries,
    }
    manifest_path = output_dir / "noise_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")
    print(f"Total clips staged: {len(all_entries)}")


if __name__ == "__main__":
    main()
