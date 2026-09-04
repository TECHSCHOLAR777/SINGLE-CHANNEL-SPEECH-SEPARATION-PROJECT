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
and the sha256 of the 8k file: not of the 16k copy, so that DNSMOS consumers
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
    required_split: str | None = None,
) -> list[dict]:
    """
    Resample all audio files under src_dir and write them to dst_dir.

    Returns a list of manifest entry dicts. Skips corrupt files and already-
    staged files (idempotent).

    required_split, when given, restricts staging to files with that split
    name as a path component (WHAM's tr/cv/tt layout). LibriMix's official
    test mixtures are built from WHAM noise, so staging from the wrong split
    (or the unfiltered whole corpus) risks the noise adapter and gate
    training on clips acoustically related to the ones the headline results
    are later scored against. See I-044. Nothing enforced this before; a
    caller that wants the guard must now pass required_split explicitly, and
    the manifest records whichever split (or "unfiltered") was actually used
    so a downstream consumer can check it, per check_noise_provenance below.
    """
    audio_files = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in _AUDIO_EXTENSIONS)
    if required_split is not None:
        audio_files = [p for p in audio_files if required_split in p.relative_to(src_dir).parts]
    if not audio_files:
        scope = f" under split {required_split!r}" if required_split else ""
        raise RuntimeError(
            f"No audio files found{scope} under {src_dir} "
            f"(checked extensions: {sorted(_AUDIO_EXTENSIONS)})."
        )

    entries: list[dict] = []

    for src in tqdm(audio_files, desc=source_name, unit="file"):
        clip_name = src.stem
        dst_8k = dst_dir / f"{clip_name}_8k.wav"
        dst_16k = dst_dir / f"{clip_name}_16k.wav"

        split = required_split if required_split is not None else "unfiltered"

        both_exist = dst_8k.exists() and dst_16k.exists()
        if both_exist:
            # Read duration from the 8k file for the manifest
            info = sf.info(str(dst_8k))
            entries.append(
                {
                    "source": source_name,
                    "clip_name": clip_name,
                    "src_path": str(src),
                    "path_8k": str(dst_8k),
                    "path_16k": str(dst_16k),
                    "duration_s": round(info.duration, 6),
                    "split": split,
                }
            )
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

        entries.append(
            {
                "source": source_name,
                "clip_name": clip_name,
                "src_path": str(src),
                "path_8k": str(dst_8k),
                "path_16k": str(dst_16k),
                "duration_s": round(float(len(audio_8k)) / target_8k, 6),
                "split": split,
            }
        )

    return entries


def check_noise_provenance(noise_dir: Path, required_split: str = "tr") -> None:
    """
    Refuse to proceed if the staged noise directory's manifest does not
    record every WHAM entry as coming from the required split.

    LibriMix's official test mixtures are built from WHAM noise, so training
    the noise adapter or the gate on any other split (or on an unfiltered
    stage that mixed splits together) risks the training data overlapping
    acoustically with the exact clips the headline results are later scored
    against (I-044). An old manifest with no "split" field at all, or one
    recording "unfiltered", is exactly the unsafe case this guard exists to
    catch, so both fail loudly rather than being treated as acceptable.

    Args:
        noise_dir: The staging output_dir passed to this module's --output-dir
            (the directory containing wham/ and dns4/ and noise_manifest.json),
            not the wham/ subdirectory itself.
        required_split: The split every wham entry must be staged from.

    Raises:
        RuntimeError: If the manifest is missing, or if any wham entry lacks
            a "split" field or does not match required_split.
    """
    manifest_path = Path(noise_dir) / "noise_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"No noise_manifest.json in {noise_dir}. Cannot verify WHAM split "
            f"provenance before training on it. Run prepare.noise_staging "
            f"with --wham-split {required_split} first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = manifest.get("clips", []) if isinstance(manifest, dict) else manifest
    bad = [e for e in clips if e.get("source") == "wham" and e.get("split") != required_split]
    if bad:
        seen = sorted({e.get("split", "MISSING") for e in bad})
        raise RuntimeError(
            f"{len(bad)} WHAM entries in {manifest_path} are not from the "
            f"required split {required_split!r} (found: {seen}). Re-stage "
            f"with `python -m coralsep.data.prepare.noise_staging --wham-split "
            f"{required_split} ...` before training the noise adapter or the "
            f"gate on this directory. See I-044."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage WHAM! and DNS-4 noise at 8 kHz and 16 kHz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--wham-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root of the WHAM! noise corpus (optional).",
    )
    parser.add_argument(
        "--dns4-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root of the DNS-4 noise corpus (optional).",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, metavar="DIR", help="Staging destination."
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=TARGET_SR_8K,
        metavar="HZ",
        help=f"Low-rate target (default {TARGET_SR_8K}). "
        "16 kHz copies are always written alongside.",
    )
    parser.add_argument(
        "--wham-split",
        default=None,
        metavar="SPLIT",
        help=(
            "Restrict WHAM staging to this split (tr/cv/tt), and record it in "
            "the manifest. LibriMix's official test mixtures are built from "
            "WHAM, so training data must come from tr, not tt or cv, and not "
            "an unfiltered stage of the whole corpus. Omitting this flag "
            "stages everything under --wham-dir and records the manifest "
            "entries as split 'unfiltered', which check_noise_provenance "
            "will refuse. See I-044."
        ),
    )
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
        split = args.wham_split if source_name == "wham" else None
        entries = stage_source(
            src_dir, dst_dir, source_name, args.target_sr, TARGET_SR_16K, required_split=split
        )
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
