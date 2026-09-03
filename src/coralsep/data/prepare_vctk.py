"""
Prepare the VCTK accent-diversity speaker pool for CoRAL-Sep, Phase 0 (P0-A4).

VCTK adds accent variety on top of LibriSpeech.  Two things must be reconciled
with ``DynamicMixer`` (data/mixer.py) before its files are usable:

  1. VCTK ships 48 kHz FLAC; the mixer hard-requires 16 kHz.  We resample to
     16 kHz mono (scipy ``resample_poly``).
  2. VCTK names files ``p225_001_mic1.flac`` (speaker ``p225``, no dash).  The
     mixer's ``_speaker_id`` splits on the first ``-``, so we rename each file
     to LibriSpeech style ``p225-001.wav``.  The pool then drops straight into
     ``DynamicMixer(train_speaker_ids=..., test_speaker_ids=...)`` with correct
     speaker-disjoint splits and no mixer changes.

Only one microphone (``mic1`` by default) is kept to avoid near-duplicate takes.

Usage
-----
    python src/coralsep/data/prepare_vctk.py --output-dir /path/to/datasets
    python src/coralsep/data/prepare_vctk.py --output-dir /path/to/datasets --max-speakers 20
    python src/coralsep/data/prepare_vctk.py --output-dir /path/to/datasets \\
        --vctk-url https://mirror.example/VCTK-Corpus-0.92.zip
"""

from __future__ import annotations

import argparse
import zipfile
from math import gcd
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ── URLs and constants ────────────────────────────────────────────────────────

# Canonical VCTK-Corpus-0.92 archive (Edinburgh DataShare). Overridable via CLI.
VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

# Directory (inside the archive) holding the silence-trimmed 48 kHz FLACs.
SPEECH_SUBDIR = "wav48_silence_trimmed"
DEFAULT_MIC = "mic1"
TARGET_SR = 16000

# ── Download ──────────────────────────────────────────────────────────────────


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_vctk(output_dir: Path, url: str = VCTK_URL) -> Path:
    """
    Download and extract VCTK; return the ``wav48_silence_trimmed`` directory.

    Skips download/extraction if the speech directory already exists and is
    non-empty.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_speech_dir(output_dir)
    if existing is not None and any(existing.iterdir()):
        print(f"  [skip] VCTK speech dir already at {existing}")
        return existing

    archive = output_dir / "VCTK-Corpus-0.92.zip"
    if not archive.exists():
        print("  [download] VCTK-Corpus-0.92.zip (large, ~11 GB)")
        print(f"    {url}")
        urlretrieve(url, str(archive), reporthook=_report_progress)
        print()
    else:
        print(f"  [skip download] archive exists: {archive.name}")

    print(f"  [extract] {archive.name} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(str(output_dir))

    speech_dir = _find_speech_dir(output_dir)
    if speech_dir is None:
        raise RuntimeError(
            f"Could not find '{SPEECH_SUBDIR}' under {output_dir} after extraction.\n"
            "The VCTK archive may have a different internal layout."
        )
    print(f"  [ok] {speech_dir}")
    return speech_dir


def _find_speech_dir(root: Path) -> Path | None:
    """Locate the wav48_silence_trimmed directory anywhere under root."""
    direct = root / SPEECH_SUBDIR
    if direct.is_dir():
        return direct
    for cand in root.rglob(SPEECH_SUBDIR):
        if cand.is_dir():
            return cand
    return None


# ── Pool building (resample + rename) ─────────────────────────────────────────


def vctk_speaker_id(path: Path) -> str:
    """Speaker ID from a renamed pool file (``p225-001.wav`` -> ``p225``)."""
    return path.stem.split("-")[0]


def _parse_vctk_name(flac: Path, mic: str) -> tuple[str, str] | None:
    """
    Parse a raw VCTK filename into (speaker, utterance), filtering by mic.

    ``p225_001_mic1.flac`` -> (``p225``, ``001``).  Returns None if the file is
    for a different microphone.  Files without a mic suffix are always kept.
    """
    parts = flac.stem.split("_")
    if len(parts) < 2:
        return None
    speaker, utt = parts[0], parts[1]
    file_mic = parts[2] if len(parts) > 2 else None
    if file_mic is not None and file_mic != mic:
        return None
    return speaker, utt


def resample_to_16k_mono(audio: np.ndarray, src_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Downmix to mono and resample to target_sr with an anti-aliasing filter."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if src_sr == target_sr:
        return audio
    g = gcd(int(src_sr), int(target_sr))
    return resample_poly(audio, target_sr // g, src_sr // g).astype(np.float32)


def build_pool(
    speech_dir: Path,
    pool_dir: Path,
    *,
    mic: str = DEFAULT_MIC,
    target_sr: int = TARGET_SR,
    max_speakers: int | None = None,
) -> Path:
    """
    Resample and rename VCTK FLACs into a flat 16 kHz pool the mixer can read.

    Writes ``pool_dir/{speaker}-{utt}.wav`` at target_sr.  Idempotent: skips if
    pool_dir already contains WAVs.  ``max_speakers`` caps the pool to the first
    N speakers (sorted) to keep the ~11 GB corpus manageable.
    """
    pool_dir.mkdir(parents=True, exist_ok=True)
    if any(pool_dir.glob("*.wav")):
        print(f"  [skip] VCTK pool already at {pool_dir}")
        return pool_dir

    parsed: list[tuple[Path, str, str]] = []
    for flac in sorted(speech_dir.rglob("*.flac")):
        info = _parse_vctk_name(flac, mic)
        if info is not None:
            parsed.append((flac, info[0], info[1]))

    if not parsed:
        raise RuntimeError(
            f"No VCTK FLACs for mic='{mic}' under {speech_dir}. "
            "Check --mic or the extracted corpus."
        )

    speakers = sorted({spk for _, spk, _ in parsed})
    if max_speakers is not None:
        speakers = speakers[:max_speakers]
    keep = set(speakers)

    written = 0
    for flac, speaker, utt in parsed:
        if speaker not in keep:
            continue
        audio, sr = sf.read(str(flac), dtype="float32", always_2d=True)
        wav = resample_to_16k_mono(audio, sr, target_sr)
        sf.write(str(pool_dir / f"{speaker}-{utt}.wav"), wav, target_sr, subtype="FLOAT")
        written += 1

    print(f"  [ok] {written} clips from {len(keep)} speakers -> {pool_dir}")
    return pool_dir


# ── Discovery + verification ──────────────────────────────────────────────────


def discover_vctk_files(pool_dir: Path) -> list[Path]:
    """Return the pool WAVs, ready to pass as DynamicMixer(source_files=...)."""
    return sorted(Path(pool_dir).glob("*.wav"))


def verify_pool(pool_dir: Path, min_speakers: int = 2) -> None:
    """Verify the pool has WAVs spanning at least min_speakers distinct speakers."""
    wavs = discover_vctk_files(pool_dir)
    if not wavs:
        raise RuntimeError(f"VCTK pool at {pool_dir} contains no WAV files.")
    speakers = {vctk_speaker_id(w) for w in wavs}
    if len(speakers) < min_speakers:
        raise RuntimeError(
            f"VCTK pool has only {len(speakers)} speaker(s); need >= {min_speakers}."
        )
    print(f"  Pool OK: {len(wavs)} clips, {len(speakers)} speakers at {pool_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download VCTK and build a 16 kHz accent-diversity pool (P0-A4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root for the download/extract. Pool lands in <DIR>/vctk_pool_16k/.",
    )
    parser.add_argument(
        "--vctk-url",
        type=str,
        default=VCTK_URL,
        metavar="URL",
        help="Override the VCTK archive URL.",
    )
    parser.add_argument(
        "--mic",
        type=str,
        default=DEFAULT_MIC,
        metavar="MIC",
        help="Microphone to keep (default: mic1).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        metavar="N",
        help="Cap the pool to the first N speakers (default: all).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    pool_dir = output_dir / "vctk_pool_16k"

    print("=" * 60)
    print("CoRAL-Sep  |  Prepare VCTK accent pool  |  Phase 0 (P0-A4)")
    print("=" * 60)
    print(f"  output dir:   {output_dir}")
    print(f"  pool dir:     {pool_dir}")
    print(f"  mic / cap:    {args.mic} / {args.max_speakers or 'all'}")
    print()

    print("Step 1 / 3  Download VCTK")
    speech_dir = download_vctk(output_dir, url=args.vctk_url)
    print()

    print("Step 2 / 3  Resample to 16 kHz + rename into pool")
    build_pool(speech_dir, pool_dir, mic=args.mic, max_speakers=args.max_speakers)
    print()

    print("Step 3 / 3  Verify pool")
    verify_pool(pool_dir)
    print()

    print("=" * 60)
    print("Done.  Feed the pool into the dynamic mixer:")
    print("  from data.prepare_vctk import discover_vctk_files")
    print("  from data.mixer import DynamicMixer")
    print(f'  files = discover_vctk_files("{pool_dir}")')
    print("  mixer = DynamicMixer(files, test_speaker_ids={'p225', 'p226'})")
    print("=" * 60)


if __name__ == "__main__":
    main()
