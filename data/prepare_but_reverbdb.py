"""
Prepare BUT ReverbDB (OpenSLR SLR17) for CALM-Sep real-RIR evaluation (BLUEPRINT §7.4).

OpenSLR SLR17 ships 1,244 measured room impulse responses from Brno University of
Technology. This script:

1. Prints download instructions when the corpus is absent (no automatic fetch).
2. Indexes local RIR WAV files into ``rir_index.json`` when present.

Expected layout after manual download::

    {root}/
      BUT_ReverbDB/
        RIRs/
          *.wav

Usage::

    python data/prepare_but_reverbdb.py --root /data/but_reverbdb
    python data/prepare_but_reverbdb.py --root /data/but_reverbdb --index-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.hashing import hash_file

SLR17_URL = "https://www.openslr.org/resources/17/but_reverbdb_release_1_0.zip"
"""Official OpenSLR SLR17 archive (BUT ReverbDB release 1.0)."""

DEFAULT_ROOT = Path("data/but_reverbdb")
INDEX_FILENAME = "rir_index.json"


@dataclass
class RirIndexEntry:
    """One indexed RIR file."""

    rir_id: str
    path: str
    sha256: str
    duration_sec: float | None = None
    sample_rate: int | None = None


def print_download_instructions(root: Path) -> None:
    """Print manual download steps when the corpus is not on disk."""
    print("BUT ReverbDB (OpenSLR SLR17) is not present locally.")
    print()
    print("Manual download steps:")
    print(f"  1. Download: {SLR17_URL}")
    print(f"  2. Extract the archive under: {root.resolve()}")
    print("  3. Ensure RIR WAV files live under BUT_ReverbDB/RIRs/ (or RIRs/ at root)")
    print("  4. Re-run this script with --index-only to build rir_index.json")
    print()
    print("SLR28 is AISHELL-2 (Mandarin ASR) — not a RIR database. Use SLR17 only.")


def _discover_rir_dirs(root: Path) -> list[Path]:
    candidates = [
        root / "BUT_ReverbDB" / "RIRs",
        root / "RIRs",
        root / "but_reverbdb" / "RIRs",
    ]
    return [p for p in candidates if p.is_dir()]


def _probe_wav(path: Path) -> tuple[float | None, int | None]:
    """Return (duration_sec, sample_rate) without loading full audio when possible."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        dur = float(info.frames) / float(info.samplerate) if info.samplerate else None
        return dur, int(info.samplerate)
    except Exception:
        return None, None


def index_rir_wavs(root: Path) -> list[RirIndexEntry]:
    """
    Walk local RIR directories and build an index of WAV files.

    Args:
        root: Root directory containing extracted BUT ReverbDB files.

    Returns:
        Sorted list of RirIndexEntry records.

    Raises:
        FileNotFoundError: When no RIR WAV files are found under root.
    """
    rir_dirs = _discover_rir_dirs(root)
    if not rir_dirs:
        raise FileNotFoundError(
            f"No RIR directory found under {root}. Expected BUT_ReverbDB/RIRs/ or RIRs/."
        )

    entries: list[RirIndexEntry] = []
    seen: set[str] = set()
    for rir_dir in rir_dirs:
        for wav in sorted(rir_dir.rglob("*.wav")):
            rel = str(wav.relative_to(root)).replace("\\", "/")
            if rel in seen:
                continue
            seen.add(rel)
            dur, sr = _probe_wav(wav)
            entries.append(
                RirIndexEntry(
                    rir_id=wav.stem,
                    path=rel,
                    sha256=hash_file(wav),
                    duration_sec=dur,
                    sample_rate=sr,
                )
            )

    if not entries:
        raise FileNotFoundError(f"No *.wav RIR files found under {root}")

    entries.sort(key=lambda e: e.rir_id)
    return entries


def write_index(root: Path, entries: list[RirIndexEntry]) -> Path:
    """Write rir_index.json under root."""
    doc = {
        "source": "BUT_ReverbDB",
        "openslr": "SLR17",
        "n_rirs": len(entries),
        "entries": [asdict(e) for e in entries],
    }
    out = root / INDEX_FILENAME
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return out


def prepare_but_reverbdb(root: Path, *, index_only: bool = False) -> Path | None:
    """
    Index local RIRs or print download instructions.

    Returns:
        Path to rir_index.json when indexing succeeds, else None.
    """
    root.mkdir(parents=True, exist_ok=True)
    rir_dirs = _discover_rir_dirs(root)
    wav_count = sum(1 for d in rir_dirs for _ in d.rglob("*.wav"))

    if wav_count == 0:
        if index_only:
            raise FileNotFoundError(f"No RIR WAV files under {root}")
        print_download_instructions(root)
        return None

    entries = index_rir_wavs(root)
    out = write_index(root, entries)
    print(f"Indexed {len(entries)} RIRs -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BUT ReverbDB (OpenSLR SLR17)")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory for BUT ReverbDB files",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Fail if RIRs are missing instead of printing download instructions",
    )
    args = parser.parse_args()

    try:
        result = prepare_but_reverbdb(args.root, index_only=args.index_only)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if result is None:
        sys.exit(0)


if __name__ == "__main__":
    main()
