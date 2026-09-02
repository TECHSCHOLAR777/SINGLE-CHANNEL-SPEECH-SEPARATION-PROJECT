"""
Download and stage the BUT ReverbDB (OpenSLR SLR17) for real-RIR evaluation (Dev A, P0-A5).

BLUEPRINT 7.4 mandates a sim-to-real evaluation tier using measured RIRs, not
simulated ones. BUT ReverbDB (Brno University of Technology, OpenSLR resource
17) is a set of measured room impulse responses from real rooms. They live at:
  https://www.openslr.org/resources/17/

The SLR17 archive contains WAV files of measured RIRs, which are resampled to
8 kHz and indexed in but_bank.json so that RirBank can load them directly with
no code change — the schema matches bank.json written by rir_bank.py, except
that t60_requested_s == t60_achieved_s (measured RIRs have no "requested" T60).

The n_peak field is found by find_direct_path_peak, which locates the largest-
magnitude sample. For real RIRs this is the direct-path arrival.

Usage
-----
    python src/coralsep/data/prepare_but_reverbdb.py \\
        --output-dir /data/but_reverbdb \\
        --sample-rate 8000
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from coralsep.data.rir_bank import RirRecord, find_direct_path_peak, measure_t60

# SLR17 ships as a set of named archives. The canonical index page is:
#   https://www.openslr.org/17/
# The actual files are WAV RIRs bundled in a zip or tarballs. We fetch the
# single combined archive that holds all rooms.
_SLR17_BASE_URL = "https://www.openslr.org/resources/17/"
_SLR17_ARCHIVES: list[tuple[str, str]] = [
    # (filename, download URL)
    ("BUT_ReverbDB_rel_19_06_RIR.tgz", _SLR17_BASE_URL + "BUT_ReverbDB_rel_19_06_RIR.tgz"),
]
# Fallback single-room zip files if the tarball isn't available
_SLR17_ZIP_ARCHIVES: list[tuple[str, str]] = [
    ("reverb_data_but.zip", _SLR17_BASE_URL + "reverb_data_but.zip"),
]


def _report_progress(block: int, block_size: int, total: int) -> None:
    if total > 0:
        pct = min(100, block * block_size * 100 // total)
        mb = block * block_size / 1_048_576
        total_mb = total / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def _download(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name}")
    print(f"    from {url}")
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_report_progress)
        print()
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed: {url}\n  {exc}") from exc


def _extract(archive: Path, dest_dir: Path) -> None:
    print(f"  Extracting {archive.name} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    name_lower = archive.name.lower()
    if name_lower.endswith((".tgz", ".tar.gz", ".tar.bz2", ".tar")):
        with tarfile.open(str(archive)) as tf:
            tf.extractall(str(dest_dir))
    elif name_lower.endswith(".zip"):
        with zipfile.ZipFile(str(archive)) as zf:
            zf.extractall(str(dest_dir))
    else:
        raise RuntimeError(f"Unsupported archive format: {archive.name}")
    print(f"  Extracted to {dest_dir}")


def _find_rir_wavs(root: Path) -> list[Path]:
    """Collect all WAV files under root, sorted by path for determinism."""
    wavs = sorted(root.rglob("*.wav")) + sorted(root.rglob("*.WAV"))
    # Deduplicate (rglob patterns are case-sensitive on Linux)
    seen: set[Path] = set()
    unique: list[Path] = []
    for w in wavs:
        r = w.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(w)
    return unique


def download_slr17(output_dir: Path) -> list[Path]:
    """
    Download and extract SLR17 archives into output_dir.

    Returns the list of extracted WAV files. Skips archives that are already
    present on disk (idempotent).
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = output_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Try the primary tarballs first, fall back to zip archives
    to_try = _SLR17_ARCHIVES + _SLR17_ZIP_ARCHIVES
    downloaded_any = False

    for fname, url in to_try:
        archive = raw_dir / fname
        if not archive.exists():
            try:
                _download(url, archive)
                downloaded_any = True
            except RuntimeError as e:
                print(f"  [warn] {e}")
                continue
        else:
            print(f"  [skip download] {fname} already present")
            downloaded_any = True

        marker = extracted_dir / f".extracted_{fname}"
        if not marker.exists():
            try:
                _extract(archive, extracted_dir)
                marker.touch()
            except Exception as exc:
                print(f"  [warn] extraction failed for {fname}: {exc}")
        else:
            print(f"  [skip extract] {fname} already extracted")

        # If we got at least one successful archive, check for WAVs
        wavs = _find_rir_wavs(extracted_dir)
        if wavs:
            return wavs

    if not downloaded_any:
        raise RuntimeError(
            "Could not download any SLR17 archive from openslr.org. "
            "Check your internet connection or manually place RIR WAV files under:\n"
            f"  {extracted_dir}\n"
            "Then re-run this script — it will skip the download and stage whatever is there."
        )

    wavs = _find_rir_wavs(extracted_dir)
    if not wavs:
        raise RuntimeError(
            f"No WAV files found under {extracted_dir} after extraction. "
            "The SLR17 archive layout may have changed. Manually inspect and re-run."
        )
    return wavs


def stage_rir(
    wav_path: Path,
    staged_dir: Path,
    target_sr: int,
    rir_idx: int,
) -> RirRecord | None:
    """
    Resample one RIR WAV to target_sr, write it to staged_dir, and return its record.

    Returns None if the file cannot be read or produces a degenerate RIR.
    """
    try:
        audio, src_sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception as exc:
        print(f"  [skip] cannot read {wav_path}: {exc}")
        return None

    rir = audio.mean(axis=1).astype(np.float32)
    if rir.size == 0:
        return None

    if src_sr != target_sr:
        from scipy.signal import resample_poly

        def _gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a

        g = _gcd(target_sr, src_sr)
        rir = resample_poly(rir, target_sr // g, src_sr // g).astype(np.float32)

    rir_id = f"but_rir_{rir_idx:05d}"
    out_path = staged_dir / f"{rir_id}.wav"
    sf.write(str(out_path), rir, target_sr)

    t60 = measure_t60(rir, target_sr)
    n_peak = find_direct_path_peak(rir)

    if t60 <= 0.0:
        # Degenerate (silent or no decay) — still include it but with t60=0
        # so the evaluator can filter it rather than causing an indexing gap.
        pass

    return RirRecord(
        rir_id=rir_id,
        path=f"{rir_id}.wav",  # relative to staged_dir, same as bank.json convention
        t60_requested_s=t60,   # measured RIRs: requested == achieved
        t60_achieved_s=t60,
        room_dim_m=[],         # unknown for measured RIRs
        source_pos_m=[],
        mic_pos_m=[],
        absorption=float("nan"),
        max_order=-1,          # image-source model was not used
        n_peak=n_peak,
        sample_rate=target_sr,
    )


def build_but_bank(
    output_dir: Path,
    sample_rate: int,
) -> list[RirRecord]:
    """
    Download SLR17, resample RIRs to sample_rate, write but_bank.json.

    Idempotent: if but_bank.json already exists, skip entirely.
    """
    bank_path = output_dir / "but_bank.json"
    if bank_path.exists():
        print(f"  [skip] but_bank.json already exists at {bank_path}")
        existing = json.loads(bank_path.read_text(encoding="utf-8"))
        return [RirRecord(**r) for r in existing["records"]]

    print("Step 1 / 3  Download SLR17 (BUT ReverbDB)")
    wav_files = download_slr17(output_dir)
    print(f"  Found {len(wav_files)} WAV files\n")

    staged_dir = output_dir / "rirs_8k"
    staged_dir.mkdir(parents=True, exist_ok=True)

    print("Step 2 / 3  Resample and measure T60")
    records: list[RirRecord] = []
    for idx, wav in enumerate(tqdm(wav_files, desc="staging RIRs", unit="file")):
        record = stage_rir(wav, staged_dir, sample_rate, idx)
        if record is not None:
            records.append(record)

    if not records:
        raise RuntimeError(
            "No valid RIRs could be staged. All WAV files were unreadable or degenerate."
        )

    print(f"\nStep 3 / 3  Write but_bank.json ({len(records)} RIRs)")
    t60s = [r.t60_achieved_s for r in records if r.t60_achieved_s > 0.0]
    index: dict = {
        "source": "BUT_ReverbDB_SLR17",
        "sample_rate": sample_rate,
        "n_rirs": len(records),
        # t60 stats for quick inspection
        "t60_mean_s": float(np.mean(t60s)) if t60s else 0.0,
        "t60_min_s": float(np.min(t60s)) if t60s else 0.0,
        "t60_max_s": float(np.max(t60s)) if t60s else 0.0,
        "records": [r.to_dict() for r in records],
    }
    bank_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"  Written to {bank_path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and stage BUT ReverbDB (OpenSLR SLR17) for CoRAL-Sep evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True, type=Path, metavar="DIR",
                        help="Destination directory for staged RIRs and but_bank.json.")
    parser.add_argument("--sample-rate", type=int, default=8000, metavar="HZ",
                        help="Target sample rate in Hz (default: 8000).")
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CoRAL-Sep  |  BUT ReverbDB staging  |  OpenSLR SLR17")
    print("=" * 60)
    print(f"  output dir:   {output_dir}")
    print(f"  sample rate:  {args.sample_rate} Hz\n")

    records = build_but_bank(output_dir, args.sample_rate)

    print(f"\nDone. {len(records)} RIRs staged.")
    print(f"Load with RirBank(bank_dir='{output_dir / 'rirs_8k'}', bank_file='../but_bank.json')")
    print("Or pass --rir-bank path to fixed_eval_generator.py for the real-RIR tier.")


if __name__ == "__main__":
    main()
