#!/usr/bin/env python3
"""
Preflight check for LibriMix generation (Dev C tooling).

Both CoRAL-Sep data-generation failures to date had the same shape: a metadata
CSV referenced audio that was not on disk, and we only found out after the
generator had already started, deep inside a ProcessPoolExecutor traceback.

  * WHAM run: metadata pointed at wham_noise/tt/*.wav before WHAM was extracted.
    Died on `LibsndfileError: System error` after the pool had spun up.
  * train split: `prepare_librimix.py --include-train` copies the *train-360*
    metadata CSV, but only downloads LibriSpeech *train-clean-100*. Every source
    path in that CSV is therefore missing.

This script answers, in seconds and before any generation starts: for the CSVs
you are about to hand the generator, does every referenced source and noise file
actually exist and open?

Usage:

    python scripts/preflight_data.py \
      --metadata-csv /workspace/tools/LibriMix/metadata/Libri3Mix/mixture_train-100_mix_both.csv \
      --librispeech-dir /workspace/data/librispeech/LibriSpeech \
      --wham-dir /workspace/data/wham_noise

    # check a whole metadata directory at once
    python scripts/preflight_data.py \
      --metadata-dir /workspace/tools/LibriMix/metadata/Libri3Mix \
      --librispeech-dir /workspace/data/librispeech/LibriSpeech \
      --wham-dir /workspace/data/wham_noise

Exit code is non-zero when anything is missing, so it composes into a shell
pipeline: preflight && generate.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_SOURCE_COL = re.compile(r"^source_\d+_path$")
_NOISE_COL = "noise_path"


def _resolve(root: Path, rel: str) -> Path:
    """LibriMix CSVs sometimes carry absolute paths from the machine that
    generated the metadata. Take only the tail that matters."""
    rel = str(rel).strip()
    candidate = Path(rel)
    if candidate.is_absolute():
        # keep the last components after the corpus root name if we can find it
        parts = candidate.parts
        for anchor in ("LibriSpeech", "wham_noise"):
            if anchor in parts:
                idx = parts.index(anchor)
                return root.joinpath(*parts[idx + 1 :])
        return candidate
    return root / rel


def check_csv(
    csv_path: Path,
    librispeech_dir: Path,
    wham_dir: Path | None,
    *,
    deep: bool,
    limit: int | None,
) -> dict[str, object]:
    if not csv_path.is_file():
        return {
            "csv": str(csv_path),
            "error": "csv not found",
            "rows": 0,
            "missing": [],
            "ok": False,
        }

    missing: list[str] = []
    unreadable: list[str] = []
    rows = 0
    checked = 0
    seen: set[Path] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        source_cols = [c for c in fields if _SOURCE_COL.match(c)]
        has_noise = _NOISE_COL in fields

        for row in reader:
            rows += 1
            if limit is not None and rows > limit:
                break

            targets: list[Path] = [
                _resolve(librispeech_dir, row[col]) for col in source_cols if row.get(col)
            ]
            if has_noise and row.get(_NOISE_COL):
                if wham_dir is None:
                    missing.append(f"<no --wham-dir given, but CSV has {_NOISE_COL}>")
                    continue
                targets.append(_resolve(wham_dir, row[_NOISE_COL]))

            for target in targets:
                if target in seen:
                    continue
                seen.add(target)
                checked += 1
                if not target.is_file():
                    missing.append(str(target))
                elif deep:
                    try:
                        import soundfile as sf

                        sf.info(str(target))
                    except Exception as exc:  # noqa: BLE001
                        unreadable.append(f"{target}  ({type(exc).__name__})")

    return {
        "csv": csv_path.name,
        "rows": rows,
        "unique_files_checked": checked,
        "source_columns": source_cols,
        "missing_count": len(missing),
        "missing": missing[:10],
        "unreadable_count": len(unreadable),
        "unreadable": unreadable[:10],
        "ok": not missing and not unreadable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metadata-csv", type=Path, nargs="+", help="One or more mixture CSVs.")
    group.add_argument("--metadata-dir", type=Path, help="Directory of mixture_*.csv files.")
    parser.add_argument(
        "--librispeech-dir",
        type=Path,
        required=True,
        help="LibriSpeech root (the dir containing train-clean-100/, dev-clean/, ...).",
    )
    parser.add_argument(
        "--wham-dir",
        type=Path,
        default=None,
        help="WHAM noise root (the dir containing tr/, cv/, tt/). Required for mix_both CSVs.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also open each file with soundfile. Catches truncated or corrupt downloads, "
        "not just missing ones. Slower.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only check the first N rows of each CSV. Use for a fast smoke check.",
    )
    args = parser.parse_args()

    if args.metadata_dir is not None:
        csvs = sorted(args.metadata_dir.glob("mixture_*.csv"))
        if not csvs:
            print(f"no mixture_*.csv found in {args.metadata_dir}", file=sys.stderr)
            raise SystemExit(2)
    else:
        csvs = list(args.metadata_csv)

    print("=" * 72)
    print("CoRAL-Sep  |  data preflight")
    print("=" * 72)
    print(f"  librispeech : {args.librispeech_dir}")
    print(f"  wham        : {args.wham_dir}")
    print(f"  deep check  : {args.deep}")
    print()

    all_ok = True
    for csv_path in csvs:
        report = check_csv(
            csv_path, args.librispeech_dir, args.wham_dir, deep=args.deep, limit=args.limit
        )
        status = "OK  " if report["ok"] else "FAIL"
        print(f"[{status}] {report['csv']}")
        if "error" in report:
            print(f"         {report['error']}")
            all_ok = False
            continue
        print(
            f"         rows={report['rows']}  unique files checked={report['unique_files_checked']}"
        )
        if report["missing_count"]:
            all_ok = False
            print(f"         MISSING {report['missing_count']} file(s). First few:")
            for path in report["missing"]:
                print(f"           - {path}")
        if report["unreadable_count"]:
            all_ok = False
            print(f"         UNREADABLE {report['unreadable_count']} file(s). First few:")
            for path in report["unreadable"]:
                print(f"           - {path}")
        print()

    if not all_ok:
        print("PREFLIGHT FAILED. Do not start generation; it will die partway through.")
        print("Common causes:")
        print("  * CSV is for train-360 but only train-clean-100 was downloaded")
        print("  * WHAM! noise not downloaded or extracted to the expected root")
        print("  * a tarball extracted to a nested dir (LibriSpeech/LibriSpeech/...)")
        raise SystemExit(1)

    print("PREFLIGHT PASSED. Every referenced file exists" + (" and opens." if args.deep else "."))


if __name__ == "__main__":
    main()
