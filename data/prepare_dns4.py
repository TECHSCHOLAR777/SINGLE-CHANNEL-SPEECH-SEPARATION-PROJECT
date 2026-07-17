"""
Stratified DNS-4 noise subset preparation (BLUEPRINT §7.3, 20 GB target).

DNS Challenge / INTERSPEECH 2022 noise tracks are large and speech-heavy.
This script defines a stratified sampling plan that caps total size near 20 GB
while avoiding over-representation of babble and crowd noise.

Actual download is optional and gated behind ``--download``; default mode writes
the stratification manifest and prints instructions only.

Usage::

    python data/prepare_dns4.py --out-dir data/dns4_subset --dry-run
    python data/prepare_dns4.py --out-dir data/dns4_subset --target-gb 20
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

TARGET_GB_DEFAULT: float = 20.0
"""BLUEPRINT §7.3 stratified subset target."""

DNS4_README = (
    "https://github.com/microsoft/DNS-Challenge/blob/master/DNS-Challenge/README.md"
)


@dataclass
class NoiseCategory:
    """One DNS-4 noise category in the stratified plan."""

    name: str
    description: str
    target_fraction: float
    max_files: int | None = None
    patterns: list[str] = field(default_factory=list)


# Stratification avoids speech-like noise dominating the subset.
STRATIFIED_CATEGORIES: tuple[NoiseCategory, ...] = (
    NoiseCategory(
        name="ambient",
        description="Steady ambient (traffic, wind, HVAC)",
        target_fraction=0.30,
        patterns=["*traffic*", "*wind*", "*hvac*", "*ambient*"],
    ),
    NoiseCategory(
        name="indoor",
        description="Indoor non-speech (keyboard, dishes, appliances)",
        target_fraction=0.20,
        patterns=["*keyboard*", "*dishes*", "*indoor*"],
    ),
    NoiseCategory(
        name="nature",
        description="Nature sounds (rain, thunder, birds)",
        target_fraction=0.15,
        patterns=["*rain*", "*thunder*", "*bird*"],
    ),
    NoiseCategory(
        name="babble",
        description="Speech-like babble — capped fraction",
        target_fraction=0.15,
        max_files=500,
        patterns=["*babble*", "*crowd*", "*speech*"],
    ),
    NoiseCategory(
        name="music",
        description="Background music",
        target_fraction=0.10,
        patterns=["*music*"],
    ),
    NoiseCategory(
        name="other",
        description="Residual categories",
        target_fraction=0.10,
        patterns=["*"],
    ),
)


@dataclass
class Dns4SubsetPlan:
    """Stratified DNS-4 subset specification."""

    target_bytes: int
    categories: list[dict]
    download_url_hint: str = DNS4_README
    notes: str = (
        "Stratify to avoid overrepresentation of speech-like noise (babble, crowd). "
        "Used by adapter_noise training (BLUEPRINT §7.2)."
    )


def build_subset_plan(target_gb: float = TARGET_GB_DEFAULT) -> Dns4SubsetPlan:
    """Build the stratified sampling plan for a target size in GB."""
    if target_gb <= 0:
        raise ValueError(f"target_gb must be positive, got {target_gb}")

    target_bytes = int(target_gb * (1024**3))
    categories: list[dict] = []
    for cat in STRATIFIED_CATEGORIES:
        cat_bytes = int(target_bytes * cat.target_fraction)
        categories.append(
            {
                "name": cat.name,
                "description": cat.description,
                "target_fraction": cat.target_fraction,
                "target_bytes": cat_bytes,
                "max_files": cat.max_files,
                "patterns": cat.patterns,
            }
        )

    return Dns4SubsetPlan(target_bytes=target_bytes, categories=categories)


def write_plan(out_dir: Path, plan: Dns4SubsetPlan) -> Path:
    """Write subset_plan.json under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "subset_plan.json"
    doc = asdict(plan)
    doc["target_gb"] = plan.target_bytes / (1024**3)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return path


def print_download_instructions(out_dir: Path, plan: Dns4SubsetPlan) -> None:
    """Print manual DNS-4 acquisition steps."""
    print(f"DNS-4 stratified subset plan written to {out_dir / 'subset_plan.json'}")
    print(f"Target size: {plan.target_bytes / (1024**3):.1f} GB")
    print()
    print("Manual download (not executed by default):")
    print(f"  1. Follow the DNS Challenge README: {DNS4_README}")
    print("  2. Download the DNS-4 / INTERSPEECH 2022 noise training set")
    print(f"  3. Place files under {out_dir.resolve()}/raw/")
    print("  4. Run with --materialize to apply stratified selection")
    print()
    print("Category fractions:")
    for cat in plan.categories:
        print(f"  - {cat['name']}: {cat['target_fraction'] * 100:.0f}%")


def materialize_subset(out_dir: Path, plan: Dns4SubsetPlan) -> int:
    """
    Apply stratified selection to files under out_dir/raw/.

    Returns:
        Number of files selected into out_dir/selected/.

    Raises:
        FileNotFoundError: When raw/ is empty or missing.
    """
    raw = out_dir / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"raw noise directory not found: {raw}")

    all_wavs = sorted(raw.rglob("*.wav"))
    if not all_wavs:
        raise FileNotFoundError(f"No WAV files under {raw}")

    selected_dir = out_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for cat in plan.categories:
        patterns = [p.lower().replace("*", "") for p in cat["patterns"]]
        matched = [
            w
            for w in all_wavs
            if any(p in w.name.lower() or p in str(w).lower() for p in patterns if p)
        ]
        if cat["max_files"] is not None:
            matched = matched[: cat["max_files"]]
        for wav in matched:
            rel = wav.relative_to(raw)
            manifest.append({"category": cat["name"], "source": str(rel)})

    manifest_path = out_dir / "selected_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Selected {len(manifest)} files (manifest only; copy/symlink in production)")
    return len(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stratified DNS-4 noise subset")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/dns4_subset"),
        help="Output directory for plan and selected noise",
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        default=TARGET_GB_DEFAULT,
        help="Target subset size in GB (default 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write plan and print instructions only (default behaviour)",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Apply stratified selection to out-dir/raw/",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Reserved: actual DNS-4 download is manual; prints instructions",
    )
    args = parser.parse_args()

    plan = build_subset_plan(args.target_gb)
    write_plan(args.out_dir, plan)

    if args.download or args.dry_run or not args.materialize:
        print_download_instructions(args.out_dir, plan)
        if args.download:
            print("\nAutomatic DNS-4 download is not implemented; use manual steps above.")
        return

    try:
        materialize_subset(args.out_dir, plan)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
