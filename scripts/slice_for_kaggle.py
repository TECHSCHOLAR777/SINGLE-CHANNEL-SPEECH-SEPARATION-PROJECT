"""
Slice the local CALM-Sep dataset to a Kaggle-uploadable subset.

What it creates (target ~900 MB uncompressed, ~600 MB zipped):
  calmsep-kaggle/
    librispeech-8k/
      train-clean-100/   100 speakers × up to 30 utterances = ~3000 files
      manifest_8k.json   updated counts, same speaker IDs (holdout logic unchanged)
    rirs/                ALL 10 000 RIR files (only 120 MB total)
    noise/
      wham/              2 000 randomly sampled noise files

Run from project root:
    python scripts/slice_for_kaggle.py
    # then zip:
    cd /tmp && zip -r calmsep-kaggle.zip calmsep-kaggle/
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────
SRC_ROOT = Path("data/calmsep-8k")
OUT_ROOT = Path("/tmp/calmsep-kaggle")
SEED = 42
N_TRAIN_SPEAKERS = 100  # unique speakers kept from train-clean-100
MAX_UTTE_PER_SPEAKER = 30  # utterances per speaker (capped)
N_NOISE = 2_000  # noise clips
KEEP_ALL_RIRS = True  # RIRs are tiny (12 KB each, 120 MB total)
# ───────────────────────────────────────────────────────────────────────────

rng = random.Random(SEED)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ── 1. speech — train-clean-100 ────────────────────────────────────────────
print("=== [1/4] Slicing train-clean-100 ===")
train_src = SRC_ROOT / "librispeech-8k" / "train-clean-100"

# Group files by speaker ID (first path component under train-clean-100)
by_speaker: dict[str, list[Path]] = defaultdict(list)
for f in sorted(train_src.rglob("*.wav")):
    spk = f.parts[f.parts.index("train-clean-100") + 1]
    by_speaker[spk].append(f)

all_speakers = sorted(by_speaker.keys())
chosen_speakers = rng.sample(all_speakers, min(N_TRAIN_SPEAKERS, len(all_speakers)))
chosen_speakers.sort()

total_copied = 0
for spk in chosen_speakers:
    files = by_speaker[spk]
    rng.shuffle(files)
    selected = files[:MAX_UTTE_PER_SPEAKER]
    for f in selected:
        rel = f.relative_to(SRC_ROOT / "librispeech-8k")
        copy_file(f, OUT_ROOT / "librispeech-8k" / rel)
        total_copied += 1

print(f"  {len(chosen_speakers)} speakers, {total_copied} files copied")

# ── 2. manifest ─────────────────────────────────────────────────────────────
print("=== [2/4] Updating manifest ===")
manifest_src = SRC_ROOT / "librispeech-8k" / "manifest_8k.json"
manifest_data = json.loads(manifest_src.read_text())

# Keep dev/test entries exactly as-is (speaker IDs drive the held_out logic).
# Update train-clean-100 to reflect the sliced speaker set.
for entry in manifest_data:
    if entry.get("split") == "train-clean-100":
        entry["speaker_ids"] = chosen_speakers
        entry["file_count"] = total_copied
        # Remove train-clean-360 entry (we didn't include that data)
    if entry.get("split") == "train-clean-360":
        entry["file_count"] = 0
        entry["speaker_ids"] = []

# Also keep dev-clean / test-clean speaker_ids intact so holdout logic works.
out_manifest = OUT_ROOT / "librispeech-8k" / "manifest_8k.json"
out_manifest.parent.mkdir(parents=True, exist_ok=True)
out_manifest.write_text(json.dumps(manifest_data, indent=2))
print("  manifest written")

# ── 3. RIRs ─────────────────────────────────────────────────────────────────
print("=== [3/4] Copying RIRs ===")
rir_src = SRC_ROOT / "rirs"
rir_dst = OUT_ROOT / "rirs"
rir_dst.mkdir(parents=True, exist_ok=True)

# Copy bank.json
shutil.copy2(rir_src / "bank.json", rir_dst / "bank.json")

rir_files = sorted(rir_src.glob("*.wav"))
if KEEP_ALL_RIRS:
    selected_rirs = rir_files
else:
    selected_rirs = rng.sample(rir_files, min(2000, len(rir_files)))
    selected_rirs.sort()

for rf in selected_rirs:
    shutil.copy2(rf, rir_dst / rf.name)
print(f"  {len(selected_rirs)} RIR files copied")

# ── 4. noise ────────────────────────────────────────────────────────────────
print("=== [4/4] Sampling noise files ===")
noise_src = SRC_ROOT / "noise" / "wham"
noise_dst = OUT_ROOT / "noise" / "wham"
noise_dst.mkdir(parents=True, exist_ok=True)

# Copy manifest
noise_manifest_src = SRC_ROOT / "noise" / "noise_manifest.json"
if noise_manifest_src.exists():
    shutil.copy2(noise_manifest_src, OUT_ROOT / "noise" / "noise_manifest.json")

noise_files = sorted(noise_src.glob("*.wav"))
selected_noise = rng.sample(noise_files, min(N_NOISE, len(noise_files)))
selected_noise.sort()
for nf in selected_noise:
    shutil.copy2(nf, noise_dst / nf.name)
print(f"  {len(selected_noise)} noise files copied")

# ── summary ─────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")

total_bytes = sum(f.stat().st_size for f in OUT_ROOT.rglob("*") if f.is_file())
print(f"Output directory : {OUT_ROOT}")
print(f"Total size       : {total_bytes / 1e9:.2f} GB")
file_count = sum(1 for f in OUT_ROOT.rglob("*") if f.is_file())
print(f"Total files      : {file_count:,}")
print("\nNext step:")
print("  cd /tmp && zip -r calmsep-kaggle.zip calmsep-kaggle/")
print("  kaggle datasets create -p calmsep-kaggle-meta/")
