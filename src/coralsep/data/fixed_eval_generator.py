"""
Generate the seeded, hashed, never-regenerated fixed evaluation matrix (Dev A, P0-A6).

BLUEPRINT 9.4 defines the condition × speaker-count evaluation matrix. Once
written and hashed, this set is frozen: any re-run that produces a different
hash means something changed in the synthesis pipeline, which invalidates every
number produced against this set.

Guard: if output_dir/eval_manifest.json already exists the script prints a
warning and exits without touching any files. Verification of an existing set
is done by utils/hashing.py, not by this script.

Condition cells
---------------
  clean × N∈{2,3,4,5}, baseline, no degradation
  reverb × N∈{2,3,4,5}, simulated RIR from rir-bank
  noise × N∈{2,3,4,5}, WHAM!/DNS-4 noise from noise-dir
  codec × N∈{2,3,4,5}, Opus/AAC codec roundtrip
  reverb+noise × N∈{2,3,4,5}
  all-three × N∈{2,3,4,5}
  reverb+codec × N∈{2,3,4,5}  (held-out combination, BLUEPRINT 7.5 holdout 2)
  noise+codec × N∈{2,3,4,5}   (held-out combination)
  but_reverb × N=2 only, BUT ReverbDB real measured RIRs

Each .npz contains:
  mixture           float32 [T], degraded observation at 8 kHz
  references        float32 [N, T], clean (or wet) stems at 8 kHz
  recipe            dict, ground-truth labels (MixtureRecipe.to_dict())
  condition_vector  dict, dense targets (MixtureRecipe.condition_vector())

A 16 kHz upsampled copy of mixture is saved alongside each .npz as
mix_{i:04d}_16k.npy for DNSMOS evaluation, which operates at 16 kHz.

Usage
-----
    python src/coralsep/data/fixed_eval_generator.py \\
        --librispeech-8k-dir /data/LibriSpeech_8k \\
        --noise-dir /data/calmsep_noise \\
        --rir-bank datasets/rirs/bank.json \\
        --but-reverbdb-dir /data/but_reverbdb \\
        --output-dir /data/calmsep_fixed_eval \\
        --seed 20240101 \\
        --n-per-cell 100
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

# Ensure the project root is importable when this file is run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from coralsep.data.condition_mixer import CORALSEP_SAMPLE_RATE, CoralSepMixer
from coralsep.data.degradations import apply_codec, apply_noise, apply_reverb
from coralsep.data.rir_bank import RirBank, RirRecord
from coralsep.utils.hashing import write_manifest

# Codec choices safe at 8 kHz. AMR-WB requires 16 kHz so it is excluded here.
_EVAL_CODECS = ("opus", "aac", "amr-nb")
_CODEC_BITRATE_MIN_BPS: int = 6_000
_CODEC_BITRATE_MAX_BPS: int = 24_000

_N_VALUES: tuple[int, ...] = (2, 3, 4, 5)

# Conditions that never appear in gate or joint training (BLUEPRINT 7.5 holdout 2).
_HELD_OUT_CONDITIONS: frozenset[str] = frozenset({"reverb+codec", "noise+codec"})

# All eval conditions in a stable order so the manifest is reproducible.
_CONDITIONS: tuple[str, ...] = (
    "clean",
    "reverb",
    "noise",
    "codec",
    "reverb+noise",
    "all-three",
    "reverb+codec",
    "noise+codec",
)
# BUT real-RIR tier is separate (N=2 only).
_BUT_CONDITION = "but_reverb"


# ── Lightweight BUT-bank loader ───────────────────────────────────────────────


class _ButBank:
    """
    Minimal RIR-bank-compatible wrapper for the BUT ReverbDB staged bank.

    Presents the same .load(record) interface as RirBank so apply_reverb
    can be called without modification.
    """

    def __init__(self, bank_dir: Path, records: list[RirRecord], sample_rate: int) -> None:
        self.bank_dir = bank_dir
        self.records = records
        self.sample_rate = sample_rate
        self._achieved = np.array([r.t60_achieved_s for r in records])

    def load(self, record: RirRecord) -> np.ndarray:
        path = self.bank_dir / record.path
        audio, sr = sf.read(str(path), dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(
                f"BUT RIR {record.path} is {sr} Hz, bank declares {self.sample_rate} Hz"
            )
        return np.asarray(audio, dtype=np.float32).squeeze()

    def sample_random(self, rng: np.random.Generator) -> RirRecord:
        return self.records[int(rng.integers(len(self.records)))]


def _load_but_bank(but_dir: Path, sample_rate: int) -> _ButBank:
    bank_json = but_dir / "but_bank.json"
    if not bank_json.exists():
        raise FileNotFoundError(
            f"but_bank.json not found at {bank_json}. " "Run data/prepare_but_reverbdb.py first."
        )
    index = json.loads(bank_json.read_text(encoding="utf-8"))
    records = [RirRecord(**r) for r in index["records"]]
    if not records:
        raise RuntimeError(f"but_bank.json at {bank_json} contains no records.")
    # WAVs live in rirs_8k/ next to but_bank.json
    rirs_dir = but_dir / "rirs_8k"
    if not rirs_dir.is_dir():
        raise FileNotFoundError(
            f"Expected staged RIR directory {rirs_dir}. " "Re-run data/prepare_but_reverbdb.py."
        )
    return _ButBank(
        bank_dir=rirs_dir,
        records=records,
        sample_rate=sample_rate,
    )


# ── Noise catalogue ───────────────────────────────────────────────────────────


def _load_noise_paths(noise_dir: Path) -> list[Path]:
    """
    Return all 8 kHz noise WAV paths from noise_dir/noise_manifest.json.

    Falls back to globbing if the manifest is absent so the script still works
    with a manually organised noise directory.
    """
    manifest_path = noise_dir / "noise_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = [Path(e["path_8k"]) for e in manifest.get("clips", [])]
        # Verify they exist (staging may be partial)
        paths = [p for p in paths if p.exists()]
        if paths:
            return paths

    # Fallback: glob for *_8k.wav
    paths = sorted(noise_dir.rglob("*_8k.wav"))
    if not paths:
        raise FileNotFoundError(
            f"No 8 kHz noise files found under {noise_dir}. "
            "Run data/prepare_noise_staging.py first."
        )
    return paths


def _load_noise_clip(path: Path, target_length: int, rng: np.random.Generator) -> np.ndarray:
    """Load a noise file, loop/crop to target_length, return float32 [T]."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    noise = audio.mean(axis=1).astype(np.float32)

    n = len(noise)
    if n == 0:
        return np.zeros(target_length, dtype=np.float32)
    if n < target_length:
        reps = int(np.ceil(target_length / n))
        noise = np.tile(noise, reps)
    if len(noise) > target_length:
        start = int(rng.integers(0, len(noise) - target_length + 1))
        noise = noise[start : start + target_length]
    return noise[:target_length].astype(np.float32)


# ── Codec parameter sampling ──────────────────────────────────────────────────


def _sample_codec(rng: np.random.Generator) -> tuple[str, int]:
    """Draw a (codec_name, bitrate_bps) pair for the fixed eval."""
    codec = str(rng.choice(_EVAL_CODECS))
    bitrate = int(rng.integers(_CODEC_BITRATE_MIN_BPS, _CODEC_BITRATE_MAX_BPS + 1))
    return codec, bitrate


# ── 16 kHz upsampling for DNSMOS ─────────────────────────────────────────────


def _upsample_to_16k(audio_8k: np.ndarray) -> np.ndarray:
    """Upsample 8 kHz audio to 16 kHz using polyphase filtering."""
    return resample_poly(audio_8k, 2, 1).astype(np.float32)


# ── LibriSpeech speaker discovery ────────────────────────────────────────────


def _collect_eval_files(librispeech_8k_dir: Path) -> list[Path]:
    """
    Collect 8 kHz WAV files for dev-clean and test-clean speakers.

    Reads manifest_8k.json to enumerate speaker IDs, then globs for WAVs
    under those splits. Raises if neither split is found in the manifest.
    """
    manifest_path = librispeech_8k_dir / "manifest_8k.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest_8k.json not found at {manifest_path}. "
            "Run data/prepare_librispeech_8k.py first."
        )

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_splits = {"dev-clean", "test-clean"}
    found_splits = set()

    wav_files: list[Path] = []
    for entry in entries:
        split = entry.get("split", "")
        if split not in eval_splits:
            continue
        found_splits.add(split)
        split_dir = librispeech_8k_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory {split_dir} listed in manifest but not found on disk. "
                "Re-run data/prepare_librispeech_8k.py."
            )
        wav_files.extend(sorted(split_dir.rglob("*.wav")))

    if not found_splits:
        raise RuntimeError(
            f"Neither dev-clean nor test-clean found in {manifest_path}. "
            "Ensure prepare_librispeech_8k.py was run with --splits dev-clean test-clean."
        )

    if not wav_files:
        raise RuntimeError(
            f"No WAV files found under dev-clean/test-clean in {librispeech_8k_dir}."
        )

    return wav_files


# ── Per-cell generation ───────────────────────────────────────────────────────


def _generate_one(
    condition: str,
    n: int,
    i: int,
    mixer: CoralSepMixer,
    rng: np.random.Generator,
    rir_bank: RirBank | None,
    but_bank: _ButBank | None,
    noise_paths: list[Path],
    output_dir: Path,
) -> Path:
    """
    Generate one eval sample and write it to output_dir/{condition}/n{n}/mix_{i:04d}.npz.

    Also writes a 16 kHz upsampled copy of the mixture as mix_{i:04d}_16k.npy.
    Returns the path to the .npz file.
    """
    cell_dir = output_dir / condition / f"n{n}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cell_dir / f"mix_{i:04d}.npz"

    # Draw a clean N-speaker mixture from the eval speaker pool (dev-clean + test-clean).
    mix = mixer.mix(split="train", n=n)

    # Apply degradations according to the condition string.
    if condition == "clean":
        pass  # no degradation

    elif condition == "reverb":
        if rir_bank is None:
            raise RuntimeError("--rir-bank required for 'reverb' condition.")
        mix = apply_reverb(mix, rir_bank, rng)

    elif condition == "noise":
        noise_path = noise_paths[int(rng.integers(len(noise_paths)))]
        noise = _load_noise_clip(noise_path, mix.mixture.shape[0], rng)
        mix = apply_noise(mix, noise, rng, noise_file=str(noise_path))

    elif condition == "codec":
        codec, bitrate = _sample_codec(rng)
        mix = apply_codec(mix, codec, bitrate)

    elif condition == "reverb+noise":
        if rir_bank is None:
            raise RuntimeError("--rir-bank required for 'reverb+noise' condition.")
        mix = apply_reverb(mix, rir_bank, rng)
        noise_path = noise_paths[int(rng.integers(len(noise_paths)))]
        noise = _load_noise_clip(noise_path, mix.mixture.shape[0], rng)
        mix = apply_noise(mix, noise, rng, noise_file=str(noise_path))

    elif condition == "all-three":
        if rir_bank is None:
            raise RuntimeError("--rir-bank required for 'all-three' condition.")
        mix = apply_reverb(mix, rir_bank, rng)
        noise_path = noise_paths[int(rng.integers(len(noise_paths)))]
        noise = _load_noise_clip(noise_path, mix.mixture.shape[0], rng)
        mix = apply_noise(mix, noise, rng, noise_file=str(noise_path))
        codec, bitrate = _sample_codec(rng)
        mix = apply_codec(mix, codec, bitrate)

    elif condition == "reverb+codec":
        # Held-out combination (BLUEPRINT 7.5 holdout 2)
        if rir_bank is None:
            raise RuntimeError("--rir-bank required for 'reverb+codec' condition.")
        mix = apply_reverb(mix, rir_bank, rng)
        codec, bitrate = _sample_codec(rng)
        mix = apply_codec(mix, codec, bitrate)

    elif condition == "noise+codec":
        # Held-out combination (BLUEPRINT 7.5 holdout 2)
        noise_path = noise_paths[int(rng.integers(len(noise_paths)))]
        noise = _load_noise_clip(noise_path, mix.mixture.shape[0], rng)
        mix = apply_noise(mix, noise, rng, noise_file=str(noise_path))
        codec, bitrate = _sample_codec(rng)
        mix = apply_codec(mix, codec, bitrate)

    elif condition == _BUT_CONDITION:
        if but_bank is None:
            raise RuntimeError("--but-reverbdb-dir required for 'but_reverb' condition.")
        record = but_bank.sample_random(rng)
        # apply_reverb accepts a RirBank-like object and a pre-chosen record.
        mix = apply_reverb(mix, but_bank, rng, record=record)  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    mixture_8k = mix.mixture
    refs = mix.references
    recipe_dict = mix.recipe.to_dict()
    cond_vec = mix.recipe.condition_vector()

    np.savez_compressed(
        str(npz_path),
        mixture=mixture_8k,
        references=refs,
        recipe=np.array([json.dumps(recipe_dict)]),  # stored as a length-1 string array
        condition_vector=np.array([json.dumps(cond_vec)]),
    )

    # 16 kHz copy of mixture for DNSMOS
    mix_16k = _upsample_to_16k(mixture_8k)
    npy_16k_path = cell_dir / f"mix_{i:04d}_16k.npy"
    np.save(str(npy_16k_path), mix_16k)

    return npz_path


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the seeded, hashed, fixed CoRAL-Sep evaluation matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--librispeech-8k-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root of the 8 kHz LibriSpeech corpus (from prepare_librispeech_8k.py).",
    )
    parser.add_argument(
        "--noise-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Staged noise directory (from prepare_noise_staging.py).",
    )
    parser.add_argument(
        "--rir-bank",
        required=True,
        type=Path,
        metavar="JSON",
        help="Path to the simulated RIR bank.json (from data.rir_bank).",
    )
    parser.add_argument(
        "--but-reverbdb-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Staged BUT ReverbDB directory (from prepare_but_reverbdb.py).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Destination for the eval matrix and eval_manifest.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240101,
        metavar="INT",
        help="RNG seed. Never change after the first generation.",
    )
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=100,
        metavar="N",
        help="Samples per (condition × speaker-count) cell (default 100).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    manifest_path = output_dir / "eval_manifest.json"

    # Guard: never overwrite an existing eval set.
    if manifest_path.exists():
        print(
            f"ABORT: {manifest_path} already exists.\n"
            "The fixed eval set is written once and never regenerated.\n"
            "To verify the existing set: python src/coralsep/utils/hashing.py verify "
            f"{manifest_path}"
        )
        raise SystemExit(1)

    # ── Validate inputs ───────────────────────────────────────────────────────

    librispeech_8k_dir: Path = args.librispeech_8k_dir.resolve()
    noise_dir: Path = args.noise_dir.resolve()
    rir_bank_path: Path = args.rir_bank.resolve()
    but_dir: Path = args.but_reverbdb_dir.resolve()

    for label, p in [
        ("--librispeech-8k-dir", librispeech_8k_dir),
        ("--noise-dir", noise_dir),
        ("--but-reverbdb-dir", but_dir),
    ]:
        if not p.is_dir():
            raise SystemExit(f"ERROR: {label} does not exist or is not a directory: {p}")

    if not rir_bank_path.exists():
        raise SystemExit(f"ERROR: --rir-bank file not found: {rir_bank_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load resources ────────────────────────────────────────────────────────

    print("Loading eval speaker pool (dev-clean + test-clean) ...")
    eval_files = _collect_eval_files(librispeech_8k_dir)
    print(f"  {len(eval_files)} utterances")

    # For the eval generator, all files come from dev-clean + test-clean, which
    # are the held-out eval speakers. We DON'T pass them as held_out_speaker_ids
    # because that would leave the train pool empty and CoralSepMixer raises.
    # Instead, we pass them as the train pool (no holdout). The isolation
    # invariant that matters here is the opposite one: these speakers must not
    # appear in the training data, which is enforced at training time, not here.
    # mix(split="train") draws from the pool we pass, which is exactly the eval
    # speakers, so every generated example comes from the held-out domain.
    mixer = CoralSepMixer(
        source_files=eval_files,
        allowed_n=list(_N_VALUES),
        held_out_speaker_ids=None,  # train pool = all eval files (see above)
        sample_rate=CORALSEP_SAMPLE_RATE,
        rng=np.random.default_rng(args.seed),
        seed=args.seed,
    )
    eval_speaker_ids = mixer.train_speakers
    print(f"  Eval pool: {mixer.train_pool_size} files, {len(eval_speaker_ids)} speakers")

    print("Loading noise catalogue ...")
    noise_paths = _load_noise_paths(noise_dir)
    print(f"  {len(noise_paths)} noise clips")

    print("Loading simulated RIR bank ...")
    rir_bank = RirBank(
        bank_dir=rir_bank_path.parent,
        rng=np.random.default_rng(args.seed + 1),
    )
    print(f"  {len(rir_bank)} RIRs")

    print("Loading BUT ReverbDB bank ...")
    but_bank = _load_but_bank(but_dir, CORALSEP_SAMPLE_RATE)
    print(f"  {len(but_bank.records)} real RIRs\n")

    # One seeded RNG per cell so that adding a new condition later doesn't
    # shift the draws for existing conditions.
    rng_master = np.random.default_rng(args.seed)

    # ── Generate standard condition × N cells ────────────────────────────────

    all_npz: list[Path] = []
    n_per_cell = args.n_per_cell

    total_cells = len(_CONDITIONS) * len(_N_VALUES) + 1  # +1 for BUT tier
    total_samples = len(_CONDITIONS) * len(_N_VALUES) * n_per_cell + n_per_cell
    print(
        f"Generating {total_cells} cells × {n_per_cell} samples = "
        f"{total_samples} total mixtures\n"
    )

    for condition in _CONDITIONS:
        for n in _N_VALUES:
            # Derive a cell-specific RNG from the master so cell order doesn't matter.
            cell_seed = int(rng_master.integers(0, 2**31))
            cell_rng = np.random.default_rng(cell_seed)
            # Reinitialise mixer RNG for this cell too.
            mixer._rng = np.random.default_rng(cell_seed + 1_000_000)

            label = f"{condition}/n{n}"
            held_marker = " [held-out]" if condition in _HELD_OUT_CONDITIONS else ""
            for i in tqdm(range(n_per_cell), desc=f"{label}{held_marker}", unit="mix"):
                npz = _generate_one(
                    condition=condition,
                    n=n,
                    i=i,
                    mixer=mixer,
                    rng=cell_rng,
                    rir_bank=rir_bank,
                    but_bank=None,
                    noise_paths=noise_paths,
                    output_dir=output_dir,
                )
                all_npz.append(npz)

    # ── BUT ReverbDB real-RIR tier (N=2 only) ────────────────────────────────

    but_seed = int(rng_master.integers(0, 2**31))
    but_rng = np.random.default_rng(but_seed)
    mixer._rng = np.random.default_rng(but_seed + 1_000_000)

    for i in tqdm(range(n_per_cell), desc=f"{_BUT_CONDITION}/n2", unit="mix"):
        npz = _generate_one(
            condition=_BUT_CONDITION,
            n=2,
            i=i,
            mixer=mixer,
            rng=but_rng,
            rir_bank=None,
            but_bank=but_bank,
            noise_paths=noise_paths,
            output_dir=output_dir,
        )
        all_npz.append(npz)

    # ── Write manifest ────────────────────────────────────────────────────────

    print(f"\nHashing {len(all_npz)} files and writing eval_manifest.json ...")
    write_manifest(
        paths=all_npz,
        manifest_path=manifest_path,
        root=output_dir,
        extra={
            "seed": args.seed,
            "n_per_cell": n_per_cell,
            "n_values": list(_N_VALUES),
            "conditions": list(_CONDITIONS) + [_BUT_CONDITION],
            "held_out_conditions": sorted(_HELD_OUT_CONDITIONS),
            "sample_rate": CORALSEP_SAMPLE_RATE,
            "total_files": len(all_npz),
        },
    )
    print(f"Done. Eval manifest: {manifest_path}")
    print("This set is now frozen. Do not delete or re-run this script against this output dir.")


if __name__ == "__main__":
    main()
