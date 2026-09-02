"""
Seeded, hashed fixed evaluation set manifests (Dev A, BLUEPRINT §7.4).

Generates JSONL recipe manifests and SHA-256 hash sidecars under
``datasets/fixed_eval/``. Does not require corpora on disk: each row records the
deterministic synthesis recipe and placeholder paths so audio can be rendered
later without changing the manifest identity.

Holdout flags mark condition-combination cells (reverb+codec, noise+codec)
that must never appear in gate or joint-training data (BLUEPRINT §7.5).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coralsep.utils.hashing import hash_bytes, hash_file

DEFAULT_EVAL_SEED: int = 42
"""Global seed for the fixed evaluation matrix (BLUEPRINT §13)."""

MANIFEST_VERSION: str = "coralsep-fixed-eval-v1"


@dataclass(frozen=True)
class EvalTierSpec:
    """One cell in the BLUEPRINT §7.4 evaluation matrix."""

    tier_id: str
    description: str
    speaker_counts: tuple[int, ...]
    n_items: int
    conditions: dict[str, Any] = field(default_factory=dict)
    gate_holdout: bool = False
    source: str = "synthesis"
    requires_reference: bool = True


# Full evaluation matrix from BLUEPRINT §7.4.
EVAL_MATRIX: tuple[EvalTierSpec, ...] = (
    EvalTierSpec(
        tier_id="clean_2_3",
        description="Libri2Mix / Libri3Mix test at 8 kHz — literature baseline",
        speaker_counts=(2, 3),
        n_items=500,
        conditions={"degradation": "clean", "overlap": "full"},
        source="librimix_test",
    ),
    EvalTierSpec(
        tier_id="sparse_overlap",
        description="SparseLibriMix test — overlap ratio 0–100%",
        speaker_counts=(2,),
        n_items=200,
        conditions={"degradation": "clean", "overlap": "sparse"},
        source="sparselibrimix_test",
    ),
    EvalTierSpec(
        tier_id="sparse_overlap_3spk",
        description="Custom 3-speaker sparse overlap (SparseLibriMix is 2-spk only)",
        speaker_counts=(3,),
        n_items=200,
        conditions={"degradation": "clean", "overlap": "sparse"},
        source="synthesis",
    ),
    EvalTierSpec(
        tier_id="reverb_noisy_primary",
        description="Primary benchmark: reverb + noise LibriMix at 8 kHz",
        speaker_counts=(2,),
        n_items=500,
        conditions={"degradation": "reverb+noise", "snr_db_range": [-6, 10], "t60_range": [0.2, 1.0]},
        source="synthesis",
    ),
    EvalTierSpec(
        tier_id="reverb_noisy_highn",
        description="Reverb-noisy at N=3,4,5",
        speaker_counts=(3, 4, 5),
        n_items=200,
        conditions={"degradation": "reverb+noise", "snr_db_range": [-6, 10], "t60_range": [0.2, 1.0]},
        source="synthesis",
    ),
    EvalTierSpec(
        tier_id="reverb_only",
        description="Clean-reverb LibriMix — isolates adapter_reverb",
        speaker_counts=(2, 3),
        n_items=200,
        conditions={"degradation": "reverb", "t60_range": [0.2, 1.0]},
        source="synthesis",
    ),
    EvalTierSpec(
        tier_id="real_rir_reverb",
        description="BUT ReverbDB measured RIRs convolved with LibriSpeech test",
        speaker_counts=(2,),
        n_items=200,
        conditions={"degradation": "real_rir", "rir_source": "but_reverbdb_slr17"},
        source="but_reverbdb",
    ),
    EvalTierSpec(
        tier_id="codec_only",
        description="LibriMix + ffmpeg codec (Opus/AAC/AMR-NB)",
        speaker_counts=(2,),
        n_items=200,
        conditions={"degradation": "codec", "codecs": ["opus_6k", "aac_16k", "amr_nb"]},
        source="synthesis",
    ),
    EvalTierSpec(
        tier_id="reverb_codec_holdout",
        description="Reverb + codec — held out of gate training",
        speaker_counts=(2, 4),
        n_items=200,
        conditions={"degradation": "reverb+codec", "t60_range": [0.2, 1.0]},
        gate_holdout=True,
        source="synthesis",
    ),
    EvalTierSpec(
        tier_id="noise_codec_holdout",
        description="Noise + codec — held out of gate training",
        speaker_counts=(2, 4),
        n_items=200,
        conditions={"degradation": "noise+codec", "snr_db_range": [-6, 10]},
        gate_holdout=True,
        source="synthesis",
    ),
    EvalTierSpec(
        tier_id="high_count_clean",
        description="Libri4Mix / Libri5Mix test — count break-point curve",
        speaker_counts=(4, 5),
        n_items=200,
        conditions={"degradation": "clean", "overlap": "full"},
        source="librimix_test",
    ),
    EvalTierSpec(
        tier_id="high_count_degraded",
        description="High count under reverb + noise",
        speaker_counts=(4, 5),
        n_items=200,
        conditions={"degradation": "reverb+noise", "snr_db_range": [-6, 10], "t60_range": [0.2, 1.0]},
        source="synthesis",
    ),
    EvalTierSpec(
        tier_id="libricss_real",
        description="LibriCSS 1ch downmix — no clean references",
        speaker_counts=(2, 3, 4, 5),
        n_items=0,
        conditions={"degradation": "real_recording"},
        source="libricss",
        requires_reference=False,
    ),
    EvalTierSpec(
        tier_id="band_recovery_gain",
        description="Matched pairs: 8 kHz pass-through vs band-recovered 16 kHz",
        speaker_counts=(2,),
        n_items=500,
        conditions={"degradation": "reverb+noise", "band_recovery_ablation": True},
        source="synthesis",
    ),
)


def _item_seed(global_seed: int, tier: str, n_speakers: int, index: int) -> int:
    """Deterministic per-item seed from tier, N, and index."""
    payload = f"{global_seed}:{tier}:{n_speakers}:{index}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def _placeholder_recipe(
    tier: str,
    n_speakers: int,
    index: int,
    item_seed: int,
    spec: EvalTierSpec,
) -> dict[str, Any]:
    """Build a recipe row when source corpora are absent."""
    degradation = spec.conditions.get("degradation", "clean")
    return {
        "n_speakers": n_speakers,
        "speaker_ids": [f"placeholder_spk_{(item_seed + k) % 1000:04d}" for k in range(n_speakers)],
        "source_files": [
            f"PLACEHOLDER/librispeech/test-clean/{(item_seed + k) % 1000:04d}.flac"
            for k in range(n_speakers)
        ],
        "snr_db": None if "noise" not in degradation else float(-6 + (item_seed % 17)),
        "t60_s": None if "reverb" not in degradation and degradation != "real_rir" else float(0.2 + (item_seed % 9) * 0.1),
        "codec_name": None if "codec" not in degradation else ["opus", "aac", "amr-nb"][item_seed % 3],
        "codec_bitrate_bps": None if "codec" not in degradation else [6000, 16000, 4750][item_seed % 3],
        "overlap_ratio": 1.0 if spec.conditions.get("overlap") != "sparse" else float((item_seed % 101) / 100.0),
        "rir_file": (
            f"PLACEHOLDER/but_reverbdb/rir_{item_seed % 1244:04d}.wav"
            if degradation == "real_rir"
            else None
        ),
        "seed": item_seed,
        "tier": tier,
        "index": index,
        "conditions": dict(spec.conditions),
        "source_status": "placeholder",
    }


def _manifest_stem(tier: str, n_speakers: int) -> str:
    return f"{tier}_n{n_speakers}"


def manifest_jsonl_path(out_dir: str | Path, tier: str, n_speakers: int) -> Path:
    """Return the JSONL path for one (tier, N) manifest."""
    return Path(out_dir) / f"{_manifest_stem(tier, n_speakers)}.jsonl"


def manifest_hash_path(out_dir: str | Path, tier: str, n_speakers: int) -> Path:
    """Return the SHA-256 sidecar path for one (tier, N) manifest."""
    return Path(out_dir) / f"{_manifest_stem(tier, n_speakers)}.sha256"


def build_eval_manifest(
    tier: str,
    n_speakers: int,
    n_items: int,
    seed: int,
    out_dir: str | Path,
) -> Path:
    """
    Write a JSONL manifest and SHA-256 hash file for one evaluation cell.

    Args:
        tier: Evaluation tier id (must match an EvalTierSpec.tier_id).
        n_speakers: Speaker count N for this manifest.
        n_items: Number of mixture rows to record.
        seed: Global seed mixed into per-item seeds.
        out_dir: Output directory (typically ``datasets/fixed_eval/``).

    Returns:
        Path to the written JSONL manifest.
    """
    if n_items < 0:
        raise ValueError(f"n_items must be non-negative, got {n_items}")

    spec = next((s for s in EVAL_MATRIX if s.tier_id == tier), None)
    if spec is None:
        raise ValueError(f"unknown tier: {tier!r}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = manifest_jsonl_path(out, tier, n_speakers)
    hash_path = manifest_hash_path(out, tier, n_speakers)

    rows: list[dict[str, Any]] = []
    for index in range(n_items):
        item_seed = _item_seed(seed, tier, n_speakers, index)
        item_id = f"{tier}_n{n_speakers}_{index:06d}"
        recipe = _placeholder_recipe(tier, n_speakers, index, item_seed, spec)
        rows.append(
            {
                "item_id": item_id,
                "tier": tier,
                "n_speakers": n_speakers,
                "index": index,
                "seed": item_seed,
                "global_seed": seed,
                "gate_holdout": spec.gate_holdout,
                "requires_reference": spec.requires_reference,
                "source": spec.source,
                "recipe": recipe,
                "paths": {
                    "mixture": f"audio/{tier}/n{n_speakers}/{item_id}_mix.wav",
                    "references": [
                        f"audio/{tier}/n{n_speakers}/{item_id}_s{k + 1}.wav"
                        for k in range(n_speakers)
                    ],
                    "mixture_16k": f"audio/{tier}/n{n_speakers}/{item_id}_mix_16k.wav",
                },
            }
        )

    header = {
        "manifest_version": MANIFEST_VERSION,
        "tier": tier,
        "n_speakers": n_speakers,
        "n_items": n_items,
        "seed": seed,
        "gate_holdout": spec.gate_holdout,
        "description": spec.description,
    }

    with jsonl_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": header}, sort_keys=True) + "\n")
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    digest = hash_file(jsonl_path)
    hash_path.write_text(f"{digest}  {jsonl_path.name}\n", encoding="utf-8")

    return jsonl_path


def generate_all_manifests(
    out_dir: str | Path,
    seed: int = DEFAULT_EVAL_SEED,
) -> list[Path]:
    """
    Generate every manifest in the BLUEPRINT §7.4 evaluation matrix.

    LibriCSS uses ``n_items=0`` as a marker for the full test set; a single
    metadata-only manifest with zero recipe rows is still written and hashed.

    Args:
        out_dir: Root directory for manifests (``datasets/fixed_eval/``).
        seed: Global seed recorded in every manifest header.

    Returns:
        List of written JSONL manifest paths.
    """
    written: list[Path] = []
    matrix_index_path = Path(out_dir) / "matrix_index.json"

    index_entries: list[dict[str, Any]] = []

    for spec in EVAL_MATRIX:
        for n_spk in spec.speaker_counts:
            n_items = spec.n_items
            path = build_eval_manifest(spec.tier_id, n_spk, n_items, seed, out_dir)
            written.append(path)
            index_entries.append(
                {
                    "tier": spec.tier_id,
                    "n_speakers": n_spk,
                    "n_items": n_items,
                    "gate_holdout": spec.gate_holdout,
                    "manifest": path.name,
                    "sha256": hash_file(path),
                }
            )

    matrix_doc = {
        "manifest_version": MANIFEST_VERSION,
        "seed": seed,
        "n_manifests": len(index_entries),
        "manifests": index_entries,
        "matrix_hash": hash_bytes(
            json.dumps(index_entries, sort_keys=True, separators=(",", ":")).encode()
        ),
    }
    matrix_index_path.write_text(json.dumps(matrix_doc, indent=2, sort_keys=True), encoding="utf-8")

    return written


def load_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Load a JSONL manifest, returning the header meta and item rows.

    Args:
        path: Path to a manifest JSONL file.

    Returns:
        (meta_dict, item_rows) where meta comes from the ``_meta`` line.
    """
    p = Path(path)
    meta: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "_meta" in row:
            meta = row["_meta"]
        else:
            items.append(row)
    return meta, items


def tier_spec(tier_id: str) -> EvalTierSpec:
    """Look up a tier spec by id."""
    for spec in EVAL_MATRIX:
        if spec.tier_id == tier_id:
            return spec
    raise KeyError(f"unknown tier: {tier_id!r}")
