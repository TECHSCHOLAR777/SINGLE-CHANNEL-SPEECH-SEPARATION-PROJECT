"""
CoRAL-Sep synthesis utilities: fixed evaluation manifests (Dev A, BLUEPRINT §7.4).

Manifest generation lives in fixed_eval.py. Full audio rendering is deferred to
the synthesis pipeline; this package pins the seeded, hashed evaluation matrix
before any model training begins.
"""

from __future__ import annotations

from coralsep.data.synthesis.fixed_eval import (
    EVAL_MATRIX,
    EvalTierSpec,
    build_eval_manifest,
    generate_all_manifests,
    load_manifest,
    manifest_hash_path,
    manifest_jsonl_path,
)

__all__ = [
    "EVAL_MATRIX",
    "EvalTierSpec",
    "build_eval_manifest",
    "generate_all_manifests",
    "load_manifest",
    "manifest_hash_path",
    "manifest_jsonl_path",
]
