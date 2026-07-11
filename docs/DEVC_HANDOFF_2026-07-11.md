# Dev C completion handoff — 2026-07-11

## What this patch completes

| Item | Result | Evidence |
|---|---|---|
| P0-INT4 | Complete | `tests/test_p0_e2e.py`: layered YAML → baseline runner → fake expert → PIT SI-SDRi → JSON/Markdown artifacts |
| P1-C3 code deliverable | Complete | `align/integration.py::run_and_align_long`, deterministic cross-chunk permutation test |
| Missing P1 integration tracking | Restored | `align/integration.py`, `tests/test_align_integration.py`, optional `tests/test_m1_real_experts.py` |
| P1-INT2 execution path | Ready; real acceptance still required | `scripts/validate_alignment.py` and `docs/RUNPOD_DEVC_VALIDATION.md` |
| Hungarian numerical warnings | Fixed | Safe row normalization, finite neutral costs for silent/invalid rows, regression tests |
| P2-C1 | Complete | Scene Analyzer → two-level router wiring in `train/trainer.py`, covered by `tests/test_e2e_forward.py` |
| P3-C1 | Code deliverable complete | Calibrated MLP, checkpoint contract, and unit tests in `models/stop_classifier.py` |
| P3-C2 | Complete | Count BCE flows from Scene Analyzer logits into `CompositeLoss`; gradient E2E test |
| P3-C3 / P3-C4 | Complete as generators | `eval/counting_report.py` emits JSON, CSV, Markdown, and dependency-free SVG artifacts |

## Verification performed in this patch

```text
Ruff:  all changed Python files passed
Black: all changed Python files passed
Pytest Dev C + trainer/cascade targeted suite: 52 passed
Pytest complete suite: 396 passed, 2 skipped
```

The two skips are the opt-in real-model tests in `tests/test_m1_real_experts.py`. They require
`RUN_REAL_EXPERTS=1`, a Libri3Mix test directory, model downloads, and a CUDA host.

## Items that must remain open

- **P1-INT2:** one real >4 s Libri3Mix clip must produce zero cross-chunk identity switches.
- **P2-INT3 / P2-INT4 / P2-INT5:** require a real short training run and validation outputs.
- **P3-C5 / P3-INT2:** require Libri2–5Mix features, a trained stop-classifier checkpoint,
  and unknown-N evaluation.
- **P6-C1:** the mock-ready UI exists, but final cascade integration and transcripts depend
  on the M5 system.

These are data/model/gate tasks, not missing local code that can be honestly marked complete
from a CPU-only repository audit.
