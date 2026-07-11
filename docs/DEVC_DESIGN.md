# Dev C: Evaluation, Alignment, Counting, Demo

Owner: Dev C. Repository audit date: 2026-07-11.

The code-side Dev C vertical is integrated and covered by the default test suite. At this
snapshot, **396 tests pass and 2 opt-in real-model tests skip**. The remaining yellow Dev C
items require real datasets, pretrained weights, training, or later milestone gates; they are
listed explicitly below rather than being hidden behind code-complete labels.

## What this vertical covers

| Module | Purpose | Key guarantee |
|---|---|---|
| `utils/config.py` | Layered YAML loader, deep merge, dot-path access | Tunables are loaded consistently; later files and explicit overrides win |
| `eval/metrics.py` | SI-SDR, SI-SDRi, PIT, counting | Framework-free NumPy implementation; mismatched unknown-N cases are explicit |
| `eval/reporting.py` | Canonical JSONL run log and report queries | Tables and figures derive from one persisted record schema |
| `eval/counting_report.py` | P3-C3/C4 artifact generator | Emits summary JSON, confusion/calibration CSV, Markdown, and SVG |
| `align/embeddings.py` | ECAPA stream embeddings | Shared embedding interface for expert and chunk alignment |
| `align/hungarian.py` | Stream alignment across experts/chunks | Embedding cost with xcorr fallback; silent/invalid rows stay finite and neutral |
| `align/chunking.py` | Persistent speaker tracks and overlap-add | Matches against track history so temporary silence does not erase identity |
| `align/integration.py` | Expert-pair and long-audio orchestration | Real wrappers and deterministic CI fakes share the same protocol |
| `models/router.py` | Two-level adaptive router | Sigmoid gates permit co-activation; includes load-balance and null losses |
| `models/stop_classifier.py` | Learned calibrated stop decision | Temperature scaling, frozen feature order, safe checkpoint loading |
| `train/train_stop_classifier.py` | Stop-classifier training | Real feature JSONL path plus a no-data self-test |
| `demo/app.py` | Mock-ready Gradio UI | Separation engine is injected; the UI does not hard-code an expert |

## P0: the metric and integration contract

`eval/metrics.py` is verified by known-answer tests:

- an oracle estimate gives a very high SI-SDR;
- using the mixture as the estimate gives 0 dB SI-SDR improvement;
- controlled orthogonal noise recovers its requested SNR;
- a permuted oracle recovers the exact permutation with Hungarian PIT.

P0-INT4 is covered by `tests/test_p0_e2e.py`. It loads layered YAML through
`utils.config.load_config`, constructs the baseline configuration, executes the baseline
runner with a deterministic injected expert, verifies permutation-invariant SI-SDRi, and
checks the persisted JSON/Markdown outputs. This is intentionally weight-free so it runs at
every gate.

## Unknown-N handling

`pit_si_sdr` handles count mismatch explicitly:

- **Over-separation:** the best subset is matched and extra streams are listed in
  `unassigned_estimates`.
- **Under-separation:** missing references follow the configured `missing_policy` and are
  listed in `missing_references`.

The P3 report path uses the same `RunLog` records. `generate_counting_report` writes a count
summary, full true-vs-estimated confusion matrix, optional confidence calibration curve,
Markdown, and SVGs. Generator completion is separate from the M3 gate: real classifier
outputs are still required before the project can claim a measured count result.

## Alignment and long-form identity

`run_and_align` executes two expert wrappers, fills missing ECAPA embeddings, computes the
Hungarian assignment, and reorders the second result into anchor order.

`run_and_align_long` uses overlapping chunks and `ChunkStitcher` to maintain persistent track
IDs before overlap-add. Deterministic integration tests deliberately permute stream order
between chunks and verify stable identity. The remaining P1-INT2 acceptance check must use
real speech because random/synthetic noise does not provide meaningful speaker identity to
ECAPA. The exact RunPod command is in `docs/RUNPOD_DEVC_VALIDATION.md`.

Alignment normalization is defensive across BLAS implementations. Silent, near-zero,
extreme-magnitude, and non-finite rows cannot create NaN or infinity costs; such rows receive
a neutral cost instead of corrupting `linear_sum_assignment`.

## SR-CorrNet and fallback boundary

No SR-CorrNet checkpoint is committed. The wrapper is configuration-gated, and
`get_expensive_expert` falls back to TF-GridNet or SepFormer when the SR-CorrNet repository
and weights are unavailable. Record checkpoint provenance before trusting any number that
uses the expensive path. Do not commit model weights or generated audio.

## What is complete versus still data-bound

Code-complete Dev C items include P0-C1–C7, P0-INT2, P0-INT4, P1-C1–C4, P2-C1–C3,
P2-B2 instrumentation, P2-INT2, P3-C1–C4, and P3-INT1.

These must remain open until their external evidence exists:

- P1-INT2: real >4 s Libri3Mix identity-lock validation;
- P2-INT3–INT5: short training, validation comparison, and measured escalation rate;
- P3-C5/P3-INT2: real Libri2–5Mix training and unknown-N evaluation;
- P6-C1 final integration: real M5 cascade and transcript path.

## Running the verified checks

```bash
python -m pip install -e ".[dev]"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q
python -m train.train_stop_classifier --self-test
python -m demo.app --mock
```

For real P1 acceptance, follow `docs/RUNPOD_DEVC_VALIDATION.md`.
