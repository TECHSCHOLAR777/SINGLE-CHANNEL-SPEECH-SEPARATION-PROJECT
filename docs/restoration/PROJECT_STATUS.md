# Project Status

**Purpose:** one-page answer to "what shape is this project in right now".

**Status:** [AMBER]

**Last verified:** 2026-09-02

**Source of truth:** the commands recorded in `VALIDATION_MATRIX.md`, run on this machine.

---

## Summary in three sentences

The codebase is more complete and more honest than its own README suggests: seven training stages, a full evaluation harness, a calibration suite, an inference pipeline, a Gradio demo and 513 tests all exist, and 504 of those tests pass. What is broken is mostly connective tissue, ten import failures caused by symbol renames that were never propagated, surviving because the CI workflow watches a branch that does not exist. What is genuinely unresolved is scientific rather than mechanical: the primary graded axis of the project, speaker count accuracy, has never been measured, because every evaluation run so far handed the true speaker count to both systems.

---

## Health by dimension

| Dimension | Status | Evidence |
|---|---|---|
| Repository integrity | [GREEN] | 158 commits, 13 branches reconciled, no divergence, no history damage, no secrets committed |
| Archive reconciliation | [GREEN] | 20 of 21 archive source files identical to `master`; 3 recoverable items identified |
| Environment | [RED] | `sr_corrnet` backbone loader is undeclared and unobtainable (I-019); requirements are unpinned despite claiming otherwise (I-020) |
| Import and runtime integrity | [AMBER] | 83 of 93 modules import; 10 fail on renamed symbols and v1 residue (I-004 to I-010) |
| Data pipeline | [AMBER] | 26 modules, all import, 6,834 lines, well tested; no dataset present locally and one config carries an unresolved TODO (I-029) |
| Model | [AMBER] | All architecture modules import and are unit-tested; the reverb adapter is measurably harmful (I-025) and the gate does not route (I-003) |
| Training | [AMBER] | All seven stage scripts import; Stage 2 was never run (I-024); Stage 4 stopped at epoch 14 of 20 |
| Inference | [RED] | `pipeline/infer.py` imports, but nothing can run without the backbone (I-019) |
| Evaluation | [RED] | `eval/matrix.py`, `eval/baselines.py` and `eval/ablation_gate.py` do not import; results carry oracle speaker count (I-002) |
| Tests | [AMBER] | 504 pass, 10 skip, 3 modules fail to collect (I-006, I-009) |
| CI | [RED] | Configured for `main`, default branch is `master`, has never executed (I-011) |
| Reproducibility | [RED] | Cannot be attempted; backbone, checkpoints and datasets are all external and only partly documented |
| Documentation | [AMBER] | Deep and unusually honest where it exists (`NUMBERS.md`, `PROJECT_HISTORY.md`), but the README contradicts it and the package metadata still names a project abandoned in July (I-016, I-017, I-018) |
| Security | [AMBER] | Repository is clean; the supplied archive carries five live credentials that must be rotated (I-001) |

---

## What is verified, and what only looks verified

| Claim | State | Basis |
|---|---|---|
| Stage 4 joint training ran for 14 epochs ending at loss 8.6809 | [VERIFIED] | epoch-by-epoch Kaggle log, wall times, checkpoint saves |
| Libri2Mix and Libri3Mix SI-SDRi improvements | [PARTIALLY_VERIFIED] | raw JSON matches the documentation exactly, but oracle N was supplied and n=30 |
| Libri5Mix SI-SDRi improvement | [PARTIALLY_VERIFIED] | same, and the raw artifact exists only in the archive |
| Stage 1 reverb adapter degrades quality | [VERIFIED] as an observation | direct diagnostic log with a correct zero-gate control |
| LoRA injection mechanism is correct | [VERIFIED] | zero-gate output matches the base model to 0.000000 |
| Gate temperature is 4.9872 | [VERIFIED] | recorded in the Stage 4c artifact and in two documents |
| Gate performs condition-aware routing | [FAILED] | at T=4.9872 the sigmoid is effectively flat |
| Speaker count accuracy | [UNVERIFIED] | never measured |
| Libri4Mix results | [UNVERIFIED] | never run |
| Stage 2 universal adapter ablation | [UNVERIFIED] | never trained |
| Calibration quality (ECE) | [UNVERIFIED] | never measured |
| v1 CA-MoSE negative results | [VERIFIED] as history | recorded with full numbers in `docs/PROJECT_HISTORY.md`, including the unflattering ones |

---

## Current blockers

1. **I-019, backbone provenance.** `sr_corrnet` is not installable. Everything downstream of the backbone is unrunnable and unverifiable until its upstream source and license are known. This needs the project owner, not more investigation.
2. **I-001, credential rotation.** Five live tokens are in the supplied archive. Rotation is the owner's action.
3. **Compute.** No GPU, no checkpoints and no datasets on this machine. Every ticket that needs a training run or a full evaluation is out of reach here and is marked [BLOCKED] for that reason rather than for a technical one.

---

## Highest value next work, in order

1. Repair the ten import failures (I-004 to I-010). No compute needed, mechanical, and it unblocks CI.
2. Recover the three uncommitted archive files (I-012, I-013, I-014) before the archive is the only copy.
3. Fix CI so the default branch is actually gated (I-011).
4. Reconcile the README with the artifacts (I-016, I-017, I-018).
5. Rename and restructure (I-031, I-028).
6. Then, and only with compute available, the scientific backlog: oracle N removal, Libri4Mix, larger n, confidence intervals, Stage 2 ablation.

The first four cost nothing but care and make every later step verifiable.
