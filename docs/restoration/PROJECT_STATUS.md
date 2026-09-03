# Project Status

**Purpose:** one-page answer to "what shape is this project in right now".

**Status:** 🟠 AMBER. Every defect that does not need compute is fixed. What remains is scientific, not mechanical.

**Last verified:** 2026-09-04

**Source of truth:** the commands recorded in `VALIDATION_MATRIX.md`, run on this machine.

---

## Summary in three sentences

The codebase was in better shape than its own documentation suggested, and it is now in better shape than that: 563 tests pass with nothing uncollectable, every module in the package imports, and lint and formatting are clean. Twenty-seven of the thirty-nine problems found are closed, most of them connective tissue that survived because CI had been watching a branch that does not exist for 158 commits. What remains unresolved is scientific rather than mechanical, and it is dominated by one fact: the primary graded axis of this project, speaker count accuracy, has never been measured, because every evaluation run handed the true count to both systems.

## Health by dimension

| Dimension | Status | Evidence |
|---|---|---|
| Repository integrity | 🟢 GREEN | 13 branches reconciled, no divergence, no history damage, no secrets committed |
| Archive reconciliation | 🟢 GREEN | all four archive-only artifacts recovered and committed |
| Environment | 🟢 GREEN | backbone pinned by commit and verified installable; every runtime import declared and enforced by a test |
| Import and runtime integrity | 🟢 GREEN | 94 of 94 modules import in an environment with the backbone installed |
| Data pipeline | 🟢 GREEN | all modules import, preparation split into its own subpackage, config placeholders resolved |
| Model | 🟠 AMBER | architecture verified and parameter counts measured; the reverb adapter is measurably harmful (I-025) and the gate does not route (I-003) |
| Training | 🟠 AMBER | all seven stage scripts import; Stage 2 was never run (I-024); Stage 4 stopped at epoch 14 of 20 with loss still falling |
| Inference | 🟢 GREEN | the full pipeline runs end to end against a mock expert, which it could not do before |
| Evaluation | 🟠 AMBER | every evaluation module imports; results still carry the oracle speaker count (I-002) |
| Tests | 🟢 GREEN | 563 passed, 11 skipped, nothing uncollectable, up from 504 with three broken modules |
| CI | 🟢 GREEN | triggers on `master`, three jobs, import sweep and credential scan |
| Reproducibility | 🟠 AMBER | a clean environment can now obtain the backbone and run the suite; past results still cannot be tied to specific weights, since no checkpoint has a hash |
| Documentation | 🟢 GREEN | README and knowledge base rewritten against the artifacts; every number traces to a file in `results/` |
| Security | 🟠 AMBER | repository clean and guarded by a pre-commit hook and a CI job; the five archive credentials still need rotating by the owner (I-001) |

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

None of these is a blocker on knowledge. Each is a blocker on access.

1. **Compute.** No GPU here. Nine open tickets need a training run or a full evaluation.
2. **Kaggle credentials.** Every checkpoint lives in datasets under one account. Reading the epoch field out of a Stage 1 checkpoint would settle I-022 in minutes.
3. **I-001, credential rotation.** Five live tokens sit in the supplied archive. None reached the repository. Rotation is the owner's action.

## Highest value next work, in order

1. **Read the Stage 1 reverb training target** (I-025). No compute needed. If the reference signal is the wet one, that explains the strongest negative result in the project, and possibly the flat gate as well.
2. **Remove the oracle speaker count from evaluation** (I-002). The counting mechanism and the metrics both already exist and are tested. The code half can be written and unit-tested here.
3. **Retain per-sample scores in the result artifact** (I-026), so confidence intervals become possible at all.
4. Then, with compute: rerun all four splits at n of at least 300, train Stage 2 for the ablation, measure calibration error.

Steps 1 to 3 need nothing but a laptop, and doing them first means the eventual GPU run answers the right question. A rerun that still supplies the oracle count would produce another set of numbers about the wrong thing.

See [`../../ISSUES.md`](../../ISSUES.md) for what each open ticket would cost to close.
