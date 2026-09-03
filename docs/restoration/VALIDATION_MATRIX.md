# Validation Matrix

**Purpose:** exactly what has been validated, with the command that validated it and the output it produced.

**Status:** [AMBER] Static and unit-level validation complete. Everything requiring the backbone, a checkpoint or a dataset is unreachable on this machine.

**Last verified:** 2026-09-04

**Environment for every row below.** Windows 11, Python 3.14.3, torch 2.10.0+cpu, torchaudio 2.11.0+cpu, numpy 2.4.3, scipy 1.17.1, CPU only, no GPU. `speechbrain`, `asteroid`, `pyroomacoustics` and `sr_corrnet` are absent. No training was run.

---

## 1. Executed checks

| # | Area | Check | Command | Expected | Actual | Status |
|---|---|---|---|---|---|---|
| V-01 | Evidence | Archive integrity | `sha256sum .restoration/archive/calm-sep-context-2026-09-01-v2.zip` | stable hash | `85129a23f8165ce373eb99d93886d8e7436d0c06d78e9828cdc5cffeb84b855e` | [PASS] |
| V-02 | Evidence | Archive unchanged after reorganisation | same command, rerun after the move | identical hash | identical | [PASS] |
| V-03 | Repository | Clone integrity | `git rev-list --count HEAD` | 158, per the restoration pack | 158 | [PASS] |
| V-04 | Repository | Working tree clean at start | `git status --porcelain` | empty | empty | [PASS] |
| V-05 | Repository | Tracked file count preserved across the root move | `git ls-files \| wc -l` | 243 | 243 | [PASS] |
| V-06 | Reconciliation | Archive versus repository, line endings normalised | `tr -d '\r' \| sha256sum` per file, 21 files | differences identified | 18 identical, 3 genuinely different | [PASS] |
| V-07 | Reconciliation | Archive files versus all 13 branches | blob hash comparison per branch | provenance established | 3 files match no branch | [PASS] |
| V-08 | Environment | Third-party import availability | `importlib.import_module` over the declared dependency list | all present | 4 of 16 absent | [FAIL] see I-019, I-020 |
| V-09 | Imports | Every non-test module imports | import sweep over 93 modules | 93 of 93 | 83 of 93 | [FAIL] see I-004 to I-010 |
| V-10 | Tests | Collection | `pytest tests/ -q --co` | no errors | 513 collected, 3 module import errors | [FAIL] see I-006, I-009 |
| V-11 | Tests | Full run, excluding the 3 uncollectable modules | `pytest tests/ -q --ignore=tests/principle2_test.py --ignore=tests/smoke_test.py --ignore=tests/test_cached_dataset.py` | all pass | **504 passed, 10 skipped, 2 warnings, 36.06 s** | [PASS] |
| V-12 | Tests | Same run after the repository root move | same command | identical counts | 504 passed, 10 skipped, 29.49 s | [PASS] |
| V-13 | Security | Credential scan of the tracked tree | `grep -rE 'hf_[A-Za-z0-9]{20,}\|KGAT_\|ak-[A-Za-z0-9]{15,}\|as-[A-Za-z0-9]{15,}'` over tracked files | no match | no match | [PASS] |
| V-14 | Security | Credential scan of the archive | same pattern over `.restoration/zip_extract/` | no match | **1 file matches: `CONTEXT.md`, 5 credentials** | [FAIL] see I-001 |
| V-15 | Hygiene | Hard-coded machine paths | grep for `/Users/`, `/teamspace/`, `~/Desktop`, `Path.home()` | none in library code | found in `demo.py`, `eval/eval_reverb_adapter.py`, notebooks | [FAIL] see I-019, I-033 |
| V-16 | Hygiene | TODO, FIXME and placeholder markers | grep across `*.py`, `*.yaml`, `*.md` | reviewed | 13 hits, 1 actionable | [PASS] see I-029 |
| V-17 | Results | Documented numbers versus raw artifacts | manual read of `calmsep_eval.json` and `calmsep_eval_5.json` against `NUMBERS.md` | agreement | exact agreement to 3 decimal places on all 12 values | [PASS] |
| V-18 | Results | Documented Stage 4 loss curve versus the raw log | manual read of the Kaggle log against `NUMBERS.md` section 3.3 | agreement | exact agreement on all 11 recorded epochs | [PASS] |
| V-19 | CI | Workflow trigger matches the default branch | read `.github/workflows/ci.yml` against `git branch -a` | match | workflow watched `main`, default is `master` | 🔴 FAIL, fixed in `df16162` |

### After repair, 2026-09-04

Same machine, with the backbone package installed into an isolated directory.

| # | Area | Check | Actual | Status |
|---|---|---|---|---|
| V-20 | Environment | Backbone installs from its pinned commit | installs cleanly | 🟢 PASS |
| V-21 | Model | Backbone loads and patches A, B and C apply | loads | 🟢 PASS |
| V-22 | Model | Backbone parameter count | **14,031,768** | 🟢 PASS, corrects three documents |
| V-23 | Model | LoRA attachment count and adapter size | 37 modules, 101,404 params each | 🟢 PASS |
| V-24 | Imports | Every module in the package imports | 94 of 94 | 🟢 PASS |
| V-25 | Tests | Full suite, no exclusions | **563 passed, 11 skipped** | 🟢 PASS |
| V-26 | Lint | `ruff check .` | All checks passed | 🟢 PASS |
| V-27 | Lint | `black --check .` | 153 files unchanged | 🟢 PASS |
| V-28 | Repro | Every third-party import is declared | 6 passed | 🟢 PASS |
| V-29 | Security | Credential scan of the tracked tree | no match | 🟢 PASS |
| V-30 | Hygiene | No em dashes in tracked source or documentation | none outside the raw evidence logs | 🟢 PASS |

---

## 2. Import sweep detail

Ten failures out of 93 modules. Recorded here because the pattern matters more than the count: nine are rename drift or v1 residue, one is a genuine missing external dependency.

| Module | Error | Category | Ticket |
|---|---|---|---|
| `models/baseline_runner.py` | `cannot import name 'CALMSEP_SR' from 'models.preprocess'` | rename drift | I-004 |
| `scripts/run_baseline.py` | same | rename drift | I-004 |
| `eval/matrix.py` | `cannot import name 'si_snr' from 'eval.metrics'` | rename drift | I-005 |
| `eval/baselines.py` | `cannot import name 'CalmSepEngine' from 'pipeline.infer'` | rename drift | I-006 |
| `eval/ablation_gate.py` | `No module named 'utils.logging'` | missing internal module | I-007 |
| `train/calibrate.py` | `No module named 'calibration.fit'` | missing internal module | I-008 |
| `train/cached_dataset.py` | `No module named 'train.trainer'` | v1 residue | I-009 |
| `scripts/build_train_cache.py` | `No module named 'models.experts.mossformer2'` | v1 residue | I-009 |
| `scripts/slice_for_kaggle.py` | `FileNotFoundError` after executing work at import time | missing main guard | I-010 |
| `eval/eval_reverb_adapter.py` | `No module named 'sr_corrnet'` | genuine external dependency | I-019 |

Test collection adds two more of the same kind: `tests/principle2_test.py` and `tests/smoke_test.py` on `CalmSepEngine` (I-006), and `tests/test_cached_dataset.py` on `models.cascade_gate` (I-009).

---

## 3. Checks that cannot be run here

Recorded so that absence is not mistaken for failure.

| Area | Check | Blocker | Ticket |
|---|---|---|---|
| Environment | `pip install -e ".[dev]"` in a clean virtual environment | `sr_corrnet` is not obtainable | I-019 |
| Model | Load the backbone and count parameters | same | I-019, I-021 |
| Model | Read the epoch field from each Stage 1 checkpoint | checkpoints are on Kaggle | I-022 |
| Data | Preflight over a real dataset | 176,000 audio files are not on this machine | none |
| Inference | Fixture inference end to end | backbone missing | I-019 |
| Training | Any stage smoke test | no GPU, no data; the owner has asked that nothing be trained | none |
| Evaluation | Reproduce any recorded SI-SDRi number | needs LibriMix, checkpoints and the backbone | I-023 |
| Evaluation | Gate output distribution per condition | needs the Stage 4 checkpoint | I-003 |
| Calibration | ECE and a reliability diagram | needs a checkpoint and a held-out set | I-034 |
| CI | A green workflow run | the workflow has never triggered | I-011 |

---

## 4. Baseline to protect

Any change made from here must leave this unchanged or better:

```
pytest tests/ -q          563 passed, 11 skipped
ruff check .              All checks passed
black --check .           153 files would be left unchanged
```

No module in the package may fail to import, and no test module may fail to collect. All of it is re-measured after every commit and recorded in `WORKLOG.md`.

The original baseline, kept for comparison: 504 passed with three modules uncollectable, and ten import failures across 93 modules.

---

## Related documents

`RESTORATION_STATE.md` · `ISSUE_LEDGER.md` · `REPRODUCTION.md` · `WORKLOG.md`
