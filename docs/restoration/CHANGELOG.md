# Changelog

**Purpose:** meaningful repository changes made during the restoration, newest first.

Format follows Keep a Changelog. Dates are the date the change was committed.

---

## [Unreleased]

### Added
- `ISSUES.md`, a plain-language companion to the ticket ledger that explains what each open problem costs and what closing it would take.
- All 39 tickets published to GitHub Issues with a type and priority label taxonomy. 27 closed with a comment naming the fix commit.
- `src/coralsep/`, a single importable package replacing eleven top-level ones.
- `utils/logging.py`, restored from the commit that introduced it alongside its consumers.
- `PipelineResult.to_separation_result`, restoring the project's own shared output contract at the top-level boundary.
- `tests/conftest.py` with a weight-free `MockExpert`, which lets the full pipeline run end to end in tests for the first time since the branch integration.
- `constraints/reproduce-2026-07.txt`, the pinned environment recovered from the Modal deployment file.
- Four new test modules covering baselines, pipeline counting, the calibration CLI, the Kaggle slicer and dependency coverage.
- README files for `docs/`, `results/`, `datasets/`, `notebooks/` and `scripts/`.
- `.gitattributes`, and pre-commit hooks rejecting credential patterns and em dashes.
- `docs/restoration/`, a populated knowledge base replacing the empty templates supplied with the restoration pack: restoration state, project status, project inventory, architecture, approach evolution, results, learnings, decisions, experiment registry, data and model inventory, issue ledger, validation matrix, reproduction and this changelog.
- `docs/restoration/protocols/`, the six operating protocols carried over from the restoration pack: commit protocol, ticketing protocol, definition of done, reconciliation protocol, reading order and the restoration rules.
- `CLAUDE.md` at the repository root, the operating contract for this restoration.
- `.gitignore` entry for `.restoration/`, with a comment recording why the evidence directory is deliberately untracked.

### Changed
- The repository is now the workspace root. It previously sat inside a `repo/` subdirectory alongside three scaffolding directories.
- Restoration evidence consolidated into a single `.restoration/` directory: the supplied archive, its extraction and the original restoration pack.

### Fixed
- Ten module import failures, from renamed symbols and abandoned v1 code.
- Inference applied random adapter gates instead of the routed vector, silently, while reporting the correct one.
- The speaker-count readout crashed on its documented numpy type and counted two attractor slots that are not speakers.
- The Kaggle slicer executed filesystem work at import time.
- CI watched a branch that does not exist, which is why none of the above was caught.
- 399 lint findings, three of which were real defects.
- `.gitignore` no longer repeats `outputs/` three times and `pretrained_models/` twice (I-027).

### Documentation
- 34 issues recorded with evidence, acceptance criteria and validation methods.
- Seven restoration decisions recorded with reasoning and consequences.
- Twelve learnings recorded, nine recovered from the project's own history and three from the restoration itself.
- Every documented result reconciled against its raw artifact. All twelve SI-SDR values and all eleven Stage 4 loss values match exactly.

### Reproducibility
- Executable baseline recorded: 83 of 93 modules import, 513 tests collected, 504 pass and 10 skip with three uncollectable modules excluded.
- Environment used for every check recorded in `VALIDATION_MATRIX.md`.
- The blocking dependency, `sr_corrnet`, documented with its evidence trail in `REPRODUCTION.md`.

### Research and experiments
- Seven executed experiments catalogued in `EXPERIMENT_REGISTRY.md` with hardware, configuration, artifacts and reproducibility status.
- Eight planned but never executed runs listed explicitly, so that silence is not mistaken for a negative result.
- The Stage 4 training record promoted from [CLAIMED] to [VERIFIED] by the recovered Kaggle log.
- The Libri5Mix result recovered; it existed in no commit.

### Security
- Five live credentials identified in the supplied archive and reported for rotation (I-001). None is or was in the repository.
- The archive's `memory/` directory, holding unrelated personal notes about named third parties, kept out of version control (I-030).

---

## Prior history

Changes before 2026-09-02 are recorded in Git history and summarised in `docs/restoration/APPROACH_EVOLUTION.md`. That period had no changelog.
