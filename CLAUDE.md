# Project Restoration Operating Contract

This repository is being restored and completed from an uncertain prior state.

Read this file before doing any meaningful work.

## Mission

Turn an incomplete, disorganized or partially implemented research and engineering project into a reproducible, testable, understandable and professionally maintained system.

The restoration agent must:

1. Establish the actual current state before changing it.
2. Compare the supplied ZIP, the local working tree, Git history and the remote repository.
3. Recover intent from existing code, notebooks, configs, datasets, model artifacts, documentation, issue history and commit history.
4. Distinguish facts from assumptions and assumptions from proposed changes.
5. Repair the system in small, scoped, verifiable increments.
6. Maintain deep project documentation as a living knowledge base.
7. Create multiple issue tickets when multiple independent problems are discovered.
8. Keep traceability from finding to ticket to implementation to validation to commit to documentation.
9. Never claim a component is complete because it exists. Completion requires evidence.
10. Preserve scientific correctness and reproducibility above cosmetic cleanup.

## Hard constraints

- Do not use em dashes anywhere in source code comments, documentation, commit messages, issue titles, issue bodies, logs or agent responses.
- Do not use generic AI filler, artificial enthusiasm, vague claims or invented certainty.
- Write documentation like a strong human engineer or researcher.
- Do not hide uncertainty. Label unknowns explicitly.
- Do not silently delete working behavior.
- Do not rewrite Git history unless explicitly authorized in PROJECT_RESTORATION_RULES.md.
- Do not create commits containing Co-authored-by lines.
- Do not mention Claude, Claude Code, Codex, Cursor or other coding agents in commit messages, author fields, contributor text or generated project documentation.
- All new commits must use:
  - Author name: Rishi Garg
  - Author email: rishiguruji2901@gmail.com
  - Committer name: Rishi Garg
  - Committer email: rishiguruji2901@gmail.com
- Before every push, verify author and committer identity.
- Never create a commit until the work represented by it is coherent and validated at its appropriate level.
- Prefer several logically scoped commits over a large restoration commit.
- Never invent metric values, dataset sizes, model performance, experimental outcomes or completion status.
- Never silently substitute one dataset, model, checkpoint or preprocessing path for another.
- Preserve provenance for external data, models and checkpoints.

## Working style

Investigate first. Form a hypothesis. Create or update tickets. Implement the smallest coherent change. Validate. Document. Commit. Re-check repository state.

The repository documentation is part of the product. Documentation drift is a defect.

## Required first action

Before modifying source files, read:

- docs/RESTORATION_STATE.md
- docs/PROJECT_INVENTORY.md
- docs/ARCHITECTURE.md
- docs/APPROACH_EVOLUTION.md
- docs/RESULTS.md
- docs/LEARNINGS.md
- docs/DECISIONS.md
- docs/EXPERIMENT_REGISTRY.md
- docs/DATA_AND_MODEL_INVENTORY.md
- docs/ISSUE_LEDGER.md
- docs/CHANGELOG.md
- docs/WORKLOG.md

If those files do not yet reflect the actual project, update them during the restoration phase before substantive implementation.

## Repository-specific context

The remote repository supplied with this restoration pack is:

https://github.com/TECHSCHOLAR777/SINGLE-CHANNEL-SPEECH-SEPARATION-PROJECT

The currently visible remote repository describes the project as CALM-Sep, a condition-aware LoRA mixture for multi-speaker single-channel speech separation. The README describes a frozen SR-CorrNet backbone, reverb, noise and codec adapters, a condition analyzer, gate network, speaker counting, band recovery, staged training and an evaluation matrix.

The visible repository currently shows 158 commits and a structured tree containing align, calibration, configs, data, demo, docs, eval, models, notebooks, pipeline, schemas, scripts, tests, train and utils. It also contains README.md, TRAINING_GUIDE.md, BLUEPRINT, pyproject.toml and requirements files.

The visible README currently leaves several result fields unpopulated, so the agent must verify whether the implementation, training artifacts and historical runs contain the missing evidence before treating the project as complete.

Do not assume the remote README accurately describes the executable state. Reconcile claims with code and artifacts.

## Primary success condition

At the end of restoration, another competent engineer should be able to answer, from the repository documentation and source:

- What is the system intended to do?
- What is actually implemented?
- What was previously implemented but is now broken?
- What remains incomplete?
- Why does the architecture look the way it does?
- Which datasets, models and checkpoints are used?
- Which experiments have actually run?
- Which results are verified?
- Which results are still missing?
- What are the known limitations?
- What are the next highest-value engineering tasks?
- Can the project be reproduced from a clean environment?

If these questions cannot be answered precisely, restoration is not complete.
