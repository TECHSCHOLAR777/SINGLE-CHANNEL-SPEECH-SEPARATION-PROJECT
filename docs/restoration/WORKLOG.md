# Worklog

**Purpose:** chronological operational record of the restoration. Every entry states what was done, what it showed and what changed.

---

## 2026-09-02, entry 1

**Phase:** 0 and 1, preservation and reconnaissance.

**Objective.** Establish the actual state of the project before changing anything.

**Actions.**
- Read the restoration contract: `CLAUDE.md`, `MASTER_AGENT_PROMPT.md`, `PROJECT_RESTORATION_RULES.md`, `AGENT_START_HERE.md`, and all 22 documents in the supplied pack. All 22 were unpopulated templates.
- Hashed the supplied archive: SHA-256 `85129a23f8165ce373eb99d93886d8e7436d0c06d78e9828cdc5cffeb84b855e`, 556,621 bytes, 60 entries.
- Extracted the archive to a separate evidence directory. Original preserved unchanged.
- Confirmed no local Git repository existed. Cloned the remote.
- Recorded branch structure, HEAD, remotes, tags and commit count.

**Findings.**
- 158 commits, matching the count named in the restoration pack. HEAD `19caf73`, 2026-07-23, author Parv Bansal. No tags.
- Thirteen remote branches. Ten strictly behind `master`.
- No local divergence to recover, because no local repository existed.

**Validation.** V-01 through V-05 in `VALIDATION_MATRIX.md`, all passing.

**Next action.** Reconcile the archive against the repository.

---

## 2026-09-02, entry 2

**Phase:** 2, archive forensics.

**Objective.** Determine which state is authoritative for each file.

**Actions.**
- Compared all 21 archive Python files against their repository counterparts byte for byte. Twenty-one reported as different.
- Recognised the false signal: the Windows checkout applies CRLF, the archive stores LF. Repeated every comparison with line endings normalised.
- Compared the three genuinely different files against all thirteen branches by blob hash.
- Compared the archive's documentation and run artifacts against the repository.

**Findings.**
- Eighteen of twenty-one archive Python files are identical to `master` modulo line endings.
- Three exist in no branch: `eval/run_eval.py` ahead of every branch, `demo.py` ahead of every branch, `modal_deploy.py` absent from all history.
- Two raw evidence artifacts are archive-only: the Libri5Mix result and the Stage 4 training log.
- Three documents are archive-only: `CONTEXT.md`, `NUMBERS.md`, `docs/PROJECT_HISTORY.md`.
- `CONTEXT.md` contains five live credentials in plaintext.
- The archive `memory/` directory holds eleven files of unrelated personal notes about named third parties.

**Decisions recorded.** DEC-002 archive is evidence not an overlay, DEC-003 normalise line endings before comparison.

**Validation.** V-06, V-07, V-14 in `VALIDATION_MATRIX.md`.

**Next action.** Reconcile the branches.

---

## 2026-09-02, entry 3

**Phase:** 2, branch reconciliation.

**Objective.** Determine whether any branch carries unmerged value.

**Actions.**
- Measured ahead and behind counts for all thirteen branches against `master`.
- For `origin/integration`, the only branch with substantial unmerged work, compared every touched file by blob hash rather than by commit count.
- Diffed the three files that still differ.

**Findings.**
- `origin/integration` is seventeen ahead and twenty-four behind. Fifteen of its eighteen touched files are already byte-identical to `master`, absorbed by commit `646ab52` on 2026-07-17.
- In all three differing files, `master` carries the later change. For `models/lora.py`, `master` adds tensor-gate support and fixes the `olora_penalty` accumulator on top of the `integration` version.
- The archive independently confirms this: its 2026-09-01 copy of `models/lora.py` matches `master`, not `integration`.
- Merging `integration` would have reverted two fixes.

**Decision recorded.** DEC-001, `master` is authoritative.

**Learning recorded.** L-011, commit counts do not settle branch precedence.

**Next action.** Establish the executable baseline.

---

## 2026-09-02, entry 4

**Phase:** 3, baseline.

**Objective.** Find out what actually runs, and what fails first.

**Constraint.** The project owner confirmed there is no GPU on this machine and that models and datasets live on Kaggle, and asked that nothing be trained. All checks below are static or unit-level.

**Actions.**
- Checked third-party dependency availability.
- Imported all 93 non-test modules individually.
- Collected and ran the test suite.
- Scanned for credentials, hard-coded machine paths and TODO markers.
- Cross-checked every documented number against its raw artifact by hand.
- Compared the CI workflow trigger against the actual default branch.

**Findings.**
- 83 of 93 modules import. Ten fail. Nine are rename drift or v1 residue; one is the genuine missing `sr_corrnet` dependency.
- 513 tests collected, three modules fail to import, **504 passed and 10 skipped in 36.06 s** with those three excluded.
- `scripts/slice_for_kaggle.py` executes filesystem work at import time.
- The repository is free of credentials.
- All twelve documented SI-SDR values match the raw JSON exactly. All eleven documented Stage 4 loss values match the raw Kaggle log exactly.
- `.github/workflows/ci.yml` triggers on `main`; the default branch is `master`. CI has never run across 158 commits, which explains how the ten broken imports survived.

**Learning recorded.** L-010, a quality gate pointed at the wrong branch is worse than no gate.

**Validation.** V-08 through V-19 in `VALIDATION_MATRIX.md`.

**Next action.** Ticket every independent problem.

---

## 2026-09-02, entry 5

**Phase:** 5, ticketing, and workspace reorganisation.

**Objective.** Convert the findings into a scoped backlog, and make the workspace look like a repository.

**Actions.**
- Opened 34 tickets in `ISSUE_LEDGER.md`, each with problem, evidence, impact, suspected cause, scope, acceptance criteria and a validation method, plus a dependency map.
- On the owner's instruction that the workspace should be a professional repository rather than four scaffolding directories, promoted the clone to the workspace root.
- Consolidated the archive, its extraction and the restoration pack into a single `.restoration/` directory, excluded by `.gitignore`.
- Promoted the six genuinely useful protocol documents from the pack into `docs/restoration/protocols/`. Left the 22 empty templates in evidence, superseded by the populated documents.
- Promoted `CLAUDE.md` to the repository root, its conventional location.
- Rewrote `.gitignore`: removed the duplicated `outputs/` and `pretrained_models/` entries, and added `.restoration/` with a comment explaining why the evidence is deliberately untracked.

**Findings.**
- Archive SHA-256 reverified after the move: unchanged.
- Tracked file count after the move: 243, unchanged.
- Test suite after the move: 504 passed, 10 skipped, unchanged.

**Decisions recorded.** DEC-004 work on `master`, DEC-005 evidence outside version control, DEC-006 rename to CoRAL-Sep, DEC-007 nothing deleted during inventory.

**Issues created.** I-001 through I-034.

**Validation.** V-02, V-05, V-12.

**Documentation updated.** The full knowledge base under `docs/restoration/`.

**Next action.** Commit the knowledge base, then begin repair at I-004.
