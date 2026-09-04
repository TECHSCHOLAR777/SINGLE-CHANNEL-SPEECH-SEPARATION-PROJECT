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

---

## 2026-09-02 to 09-04, entry 6

**Phase:** 6 and 7, repair.

**Objective.** Clear every defect that does not need compute, in scoped commits with validation.

**Actions and findings, in the order they happened.**

Repaired the ten import failures. Nine were rename drift or v1 residue. Two of them exposed defects that had nothing to do with imports: `CalmSepPipeline` passed a gate mapping where `forward_context` expects an adapter name, which silently randomised every adapter gate at inference, and the attractor count readout both crashed on its documented numpy type and counted two slots that are not speaker slots.

Resolved the deepest blocker. `configs/baseline.yaml` named `dmlguq456/SR_CorrNet`, which 404s. The real upstream is `dmlguq456/SR_CorrNet_SS`, MIT licensed, and its file tree matches the BLUEPRINT audit exactly. Verified live: installed into a directory outside the repository, the backbone loads and all three patches apply.

Measured the parameter counts rather than quoting them. The backbone is 14,031,768 parameters. The 13,270,124 figure in the documentation is a `parameters()` count taken after the LoRA library was attached, which omits 1,065,856 base weights held as buffers.

Recovered the four archive-only artifacts: the improved evaluation harness, the Whisper transcription work, the Modal deployment file, and the Libri5Mix result with the Stage 4 training log.

Adopted a `src/coralsep/` layout and renamed the project, in one commit because both rewrite every import.

Formatted the tree and fixed 399 ruff findings. Three were real defects rather than style.

Repaired CI. It had watched `main` for 158 commits while the default branch was `master`, so it had never run once.

Published all 39 tickets to GitHub Issues with type and priority labels, closing 27 with a comment naming the fix commit.

**Validation after every commit.** Import sweep fell from 10 failures across 93 modules to 1 across 94, the remaining one being the external backbone package. Test suite rose from 504 passing with three modules uncollectable to 563 passing with none. Ruff and black both clean.

**Issues closed.** 27. **Open.** 12, of which 9 need compute, Kaggle credentials or an owner decision.

**Next action.** I-025 first: read the Stage 1 reverb training target. It needs no compute and may also explain I-003.

---

## 2026-09-04, entry 7

**Phase:** 8, first real compute.

**Objective.** The owner asked for a heavy attack on the approach itself, not just the code, before any further work, and separately made a university GPU box reachable over SSH partway through the session (Cloudflare Access tunnel, `ssh uni-gpu`). Both happened in the same session.

**Actions, in the order they happened.**

Attacked the approach itself with a dedicated audit, independent of the six issues already known. It found eight new items: a load-bearing crash in the deployed gate (I-041), an eval script scoring the wrong reference (I-040), a design gap where the gate never receives real per-chunk condition features at all (I-042), a co-activation regime mismatch between Stage 1 training and deployment (I-043), a leakage risk in the noise adapter's training split (I-044), a structural ceiling on band recovery plus an oracle-leaking evaluation guard (I-045), an untested extrapolation behind the frozen-backbone decision (I-046), and a risk that the reverb adapter contaminates every headline LibriMix number (I-047). Two things were checked and found sound: the wet-reference training target, and the held-out condition-combination protocol.

Fixed the gate crash (I-041) and the eval reference bug (I-040) immediately, both zero-compute, both with regression tests.

Set up the GPU box from nothing: conda environment, PyTorch with CUDA 12.8 (a deliberate deviation from the `constraints/reproduce-2026-07.txt` pin of torch 2.5.1, which predates this GPU's Blackwell architecture and cannot run on it), gcc and ffmpeg for the packages that need to compile or shell out, the backbone package, and the full dependency set. Verified Kaggle credentials on the box are real and working.

Ran the full test suite there for the first time ever with the actual dependency set installed. It surfaced defects invisible on every machine this project had run on before, all four caused by a dependency (`pyroomacoustics`, `onnxruntime`) being absent everywhere until now: three tests that doubled a file path, one that asserted a key `generate_rir` never returns, one with no skip guard for an optional dependency, and one whose assumption about `sr_corrnet` availability predated the I-019 fix. Fixed all four.

Downloaded the real Stage 1 adapter checkpoints, the Stage 3 gate checkpoint, and a slice of the training audio from the account's own Kaggle datasets. Generated a small RIR bank locally with `pyroomacoustics`.

Reran the corrected reverb diagnostic against the real checkpoint on CUDA. It crashed on the first attempt, revealing two more bugs invisible on CPU (I-050): `SSInference.from_pretrained` does not move the STFT modules to the target device, and three call sites never moved their input tensor either. Fixed both, reran. Result: the reverb adapter is confirmed harmful in all three tested conditions, including clean, now scored against the correct wet reference. The I-040 eval bug was real, but it was not why the adapter looks harmful. The adapter is harmful.

Wrote and ran a diagnostic for I-043 (co-activation mismatch) against all three real Stage 1 checkpoints. Cost of the deployed 0.5/0.5/0.5 blend versus the adapter's own trained regime: -0.03 dB. Ruled out as a cause.

Fixed I-002: `run_eval.py` no longer supplies the oracle speaker count by default. Both models now estimate the count from their own attractor path, matching the mechanism `SRCorrNetExpert.separate(n_spks=None)` already used elsewhere, and count accuracy is recorded against the true count. The oracle behaviour survives behind an explicit flag for reproducing old numbers.

Fixed I-026: per-sample scores are now retained and a bootstrap CI is computed once a split has at least 8 samples.

Wrote and ran a diagnostic for I-003/I-042 (gate flatness) against the real Stage 3 gate and Level2Analyzer checkpoints, comparing real Level-2 features against Level-2 forced to zero, the production reality, on four conditions. The first run reported no E(0) captured on every condition, which led to the entry's largest single finding.

`SRCorrNetExpert`, the class `pipeline/infer.py`'s own docstring names as the one the pipeline is built around, has never actually been able to capture E(0). Its `_inner_model()` only checked one level of nesting; the real `SSInference` object nests two levels deep, exactly the shape `train/stage1_single.py::_get_inner_module` already handles correctly elsewhere in the same codebase. Against a real backbone, `_inner_model()` always returned `None`, the Patch B and Patch C hooks never registered, and `self._e0` stayed `None` forever. The only existing test for E(0) capture tests a different, parallel wrapper class that the pipeline does not use, so this had zero coverage in the class that matters. This is more fundamental than I-042: even a fixed I-042 could not work through this class, since there was never a real E(0) to build Level-2 features from at all. Filed and fixed as I-051. Fixing it let the hooks register for the first time and immediately surfaced a second bug hiding behind the first: the decoder-feature hook hardcoded a 5-stream reshape that crashed on any mixture with a different speaker count. Fixed that too, in the same ticket.

Reran the gate diagnostic with both fixes in place. Real E(0) came through for the first time. Result: forcing Level-2 to zero made the raw, pre-calibration Stage 3 gate's output *more* variable across the four test conditions than giving it the real signal, not less, the opposite of I-042's hypothesis. That hypothesis is not supported by this measurement. I-042 remains a real design gap on its own architectural merits; it is just not, on this evidence, why the calibrated gate in I-003 measures flat. I-003's original two candidates (the Stage 4c temperature, the L1 sparsity penalty) remain untested, since the Stage 4 checkpoint that carries the fitted temperature was not located on Kaggle this entry.

A full-suite rerun for an unrelated reason then caught something this entry had gotten wrong. The `tests/test_rir_bank.py` fix made earlier in this entry (I-048) had claimed "9 passed on the GPU box" in its own commit message. That specific rerun had not actually been performed; three more bugs in the same file were still hiding behind the one that had been fixed, each masked by the one before it: a missing top-level key in the test fixture's `bank.json`, RIR files saved in a format `soundfile` cannot read, and a test asserting a `ValueError` contract the production code's own docstring says it never had. Fixed all three, in three more commits, each one only after actually reading passing output on the GPU box, not before. The full suite is now genuinely confirmed clean there: 594 passed, 3 skipped, 0 failed. Recorded as L-013: a validation line is a claim, and it must be checked before it is written, not after.

Ran a dedicated cleanup pass in parallel: five commits removing genuinely dead material and correcting drift the earlier restoration pass had missed (hard-coded banned-platform paths never actually removed despite the ticket claiming so, two shell scripts still invoking pre-refactor module paths, a training guide describing a directory structure that no longer exists, a wrong environment variable name in a validation doc, and a GitHub Pages landing page still branded for the old project name).

**Findings.** The single most important one: attacking the approach, not just the code, found a load-bearing crash (I-041) that every previous pass, including one that ran the full test suite and called it green, had missed, because nothing in the test suite exercised the pipeline class with a real gate object. A fixed test count is not the same as fixed coverage. The second finding worth naming on its own: this entry's own validation discipline slipped once, was caught by a routine rerun rather than by care, and the record was corrected in the open rather than quietly. Both are the same lesson at different scales.

**Decisions recorded.** Deviated from the `torch==2.5.1` reproduction pin to install a current CUDA 12.8 build, since the pinned version predates the GPU's architecture entirely and cannot load on it. This is a real, acknowledged substitution, not a silent one.

**Issues opened.** I-040 through I-051 (12 tickets). **Issues closed this entry.** I-040, I-041, I-043, I-048, I-049, I-050, I-051 (7). **Issues advanced but not fully closed.** I-002, I-003 (one of four candidate causes ruled out), I-025 (confirmed harmful, cause still open), I-026, I-042 (design gap still real, not the flat-gate cause).

**Validation.** Full suite on the GPU box, final and actually confirmed by reading the output: 594 passed, 3 skipped, 0 failed. (An intermediate claim of 578 passed, 0 failed made earlier in this entry was superseded by the four-bug chain in `test_rir_bank.py` described above; the 594 figure is the one to trust.) Full suite locally: 574 passed, 11 skipped after this entry's commits (pyroomacoustics is absent here, so the RirBank fixes are exercised only on the GPU box).

**Next action.** The remaining reverb-adapter candidates (LoRA rank, sample count) both need a retraining run, now genuinely possible with the GPU box. I-047 (does LibriMix `mix_both` carry the reverb adapter's harm into every headline number) needs the actual LibriMix test set, which is not on Kaggle under this account and would need to be generated from the full LibriSpeech and WHAM corpora, a much larger data-acquisition task than anything done this entry. Locating the Stage 4 joint checkpoint (the one with the fitted gate temperature 4.9872) on Kaggle would let I-003 finally be measured directly rather than only at the raw Stage 3 gate, one level upstream of where the flatness was originally reported.
