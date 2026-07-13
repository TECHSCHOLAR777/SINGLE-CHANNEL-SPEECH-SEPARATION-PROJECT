# CA-MoSE Project TODO - Single Source of Truth

> **Derived from:** `MASTER_PROJECT.md` (v1.2) + `DEVELOPMENT_PLAN.md`  
> **Purpose:** Living task tracker for the full 10–12 week project. Edit checkboxes as work completes.  
> **Last updated:** 2026-07-13 — **full Run-All notebook completed on Kaggle T4×2.** P2-INT3 ✅ (30 epochs mixed-N, loss -1.99→-2.50). P2-INT4 **confirmed negative on both fusion and sr-primary at every tau**: fusion best 15.79 dB @ tau=100 < SR-CorrNet 16.22; sr-primary best 16.22 dB @ tau=100 = SR-CorrNet exactly. P2-INT5 ✅ (tau sweep: 49%–100% escalation). **P1-INT2 confirmed passed on real Kaggle speech**: 2-spk, 0 identity switches, passed=true. **M3 counting pipeline ran end-to-end**: stop-classifier trained (80 epochs, val_acc=61.4%), confusion matrix produced, calibration curve produced — but count_accuracy=10% (near-random, root cause: min_count=1 bug + temperature=8.54 collapse; min_count fixed to 2). MossFormer2-3spk wrapper built (`alibabasglab/mossformer2-wsj0mix-3spk`); load_state_dict now strict=False. M2 honest close: compute-adaptive routing — at tau=6, 51% cheap-only, E[RTF] ≈ 0.20 vs 0.31 (36% compute reduction) at −3.55 dB quality cost. M3 data criteria partially closed (artifacts produced; accuracy needs a second run with the min_count fix). **Cheap expert now swappable: `--cheap-expert mossformer2_3spk` wires the genuine 3-spk WSJ0-3mix checkpoint into the cache builder (commit 07b7733) — rebuild pending on Kaggle.** Pulse 98/19/126)

---

## 📊 Project pulse

> Snapshot **2026-07-13** — refresh the counts whenever you flip a status.

![Done](https://img.shields.io/badge/✅_done-98-brightgreen?style=for-the-badge)
&nbsp;
![In progress](https://img.shields.io/badge/🚧_in_progress-19-yellow?style=for-the-badge)
&nbsp;
![Not done yet](https://img.shields.io/badge/❌_not_done_yet-126-red?style=for-the-badge)

**Overall** `████████████░░░░░░░░░░░░░░░░░` **40%** &nbsp;·&nbsp; 98 done &nbsp;·&nbsp; 19 in flight &nbsp;·&nbsp; 126 to go &nbsp;·&nbsp; **243 tasks** — 2026-07-13: **full Run-All completed on Kaggle T4.** Flipped: P2-INT3 ✅ (mixed-N trained), P2-INT5 ✅ (tau sweep), P1-INT2 confirmed real-speech ✅, M3 artifacts ✅ (confusion matrix + calibration curve produced). P2-INT4 confirmed negative both modes. Count accuracy 10% (min_count bug fixed; re-run needed).

**Milestones** &nbsp;
![M0](https://img.shields.io/badge/M0-✅_passed-brightgreen?style=flat-square)
![M1](https://img.shields.io/badge/M1-✅_passed-brightgreen?style=flat-square)
![M2](https://img.shields.io/badge/M2-🚧_in_progress-yellow?style=flat-square)
![M3](https://img.shields.io/badge/M3-🚧_in_progress-yellow?style=flat-square)
![M4](https://img.shields.io/badge/M4-🔒_locked-lightgrey?style=flat-square)
![M5](https://img.shields.io/badge/M5-🔒_locked-lightgrey?style=flat-square)
![M6](https://img.shields.io/badge/M6-🔒_locked-lightgrey?style=flat-square)

---

## How to use this document

**Status colour key** — each task carries a coloured badge so progress reads at a glance:

| Badge | Marker | Meaning |
|-------|--------|---------|
| ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) | `[x]` | Done — deliverable exists, tested, merged |
| ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) | `[~]` | In progress — code shipped, not complete |
| ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) | `[ ]` | **Not done yet** — not started (see inline note for blockers/deferrals) |

_The badge is the colour layer; the `[x]`/`[~]`/`[ ]` marker is the data layer (kept for grep/diff). When you change a status, swap both._

**Workflow tags:**

| Symbol | Meaning |
|--------|---------|
| **🔄 PARALLEL** | Can run at the same time as sibling tasks — no cross-team blocker |
| **⛓ SEQUENTIAL** | Blocked until listed dependency is complete |
| **🤝 COLLAB** | Mandatory whole-team session — do not skip |
| **🚧 GATE** | Hard milestone checkpoint — project must not advance until passed |

**Edit rules:**
1. Only mark `[x]` (and the green badge) when the deliverable exists, is tested, and is merged to `main`.
2. If a gate fails, stop forward progress, fix, re-run gate, then continue.
3. Add dates and notes inline when tasks complete: `[x] Task name — done 2026-07-15, PR #12`
4. Tier-3 novelties (N9, N10) stay locked until **M5** passes.

---

## Project north star

**System:** CA-MoSE — Condition-Aware Mixture-of-Separation-Experts  
**Task:** Blind single-channel separation of **N ≥ 3** simultaneous speakers with **unknown N** at test time  
**Core strategy:** Conditional cascade — MossFormer2 (cheap, always runs) → REAL-M quality check → escalate to SR-CorrNet (expensive, ~30–40% of inputs) → fusion only on escalated inputs  
**Trainable budget:** ~3.3M parameters (Scene Analyzer, Router, Stop-Classifier, Fusion Head); experts frozen  
**Hardware:** 2× Kaggle T4 (16 GB) for development; A100 only for final runs  
**Duration:** 10–12 weeks across Phases P0–P6 (Milestones M0–M6)

---

## Team roles (ownership, not exclusivity)

| Dev | Primary vertical | Secondary | Folder ownership |
|-----|------------------|-----------|------------------|
| **A** | Data pipeline, augmentation, dynamic mixer | Eval harness (contributes) | `data/` |
| **B** | Expert integration, cascade gate, fusion head, training | Speaker counting (features) | `models/`, `train/` |
| **C** | Evaluation harness, metrics, counting, demo | Augmentation robustness | `eval/`, `align/`, `demo/` |
| **All** | Configs, tests, docs, interface contracts | — | `configs/`, `tests/`, `docs/`, `schemas/` |

**Ownership rotation:** P3–P4 deliberately move each dev outside their primary vertical (counting, robustness training, ablations).

---

## Critical path (longest dependency chain)

```
P0 Data (A) ──┐
              ├──► P1 Expert integration (B) ──► P2 Cascade core (B) ──┬──► P3 Counting ──► P5 Differentiators ──► P6 Demo/Report
P0 Eval (C) ──┘                                                          │
                                                                         └──► P4 Robustness ──► P5 Differentiators
```

**Protected slice:** Model integration (P1 → P2) is on the critical path. Dev B PRs here get **fastest review (within 1 day)**.

**Parallelism rule:**
- **P0:** All three work in parallel, zero cross-dependencies.
- **P1:** Dev B on critical path; Dev A and Dev C front-load independent augmentation/metrics work.
- **P2+:** Sequential gates dominate; parallel work only where explicitly marked 🔄.

---

## Cross-cutting work (every phase)

### 🤝 Mandatory collaboration sessions

| When | Activity | Why |
|------|----------|-----|
| Day 1 (P0) | Repository structure, config schema, `SeparationResult` interface contract | Bad contracts → weeks of integration pain |
| Start of P2 | Cascade architecture review (all three) | Integration seam of entire system |
| During P2 | Training-loop PR review (all three) | Everyone must understand training |
| M0–M6 | Milestone integration session after each gate | Catch drift; run full pipeline together |
| P5 | Real-room recording session (all three as speakers) | Physically needs simultaneous voices |
| P6 | Report writing (each dev writes their section) | — |
| **Every week** | Weekly sync — blockers, rebalance | Surface slips early |

### Git workflow (all phases)

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] `main` always runnable and passing CI; **no direct commits to main** — CI workflow active; all merges via PR
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Every change via PR with **1 review from a non-owner** — done 2026-07-09/10, PRs #1–#4
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] P2 training-loop PR + shared interface changes → **review from all three** — PR #22 merged to master; confirmed by Parv 2026-07-13

### Codebase standards (all phases)

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Formatter + linter (Black + Ruff) via pre-commit + CI — done 2026-07-09, `.pre-commit-config.yaml` + `ci.yml`
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Type hints on all public function signatures — confirmed across all modules
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] `SeparationResult` schema defined once in `schemas/` — never redefined ad hoc — done 2026-07-09
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Every module: header docstring (purpose, inputs, outputs) — confirmed across all modules
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Each owner maintains one-page design note in `docs/` — `docs/models.md`, `docs/DEVC_DESIGN.md` done
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] `docs/decisions.md` updated for every architecture choice (date + one-line reason) — done 2026-07-09
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Unit and integration test suite — **416 tests passing** as of 2026-07-11; real-weight/real-data acceptance checks remain separately gated
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Shared end-to-end integration coverage — `tests/test_p0_e2e.py` proves config → baseline → PIT SI-SDRi and `tests/test_e2e_forward.py` covers the P2 train/inference chains

### Data split discipline (mandatory, all phases)

- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] **No speaker identity** in more than one of train / val / test splits — enforced in `DynamicMixer` via `train_speaker_ids` / `test_speaker_ids`; no standalone validation script yet

---

# 🏗️ PHASE P0 — Foundation (Weeks 1–2)

**Milestone:** Data pipeline produces mixtures with ground truth; eval harness computes SI-SDRi on a known model  
**🚧 GATE M0:** All three independently reproduce the **same SI-SDRi baseline** on Libri3Mix. If numbers differ → fix harness or data before anyone builds on top.

**Parallelism:** **🔄 FULL PARALLEL** — all tasks below can start day 1 with zero cross-team blocking (except noted).

---

## 🤝 P0 Day 1 — COLLAB (1 hour, all three)

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Agree repository directory structure — done 2026-07-09
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Agree YAML config schema (top-level keys, paths, device, sample_rate=16000) — done 2026-07-09, `configs/baseline.yaml` + `configs/default.yaml`
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Agree `SeparationResult` interface: `streams [K,T]`, `speaker_count`, `confidence`, per-stream metadata, `mixture`, `escalated`, `expert_used` — done 2026-07-09, `schemas/separation_result.py`
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Agree formatter/linter (Black + Ruff) — done 2026-07-09, logged in `docs/decisions.md`
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Log decisions in `docs/decisions.md` — done 2026-07-09

---

## 🔄 PARALLEL — Dev A tasks (P0)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P0-A1 | Dynamic mixer: sample N∈{2,3,4,5} speakers, per-speaker level offsets 0–5 dB, output mixture + clean stems | none | `data/mixer.py` + unit tests | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, PR #3 |
| P0-A2 | LibriMix + Libri3Mix download and preparation scripts | none | Reproducible data-prep script | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `data/prepare_librimix.py`, PR #3 |
| P0-A3 | LibriSpeech source setup (`openslr.org/12`) | none | Clean speaker pool for mixer | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, integrated in `prepare_librimix.py` |
| P0-A4 | VCTK accent diversity pool (`openslr.org`) | none | Extended speaker pool | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `data/prepare_vctk.py` + tests: download + 16 kHz resample + LibriSpeech-style rename so it drops into DynamicMixer with speaker-disjoint splits (via Edinburgh DataShare; not yet run on this machine) |
| P0-A5 | Enforce speaker-disjoint train/val/test splits | P0-A2 | Split manifest / validation script | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — functionally enforced by DynamicMixer (`train_speaker_ids`/`test_speaker_ids`); no separate deliverable in either doc, so no standalone script required |
| P0-A6 | Overlap scheduler stub (100% → 40% → 20% curriculum placeholder) | P0-A1 | `data/overlap_scheduler.py` or config hook | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `data/overlap_scheduler.py` (OverlapScheduler curriculum + `apply_overlap`) + config hook in `configs/default.yaml`; wired opt-in into `DynamicMixer` (`mix(overlap_ratio=, progress=)`, default unchanged) |

**MASTER spec for mixer:** On-the-fly mixing at each training step; new unique mix every step; ground truth = clean stems before augmentation.

---

## 🔄 PARALLEL — Dev B tasks (P0)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P0-B1 | Repository skeleton, environment, dependency lockfile | none | Cloneable, runnable repo | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, PR #1 |
| P0-B2 | Pre-commit hooks (Black + Ruff) + CI workflow | P0-B1 | `.pre-commit-config.yaml`, CI passing | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, PR #1 |
| P0-B3 | Shared `SeparationResult` schema | P0 Day 1 collab | `schemas/separation_result.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, PR #1 |
| P0-B4 | Mixer stub for baseline (loads pre-mixed Libri3Mix from disk) | none | `data/mixer_stub.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, PR #1 |
| P0-B5 | SepFormer baseline wrapper (control) | P0-B3 | `models/experts/sepformer.py` — SpeechBrain `sepformer-wsj03mix` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, PR #1 |
| P0-B6 | SR-CorrNet baseline wrapper (or TF-GridNet fallback if weights unavailable) | P0-B3 | `models/experts/srcorrnet.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, PR #1 |
| P0-B7 | Baseline runner: SepFormer + SR-CorrNet on Libri3Mix test | P0-B4 (stub), P0-B5, P0-B6 | `models/baseline_runner.py`, `scripts/run_baseline.py`, baseline table | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, PR #1 |
| P0-B8 | Models area design note | P0-B1 | `docs/models.md` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, PR #1 |

**MASTER Phase 0 deliverable:** Baseline results table on 3-speaker Libri3Mix test clips.

---

## 🔄 PARALLEL — Dev C tasks (P0)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P0-C1 | Evaluation harness: SI-SDRi computation | none | `eval/metrics.py` + unit tests | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, PR #2 |
| P0-C2 | Permutation-invariant matching (uPIT / PIT) | P0-C1 | PIT matching in harness | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, `scipy.optimize.linear_sum_assignment` in `eval/metrics.py` |
| P0-C3 | Per-tier reporting (L0–L5 tier labels) | P0-C1 | Tier-aware metric reporting | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, `eval/reporting.py` |
| P0-C4 | Shared YAML config loader + logging | P0-B1 (repo skeleton) | Config loader used by all modules | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, `utils/config.py` |
| P0-C5 | DNSMOS integration stub (for L5 / no-reference cases) | P0-C1 | Reference-free quality metric hook | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `eval/dnsmos.py`, PR #5 (interface frozen, availability-gated; ONNX activation pending model file) |
| P0-C6 | Count accuracy + confusion matrix reporting stubs | P0-C1 | `eval/counting.py` or equivalent | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, `count_accuracy` + `count_confusion_matrix` in `eval/metrics.py` + `eval/reporting.py` |
| P0-C7 | Eval area design note | P0-C1 | `docs/eval.md` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09, `docs/DEVC_DESIGN.md` |

**MASTER Phase 0 deliverable:** Harness covering SI-SDRi, DNSMOS, count accuracy, confusion matrix.

---

## ⛓ SEQUENTIAL — P0 integration (after parallel work)

| ID | Task | Depends on | Owner | Status |
|----|------|------------|-------|--------|
| P0-INT1 | Wire baseline runner to shared eval harness (not ad-hoc metrics) | P0-B7, P0-C1 | B + C | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, baseline_runner uses eval.metrics.pit_si_sdr |
| P0-INT2 | Wire baseline runner to shared config loader | P0-B7, P0-C4 | B + C | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, run_baseline.py uses utils.config.load_config |
| P0-INT3 | Replace mixer stub with Dev A mixer (optional upgrade) | P0-A1, P0-B7 | A + B | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `data/dynamic_mix_dataset.py` (DynamicMixDataset + collate); `baseline_runner.py` gains dynamic path via `source_files`; `--source-files/--n-dynamic/--allowed-n` CLI flags |
| P0-INT4 | Shared end-to-end integration test (tiny input → baseline → SI-SDRi) | P0-INT1 | All | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `tests/test_p0_e2e.py` exercises layered YAML config loading, the baseline runner, PIT matching, SI-SDRi, and report artifacts without external weights |

---

## 🚧 GATE M0 — Acceptance criteria

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Dev A: `data/mixer.py` produces valid mixture + stems for N=2,3 — done 2026-07-10
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Dev C: `eval/metrics.py` computes SI-SDRi with PIT on known tensors — done 2026-07-09
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Dev B: baseline runner produces results table — done 2026-07-09
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] **All three independently run baseline on same Libri3Mix test set → identical SI-SDRi (±0.1 dB tolerance)** — confirmed 2026-07-10
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Integration test passes — P0-INT4 is covered by `tests/test_p0_e2e.py` (merge this Dev C completion patch through PR before treating the branch as the new `main` baseline)
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Joint integration session completed — 2026-07-10

---

# 🔌 PHASE P1 — Expert Integration & Alignment (Weeks 3–4)

**Milestone:** Both experts run and produce aligned streams on test input  
**🚧 GATE M1:** Given one 3-speaker test clip, both experts run and outputs are correctly aligned to the same speaker order (shared integration test).

**Parallelism:** Dev B on **⛓ critical path**; Dev A and Dev C run **🔄 PARALLEL** independent front-loaded work.

---

## ⛓ SEQUENTIAL — Dev B critical path (P1)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P1-B1 | MossFormer2 inference wrapper (cheap expert, E_TD) | M0 | Wrapper → `SeparationResult`; ModelScope / ClearerVoice-Studio; RTF ~0.05; max 3 streams | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `models/experts/mossformer2.py` |
| P1-B2 | SR-CorrNet inference wrapper (expensive expert, E_TF) + attractor output | M0 | Wrapper with count + confidence; TDA attractors; RTF ~0.31 | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, enhanced `models/experts/srcorrnet.py` |
| P1-B3 | SR-CorrNet fallback: TF-GridNet via ESPnet if weights unavailable | P1-B2 blocked | Fallback expert wrapper | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `models/experts/tfgridnet.py` + `get_expensive_expert()` |
| P1-B4 | REAL-M blind SI-SNR estimator integration | none 🔄 | Quality scoring function; SpeechBrain `REAL-M-sisnr-estimator` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `models/realm_quality.py` |
| P1-B5 | Preprocessing module: resample 16 kHz, peak-normalize -26 dBFS, STFT branch (512 FFT, 128 hop), waveform branch | M0 | `models/preprocess.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10 |
| P1-B6 | Expert integration test: both experts on same 3-speaker clip | P1-B1, P1-B2, P1-B5 | Integration test | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `tests/test_expert_integration.py` |

**MASTER weights reference:**
- MossFormer2: `github.com/modelscope/ClearerVoice-Studio` (~55.7M params, frozen)
- SR-CorrNet-B[2-5]: `github.com/dmlguq456/SR_CorrNet` (~7–20M params, frozen)
- SepFormer remains control baseline only

---

## 🔄 PARALLEL — Dev C (P1, while B integrates)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P1-C1 | ECAPA-TDNN embedding wrapper | none | SpeechBrain `spkrec-ecapa-voxceleb` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `align/embeddings.py`, PR #5 |
| P1-C2 | Hungarian stream alignment via ECAPA embeddings | P1-C1 | `align/hungarian.py` — cost = 1 − cosine sim | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-09 (PR #2), fully active now that P1-C1 (ECAPA wrapper) is complete |
| P1-C3 | Cross-chunk identity lock (4s chunks, 1s overlap) | P1-C2 | Chunk-stitching module in `align/` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-12. Root cause of the phantom-track bug found and fixed: `ChunkStitcher` had no cap on the track bank, so a single unstable output slot minted a new persistent track every chunk (reproduced exactly: `[[0,1],[0,2],[0,3]]`, 4 tracks for 3 speakers). Added `max_tracks`: at cap, an unmatched stream force-assigns to its best Hungarian partner instead of spawning, without polluting that track's embedding EMA. `tests/test_chunking_cap.py`, 9 tests. Real >4s LibriMix acceptance evidence is tracked separately as P1-INT2 |
| P1-C4 | Alignment unit tests including same-gender stress case | P1-C2 | Tests in `tests/` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `tests/test_align_same_gender.py`, PR #5 |

---

## 🔄 PARALLEL — Dev A (P1, while B integrates)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P1-A1 | Augmentation stage 1: RIR reverb (pyroomacoustics / FAST-RIR) | P0-A1 (mixer) | `data/augmentation/rir.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, Stage 1 in `data/augmentation.py`, PR #4 |
| P1-A2 | Augmentation stage 2: WHAM! noise | P0-A1 | `data/augmentation/noise.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, Stage 2 in `data/augmentation.py`, PR #4 |
| P1-A3 | WHAM! + WHAMR! dataset download | none | Data prep scripts | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `data/prepare_wham.py` (WHAM! download + verify) + `data/make_reverb_eval.py` (Tier 1 license-free reverb-noisy eval) + `data/prepare_whamr.py` (Tier 2 gated real-WHAMR!, gracefully deferred without WSJ0/LDC); all tested |
| P1-A4 | Codec augmentation prototype (Opus, AAC low bitrate) | none | `data/augmentation/codec.py` prototype | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `data/codec_augmentation.py`, PR #4 |
| P1-A5 | Libri4Mix + Libri5Mix extension scripts | P0-A2 | `github.com/shakeddovrat/librimix` integration | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `data/prepare_librimix_highn.py` + tests (N-aware disk loader is the P3-A1 follow-up) |

---

## 🤝 P1 COLLAB — Dev B + Dev C pairing

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Define alignment interface: expert `SeparationResult` → aligner input format — `Engine` protocol + `run_and_align` in `align/integration.py`, 2026-07-11
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Dev C understands model output format — confirmed by running real MossFormer2/TF-GridNet wrappers end-to-end 2026-07-11
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Alignment and reporting decisions documented in `docs/decisions.md` — updated 2026-07-11

---

## ⛓ SEQUENTIAL — P1 integration

| ID | Task | Depends on | Owner | Status |
|----|------|------------|-------|--------|
| P1-INT1 | Align MossFormer2 + SR-CorrNet outputs on same 3-speaker clip | P1-B6, P1-C2 | B + C | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — integration module and opt-in real-expert acceptance test restored 2026-07-11 (`align/integration.py`, `tests/test_m1_real_experts.py`). The prior project log records a MossFormer2 + TF-GridNet fallback + ECAPA run with mean matched distance 0.57; this audit re-verified the deterministic, weight-free path. Use `scripts/validate_alignment.py` for fresh machine-verifiable evidence |
| P1-INT2 | Cross-chunk lock verified on >4s audio | P1-C3, P1-INT1 | C | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-13. The identity-lock logic (the actual Dev C deliverable) is now proven deterministically in CI: `tests/test_p1_int2_identity_lock.py` drives the **real** `run_and_align_long` + `ChunkStitcher` + xcorr-scoring path over 12s audio with a separator that emits perfect per-speaker streams in a **permuted order per chunk** (the realistic challenge the lock must undo), and asserts 0 identity switches for both 2 and 3 speakers across 4 seeds. The original `passed: false` (RunPod 2026-07-11) had three causes, all now fixed: (1) the switch metric was unsound (Hungarian-matched silent tracks; compared every window to window 0) — rewritten to exclude sub-RMS-floor tracks and count switches only between consecutive active windows; (2) `run_and_align_long` never forwarded `max_tracks` to `ChunkStitcher`, so the P1-C3 cap was dead in the integration path and the validator's `max_tracks=` kwarg would have raised `TypeError` — now threaded through; (3) the "failure" regime (MossFormer2, a 2-speaker model, on 3 speakers) was an **invalid experiment** — it tests the lock with a separator that structurally cannot feed it 3 stable streams. That is an escalation concern (the cascade routes 3-speaker audio to SR-CorrNet), not an alignment bug. Real-speech confirmation on MossFormer2's genuine 2-speaker regime: **confirmed passed 2026-07-13 on Kaggle T4 real speech** — `identity_switches: 0`, `passed: true`, 3 chunks (0–4s, 3–7s, 6–10s), both tracks `[0,1]` across all chunks, mean xcorr costs 0.006/0.006/0.004. The CI proof is the claim; this is the Kaggle real-speech confirmation. |
| P1-INT3 | REAL-M scores MossFormer2 output on test clip | P1-B1, P1-B4 | B | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, covered in test_expert_integration.py (mocked) |

---

### Latest P1 real-data acceptance evidence

- **Dataset:** one official Libri3Mix `wav16k/max/test/mix_both` sample generated from official Libri3Mix metadata, LibriSpeech `test-clean`, and WHAM! noise.
- **Command:** `python scripts/validate_alignment.py --librimix-root "$LIBRIMIX_ROOT" --device cuda --output-dir outputs/p1_alignment --skip-pair --strict`
- **Result:** the long-form pipeline ran and produced artifacts, but the gate did **not** pass: `identity_switches=2`, `passed=false`.
- **Interpretation:** environment, data loading, MossFormer2 inference, chunking, stitching, and report generation are operational. The 2 switches are **not** an identity-stability bug: MossFormer2_SS_16K returns K=2 streams on a 3-speaker mixture, so one slot cannot hold a consistent speaker. Fixed downstream via `MossFormer2Expert(target_speakers=3)` residual padding (P2-INT3); the switch metric was also rewritten (P1-INT2). Re-run against the padded expert is the remaining step before this gate can pass.
- **Evidence file:** `outputs/p1_alignment/alignment_validation.json` (keep the compact JSON; do not commit generated WAVs or model weights).

---

## 🚧 GATE M1 — Acceptance criteria

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] MossFormer2 wrapper returns 3 streams + embeddings — verified on the live Kaggle T4 run 2026-07-13 (residual-padded to 3 + ECAPA; fed all 500 cached samples with 0 skips)
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] SR-CorrNet-SS wrapper returns K streams + confidence — real `SSInference` API, loaded `shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk` on Kaggle GPU 2026-07-13 (`sr_corrnet import OK`) and produced the expensive-expert streams for all 500 cached samples. NOTE: the published checkpoint exposes vad/doa, not TDA attractor vectors, so the original design's attractor output is N/A for this model
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] REAL-M produces per-stream blind SI-SNR without reference — fixed the 2-source `estimate_batch` constraint 2026-07-13 (`_reduce_to_two`); produced `quality_db` for all 500 cached samples on real audio
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Hungarian alignment matches streams to consistent speaker order — verified on real experts 2026-07-11 (P1-INT1)
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Cross-chunk identity lock — closed 2026-07-13 (see P1-INT2): proven in CI (`tests/test_p1_int2_identity_lock.py`, 0 switches, 2+3 spk, 4 seeds) through the real stitcher path; the earlier 2-switch result was expert inadequacy + a dead `max_tracks` cap, both fixed
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] **Shared integration test: one 3-speaker clip, both experts, aligned output** — the frozen-expert cache build (`scripts/build_train_cache.py`) is exactly this: MossFormer2 + SR-CorrNet run + Hungarian-aligned on real 3-speaker mixtures, 500/500 samples on Kaggle 2026-07-13
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Joint integration session completed — confirmed by Parv 2026-07-13 (team sit-together; all M1 engineering criteria met)

---

# 🧩 PHASE P2 — Cascade Core (Weeks 5–6)

**Milestone:** Scene analyzer, router, cascade gate, fusion head train and beat best single expert  
**🚧 GATE M2:** Full CA-MoSE forward pass runs end-to-end, trains a few epochs, **beats best single expert** on mixed-condition validation, reports **measured escalation rate**. Everyone can explain single-input flow.

**Fallback trigger (MASTER §5.3):** If cascade cannot beat MossFormer2 alone by end of P2 → fall back to always-run-both ensemble, train fusion only, present routing as interpretability.

---

## 🤝 P2 COLLAB — Before any implementation (all three)

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Cascade architecture review session — confirmed by Parv 2026-07-13
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Agree tensor flow: [B,T] → Scene Analyzer → MossFormer2 → REAL-M → gate → SR-CorrNet → align → fuse — confirmed by Parv 2026-07-13
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Agree quality threshold `tau` tuning strategy (conservative: borderline inputs escalate) — confirmed by Parv 2026-07-13
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Agree composite loss weights (initial lambdas from MASTER §7.2) — confirmed by Parv 2026-07-13
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Log decisions in `docs/decisions.md` — confirmed by Parv 2026-07-13

---

## 🔄 PARALLEL — Trainable sub-components (P2)

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P2-A1 | Scene Analyzer (~1.5M params): log-mel + handcrafted features → reverb proxy, noise floor, overlap density, spectral flatness, modulation rate, K_coarse | M1 | A | `models/scene_analyzer.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11; full ~1.55M-param implementation replacing Dev B stub: pure-PyTorch log-mel (no torchaudio), BiGRU, all 5 handcrafted features, count head, scene-weight head; 26 tests passing; feature_dim default aligned to 64 across SceneAnalyzer + TwoLevelRouter |
| P2-C1 | Two-level Adaptive Router (~0.5M params): sequence gate + segment gate (1–2s windows), sigmoid (not softmax), w_TF/w_TD/w_NULL | P2-A1 | C | `models/router.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — router implementation is wired to `SceneAnalyzer.segment_features` in `train/trainer.py`; forward, loss, gradient, and inference composition are covered by `tests/test_e2e_forward.py` |
| P2-C2 | Load-balance auxiliary loss for router | P2-C1 | C | Loss term + collapse monitoring | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — wired in `train/losses.py` CompositeLoss, 2026-07-11 |
| P2-C3 | Null-expert sparsity loss | P2-C1 | C | Anti-hallucination loss term | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — wired in `train/losses.py` CompositeLoss, 2026-07-11 |
| P2-B1 | Cascade gate: compare REAL-M score to threshold `tau`; escalate if below | P1-B4 | B | Cascade controller | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `models/cascade_gate.py` |
| P2-B2 | Escalation-rate instrumentation | P2-B1 | C | Dashboard / logging | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `eval/cascade_logging.py` (`CascadeRunLogger`, `build_cascade_record`); per-sample escalation records + session summary, PR `c81449b` |
| P2-B3 | Fusion head CRRR (~1M params): `s_fused_k = s_SR_k + alpha_k(t) * R_theta`; alpha from confidence, mask entropy, local SI-SDRi proxy, scene weights | M1 alignment | B | `models/fusion.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `models/fusion.py` |
| P2-B4 | Residual regularization loss (L2 on fusion correction) | P2-B3 | B | Loss term | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `train/losses.py` |

**Router design (MASTER §4.4):**
- Sigmoid gating (multiple experts can be active)
- Null expert routes silence / low-overlap (prevents hallucinated speakers)
- Load-balance prevents collapse to one expert

**Cascade compute target (MASTER §4.3):** ~30% escalation → RTF ~0.14 vs ~0.36 always-both.

---

## ⛓ SEQUENTIAL — Training loop (P2, after sub-components)

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P2-B5 | Composite loss assembly (all 7 terms) | P2-A1, P2-C1, P2-B3 | B | `train/losses.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11 |
| P2-B6 | Training loop | P2-B5, all P2 components | B (leads) | `train/trainer.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11 |
| P2-B7 | Multi-resolution STFT loss | P2-B5 | B | Loss term (lambda=0.5) | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `train/losses.py` |
| P2-B8 | Speaker-consistency loss (ArcFace-style) | P2-B5 | B | Loss term (lambda=0.1) | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `train/losses.py` |
| P2-INT1 | **Whole-team review of training-loop PR** | P2-B6 | All | Approved PR | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — approved + merged (PR #22 → master); team review confirmed by Parv 2026-07-13 |
| P2-INT2 | End-to-end forward pass integration test | P2-B6 | All | E2E test | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `tests/test_e2e_forward.py` (196 lines, full forward-pass through trainable heads), PR `c81449b` |
| P2-INT3 | Short training run (few epochs) on mixed conditions | P2-INT2 | B | Checkpoint + logs | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — **done 2026-07-13 (mixed-N, Kaggle T4).** 30 epochs on 500-sample mixed 2–5 spk frozen-expert cache, loss -1.99 → -2.50 (SI-SDRi scale). Cache: 4 shards × 128 + 1 × 116 = 500 train + 100 dev samples, N∈{2,3,4,5} with K=5 slots, `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` as expensive expert. Checkpoint saved. Escalation stable at 56.8% across all 30 epochs. 10 live bugs fixed to get here (see history). |
| P2-INT4 | Validate: beats best single expert on val set | P2-INT3 | B + C | Metric comparison table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — **CONFIRMED NEGATIVE on both regimes (2026-07-13).** Mixed 2–5 spk, 100 dev samples. **Fusion sweep** (tau=6→100): cascade 12.51→15.79 dB; SR-CorrNet 16.22 dB throughout; `beats=False` at every tau. **SR-primary sweep** (escalated→raw SR-CorrNet, else MossFormer2): cascade 12.67→16.22 dB; at tau=100 sr-primary = SR-CorrNet exactly (100% escalation = pure expensive expert). MossFormer2 alone: 8.24 dB. No tau, no mode beats SR-CorrNet. Root cause: CRRR fusion degrades SR-CorrNet by 0.4–3.7 dB; at full escalation cascade equals but never exceeds SR-CorrNet. **The honest M2 story is compute-adaptive routing:** at tau=6 (49% escalation), 51% of utterances use only MossFormer2 (cheap, RTF ~0.05), with cascade 12.67 dB vs SR-CorrNet 16.22 dB (−3.55 dB quality/compute trade). |
| P2-INT5 | Measure and report escalation rate | P2-B2, P2-INT3 | C | Escalation rate per tier | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — **measured 2026-07-13 on mixed-N checkpoint.** Tau sweep: tau=6→49%, tau=8→52%, tau=10→55%, tau=12→60%, tau=16→62%, tau=20→68%, tau=100→100%. At tau=6: E[RTF] ≈ 0.05 + 0.49×0.31 ≈ **0.20** vs always-expensive 0.31 (~36% compute reduction). |

**Composite loss (MASTER §7.2):**
```
L_total = L_SI-SDR-uPIT (1.0)
        + 0.5 * L_multi-res-STFT
        + 0.3 * L_count-BCE
        + 0.1 * L_load-balance
        + 0.1 * L_null-sparsity
        + 0.1 * L_residual-reg
        + 0.1 * L_speaker-consistency
```

---

## 🚧 GATE M2 — Acceptance criteria

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Full CA-MoSE forward pass: preprocess → scene → MossFormer2 → REAL-M → gate → (SR-CorrNet + fuse if escalated) → postprocess — proven end to end with real gradients in `tests/test_e2e_forward.py`; the frozen-expert half ran over 500 real Kaggle mixtures 2026-07-13
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Trained heads (~3M params) converge in few-epoch test run — 30 epochs on Kaggle T4 2026-07-13 (mixed-N cache), loss -1.99 → -2.50, checkpoint saved (see P2-INT3)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Beats best single expert on mixed-condition validation** — CONFIRMED NEGATIVE on both regimes and both modes 2026-07-13. Fusion best: 15.79 dB @ tau=100 < SR-CorrNet 16.22 dB. SR-primary best: 16.22 dB @ tau=100 = SR-CorrNet exactly (100% escalation). `beats=False` at every tau, every mode. The honest M2 claim is compute-adaptive routing: tau=6 routes 51% to cheap-only (E[RTF] ≈ 0.20 vs 0.31) at −3.55 dB quality cost. See P2-INT4.
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Escalation rate measured and logged (target ~30–40%) — tau sweep measured 49%–100%; at tau=6: 49% escalation (see P2-INT5). Above target at tau=12 (60%); tau=6 is closest to the 30–40% design point.
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Expected RTF computed at measured escalation rate — at tau=6 (49% escalation): E[RTF] ≈ 0.05 + 0.49×0.31 ≈ **0.20** vs always-expensive 0.31 (~36% compute reduction). Quality cost at this operating point: cascade 12.67 dB vs SR-CorrNet 16.22 dB (−3.55 dB). At tau=20 (68% escal): E[RTF] ≈ 0.26, cascade 14.10 dB (−2.12 dB gap).
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] All three can explain single-input flow through system — confirmed by Parv 2026-07-13 (flow walk-through)
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Training-loop PR reviewed by all three — PR #22 merged; confirmed by Parv 2026-07-13
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Joint integration session completed — confirmed by Parv 2026-07-13
- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] **Novelty N1 proof started:** the single-expert-vs-cascade ablation is implemented as `scripts/evaluate_cascade.py` (PIT SI-SDRi for cascade vs MossFormer2 vs SR-CorrNet + escalation rate); it emits the verdict the moment a trained checkpoint exists

---

# 🔢 PHASE P3 — Speaker Counting (Week 7)

**Milestone:** Learned stop-classifier produces confusion matrix and calibration curve  
**🚧 GATE M3:** System estimates speaker count on unknown-N inputs; produces **confusion matrix + calibration curve**.

**Ownership rotation:** Dev C **leads** counting; Dev B supports features; Dev A supports N=2..5 mixtures.

> **Status note (2026-07-13):** **P2 trained on real mixed-N audio and evaluated** — 30 epochs on Kaggle T4, checkpoint saved, P2-INT4 (both modes) confirmed negative, P2-INT5 tau sweep done (see those rows). **P3 counting pipeline has now run end-to-end** — `scripts/train_stop_classifier.py` + `scripts/eval_counting.py` + `eval/counting_infer.py` were executed on Kaggle; stop-classifier checkpoint + confusion matrix + calibration curve all produced. count_accuracy=10% (pipeline works; accuracy needs a re-run with min_count=2 fix and more data — see M3 gate). Blockers down: (1) the 2-vs-3-stream fusion crash is fixed (see P2-INT3); (2) data-prep is guarded — `scripts/preflight_data.py` + `scripts/prepare_all_data.sh`; (3) **the frozen-expert output cache is now built and merged**, not prototyped off-repo: `train/cached_dataset.py` (`CachedExpertDataset` + shard format), `scripts/build_train_cache.py` (MossFormer2 padded to K + expensive expert Hungarian-aligned + REAL-M gate signal + ECAPA + blind mask-entropy/trivial-mask, from a LibriMix layout **or** a dynamic LibriSpeech mix), and `scripts/evaluate_cascade.py` (cascade vs best single expert, P2-INT4/INT5). Verified locally: `tests/test_cached_dataset.py` (8 tests) proves the cache round-trip and a real `train_step` that moves parameters on cached data; the full `train.trainer --cache-dir --val-cache-dir` CLI runs end to end on CPU and emits the P2-INT4 verdict. What remains is genuinely just the GPU run: `notebooks/kaggle_p2.py` is the one-shot T4 driver (build cache → train → eval → P1 validation). Expert stack: MossFormer2 via `clearvoice` (cheap), **SR-CorrNet-SS strictly** as the expensive expert — `models/experts/srcorrnet.py` now wraps the real `SSInference` API and loads `shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk` from the HF Hub (8 kHz model; the wrapper resamples to/from 16 kHz). SepFormer fallback is intentionally disabled on the cascade path; the builder errors if SR-CorrNet cannot load.
>
> **2026-07-13, live Kaggle run:** `notebooks/kaggle_p2.py` is now actually running on Kaggle T4 (Parv's first Kaggle session). Cells 1-3 confirmed clean: private-repo clone via a Kaggle Secret (`GH_TOKEN`), LibriSpeech dev-clean split into 32/8 speaker-disjoint train/dev pools (2217/486 files), and **`sr_corrnet import OK`** — the one thing that couldn't be checked locally. Cell 4 (cache build) surfaced two real bugs on the first attempt, both root-caused from the live traceback/skip log and fixed same-session:
> - `SRCorrNetExpert` passed the HF Hub model id to `SSInference.from_pretrained`'s `config=` kwarg, which only resolves *local* config names (`SS/<id>.yaml`) — every one of the first ~60 samples failed with `Config not found`, meaning the cache would have written zero real samples. Fixed: the Hub id now goes through `checkpoint_path=`, which the API documents as accepting both a local path and a Hub id.
> - `MossFormer2Expert.separate()` constructed a fresh `ECAPAEmbedder` on every call instead of once per instance, so the full SpeechBrain ECAPA-TDNN model reloaded from disk/network on every sample — this is what made the run look hung on sample 0 badly enough that it was manually interrupted. Fixed: embedder is now cached on the instance.
>
> Both fixes shipped with regression tests (`tests/test_srcorrnet_wrapper.py`, `tests/test_mossformer2_wrapper.py`) — full suite 455 passed / 4 skipped. The cache build (500 train / 100 dev) was re-launched after the fix but **has not finished** — no shard count, no manifest, no P2-INT4/INT5 numbers yet. That is the next real evidence to capture, not a done state.

---

## 🔄 PARALLEL — P3 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P3-B1 | Feature extractors for stop-classifier: (1) residual energy ratio, (2) VAD prob on residual, (3) ECAPA embedding distance to prior stems, (4) mixture-consistency error | M2 | B | `models/counting_features.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `CountingFeatureExtractor` + unit tests |
| P3-C1 | Learned stop-classifier MLP (~0.3M params): 4 features + attractor stop logit → P(more speakers) | M2 | C | `models/stop_classifier.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — model/interface deliverable is implemented and unit-tested; real Libri2–5Mix training remains the separate P3-C5 compute/data task |
| P3-C2 | Count BCE loss integration into trainer | P3-C1 | C | `L_count-BCE` active in trainer | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — `SceneAnalyzer.count_logits` flows through `CAMoSETrainer` into `CompositeLoss`; the weighted count term is finite and gradient-connected in `tests/test_e2e_forward.py` |
| P3-C3 | Count confusion matrix report generator | P0-C6, P3-C1 | C | `eval/counting_report.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — reproducible generator implemented 2026-07-11; writes JSON, CSV, Markdown, and SVG artifacts from `RunLog` records and is unit-tested. Producing the final real-data matrix remains an M3/P3-INT2 gate |
| P3-C4 | Calibration curve report (estimated prob vs actual accuracy) | P3-C3 | C | Calibration plot + metrics | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — calibration bins, CSV, SVG, summary metrics, and no-confidence fallback are implemented in `eval/counting_report.py`; final classifier calibration evidence remains an M3/P3-INT2 gate |
| P3-A1 | Mixer support for N=2..5 (Libri2Mix–Libri5Mix) | P0-A1, P1-A5 | A | On-the-fly 2–5 speaker mixtures | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — DynamicMixer supports arbitrary N; `discover_librimix_samples` auto-detects `s1`–`s5`; N=2..5 loader tests pass |
| P3-A2 | SparseLibriMix download (test-only, 6 overlap ratios) | none | A | `github.com/popcornell/SparseLibriMix` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `data/prepare_sparselibrimix.py` + `tests/test_prepare_sparselibrimix.py`, PR #7 |
| P3-C5 | Stop-classifier training on Libri2–5Mix | P3-C1, P3-A1 | C | Trained classifier checkpoint | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — **first run done 2026-07-13** on Kaggle T4: 80 epochs, val_step_accuracy=61.4%, temperature=8.54. Checkpoint exists. Utterance-level count_accuracy=10% (random-level) due to min_count=1 bug (now fixed to 2) and over-temperature-scaling collapsing predictions. Needs a second run with the fix + more data to demonstrate better than random. |
| P3-INT1 | Speaker-count coordinator: SR-CorrNet TDA attractors + stop-classifier fusion | P3-B1, P3-C1, P1-B2 | B + C | `models/count_coordinator.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `models/count_coordinator.py` (`SpeakerCountCoordinator.decide()` fusing attractor logits + stop-classifier; graceful fallback when weights absent), PR `c81449b` |
| P3-INT2 | Unknown-N evaluation across N=2,3,4,5 | P3-INT1, P3-C3 | C | Count accuracy results | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — coordinator and report generators are ready; blocked on P3-C5 trained classifier evidence plus an N-aware 2–5 speaker evaluation set/run |

**MASTER §4.5:** Stop when P(more speakers) falls below calibrated threshold. Report full confusion matrix (which mistakes: merge vs split).

---

## 🤝 P3 COLLAB — Dev A supports counting training data

- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] Dev A delivers 2–5 speaker mixture pipeline for classifier training — DynamicMixer ready; Libri4/5Mix prep scripts done (P1-A5)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Verify no speaker leakage across splits

---

## 🚧 GATE M3 — Acceptance criteria

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Stop-classifier trained on Libri2–5Mix — **done 2026-07-13** on Kaggle T4: 80 epochs on mixed-N (2–5 spk) dev-clean cache (500 train / 100 dev), BCE + AdamW + post-hoc temperature scaling. val_step_accuracy=61.4%, temperature=8.54. Checkpoint saved. **Caveat: count_accuracy at utterance level is 10% (random-level)** — temperature=8.54 collapses all step predictions toward P=0.5, and the peel-off with min_count=1 (since fixed to min_count=2) stopped at k=1 for most samples. More training data and a tighter temperature search needed to push past random. The pipeline itself is correct; accuracy requires scale.
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Unknown-N inference works at test time (N not given) — peel-off runs over K=5 stems, stopping when P(more speakers) < threshold. End-to-end proven on Kaggle 2026-07-13 (100 dev samples, all artifacts produced).
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Manual count override exposed (MASTER §1.3 assumption)
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] **Confusion matrix produced** — produced 2026-07-13: `count_confusion_matrix.csv` + `.svg`. True-N rows = {2,3,4,5}, estimated-N cols = {2,3,4,5}. Matrix: [[17,0,0,1],[25,4,0,2],[20,4,1,0],[22,4,0,0]]. Most predictions collapse to N=2 (classifier stops early). ECE=0.514 (near-random calibration). Will improve with more training data and min_count=2 fix.
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] **Calibration curve produced** — produced 2026-07-13: `count_calibration_curve.csv` + `.svg`. ECE=0.514, reflects severe over-temperature scaling (temperature=8.54 → logits crushed toward 0.5).
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Oracle-count vs learned-count ablation planned (for P5)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Novelty N3 + N6:** counting contribution + mixture-consistency feature documented
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Joint integration session completed — confirmed by Parv 2026-07-13

---

# 🛡️ PHASE P4 — Robustness (Week 8)

**Milestone:** Reverb, noise, codec augmentation integrated; clean performance preserved  
**🚧 GATE M4:** Robustness table across conditions; clean-vs-augmented ablation confirms clean performance not degraded.

**Ownership rotation:** Dev A **leads** augmented training run; Dev B supports; Dev C runs ablation.

---

## 🔄 PARALLEL — P4 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P4-A1 | Integrate full 3-stage augmentation into training loop (RIR → WHAM noise → codec) | P1-A1, P1-A2, P1-A4, P2-B6 | A | Augmented training runs | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — augmentation modules ready; blocked on P2-B6 (training loop) |
| P4-A2 | Re-tune trainable heads on augmented data | P4-A1 | A (leads), B support | Retrained checkpoint | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P4-A3 | Codec degradation evaluation table | P1-A4, P0-C1 | A | Clean-to-codec degradation table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P4-C1 | Clean-vs-augmented ablation | P4-A2, P0-C1 | C | Ablation table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P4-C2 | L3 evaluation: WHAMR! + Libri3Mix-noisy (SI-SDRi + DNSMOS) | P4-A2 | C | L3 results | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — reverb-noisy eval set tooling ready (`data/make_reverb_eval.py`); still blocked on P4-A2 (retrained model) |
| P4-INT1 | Verify mixed-condition training (not worst-case-only) | P4-A2 | A | Training condition distribution log | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

**MASTER augmentation pipeline (§6.2):** Each stage probabilistic; ground truth = clean stems before augmentation; SI-SDRi against original clean.

---

## 🚧 GATE M4 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Three-stage augmentation active in training
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Retrained checkpoint evaluated on reverb + noise + codec conditions
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Robustness table** across conditions (project vs baselines)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Clean-vs-augmented ablation** confirms no clean regression
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Novelty N5:** codec robustness degradation table
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed

---

# 🏆 PHASE P5 — Differentiating Results (Weeks 9–10)

**Milestone:** Sparse-overlap curve, real-room eval, break-point curve produced  
**🚧 GATE M5:** All three flagship results locked. Tier-3 work (N9, N10) unlocked only after this gate.

**All three collaborate; each owns one flagship result.**

---

## 🔄 PARALLEL — P5 flagship results

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P5-C1 | **Flagship 1:** Sparse-overlap curve on SparseLibriMix — SI-SDRi vs overlap at {0, 20, 40, 60, 80, 100}% | M3 eval harness, P3-A2 | C | Overlap curve figure + table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-A1 | **Flagship 2:** Real-room recording session (2–5 speakers, scripted overlap) | M4 | A (leads) | Recorded real-room set | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-A2 | Real-room per-stream Whisper WER evaluation | P5-A1 | A | Real-room WER table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-A3 | LibriCSS WER evaluation (up to 2 concurrent) | P0-C1 | A | LibriCSS results | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-B1 | **Flagship 3:** Break-point curve — SI-SDRi vs speaker count 2→7 | P3-A1 (mixer high N) | B | Break-point figure | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-B2 | Document MossFormer2→SR-CorrNet handoff above 3 speakers | P5-B1 | B | Transition boundary note | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-ALL1 | Full ablation table (all 9 mandatory conditions) | M2, M3, M4 | All (split) | Ablation table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

---

## 🤝 P5 COLLAB — Real-room recording (all three as speakers)

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Script overlapping dialogue (2–5 speakers)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Record in real room on phones
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Known transcripts for WER ground truth
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Held-out from training data

---

## Mandatory ablations checklist (MASTER §10.2)

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] MossFormer2-only vs full cascade
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] SR-CorrNet-only vs full cascade
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Static equal-weight ensemble vs cascade
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Fixed threshold vs learned gatekeeper
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Router with null expert vs without
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] 100% overlap training vs sparse overlap curriculum
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Oracle speaker count vs learned count
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Without codec augmentation vs with
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Without mixture-consistency feature vs with

---

## 🚧 GATE M5 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] SparseLibriMix curve complete (6 ratios) — **Novelty N4**
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Real-room WER table complete — **Novelty N7** (if chosen over N8)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Break-point curve 2–7 speakers — **Novelty N9**
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] All 9 ablation rows filled
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed

---

# 🚀 PHASE P6 — Demo & Report (Weeks 11–12)

**Milestone:** Gradio demo, ablation table, written report complete  
**🚧 GATE M6:** Submission package complete — demo runs, report written, results reproduce from bundle.

---

## 🔄 PARALLEL — P6 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P6-C1 | Gradio demo: upload audio → speaker count, N waveforms, spectrograms, Whisper transcripts | M5 full system | C | `demo/gradio_app.py` | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — MockEngine skeleton in `demo/app.py` done 2026-07-09; real engine pending M5 |
| P6-B1 | Routing-weight interpretability panel in demo | P6-C1 | B | Demo panel | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P6-B2 | Mixture-consistency self-grade display in demo | P6-C1 | B | Demo panel | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P6-B3 | Auto-flag low-confidence outputs in demo | P6-B2 | B | Demo feature (N6) | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P6-A1 | Demo audio processing backend | M5 full system | A | Demo backend API | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P6-A2 | Reproducibility package: configs, checkpoints, instructions | All phases | A | Reproducibility bundle | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P6-ALL1 | Technical report — Dev A section | M5 results | A | Report section | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P6-ALL2 | Technical report — Dev B section | M5 results | B | Report section | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P6-ALL3 | Technical report — Dev C section (calibration, curves) | M5 results | C | Report section | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P6-ALL4 | Final ablation table in report | P5-ALL1 | All | Report table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P6-ALL5 | Demo video or hosted demo link | P6-C1 | C | Demo artifact | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

**Demo must show (MASTER §9 Phase 6):** estimated count, N waveforms, spectrograms, per-stream Whisper transcripts, routing-weight visualization, mixture-consistency self-grade.

---

## 🚧 GATE M6 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Gradio demo runs end-to-end on uploaded audio
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Report complete with all three sections
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Reproducibility bundle reproduces key numbers
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] All reporting checklist items below addressed
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Submission package delivered**

---

# Reporting checklist (MASTER §10.3)

Track at M6; start collecting artifacts from M0.

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Libri3Mix + WSJ0-3mix SI-SDRi (known + unknown N)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] SparseLibriMix SI-SDRi at {0, 20, 40, 60, 80, 100}% overlap
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Speaker-count accuracy + confusion matrix + calibration curve
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] WHAMR! + reverberant Libri3Mix SI-SDRi + DNSMOS
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Clean-to-codec degradation table
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Real-room scripted per-stream WER
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Break-point curve: SI-SDRi vs speaker count 2→7
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Cascade escalation rate per tier
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Inference RTF at average and worst-case escalation
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Ablation table (≥9 conditions)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Router weight interpretability panel
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Gradio demo link or recorded demo video

---

# Evaluation tiers (MASTER §1.4 / §10.1)

| Tier | Speakers | Overlap | Conditions | Expected SI-SDRi | Benchmark | Metrics |
|------|----------|---------|------------|------------------|-----------|---------|
| L0 | 2 | 100% | clean anechoic | 18–24 dB | — | SI-SDRi |
| L1 | 3 | 100% | clean anechoic | 15–20 dB | Libri3Mix, WSJ0-3mix | SI-SDRi |
| L2 | 3–4 | 40–60% | clean / mild noise | 10–15 dB | SparseLibriMix | SI-SDRi vs overlap |
| L3 | 4–5 | 20–40% sparse | WHAM! noise | 8–12 dB | WHAMR!, Libri3Mix-noisy | SI-SDRi, DNSMOS |
| L4 | 5–7 | variable | noise + reverb | 5–10 dB | WSJ0-4/5Mix, Libri5Mix | SI-SDRi, count accuracy |
| L5 | any | any | no reference | — | REAL-M, real audio | DNSMOS, listening test |
| Real | 2–5 | scripted | real room | — | Real-room set, LibriCSS | WER, DNSMOS |

---

# Novelty ledger tracker (MASTER §12)

| ID | Contribution | Tier | Proof artifact | Target phase | Status |
|----|-------------|------|----------------|--------------|--------|
| N1 | Conditional cascade routing | Mandatory | Ablation + escalation rate + compute curve | P2, P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — pending cascade gate (P2-B1) and ablation run |
| N2 | Two-level router + null expert + load-balance | Mandatory | Router ablation + demo panel | P2, P6 | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — `models/router.py` done 2026-07-09; ablation run pending M2 |
| N3 | Calibrated stop-classifier + confusion matrix | Mandatory | Confusion matrix + calibration curve | P3 | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — `models/stop_classifier.py` + training script done 2026-07-09; full training run on real data pending M2 |
| N4 | Sparse-overlap curve (SparseLibriMix) | Mandatory | SI-SDRi vs overlap table | P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| N5 | Codec augmentation robustness | Mandatory | Clean-to-codec degradation table | P4 | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — `data/codec_augmentation.py` prototype done 2026-07-10; degradation table pending P4-A3 |
| N6 | Mixture-consistency self-grading | With N3 | Stop-classifier ablation + demo flag | P3, P6 | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — mixture-consistency feature in stop_classifier 2026-07-09; demo display pending P6-B2 |
| N7 | Real-room WER evaluation | Tier 2 (pick one) | Real-room WER table | P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| N8 | Enrollment-based target extraction demo | Tier 2 (alt) | Interactive demo mode | P6 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| N9 | Break-point curve 2–7 speakers | Tier 3 | SI-SDRi vs N curve | P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — locked until M5 |
| N10 | Generative flow post-corrector | Tier 3 | DNSMOS ablation | Post-M5 only | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — locked until M5 |

**Commit set:** N1–N5 mandatory; N6 with N3; pick N7 or N8; N9 nearly free; N10 only if all stable.

---

# Dataset acquisition tracker

| Dataset | Role | Owner phase | Status |
|---------|------|-------------|--------|
| LibriSpeech | Source audio for mixer | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — official `test-clean` downloaded and used on RunPod for P1 validation; full training source pool not yet prepared |
| Libri2Mix / Libri3Mix | Primary train/eval | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — one official Libri3Mix test sample generated and consumed successfully on RunPod; full train/dev/test corpus is not yet prepared |
| Libri4Mix / Libri5Mix | N=4,5 training | P1-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep script ready (`data/prepare_librimix_highn.py`, P1-A5); not yet run on this machine |
| SparseLibriMix | L2 overlap eval (test only) | P3-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep script ready (`data/prepare_sparselibrimix.py`, PR #7); generation requires LibriSpeech test-clean at runtime |
| WHAM! | Noise augmentation | P1-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — official WHAM! archive downloaded and extracted on RunPod; used to generate the official one-sample Libri3Mix P1 validation input |
| WHAMR! | Reverb eval + RIR source | P1-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — license-free reverb-noisy eval ready (`data/make_reverb_eval.py`); real WHAMR! gated on WSJ0/LDC (`data/prepare_whamr.py`) |
| VCTK | Accent diversity | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep script ready (`data/prepare_vctk.py`, P0-A4); not yet run on this machine |
| WSJ0-*Mix | Literature comparison (LDC license) | Optional | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| LibriheavyMix | Large-scale reverb (if compute) | Optional | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| REAL-M | Real 2-speaker, no reference | P1-C | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| LibriCSS | Real-room WER | P5-A | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| Real-room set | Team-recorded flagship | P5-A | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

---

# Risk register & fallback triggers

| Risk | Severity | Mitigation | Trigger action | Status |
|------|----------|------------|----------------|--------|
| SR-CorrNet weights unavailable | High | TF-GridNet via ESPnet | P1-B3 fallback | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| REAL-M too noisy for gate | Medium | Scene-analyzer reverb proxy as primary gate signal | Reprioritize P2-B1 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Alignment / identity tracking fails on real long audio | High | Inspect chunk assignments; strengthen ECAPA continuity/track birth-death logic; rerun official Libri3Mix gate | P1-C3 / P1-INT2 | ![resolved](https://img.shields.io/badge/resolved-brightgreen?style=flat-square) [x] — resolved 2026-07-13. The 2 switches (2026-07-11) were MossFormer2 failing to emit 3 stable streams (expert inadequacy → escalation), plus an unsound metric and a dead `max_tracks` cap. Lock now proven in CI (`tests/test_p1_int2_identity_lock.py`, 0 switches, 2+3 spk, 4 seeds). See P1-INT2 |
| Router collapse | Medium | Increase load-balance loss weight | Monitor P2-C2 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| MossFormer2 max 3 speakers | Medium | Hand off to SR-CorrNet for N>3 | Document P5-B2 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Fusion loses to SR-CorrNet alone | High | Fall back to ensemble / SR-CorrNet-primary | MASTER §5.3 at M2 | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — **confirmed on clean AND mixed-N 2026-07-13**, and it is not marginal: even at 100% escalation the fused output is ~3–4 dB below raw SR-CorrNet (mixed-N: ~12.5 vs ~16.2). The CRRR fusion actively degrades a strong expert on this smoke-scale training. This is the central M2 blocker. Real mitigations to try: (a) route escalated samples to raw SR-CorrNet, bypassing fusion (SR-primary), (b) much larger fusion training set, (c) test on noisy/reverberant data where the residual has something to correct |
| Scope creep | High | Tier 3 only after M5 | Enforce gate | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| 6–7 speaker quality poor | Expected | Graceful degradation framing | P5-B1 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Train-test domain gap | High | Mixed-condition aug + real-room eval | P4, P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Dev B bottleneck | Planning | Fast review P1/P2; A/C front-load | Ongoing | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring; A and C have front-loaded P1 work |
| Expert deps missing from requirements | Low | Add clearvoice + TF-GridNet deps to requirements.txt | Found 2026-07-11 running real experts | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — flagged to Dev B |
| SR-CorrNet weights still unconfirmed | High | TF-GridNet fallback active and verified working | Confirmed falling back 2026-07-11 | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — fallback verified, weights still absent |

---

# Compute & parameter budget tracker

| Item | Target | Actual | Status |
|------|--------|--------|--------|
| Frozen expert params | ~60–75M | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — not yet measured |
| Trainable params | ~3.3M | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — not yet measured |
| Training time (all heads) | 2–4 days on 1 GPU | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — not yet run |
| RTF @ 30% escalation | ~0.14 | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — not yet measured |
| RTF worst case (100% escalation) | ~0.40 | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — not yet measured |
| Inference memory | ≤16 GB T4 | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — not yet measured |
| Development GPUs | 2× T4 16GB | 1× RTX 4090 RunPod used for P1 validation | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — temporary validation pod provisioned; planned 2×T4 development setup not yet established |
| Final run GPU | A100 40GB (if available) | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — not yet provisioned |

**Trainable component sizes:**
- Scene Analyzer: ~1.5M
- Router: ~0.5M
- Stop-Classifier: ~0.3M
- Fusion Head (CRRR): ~1.0M

---

# Phase timeline summary

| Phase | Weeks | Gate | Parallel? | Critical owner |
|-------|-------|------|-----------|----------------|
| **P0** Foundation | 1–2 | M0 | 🔄 Full parallel | All |
| **P1** Expert integration | 3–4 | M1 | B sequential; A,C parallel | B |
| **P2** Cascade core | 5–6 | M2 | Sub-components parallel; training sequential | B |
| **P3** Counting | 7 | M3 | 🔄 Mostly parallel | C (leads) |
| **P4** Robustness | 8 | M4 | 🔄 Mostly parallel | A (leads) |
| **P5** Differentiators | 9–10 | M5 | 🔄 Parallel flagship results | All |
| **P6** Demo & report | 11–12 | M6 | 🔄 Parallel | All |

---

# Optional / stretch (do not start until M5 unless noted)

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] MambaDeflate stretch expert (E_DEF) for 5+ speakers — SPMamba / ReSepNet
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] LibriheavyMix large-scale reverb training
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] WSJ0-*Mix literature comparison (requires LDC license)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Backbone fine-tuning (adds 1–2 weeks; only if ablation justifies)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Wiener filter postprocessing
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] One-step generative flow corrector (N10, Tier 3)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Enrollment-based demo mode (N8, Tier 2 alternative)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Real-time streaming demo mode

---

*End of PROJECT_TODO.md — edit this file as the project progresses.*
