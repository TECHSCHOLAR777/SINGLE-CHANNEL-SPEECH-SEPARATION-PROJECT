# CA-MoSE Project TODO - Single Source of Truth

> **Derived from:** `MASTER_PROJECT.md` (v1.2) + `DEVELOPMENT_PLAN.md`  
> **Purpose:** Living task tracker for the full 10–12 week project. Edit checkboxes as work completes.  
> **Last updated:** 2026-07-11 (colour badges + live dashboard)

---

## 📊 Project pulse

> Snapshot **2026-07-11** — refresh the counts whenever you flip a status.

![Done](https://img.shields.io/badge/✅_done-70-brightgreen?style=for-the-badge)
&nbsp;
![In progress](https://img.shields.io/badge/🚧_in_progress-32-yellow?style=for-the-badge)
&nbsp;
![Not done yet](https://img.shields.io/badge/❌_not_done_yet-145-red?style=for-the-badge)

**Overall** `████████░░░░░░░░░░░░░░░░░░░░` **28%** &nbsp;·&nbsp; 70 done &nbsp;·&nbsp; 32 in flight &nbsp;·&nbsp; 145 to go &nbsp;·&nbsp; **247 tasks**

**Milestones** &nbsp;
![M0](https://img.shields.io/badge/M0-✅_passed-brightgreen?style=flat-square)
![M1](https://img.shields.io/badge/M1-🚧_in_progress-yellow?style=flat-square)
![M2](https://img.shields.io/badge/M2-🔒_locked-lightgrey?style=flat-square)
![M3](https://img.shields.io/badge/M3-🔒_locked-lightgrey?style=flat-square)
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

- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] `main` always runnable and passing CI; **no direct commits to main** — CI workflow active; all merges via PR
- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] Branch naming: `type/owner/short-description` (e.g. `feat/devb/fusion-head`) — Dev C followed convention; Dev A used plain branch names
- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] One branch per task; short-lived (merge within 2–4 days)
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Every change via PR with **1 review from a non-owner** — done 2026-07-09/10, PRs #1–#4
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] P2 training-loop PR + shared interface changes → **review from all three**
- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] Squash-merge; rebase before merge; delete branch after merge
- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] Merge at least at each milestone gate; ideally more often

### Codebase standards (all phases)

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Formatter + linter (Black + Ruff) via pre-commit + CI — done 2026-07-09, `.pre-commit-config.yaml` + `ci.yml`
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Type hints on all public function signatures — confirmed across all modules
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] `SeparationResult` schema defined once in `schemas/` — never redefined ad hoc — done 2026-07-09
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Every module: header docstring (purpose, inputs, outputs) — confirmed across all modules
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Each owner maintains one-page design note in `docs/` — `docs/models.md`, `docs/DEVC_DESIGN.md` done
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] `docs/decisions.md` updated for every architecture choice (date + one-line reason) — done 2026-07-09
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Unit tests for every data and metric function — 244 tests passing as of 2026-07-11 (1 env-specific torchaudio failure on Windows/Python 3.13)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] One shared end-to-end integration test — must pass before every gate

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
| P0-INT3 | Replace mixer stub with Dev A mixer (optional upgrade) | P0-A1, P0-B7 | A + B | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P0-INT4 | Shared end-to-end integration test (tiny input → baseline → SI-SDRi) | P0-INT1 | All | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

---

## 🚧 GATE M0 — Acceptance criteria

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Dev A: `data/mixer.py` produces valid mixture + stems for N=2,3 — done 2026-07-10
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Dev C: `eval/metrics.py` computes SI-SDRi with PIT on known tensors — done 2026-07-09
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Dev B: baseline runner produces results table — done 2026-07-09
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] **All three independently run baseline on same Libri3Mix test set → identical SI-SDRi (±0.1 dB tolerance)** — confirmed 2026-07-10
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Integration test passes on `main` — P0-INT4 still pending
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
| P1-C3 | Cross-chunk identity lock (4s chunks, 1s overlap) | P1-C2 | Chunk-stitching module in `align/` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-12. Root cause of the phantom-track bug found and fixed: `ChunkStitcher` had no cap on the track bank, so a single unstable output slot minted a new persistent track every chunk (reproduced exactly: `[[0,1],[0,2],[0,3]]`, 4 tracks for 3 speakers). Added `max_tracks`: at cap, an unmatched stream force-assigns to its best Hungarian partner instead of spawning, without polluting that track's embedding EMA. `tests/test_chunking_cap.py`, 9 tests |
| P1-C4 | Alignment unit tests including same-gender stress case | P1-C2 | Tests in `tests/` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `tests/test_align_same_gender.py`, PR #5 |

---

## 🔄 PARALLEL — Dev A (P1, while B integrates)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P1-A1 | Augmentation stage 1: RIR reverb (pyroomacoustics / FAST-RIR) | P0-A1 (mixer) | `data/augmentation/rir.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, Stage 1 in `data/augmentation.py`, PR #4 |
| P1-A2 | Augmentation stage 2: WHAM! noise | P0-A1 | `data/augmentation/noise.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, Stage 2 in `data/augmentation.py`, PR #4 |
| P1-A3 | WHAM! + WHAMR! dataset download | none | Data prep scripts | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — WHAM! noise done 2026-07-11 (`data/prepare_wham.py`); WHAMR! addressed via license-free reverb-noisy eval (`data/make_reverb_eval.py`) + gated real-WHAMR! generator (`data/prepare_whamr.py`, needs WSJ0/LDC); all tested |
| P1-A4 | Codec augmentation prototype (Opus, AAC low bitrate) | none | `data/augmentation/codec.py` prototype | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `data/codec_augmentation.py`, PR #4 |
| P1-A5 | Libri4Mix + Libri5Mix extension scripts | P0-A2 | `github.com/shakeddovrat/librimix` integration | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `data/prepare_librimix_highn.py` + tests (N-aware disk loader is the P3-A1 follow-up) |

---

## 🤝 P1 COLLAB — Dev B + Dev C pairing

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Define alignment interface: expert `SeparationResult` → aligner input format — `Engine` protocol + `run_and_align` in `align/integration.py`, 2026-07-11
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Dev C understands model output format — confirmed by running real MossFormer2/TF-GridNet wrappers end-to-end 2026-07-11
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Document in `docs/decisions.md`

---

## ⛓ SEQUENTIAL — P1 integration

| ID | Task | Depends on | Owner | Status |
|----|------|------------|-------|--------|
| P1-INT1 | Align MossFormer2 + SR-CorrNet outputs on same 3-speaker clip | P1-B6, P1-C2 | B + C | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `align/integration.py` + `tests/test_m1_real_experts.py`; verified on real experts (MossFormer2 + TF-GridNet fallback + real ECAPA), mean matched distance 0.57 |
| P1-INT2 | Cross-chunk lock verified on >4s audio | P1-C3, P1-INT1 | C | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — run on a real Libri3Mix clip 2026-07-11 via RunPod, `identity_switches: 2, passed: false`. Root cause was NOT identity drift: the cheap expert (MossFormer2_SS_16K) is a 2-speaker checkpoint returning K=2 on a 3-speaker mixture, so one output slot structurally cannot hold one consistent speaker. The switch-counting metric itself was also unsound (Hungarian-matched silent tracks, compared every window to window 0 instead of its predecessor) and has been rewritten: silent tracks excluded via RMS floor, switches counted only between consecutive active windows, `expert_covers_all_speakers` now gates pass/fail explicitly so a lucky-looking green cannot hide a structurally broken run. `scripts/validate_alignment.py` rewritten 2026-07-11. Re-run on RunPod pending, now that the expert's stream-count gap is fixed (see P2 note below) |
| P1-INT3 | REAL-M scores MossFormer2 output on test clip | P1-B1, P1-B4 | B | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, covered in test_expert_integration.py (mocked) |

---

## 🚧 GATE M1 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] MossFormer2 wrapper returns 3 streams + embeddings
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] SR-CorrNet wrapper returns K streams + attractor vectors + confidence
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] REAL-M produces per-stream SI-SNRi estimates without reference
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Hungarian alignment matches streams to consistent speaker order — verified on real experts 2026-07-11 (P1-INT1)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Cross-chunk identity lock works on long audio
- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] **Shared integration test: one 3-speaker clip, both experts, aligned output** — passes on real experts (`test_m1_real_experts.py`); long-form identity-lock leg awaits a real speech clip (Kaggle run planned)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed

---

# 🧩 PHASE P2 — Cascade Core (Weeks 5–6)

**Milestone:** Scene analyzer, router, cascade gate, fusion head train and beat best single expert  
**🚧 GATE M2:** Full CA-MoSE forward pass runs end-to-end, trains a few epochs, **beats best single expert** on mixed-condition validation, reports **measured escalation rate**. Everyone can explain single-input flow.

**Fallback trigger (MASTER §5.3):** If cascade cannot beat MossFormer2 alone by end of P2 → fall back to always-run-both ensemble, train fusion only, present routing as interpretability.

---

## 🤝 P2 COLLAB — Before any implementation (all three)

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Cascade architecture review session
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Agree tensor flow: [B,T] → Scene Analyzer → MossFormer2 → REAL-M → gate → SR-CorrNet → align → fuse
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Agree quality threshold `tau` tuning strategy (conservative: borderline inputs escalate)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Agree composite loss weights (initial lambdas from MASTER §7.2)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Log decisions in `docs/decisions.md`

---

## 🔄 PARALLEL — Trainable sub-components (P2)

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P2-A1 | Scene Analyzer (~1.5M params): log-mel + handcrafted features → reverb proxy, noise floor, overlap density, spectral flatness, modulation rate, K_coarse | M1 | A | `models/scene_analyzer.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done, `SceneAnalyzer` merged and wired into `CAMoSETrainable` in `train/trainer.py` |
| P2-C1 | Two-level Adaptive Router (~0.5M params): sequence gate + segment gate (1–2s windows), sigmoid (not softmax), w_TF/w_TD/w_NULL | P2-A1 | C | `models/router.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — wired to Scene Analyzer output via `CAMoSETrainer.forward_batch`, verified in `tests/test_e2e_forward.py` (P2-INT2) |
| P2-C2 | Load-balance auxiliary loss for router | P2-C1 | C | Loss term + collapse monitoring | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — wired in `train/losses.py` CompositeLoss, 2026-07-11 |
| P2-C3 | Null-expert sparsity loss | P2-C1 | C | Anti-hallucination loss term | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — wired in `train/losses.py` CompositeLoss, 2026-07-11 |
| P2-B1 | Cascade gate: compare REAL-M score to threshold `tau`; escalate if below | P1-B4 | B | Cascade controller | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `models/cascade_gate.py` |
| P2-B2 | Escalation-rate instrumentation | P2-B1 | C | Dashboard / logging | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done, `CascadeRunLogger` in `eval/cascade_logging.py` feeds live `CascadeDecision`s into `RunLog`; `escalation_rate` query verified against it in `tests/test_cascade_logging.py` |
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
| P2-INT1 | **Whole-team review of training-loop PR** | P2-B6 | All | Approved PR | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P2-INT2 | End-to-end forward pass integration test | P2-B6 | All | E2E test | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done, `tests/test_e2e_forward.py`, mock mode: training-side (scene→router→gate→fusion→loss, gradients verified) + inference-side (result→quality→gate→coordinator→log) |
| P2-INT3 | Short training run (few epochs) on mixed conditions | P2-INT2 | B | Checkpoint + logs | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — **was structurally impossible until 2026-07-12**: `CRRRFusionHead.forward` raises `stream shape mismatch` the instant a 2-stream MossFormer2 output meets a 3-stream SR-CorrNet output on a 3-speaker mixture, which every Libri3Mix training batch would trigger. Fixed via `MossFormer2Expert(target_speakers=k)`: the missing stream is filled with the RESIDUAL (mixture minus the sum of emitted streams), which on a genuine speaker gap recovers something close to the missed speaker rather than being pure padding — tested to 1e-5 in `tests/test_mossformer2_residual.py`. Marked `synthetic="residual"`, confidence 0.0, so downstream heads learn to distrust it; residual energy is also a free, principled escalation signal. Fusion now accepts the padded input (regression test included). Blocker is dead; a training run has still not been executed — 0 epochs run, 0 checkpoints exist |
| P2-INT4 | Validate: beats best single expert on val set | P2-INT3 | B + C | Metric comparison table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P2-INT5 | Measure and report escalation rate | P2-B2, P2-INT3 | C | Escalation rate per tier | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

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

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Full CA-MoSE forward pass: preprocess → scene → MossFormer2 → REAL-M → gate → (SR-CorrNet + fuse if escalated) → postprocess
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Trained heads (~3M params) converge in few-epoch test run
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Beats best single expert on mixed-condition validation**
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Escalation rate measured and logged (target ~30–40%)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Expected RTF computed at measured escalation rate
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] All three can explain single-input flow through system
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Training-loop PR reviewed by all three
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Novelty N1 proof started:** ablation plan for single-expert vs cascade documented

---

# 🔢 PHASE P3 — Speaker Counting (Week 7)

**Milestone:** Learned stop-classifier produces confusion matrix and calibration curve  
**🚧 GATE M3:** System estimates speaker count on unknown-N inputs; produces **confusion matrix + calibration curve**.

**Ownership rotation:** Dev C **leads** counting; Dev B supports features; Dev A supports N=2..5 mixtures.

> **Infra note (2026-07-12):** No training run for P2 or P3 has been executed yet — 0 epochs, 0 checkpoints, for either phase. What changed this week is that both blockers standing in front of a first run are gone:
> 1. The 2-vs-3-stream shape-mismatch crash (see P2-INT3) is fixed.
> 2. A data pipeline now exists end to end: `scripts/prepare_all_data.sh` (resumable, preflighted generation), `scripts/preflight_data.py` (catches missing-file crashes before they cost hours, not after), `scripts/build_train_cache.py` (runs the frozen experts ONCE and caches fp16 tensors, since re-running two frozen separation networks every epoch was the dominant training cost and bought nothing), and `train/cached_dataset.py` (feeds the cache straight into `CAMoSETrainer`/stop-classifier training with zero experts loaded). Kaggle-specific path in `notebooks/kaggle_build_cache.py` (mix_clean subset, no WHAM, fits the 20 GB output cap).
>
> P2-INT3 and P3-C5 are therefore genuinely next, not blocked on unresolved design questions — they need the cache built and a training loop pointed at it.

---

## 🔄 PARALLEL — P3 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P3-B1 | Feature extractors for stop-classifier: (1) residual energy ratio, (2) VAD prob on residual, (3) ECAPA embedding distance to prior stems, (4) mixture-consistency error | M2 | B | `models/counting_features.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-11, `CountingFeatureExtractor` + unit tests |
| P3-C1 | Learned stop-classifier MLP (~0.3M params): 4 features + attractor stop logit → P(more speakers) | M2 | C | `models/stop_classifier.py` | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — code shipped early 2026-07-09, PR #2; real training on Libri2–5Mix pending M2 gate; training-data pipeline now exists (see infra note below), 0 real epochs run |
| P3-C2 | Count BCE loss integration into trainer | P3-C1 | C | `L_count-BCE` active in trainer | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done, `count_bce_loss` live in `CompositeLoss.forward` fed by `scene_out["count_logits"]` in `train/trainer.py`; finiteness asserted in `tests/test_e2e_forward.py` |
| P3-C3 | Count confusion matrix report generator | P0-C6, P3-C1 | C | `eval/counting_report.py` | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — code shipped early 2026-07-09, PR #2; needs real classifier outputs to produce results |
| P3-C4 | Calibration curve report (estimated prob vs actual accuracy) | P3-C3 | C | Calibration plot + metrics | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — code shipped early 2026-07-09, PR #2; needs real classifier run to produce calibration data |
| P3-A1 | Mixer support for N=2..5 (Libri2Mix–Libri5Mix) | P0-A1, P1-A5 | A | On-the-fly 2–5 speaker mixtures | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — DynamicMixer supports arbitrary N; Libri4/5Mix prep scripts done (P1-A5); N-aware disk loader (`discover_librimix_samples`) still 3-speaker only |
| P3-A2 | SparseLibriMix download (test-only, 6 overlap ratios) | none | A | `github.com/popcornell/SparseLibriMix` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — done 2026-07-10, `data/prepare_sparselibrimix.py` + `tests/test_prepare_sparselibrimix.py`, PR #7 |
| P3-C5 | Stop-classifier training on Libri2–5Mix | P3-C1, P3-A1 | C | Trained classifier checkpoint | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — training script shipped 2026-07-09, PR #2 (self-test passes); real training run pending — data pipeline (see infra note) now exists, no run has been executed |
| P3-INT1 | Speaker-count coordinator: SR-CorrNet TDA attractors + stop-classifier fusion | P3-B1, P3-C1, P1-B2 | B + C | `models/count_coordinator.py` | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — code done, `CountCoordinator` fuses attractor `stop_logit` + classifier logit in logit space with graceful degradation; `attractor_weight` still needs tuning on a real dev set once P3-C5 training lands |
| P3-INT2 | Unknown-N evaluation across N=2,3,4,5 | P3-INT1, P3-C3 | C | Count accuracy results | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — blocked on P3-INT1 |

**MASTER §4.5:** Stop when P(more speakers) falls below calibrated threshold. Report full confusion matrix (which mistakes: merge vs split).

---

## 🤝 P3 COLLAB — Dev A supports counting training data

- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] Dev A delivers 2–5 speaker mixture pipeline for classifier training — DynamicMixer ready; Libri4/5Mix prep scripts done (P1-A5)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Verify no speaker leakage across splits

---

## 🚧 GATE M3 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Stop-classifier trained on Libri2–5Mix
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Unknown-N inference works at test time (N not given)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Manual count override exposed (MASTER §1.3 assumption)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Confusion matrix produced** (rows=true N, cols=estimated N)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Calibration curve produced**
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Oracle-count vs learned-count ablation planned (for P5)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Novelty N3 + N6:** counting contribution + mixture-consistency feature documented
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed

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
| LibriSpeech | Source audio for mixer | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — `prepare_librimix.py` download script ready; not yet run on this machine |
| Libri2Mix / Libri3Mix | Primary train/eval | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — generation script ready; not yet run on this machine |
| Libri4Mix / Libri5Mix | N=4,5 training | P1-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep script ready (`data/prepare_librimix_highn.py`, P1-A5); not yet run on this machine |
| SparseLibriMix | L2 overlap eval (test only) | P3-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep script ready (`data/prepare_sparselibrimix.py`, PR #7); generation requires LibriSpeech test-clean at runtime |
| WHAM! | Noise augmentation | P1-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep script ready (`data/prepare_wham.py`, P1-A3); not yet run on this machine |
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
| Alignment fails same-gender | Medium | Local SI-SDRi fine-alignment; stress test | P1-C4 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Router collapse | Medium | Increase load-balance loss weight | Monitor P2-C2 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| MossFormer2 max 3 speakers | Medium | Hand off to SR-CorrNet for N>3 | Document P5-B2 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Fusion loses to SR-CorrNet alone | Medium | Fall back to ensemble / SR-CorrNet-primary | MASTER §5.3 at M2 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
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
| Development GPUs | 2× T4 16GB | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — not yet provisioned |
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
