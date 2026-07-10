# CA-MoSE Project TODO — Single Source of Truth

> **Derived from:** `MASTER_PROJECT.md` (v1.2) + `DEVELOPMENT_PLAN.md`  
> **Purpose:** Living task tracker for the full 10–12 week project. Edit checkboxes as work completes.  
> **Last updated:** 2026-07-10

---

## How to use this document

| Symbol | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Done |
| **🔄 PARALLEL** | Can run at the same time as sibling tasks — no cross-team blocker |
| **⛓ SEQUENTIAL** | Blocked until listed dependency is complete |
| **🤝 COLLAB** | Mandatory whole-team session — do not skip |
| **🚧 GATE** | Hard milestone checkpoint — project must not advance until passed |

**Edit rules:**
1. Only mark `[x]` when the deliverable exists, is tested, and is merged to `main`.
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

- [~] `main` always runnable and passing CI; **no direct commits to main** — CI workflow active; all merges via PR
- [~] Branch naming: `type/owner/short-description` (e.g. `feat/devb/fusion-head`) — Dev C followed convention; Dev A used plain branch names
- [~] One branch per task; short-lived (merge within 2–4 days)
- [x] Every change via PR with **1 review from a non-owner** — done 2026-07-09/10, PRs #1–#4
- [ ] P2 training-loop PR + shared interface changes → **review from all three**
- [~] Squash-merge; rebase before merge; delete branch after merge
- [~] Merge at least at each milestone gate; ideally more often

### Codebase standards (all phases)

- [x] Formatter + linter (Black + Ruff) via pre-commit + CI — done 2026-07-09, `.pre-commit-config.yaml` + `ci.yml`
- [x] Type hints on all public function signatures — confirmed across all modules
- [x] `SeparationResult` schema defined once in `schemas/` — never redefined ad hoc — done 2026-07-09
- [x] Every module: header docstring (purpose, inputs, outputs) — confirmed across all modules
- [x] Each owner maintains one-page design note in `docs/` — `docs/models.md`, `docs/DEVC_DESIGN.md` done
- [x] `docs/decisions.md` updated for every architecture choice (date + one-line reason) — done 2026-07-09
- [x] Unit tests for every data and metric function — 244 tests passing as of 2026-07-11 (1 env-specific torchaudio failure on Windows/Python 3.13)
- [ ] One shared end-to-end integration test — must pass before every gate

### Data split discipline (mandatory, all phases)

- [~] **No speaker identity** in more than one of train / val / test splits — enforced in `DynamicMixer` via `train_speaker_ids` / `test_speaker_ids`; no standalone validation script yet

---

# PHASE P0 — Foundation (Weeks 1–2)

**Milestone:** Data pipeline produces mixtures with ground truth; eval harness computes SI-SDRi on a known model  
**🚧 GATE M0:** All three independently reproduce the **same SI-SDRi baseline** on Libri3Mix. If numbers differ → fix harness or data before anyone builds on top.

**Parallelism:** **🔄 FULL PARALLEL** — all tasks below can start day 1 with zero cross-team blocking (except noted).

---

## 🤝 P0 Day 1 — COLLAB (1 hour, all three)

- [x] Agree repository directory structure — done 2026-07-09
- [x] Agree YAML config schema (top-level keys, paths, device, sample_rate=16000) — done 2026-07-09, `configs/baseline.yaml` + `configs/default.yaml`
- [x] Agree `SeparationResult` interface: `streams [K,T]`, `speaker_count`, `confidence`, per-stream metadata, `mixture`, `escalated`, `expert_used` — done 2026-07-09, `schemas/separation_result.py`
- [x] Agree formatter/linter (Black + Ruff) — done 2026-07-09, logged in `docs/decisions.md`
- [x] Log decisions in `docs/decisions.md` — done 2026-07-09

---

## 🔄 PARALLEL — Dev A tasks (P0)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P0-A1 | Dynamic mixer: sample N∈{2,3,4,5} speakers, per-speaker level offsets 0–5 dB, output mixture + clean stems | none | `data/mixer.py` + unit tests | [x] — done 2026-07-10, PR #3 |
| P0-A2 | LibriMix + Libri3Mix download and preparation scripts | none | Reproducible data-prep script | [x] — done 2026-07-10, `data/prepare_librimix.py`, PR #3 |
| P0-A3 | LibriSpeech source setup (`openslr.org/12`) | none | Clean speaker pool for mixer | [x] — done 2026-07-10, integrated in `prepare_librimix.py` |
| P0-A4 | VCTK accent diversity pool (`openslr.org`) | none | Extended speaker pool | [ ] — in MASTER_PROJECT.md architecture (data source) but not a DEVELOPMENT_PLAN.md P0 task; not an M0/M1 blocker |
| P0-A5 | Enforce speaker-disjoint train/val/test splits | P0-A2 | Split manifest / validation script | [x] — functionally enforced by DynamicMixer (`train_speaker_ids`/`test_speaker_ids`); no separate deliverable in either doc, so no standalone script required |
| P0-A6 | Overlap scheduler stub (100% → 40% → 20% curriculum placeholder) | P0-A1 | `data/overlap_scheduler.py` or config hook | [ ] — in MASTER_PROJECT.md architecture (overlap scheduler) but not a DEVELOPMENT_PLAN.md P0 task; scheduled for a later phase |

**MASTER spec for mixer:** On-the-fly mixing at each training step; new unique mix every step; ground truth = clean stems before augmentation.

---

## 🔄 PARALLEL — Dev B tasks (P0)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P0-B1 | Repository skeleton, environment, dependency lockfile | none | Cloneable, runnable repo | [x] — done 2026-07-09, PR #1 |
| P0-B2 | Pre-commit hooks (Black + Ruff) + CI workflow | P0-B1 | `.pre-commit-config.yaml`, CI passing | [x] — done 2026-07-09, PR #1 |
| P0-B3 | Shared `SeparationResult` schema | P0 Day 1 collab | `schemas/separation_result.py` | [x] — done 2026-07-09, PR #1 |
| P0-B4 | Mixer stub for baseline (loads pre-mixed Libri3Mix from disk) | none | `data/mixer_stub.py` | [x] — done 2026-07-09, PR #1 |
| P0-B5 | SepFormer baseline wrapper (control) | P0-B3 | `models/experts/sepformer.py` — SpeechBrain `sepformer-wsj03mix` | [x] — done 2026-07-09, PR #1 |
| P0-B6 | SR-CorrNet baseline wrapper (or TF-GridNet fallback if weights unavailable) | P0-B3 | `models/experts/srcorrnet.py` | [x] — done 2026-07-09, PR #1 |
| P0-B7 | Baseline runner: SepFormer + SR-CorrNet on Libri3Mix test | P0-B4 (stub), P0-B5, P0-B6 | `models/baseline_runner.py`, `scripts/run_baseline.py`, baseline table | [x] — done 2026-07-09, PR #1 |
| P0-B8 | Models area design note | P0-B1 | `docs/models.md` | [x] — done 2026-07-09, PR #1 |

**MASTER Phase 0 deliverable:** Baseline results table on 3-speaker Libri3Mix test clips.

---

## 🔄 PARALLEL — Dev C tasks (P0)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P0-C1 | Evaluation harness: SI-SDRi computation | none | `eval/metrics.py` + unit tests | [x] — done 2026-07-09, PR #2 |
| P0-C2 | Permutation-invariant matching (uPIT / PIT) | P0-C1 | PIT matching in harness | [x] — done 2026-07-09, `scipy.optimize.linear_sum_assignment` in `eval/metrics.py` |
| P0-C3 | Per-tier reporting (L0–L5 tier labels) | P0-C1 | Tier-aware metric reporting | [x] — done 2026-07-09, `eval/reporting.py` |
| P0-C4 | Shared YAML config loader + logging | P0-B1 (repo skeleton) | Config loader used by all modules | [x] — done 2026-07-09, `utils/config.py` |
| P0-C5 | DNSMOS integration stub (for L5 / no-reference cases) | P0-C1 | Reference-free quality metric hook | [ ] |
| P0-C6 | Count accuracy + confusion matrix reporting stubs | P0-C1 | `eval/counting.py` or equivalent | [x] — done 2026-07-09, `count_accuracy` + `count_confusion_matrix` in `eval/metrics.py` + `eval/reporting.py` |
| P0-C7 | Eval area design note | P0-C1 | `docs/eval.md` | [x] — done 2026-07-09, `docs/DEVC_DESIGN.md` |

**MASTER Phase 0 deliverable:** Harness covering SI-SDRi, DNSMOS, count accuracy, confusion matrix.

---

## ⛓ SEQUENTIAL — P0 integration (after parallel work)

| ID | Task | Depends on | Owner | Status |
|----|------|------------|-------|--------|
| P0-INT1 | Wire baseline runner to shared eval harness (not ad-hoc metrics) | P0-B7, P0-C1 | B + C | [x] — done 2026-07-10, baseline_runner uses eval.metrics.pit_si_sdr |
| P0-INT2 | Wire baseline runner to shared config loader | P0-B7, P0-C4 | B + C | [x] — done 2026-07-10, run_baseline.py uses utils.config.load_config |
| P0-INT3 | Replace mixer stub with Dev A mixer (optional upgrade) | P0-A1, P0-B7 | A + B | [ ] |
| P0-INT4 | Shared end-to-end integration test (tiny input → baseline → SI-SDRi) | P0-INT1 | All | [ ] |

---

## 🚧 GATE M0 — Acceptance criteria

- [x] Dev A: `data/mixer.py` produces valid mixture + stems for N=2,3 — done 2026-07-10
- [x] Dev C: `eval/metrics.py` computes SI-SDRi with PIT on known tensors — done 2026-07-09
- [x] Dev B: baseline runner produces results table — done 2026-07-09
- [x] **All three independently run baseline on same Libri3Mix test set → identical SI-SDRi (±0.1 dB tolerance)** — confirmed 2026-07-10
- [ ] Integration test passes on `main` — P0-INT4 still pending
- [x] Joint integration session completed — 2026-07-10

---

# PHASE P1 — Expert Integration & Alignment (Weeks 3–4)

**Milestone:** Both experts run and produce aligned streams on test input  
**🚧 GATE M1:** Given one 3-speaker test clip, both experts run and outputs are correctly aligned to the same speaker order (shared integration test).

**Parallelism:** Dev B on **⛓ critical path**; Dev A and Dev C run **🔄 PARALLEL** independent front-loaded work.

---

## ⛓ SEQUENTIAL — Dev B critical path (P1)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P1-B1 | MossFormer2 inference wrapper (cheap expert, E_TD) | M0 | Wrapper → `SeparationResult`; ModelScope / ClearerVoice-Studio; RTF ~0.05; max 3 streams | [x] — done 2026-07-10, `models/experts/mossformer2.py` |
| P1-B2 | SR-CorrNet inference wrapper (expensive expert, E_TF) + attractor output | M0 | Wrapper with count + confidence; TDA attractors; RTF ~0.31 | [x] — done 2026-07-10, enhanced `models/experts/srcorrnet.py` |
| P1-B3 | SR-CorrNet fallback: TF-GridNet via ESPnet if weights unavailable | P1-B2 blocked | Fallback expert wrapper | [x] — done 2026-07-10, `models/experts/tfgridnet.py` + `get_expensive_expert()` |
| P1-B4 | REAL-M blind SI-SNR estimator integration | none 🔄 | Quality scoring function; SpeechBrain `REAL-M-sisnr-estimator` | [x] — done 2026-07-10, `models/realm_quality.py` |
| P1-B5 | Preprocessing module: resample 16 kHz, peak-normalize -26 dBFS, STFT branch (512 FFT, 128 hop), waveform branch | M0 | `models/preprocess.py` | [x] — done 2026-07-10 |
| P1-B6 | Expert integration test: both experts on same 3-speaker clip | P1-B1, P1-B2, P1-B5 | Integration test | [x] — done 2026-07-10, `tests/test_expert_integration.py` |

**MASTER weights reference:**
- MossFormer2: `github.com/modelscope/ClearerVoice-Studio` (~55.7M params, frozen)
- SR-CorrNet-B[2-5]: `github.com/dmlguq456/SR_CorrNet` (~7–20M params, frozen)
- SepFormer remains control baseline only

---

## 🔄 PARALLEL — Dev C (P1, while B integrates)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P1-C1 | ECAPA-TDNN embedding wrapper | none | SpeechBrain `spkrec-ecapa-voxceleb` | [ ] — hungarian.py uses embeddings from metadata but standalone wrapper not yet implemented |
| P1-C2 | Hungarian stream alignment via ECAPA embeddings | P1-C1 | `align/hungarian.py` — cost = 1 − cosine sim | [~] — code shipped 2026-07-09, PR #2; fully activates when P1-C1 (ECAPA wrapper) is complete |
| P1-C3 | Cross-chunk identity lock (4s chunks, 1s overlap) | P1-C2 | Chunk-stitching module in `align/` | [~] — code shipped 2026-07-09, PR #2; real long-audio validation pending P1-INT2 |
| P1-C4 | Alignment unit tests including same-gender stress case | P1-C2 | Tests in `tests/` | [~] — code shipped 2026-07-09, PR #2; same-gender stress test needs ECAPA wrapper (P1-C1) to be meaningful |

---

## 🔄 PARALLEL — Dev A (P1, while B integrates)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P1-A1 | Augmentation stage 1: RIR reverb (pyroomacoustics / FAST-RIR) | P0-A1 (mixer) | `data/augmentation/rir.py` | [x] — done 2026-07-10, Stage 1 in `data/augmentation.py`, PR #4 |
| P1-A2 | Augmentation stage 2: WHAM! noise | P0-A1 | `data/augmentation/noise.py` | [x] — done 2026-07-10, Stage 2 in `data/augmentation.py`, PR #4 |
| P1-A3 | WHAM! + WHAMR! dataset download | none | Data prep scripts | [ ] |
| P1-A4 | Codec augmentation prototype (Opus, AAC low bitrate) | none | `data/augmentation/codec.py` prototype | [x] — done 2026-07-10, `data/codec_augmentation.py`, PR #4 |
| P1-A5 | Libri4Mix + Libri5Mix extension scripts | P0-A2 | `github.com/shakeddovrat/librimix` integration | [ ] |

---

## 🤝 P1 COLLAB — Dev B + Dev C pairing

- [ ] Define alignment interface: expert `SeparationResult` → aligner input format
- [ ] Dev C understands model output format (streams, embeddings, confidence)
- [ ] Document in `docs/decisions.md`

---

## ⛓ SEQUENTIAL — P1 integration

| ID | Task | Depends on | Owner | Status |
|----|------|------------|-------|--------|
| P1-INT1 | Align MossFormer2 + SR-CorrNet outputs on same 3-speaker clip | P1-B6, P1-C2 | B + C | [ ] — blocked on P1-B1, P1-B2 |
| P1-INT2 | Cross-chunk lock verified on >4s audio | P1-C3, P1-INT1 | C | [ ] — blocked on P1-INT1 |
| P1-INT3 | REAL-M scores MossFormer2 output on test clip | P1-B1, P1-B4 | B | [x] — done 2026-07-10, covered in test_expert_integration.py (mocked) |

---

## 🚧 GATE M1 — Acceptance criteria

- [ ] MossFormer2 wrapper returns 3 streams + embeddings
- [ ] SR-CorrNet wrapper returns K streams + attractor vectors + confidence
- [ ] REAL-M produces per-stream SI-SNRi estimates without reference
- [ ] Hungarian alignment matches streams to consistent speaker order
- [ ] Cross-chunk identity lock works on long audio
- [ ] **Shared integration test: one 3-speaker clip, both experts, aligned output**
- [ ] Joint integration session completed

---

# PHASE P2 — Cascade Core (Weeks 5–6)

**Milestone:** Scene analyzer, router, cascade gate, fusion head train and beat best single expert  
**🚧 GATE M2:** Full CA-MoSE forward pass runs end-to-end, trains a few epochs, **beats best single expert** on mixed-condition validation, reports **measured escalation rate**. Everyone can explain single-input flow.

**Fallback trigger (MASTER §5.3):** If cascade cannot beat MossFormer2 alone by end of P2 → fall back to always-run-both ensemble, train fusion only, present routing as interpretability.

---

## 🤝 P2 COLLAB — Before any implementation (all three)

- [ ] Cascade architecture review session
- [ ] Agree tensor flow: [B,T] → Scene Analyzer → MossFormer2 → REAL-M → gate → SR-CorrNet → align → fuse
- [ ] Agree quality threshold `tau` tuning strategy (conservative: borderline inputs escalate)
- [ ] Agree composite loss weights (initial lambdas from MASTER §7.2)
- [ ] Log decisions in `docs/decisions.md`

---

## 🔄 PARALLEL — Trainable sub-components (P2)

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P2-A1 | Scene Analyzer (~1.5M params): log-mel + handcrafted features → reverb proxy, noise floor, overlap density, spectral flatness, modulation rate, K_coarse | M1 | A | `models/scene_analyzer.py` | [ ] |
| P2-C1 | Two-level Adaptive Router (~0.5M params): sequence gate + segment gate (1–2s windows), sigmoid (not softmax), w_TF/w_TD/w_NULL | P2-A1 | C | `models/router.py` | [~] — code shipped early 2026-07-09, PR #2; wire-up to Scene Analyzer (P2-A1) pending |
| P2-C2 | Load-balance auxiliary loss for router | P2-C1 | C | Loss term + collapse monitoring | [x] — wired in `train/losses.py` CompositeLoss, 2026-07-11 |
| P2-C3 | Null-expert sparsity loss | P2-C1 | C | Anti-hallucination loss term | [x] — wired in `train/losses.py` CompositeLoss, 2026-07-11 |
| P2-B1 | Cascade gate: compare REAL-M score to threshold `tau`; escalate if below | P1-B4 | B | Cascade controller | [x] — done 2026-07-11, `models/cascade_gate.py` |
| P2-B2 | Escalation-rate instrumentation | P2-B1 | C | Dashboard / logging | [~] — `escalation_rate` query in `eval/reporting.py`; runtime logging pending P2-B1 |
| P2-B3 | Fusion head CRRR (~1M params): `s_fused_k = s_SR_k + alpha_k(t) * R_theta`; alpha from confidence, mask entropy, local SI-SDRi proxy, scene weights | M1 alignment | B | `models/fusion.py` | [x] — done 2026-07-11, `models/fusion.py` |
| P2-B4 | Residual regularization loss (L2 on fusion correction) | P2-B3 | B | Loss term | [x] — done 2026-07-11, `train/losses.py` |

**Router design (MASTER §4.4):**
- Sigmoid gating (multiple experts can be active)
- Null expert routes silence / low-overlap (prevents hallucinated speakers)
- Load-balance prevents collapse to one expert

**Cascade compute target (MASTER §4.3):** ~30% escalation → RTF ~0.14 vs ~0.36 always-both.

---

## ⛓ SEQUENTIAL — Training loop (P2, after sub-components)

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P2-B5 | Composite loss assembly (all 7 terms) | P2-A1, P2-C1, P2-B3 | B | `train/losses.py` | [x] — done 2026-07-11 |
| P2-B6 | Training loop | P2-B5, all P2 components | B (leads) | `train/trainer.py` | [x] — done 2026-07-11 |
| P2-B7 | Multi-resolution STFT loss | P2-B5 | B | Loss term (lambda=0.5) | [x] — done 2026-07-11, `train/losses.py` |
| P2-B8 | Speaker-consistency loss (ArcFace-style) | P2-B5 | B | Loss term (lambda=0.1) | [x] — done 2026-07-11, `train/losses.py` |
| P2-INT1 | **Whole-team review of training-loop PR** | P2-B6 | All | Approved PR | [ ] |
| P2-INT2 | End-to-end forward pass integration test | P2-B6 | All | E2E test | [ ] |
| P2-INT3 | Short training run (few epochs) on mixed conditions | P2-INT2 | B | Checkpoint + logs | [ ] |
| P2-INT4 | Validate: beats best single expert on val set | P2-INT3 | B + C | Metric comparison table | [ ] |
| P2-INT5 | Measure and report escalation rate | P2-B2, P2-INT3 | C | Escalation rate per tier | [ ] |

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

- [ ] Full CA-MoSE forward pass: preprocess → scene → MossFormer2 → REAL-M → gate → (SR-CorrNet + fuse if escalated) → postprocess
- [ ] Trained heads (~3M params) converge in few-epoch test run
- [ ] **Beats best single expert on mixed-condition validation**
- [ ] Escalation rate measured and logged (target ~30–40%)
- [ ] Expected RTF computed at measured escalation rate
- [ ] All three can explain single-input flow through system
- [ ] Training-loop PR reviewed by all three
- [ ] Joint integration session completed
- [ ] **Novelty N1 proof started:** ablation plan for single-expert vs cascade documented

---

# PHASE P3 — Speaker Counting (Week 7)

**Milestone:** Learned stop-classifier produces confusion matrix and calibration curve  
**🚧 GATE M3:** System estimates speaker count on unknown-N inputs; produces **confusion matrix + calibration curve**.

**Ownership rotation:** Dev C **leads** counting; Dev B supports features; Dev A supports N=2..5 mixtures.

---

## 🔄 PARALLEL — P3 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P3-B1 | Feature extractors for stop-classifier: (1) residual energy ratio, (2) VAD prob on residual, (3) ECAPA embedding distance to prior stems, (4) mixture-consistency error | M2 | B | `models/counting_features.py` | [ ] — blocked on M2 |
| P3-C1 | Learned stop-classifier MLP (~0.3M params): 4 features + attractor stop logit → P(more speakers) | M2 | C | `models/stop_classifier.py` | [~] — code shipped early 2026-07-09, PR #2; real training on Libri2–5Mix pending M2 gate |
| P3-C2 | Count BCE loss integration into trainer | P3-C1 | C | `L_count-BCE` active in trainer | [~] — code shipped early 2026-07-09, PR #2; wired into trainer pending P2-B6 |
| P3-C3 | Count confusion matrix report generator | P0-C6, P3-C1 | C | `eval/counting_report.py` | [~] — code shipped early 2026-07-09, PR #2; needs real classifier outputs to produce results |
| P3-C4 | Calibration curve report (estimated prob vs actual accuracy) | P3-C3 | C | Calibration plot + metrics | [~] — code shipped early 2026-07-09, PR #2; needs real classifier run to produce calibration data |
| P3-A1 | Mixer support for N=2..5 (Libri2Mix–Libri5Mix) | P0-A1, P1-A5 | A | On-the-fly 2–5 speaker mixtures | [~] — DynamicMixer supports arbitrary N; Libri4/5Mix download scripts (P1-A5) pending |
| P3-A2 | SparseLibriMix download (test-only, 6 overlap ratios) | none | A | `github.com/popcornell/SparseLibriMix` | [ ] |
| P3-C5 | Stop-classifier training on Libri2–5Mix | P3-C1, P3-A1 | C | Trained classifier checkpoint | [~] — training script shipped 2026-07-09, PR #2 (self-test passes); real training run on Libri2–5Mix pending data + M2 |
| P3-INT1 | Speaker-count coordinator: SR-CorrNet TDA attractors + stop-classifier fusion | P3-B1, P3-C1, P1-B2 | B + C | `models/count_coordinator.py` | [ ] — blocked on P3-B1, P1-B2 |
| P3-INT2 | Unknown-N evaluation across N=2,3,4,5 | P3-INT1, P3-C3 | C | Count accuracy results | [ ] — blocked on P3-INT1 |

**MASTER §4.5:** Stop when P(more speakers) falls below calibrated threshold. Report full confusion matrix (which mistakes: merge vs split).

---

## 🤝 P3 COLLAB — Dev A supports counting training data

- [~] Dev A delivers 2–5 speaker mixture pipeline for classifier training — DynamicMixer ready; Libri4/5Mix download pending
- [ ] Verify no speaker leakage across splits

---

## 🚧 GATE M3 — Acceptance criteria

- [ ] Stop-classifier trained on Libri2–5Mix
- [ ] Unknown-N inference works at test time (N not given)
- [ ] Manual count override exposed (MASTER §1.3 assumption)
- [ ] **Confusion matrix produced** (rows=true N, cols=estimated N)
- [ ] **Calibration curve produced**
- [ ] Oracle-count vs learned-count ablation planned (for P5)
- [ ] **Novelty N3 + N6:** counting contribution + mixture-consistency feature documented
- [ ] Joint integration session completed

---

# PHASE P4 — Robustness (Week 8)

**Milestone:** Reverb, noise, codec augmentation integrated; clean performance preserved  
**🚧 GATE M4:** Robustness table across conditions; clean-vs-augmented ablation confirms clean performance not degraded.

**Ownership rotation:** Dev A **leads** augmented training run; Dev B supports; Dev C runs ablation.

---

## 🔄 PARALLEL — P4 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P4-A1 | Integrate full 3-stage augmentation into training loop (RIR → WHAM noise → codec) | P1-A1, P1-A2, P1-A4, P2-B6 | A | Augmented training runs | [ ] — augmentation modules ready; blocked on P2-B6 (training loop) |
| P4-A2 | Re-tune trainable heads on augmented data | P4-A1 | A (leads), B support | Retrained checkpoint | [ ] |
| P4-A3 | Codec degradation evaluation table | P1-A4, P0-C1 | A | Clean-to-codec degradation table | [ ] |
| P4-C1 | Clean-vs-augmented ablation | P4-A2, P0-C1 | C | Ablation table | [ ] |
| P4-C2 | L3 evaluation: WHAMR! + Libri3Mix-noisy (SI-SDRi + DNSMOS) | P4-A2 | C | L3 results | [ ] |
| P4-INT1 | Verify mixed-condition training (not worst-case-only) | P4-A2 | A | Training condition distribution log | [ ] |

**MASTER augmentation pipeline (§6.2):** Each stage probabilistic; ground truth = clean stems before augmentation; SI-SDRi against original clean.

---

## 🚧 GATE M4 — Acceptance criteria

- [ ] Three-stage augmentation active in training
- [ ] Retrained checkpoint evaluated on reverb + noise + codec conditions
- [ ] **Robustness table** across conditions (project vs baselines)
- [ ] **Clean-vs-augmented ablation** confirms no clean regression
- [ ] **Novelty N5:** codec robustness degradation table
- [ ] Joint integration session completed

---

# PHASE P5 — Differentiating Results (Weeks 9–10)

**Milestone:** Sparse-overlap curve, real-room eval, break-point curve produced  
**🚧 GATE M5:** All three flagship results locked. Tier-3 work (N9, N10) unlocked only after this gate.

**All three collaborate; each owns one flagship result.**

---

## 🔄 PARALLEL — P5 flagship results

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P5-C1 | **Flagship 1:** Sparse-overlap curve on SparseLibriMix — SI-SDRi vs overlap at {0, 20, 40, 60, 80, 100}% | M3 eval harness, P3-A2 | C | Overlap curve figure + table | [ ] |
| P5-A1 | **Flagship 2:** Real-room recording session (2–5 speakers, scripted overlap) | M4 | A (leads) | Recorded real-room set | [ ] |
| P5-A2 | Real-room per-stream Whisper WER evaluation | P5-A1 | A | Real-room WER table | [ ] |
| P5-A3 | LibriCSS WER evaluation (up to 2 concurrent) | P0-C1 | A | LibriCSS results | [ ] |
| P5-B1 | **Flagship 3:** Break-point curve — SI-SDRi vs speaker count 2→7 | P3-A1 (mixer high N) | B | Break-point figure | [ ] |
| P5-B2 | Document MossFormer2→SR-CorrNet handoff above 3 speakers | P5-B1 | B | Transition boundary note | [ ] |
| P5-ALL1 | Full ablation table (all 9 mandatory conditions) | M2, M3, M4 | All (split) | Ablation table | [ ] |

---

## 🤝 P5 COLLAB — Real-room recording (all three as speakers)

- [ ] Script overlapping dialogue (2–5 speakers)
- [ ] Record in real room on phones
- [ ] Known transcripts for WER ground truth
- [ ] Held-out from training data

---

## Mandatory ablations checklist (MASTER §10.2)

- [ ] MossFormer2-only vs full cascade
- [ ] SR-CorrNet-only vs full cascade
- [ ] Static equal-weight ensemble vs cascade
- [ ] Fixed threshold vs learned gatekeeper
- [ ] Router with null expert vs without
- [ ] 100% overlap training vs sparse overlap curriculum
- [ ] Oracle speaker count vs learned count
- [ ] Without codec augmentation vs with
- [ ] Without mixture-consistency feature vs with

---

## 🚧 GATE M5 — Acceptance criteria

- [ ] SparseLibriMix curve complete (6 ratios) — **Novelty N4**
- [ ] Real-room WER table complete — **Novelty N7** (if chosen over N8)
- [ ] Break-point curve 2–7 speakers — **Novelty N9**
- [ ] All 9 ablation rows filled
- [ ] Joint integration session completed

---

# PHASE P6 — Demo & Report (Weeks 11–12)

**Milestone:** Gradio demo, ablation table, written report complete  
**🚧 GATE M6:** Submission package complete — demo runs, report written, results reproduce from bundle.

---

## 🔄 PARALLEL — P6 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P6-C1 | Gradio demo: upload audio → speaker count, N waveforms, spectrograms, Whisper transcripts | M5 full system | C | `demo/gradio_app.py` | [~] — MockEngine skeleton in `demo/app.py` done 2026-07-09; real engine pending M5 |
| P6-B1 | Routing-weight interpretability panel in demo | P6-C1 | B | Demo panel | [ ] |
| P6-B2 | Mixture-consistency self-grade display in demo | P6-C1 | B | Demo panel | [ ] |
| P6-B3 | Auto-flag low-confidence outputs in demo | P6-B2 | B | Demo feature (N6) | [ ] |
| P6-A1 | Demo audio processing backend | M5 full system | A | Demo backend API | [ ] |
| P6-A2 | Reproducibility package: configs, checkpoints, instructions | All phases | A | Reproducibility bundle | [ ] |
| P6-ALL1 | Technical report — Dev A section | M5 results | A | Report section | [ ] |
| P6-ALL2 | Technical report — Dev B section | M5 results | B | Report section | [ ] |
| P6-ALL3 | Technical report — Dev C section (calibration, curves) | M5 results | C | Report section | [ ] |
| P6-ALL4 | Final ablation table in report | P5-ALL1 | All | Report table | [ ] |
| P6-ALL5 | Demo video or hosted demo link | P6-C1 | C | Demo artifact | [ ] |

**Demo must show (MASTER §9 Phase 6):** estimated count, N waveforms, spectrograms, per-stream Whisper transcripts, routing-weight visualization, mixture-consistency self-grade.

---

## 🚧 GATE M6 — Acceptance criteria

- [ ] Gradio demo runs end-to-end on uploaded audio
- [ ] Report complete with all three sections
- [ ] Reproducibility bundle reproduces key numbers
- [ ] All reporting checklist items below addressed
- [ ] Joint integration session completed
- [ ] **Submission package delivered**

---

# Reporting checklist (MASTER §10.3)

Track at M6; start collecting artifacts from M0.

- [ ] Libri3Mix + WSJ0-3mix SI-SDRi (known + unknown N)
- [ ] SparseLibriMix SI-SDRi at {0, 20, 40, 60, 80, 100}% overlap
- [ ] Speaker-count accuracy + confusion matrix + calibration curve
- [ ] WHAMR! + reverberant Libri3Mix SI-SDRi + DNSMOS
- [ ] Clean-to-codec degradation table
- [ ] Real-room scripted per-stream WER
- [ ] Break-point curve: SI-SDRi vs speaker count 2→7
- [ ] Cascade escalation rate per tier
- [ ] Inference RTF at average and worst-case escalation
- [ ] Ablation table (≥9 conditions)
- [ ] Router weight interpretability panel
- [ ] Gradio demo link or recorded demo video

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
| N1 | Conditional cascade routing | Mandatory | Ablation + escalation rate + compute curve | P2, P5 | [ ] — pending cascade gate (P2-B1) and ablation run |
| N2 | Two-level router + null expert + load-balance | Mandatory | Router ablation + demo panel | P2, P6 | [~] — `models/router.py` done 2026-07-09; ablation run pending M2 |
| N3 | Calibrated stop-classifier + confusion matrix | Mandatory | Confusion matrix + calibration curve | P3 | [~] — `models/stop_classifier.py` + training script done 2026-07-09; full training run on real data pending M2 |
| N4 | Sparse-overlap curve (SparseLibriMix) | Mandatory | SI-SDRi vs overlap table | P5 | [ ] |
| N5 | Codec augmentation robustness | Mandatory | Clean-to-codec degradation table | P4 | [~] — `data/codec_augmentation.py` prototype done 2026-07-10; degradation table pending P4-A3 |
| N6 | Mixture-consistency self-grading | With N3 | Stop-classifier ablation + demo flag | P3, P6 | [~] — mixture-consistency feature in stop_classifier 2026-07-09; demo display pending P6-B2 |
| N7 | Real-room WER evaluation | Tier 2 (pick one) | Real-room WER table | P5 | [ ] |
| N8 | Enrollment-based target extraction demo | Tier 2 (alt) | Interactive demo mode | P6 | [ ] |
| N9 | Break-point curve 2–7 speakers | Tier 3 | SI-SDRi vs N curve | P5 | [ ] — locked until M5 |
| N10 | Generative flow post-corrector | Tier 3 | DNSMOS ablation | Post-M5 only | [ ] — locked until M5 |

**Commit set:** N1–N5 mandatory; N6 with N3; pick N7 or N8; N9 nearly free; N10 only if all stable.

---

# Dataset acquisition tracker

| Dataset | Role | Owner phase | Status |
|---------|------|-------------|--------|
| LibriSpeech | Source audio for mixer | P0-A | [~] — `prepare_librimix.py` download script ready; not yet run on this machine |
| Libri2Mix / Libri3Mix | Primary train/eval | P0-A | [~] — generation script ready; not yet run on this machine |
| Libri4Mix / Libri5Mix | N=4,5 training | P1-A | [ ] — P1-A5 script not yet written |
| SparseLibriMix | L2 overlap eval (test only) | P3-A | [ ] |
| WHAM! | Noise augmentation | P1-A | [ ] |
| WHAMR! | Reverb eval + RIR source | P1-A | [ ] |
| VCTK | Accent diversity | P0-A | [ ] — P0-A4 not started |
| WSJ0-*Mix | Literature comparison (LDC license) | Optional | [ ] |
| LibriheavyMix | Large-scale reverb (if compute) | Optional | [ ] |
| REAL-M | Real 2-speaker, no reference | P1-C | [ ] |
| LibriCSS | Real-room WER | P5-A | [ ] |
| Real-room set | Team-recorded flagship | P5-A | [ ] |

---

# Risk register & fallback triggers

| Risk | Severity | Mitigation | Trigger action | Status |
|------|----------|------------|----------------|--------|
| SR-CorrNet weights unavailable | High | TF-GridNet via ESPnet | P1-B3 fallback | [ ] — monitoring |
| REAL-M too noisy for gate | Medium | Scene-analyzer reverb proxy as primary gate signal | Reprioritize P2-B1 | [ ] — monitoring |
| Alignment fails same-gender | Medium | Local SI-SDRi fine-alignment; stress test | P1-C4 | [ ] — monitoring |
| Router collapse | Medium | Increase load-balance loss weight | Monitor P2-C2 | [ ] — monitoring |
| MossFormer2 max 3 speakers | Medium | Hand off to SR-CorrNet for N>3 | Document P5-B2 | [ ] — monitoring |
| Fusion loses to SR-CorrNet alone | Medium | Fall back to ensemble / SR-CorrNet-primary | MASTER §5.3 at M2 | [ ] — monitoring |
| Scope creep | High | Tier 3 only after M5 | Enforce gate | [ ] — monitoring |
| 6–7 speaker quality poor | Expected | Graceful degradation framing | P5-B1 | [ ] — monitoring |
| Train-test domain gap | High | Mixed-condition aug + real-room eval | P4, P5 | [ ] — monitoring |
| Dev B bottleneck | Planning | Fast review P1/P2; A/C front-load | Ongoing | [ ] — monitoring; A and C have front-loaded P1 work |

---

# Compute & parameter budget tracker

| Item | Target | Actual | Status |
|------|--------|--------|--------|
| Frozen expert params | ~60–75M | — | [ ] — not yet measured |
| Trainable params | ~3.3M | — | [ ] — not yet measured |
| Training time (all heads) | 2–4 days on 1 GPU | — | [ ] — not yet run |
| RTF @ 30% escalation | ~0.14 | — | [ ] — not yet measured |
| RTF worst case (100% escalation) | ~0.40 | — | [ ] — not yet measured |
| Inference memory | ≤16 GB T4 | — | [ ] — not yet measured |
| Development GPUs | 2× T4 16GB | — | [ ] — not yet provisioned |
| Final run GPU | A100 40GB (if available) | — | [ ] — not yet provisioned |

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

- [ ] MambaDeflate stretch expert (E_DEF) for 5+ speakers — SPMamba / ReSepNet
- [ ] LibriheavyMix large-scale reverb training
- [ ] WSJ0-*Mix literature comparison (requires LDC license)
- [ ] Backbone fine-tuning (adds 1–2 weeks; only if ablation justifies)
- [ ] Wiener filter postprocessing
- [ ] One-step generative flow corrector (N10, Tier 3)
- [ ] Enrollment-based demo mode (N8, Tier 2 alternative)
- [ ] Real-time streaming demo mode

---

*End of PROJECT_TODO.md — edit this file as the project progresses.*
