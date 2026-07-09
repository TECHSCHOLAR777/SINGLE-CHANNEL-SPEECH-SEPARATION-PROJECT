# CA-MoSE: Condition-Aware Mixture-of-Separation-Experts

Multi-speaker blind speech separation for three or more concurrent speakers using conditional cascade routing between MossFormer2 and SR-CorrNet.

See [MASTER_PROJECT.md](MASTER_PROJECT.md) for architecture and [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for team workflow.

## Quick start

```bash
# Create environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"
pre-commit install

# Run unit tests (no GPU or pretrained weights required)
pytest tests/ -v

# Run Phase 0 baseline (requires GPU + downloaded Libri3Mix data)
python scripts/run_baseline.py --config configs/baseline.yaml
```

## Repository layout

| Directory | Owner | Purpose |
|-----------|-------|---------|
| `data/` | Dev A | Mixer, augmentation, dataset prep |
| `models/` | Dev B | Experts, router, fusion, cascade |
| `train/` | Dev B | Training loops |
| `eval/` | Dev C | Metrics and evaluation harness |
| `align/` | Dev C | Stream alignment |
| `demo/` | Dev C | Gradio demo |
| `schemas/` | Shared | Interface contracts (e.g. `SeparationResult`) |
| `configs/` | Shared | YAML configuration files |
| `tests/` | Shared | Unit and integration tests |
| `docs/` | Shared | Design notes and decision log |

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
- [x] Unit tests for every data and metric function — 173 tests passing as of 2026-07-10
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
| P2-C2 | Load-balance auxiliary loss for router | P2-C1 | C | Loss term + collapse monitoring | [~] — code shipped early 2026-07-09, PR #2; active in training loop pending P2-B5 |
| P2-C3 | Null-expert sparsity loss | P2-C1 | C | Anti-hallucination loss term | [~] — code shipped early 2026-07-09, PR #2; active in training loop pending P2-B5 |
| P2-B1 | Cascade gate: compare REAL-M score to threshold `tau`; escalate if below | P1-B4 | B | Cascade controller | [ ] — blocked on P1-B4 (REAL-M) |
| P2-B2 | Escalation-rate instrumentation | P2-B1 | C | Dashboard / logging | [~] — `escalation_rate` query in `eval/reporting.py`; runtime logging pending P2-B1 |
| P2-B3 | Fusion head CRRR (~1M params): `s_fused_k = s_SR_k + alpha_k(t) * R_theta`; alpha from confidence, mask entropy, local SI-SDRi proxy, scene weights | M1 alignment | B | `models/fusion.py` | [ ] — blocked on M1 |
| P2-B4 | Residual regularization loss (L2 on fusion correction) | P2-B3 | B | Loss term | [ ] — blocked on P2-B3 |

**Router design (MASTER §4.4):**
- Sigmoid gating (multiple experts can be active)
- Null expert routes silence / low-overlap (prevents hallucinated speakers)
- Load-balance prevents collapse to one expert

**Cascade compute target (MASTER §4.3):** ~30% escalation → RTF ~0.14 vs ~0.36 always-both.

---

## ⛓ SEQUENTIAL — Training loop (P2, after sub-components)

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P2-B5 | Composite loss assembly (all 7 terms) | P2-A1, P2-C1, P2-B3 | B | `train/losses.py` | [ ] |
| P2-B6 | Training loop | P2-B5, all P2 components | B (leads) | `train/trainer.py` | [ ] |
| P2-B7 | Multi-resolution STFT loss | P2-B5 | B | Loss term (lambda=0.5) | [ ] |
| P2-B8 | Speaker-consistency loss (ArcFace-style) | P2-B5 | B | Loss term (lambda=0.1) | [ ] |
| P2-INT1 | **Whole-team review of training-loop PR** | P2-B6 | All | Approved PR | [ ] |
| P2-INT2 | End-to-end forward pass integration test | P2-B6 | All | E2E test | [ ] |
| P2-INT3 | Short training run (few epochs) on mixed conditions | P2-INT2 | B | Checkpoint + logs | [ ] |
| P2-INT4 | Validate: beats best single expert on val set | P2-INT3 | B + C | Metric comparison table | [ ] |
| P2-INT5 | Measure and report escalation rate | P2-B2, P2-INT3 | C | Escalation rate per tier | [ ] |

**Composite loss (MASTER §7.2):**
```
L_total = L_SI-SDR-uPIT (1.0)
        + 0.5 * L_multi-res-STFT
        + 0.3 * L_count-BCE        [placeholder until P3]
        + 0.1 * L_load-balance
        + 0.1 * L_null-sparsity
        + 0.1 * L_residual-reg
        + 0.1 * L_speaker-consistency
```

Set `data_root` in `configs/baseline.yaml` to your Libri3Mix test directory before running.
