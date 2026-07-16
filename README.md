# CALM-Sep Project TODO — Single Source of Truth

> **Derived from:** `BLUEPRINT` (Master Project Blueprint, CALM-Sep) — the sole authority.
> **Purpose:** Living task tracker for the full project. Edit checkboxes as work completes.
> **Supersedes:** the CA-MoSE cascade tracker (MossFormer2 → REAL-M → SR-CorrNet). That architecture is retired. See **§ Migration from CA-MoSE** below for what carried over and what was deleted.
> **Last updated:** 2026-07-17 — tracker re-baselined onto CALM-Sep. Reusable foundation (repo infra, eval metrics, alignment/stitching, data-prep, augmentation, dynamic mixer) carried forward as done; all CALM-Sep-specific components (LoRA library, condition analyzer, gate, attractor counting, band recovery, calibration) reset to not-started.

---

## 📊 Project pulse

> Snapshot **2026-07-17** — refresh the counts whenever you flip a status.

![Done](https://img.shields.io/badge/✅_done-31-brightgreen?style=for-the-badge)
&nbsp;
![In progress](https://img.shields.io/badge/🚧_in_progress-2-yellow?style=for-the-badge)
&nbsp;
![Not done yet](https://img.shields.io/badge/❌_not_done_yet-74-red?style=for-the-badge)

**Overall** `████████░░░░░░░░░░░░░░░░░░░░░` **29%** &nbsp;·&nbsp; 31 done &nbsp;·&nbsp; 2 in flight &nbsp;·&nbsp; 74 to go &nbsp;·&nbsp; **107 tasks** — architecture pivot from cascade to frozen-backbone + LoRA mixture. "Done" now counts only work that is valid **under CALM-Sep** (reusable infra/data/eval/alignment). Cascade-only deliverables were removed from the count, not carried as done.

**Milestones** &nbsp;
![M0](https://img.shields.io/badge/M0-🚧_in_progress-yellow?style=flat-square)
![M1](https://img.shields.io/badge/M1-🔒_locked-lightgrey?style=flat-square)
![M1b](https://img.shields.io/badge/M1b-🔒_locked-lightgrey?style=flat-square)
![M2](https://img.shields.io/badge/M2-🔒_locked-lightgrey?style=flat-square)
![M3](https://img.shields.io/badge/M3-🔒_locked-lightgrey?style=flat-square)
![M4](https://img.shields.io/badge/M4-🔒_locked-lightgrey?style=flat-square)
![M5](https://img.shields.io/badge/M5-🔒_locked-lightgrey?style=flat-square)

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
3. Add dates and notes inline when tasks complete: `[x] Task name — done 2026-07-20, PR #NN`
4. **Fixed constraints are never revisited** (see north star). A task that reopens one is out of scope.

---

## Project north star

**System:** CALM-Sep — Condition-Aware LoRA Mixture for Multi-Speaker Speech Separation
**Task:** Blind single-channel separation of **N ∈ {2,3,4,5}** simultaneous speakers with **unknown N** at test time, under reverb / noise / codec degradation
**Core strategy:** One **frozen** SR-CorrNet var-2-5 backbone; three small **LoRA adapters** (reverb, noise, codec) blended in weight space by a supervised **two-level condition analyzer** + **gate**; speaker count read from the backbone's **attractor probabilities `p_k`**; **band recovery** head extends 8 kHz output to 16 kHz; residual-energy **completeness** detector guards missed speakers.
**Trainable budget:** ~3–4M new parameters (condition analyzer, gate MLP, counting fusion, calibration heads, 3 LoRA adapters) against a **13.6M frozen** base.
**Fixed constraints (never revisited):**
- Base checkpoint `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` — downloaded once, frozen forever, never fine-tuned.
- Speaker count N = 2–5 only.
- Sample rate **8 kHz** internal (128/64 STFT), locked by the checkpoint.
- Quality path = band recovery (Option B) only. No backbone retraining.

**Hardware:** Kaggle T4 / Colab T4 (free tier). Largest single run ≈ one adapter, 20–40 GPU-hours. Total < 150 GPU-hours.
**Published base performance (clean WSJ0-mix):** count 100/99.7/97.7/96.9% and 24.8/24.4/21.9/19.9 dB SI-SDRi at N=2/3/4/5.

---

## Team roles (ownership, not exclusivity)

| Dev | Primary vertical | Secondary | Folder ownership |
|-----|------------------|-----------|------------------|
| **A** | Data & synthesis: dynamic 8 kHz mixing, RIR bank, WHAM!/DNS-4 staging, codec transforms, fixed+hashed eval sets | Eval data hygiene | `data/` |
| **B** | Model core: SR-CorrNet wrapper patches, LoRA library, adapters (Stage 1/2/4), condition analyzer, gate, attractor counting | Confidence/completeness | `models/`, `train/` |
| **C** | Evaluation & pipeline: metrics matrix, stats, band-recovery guard, calibration, chunker/stitcher, demo | Robustness ablations | `eval/`, `pipeline/`, `calibration/`, `align/`, `demo/` |
| **All** | Configs, tests, docs, interface contracts | — | `configs/`, `tests/`, `docs/`, `schemas/` |

---

## Critical path (longest dependency chain)

```
P0 Verify checkpoint + expose p_k/E(0) (B) ──► P1 Adapter library (B) ──► P1b Universal-adapter gate (B) ──► P2 Condition analyzer + gate (B/C) ──► P3 Joint polish + band recovery + calibration (B/C) ──► P4 Demo/CLI/efficiency ──► P5 Full eval + report
        ▲                                                              ▲
P0 Synthesis pipeline + hashed eval sets (A) ──────────────────────────┘ (feeds every training/eval stage)
```

**Protected slice:** The Phase 0 wrapper patches (expose `p_k`, `E(0)`, decoder-stage features) block **everything**. `attractor_test.py` is the gate; nothing downstream starts until it passes.

**Parallelism rule:**
- **P0:** Dev B (wrapper patches) and Dev A (synthesis + eval sets) run fully parallel. Zero GPU spent.
- **P1+:** Adapter training is sequential per adapter; Dev A/C front-load eval matrix and band-recovery tooling.

---

## Cross-cutting work (every phase)

### 🤝 Mandatory collaboration sessions

| When | Activity | Why |
|------|----------|-----|
| Day 1 (P0) | Repo structure, config schema, `SeparationResult` contract (add `p_k`, gate vector, completeness) | Bad contracts → integration pain |
| Start of P1 | LoRA attachment + adapter-training review (all three) | Weight-space composition is the whole thesis |
| Before P1b | Universal-adapter decision protocol (pre-commit the verdict rule) | The verdict is irreversible; agree it first |
| Start of P2 | Condition-analyzer / gate review (all three) | Circularity resolution must be understood |
| M0–M5 | Milestone integration session after each gate | Catch drift; run pipeline together |
| P4 | Real-recording session (all three as speakers) | Physically needs simultaneous voices |
| P5 | Report writing (each dev writes their section) | — |
| **Every week** | Weekly sync — blockers, rebalance | Surface slips early |

### Git workflow (all phases)

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] `main` always runnable and passing CI; **no direct commits to main** — CI workflow active; all merges via PR
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Every change via PR with **1 review from a non-owner**
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Model-core PRs (wrapper patches, LoRA, gate) → **review from all three** (they are the integration seam)

### Codebase standards (all phases)

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Formatter + linter (Black + Ruff) via pre-commit + CI — `.pre-commit-config.yaml` + `ci.yml`
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Type hints on all public function signatures
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] `SeparationResult` schema defined once in `schemas/` — never redefined ad hoc
- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] Extend `SeparationResult` with `p_k`, gate vector, completeness probability, OOD flag (BLUEPRINT §6.5)
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Every module: header docstring (purpose, inputs, outputs)
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] `docs/decisions.md` updated for every architecture choice (date + one-line reason)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Config-hash in every artifact** — SHA-256 of the producing config recorded on every checkpoint/result (BLUEPRINT §13)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Every mechanism ships with its off switch** — adapters, gate, residual sweep, band recovery each disabled by config; baselines are one-line runs

### Data split discipline (mandatory, all phases)

- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] **Three-way holdout** (BLUEPRINT §7.5): (1) speaker holdout — `dev-clean`/`test-clean` never in training (enforced in `DynamicMixer`); (2) condition-combination holdout — reverb+codec & noise+codec never in gate/joint training; (3) severity holdout — T60>0.9s and SNR<−4dB underrepresented in training. Only (1) is currently enforced.

---

## Migration from CA-MoSE (what changed)

The cascade design is retired. This table is the authority for the file audit.

| CA-MoSE concept | CALM-Sep replacement | Fate of old code |
|-----------------|----------------------|------------------|
| MossFormer2 cheap expert | (none — single frozen backbone) | **deleted** |
| SepFormer / TF-GridNet experts + fallback | (none — SR-CorrNet is sole foundation) | **deleted** |
| REAL-M blind gate | attractor `p_k` + residual energy | **deleted** |
| Cascade gate / escalation (tau) | continuous LoRA gate (no escalation) | **deleted** |
| CRRR fusion head | (none — one forward pass, one split) | **deleted** |
| Scene analyzer (log-mel → scene weights) | two-level condition analyzer (DSP + E(0)) | **deleted / re-designed** |
| Two-level router (w_TF/w_TD/w_NULL) | gate MLP scaling LoRA in weight space | **deleted / re-designed** |
| Stop-classifier + count coordinator (peel-off) | attractor readout + bounded residual sweep | **deleted; residual/VAD features salvaged** |
| Frozen-expert output cache | dynamic on-the-fly mixing (no pre-render) | **deleted** |
| SR-CorrNet 2-3spk @ resampled 16 kHz | SR-CorrNet **var-2-5** @ native 8 kHz + p_k/E(0) patches | **rewritten** |
| Eval metrics / alignment / data-prep / augmentation / mixer | unchanged in role | **reused** |

---

# 🏗️ PHASE P0 — Verify checkpoint & build synthesis pipeline (Weeks 1–2)

**Milestone:** Frozen checkpoint loads and exposes `p_k`/`E(0)`/decoder-stage features; 8 kHz synthesis pipeline produces labelled mixtures; all evaluation sets generated once, seeded, and hashed.
**🚧 GATE M0:** `attractor_test.py` passes (p_k varies with true count at N=2,3,4,5); eval sets hashed; frozen-base corpus-transfer SI-SDRi recorded. **Zero GPU hours.**

**Parallelism:** **🔄 FULL PARALLEL** — Dev B on wrapper patches, Dev A on synthesis + eval sets.

---

## 🤝 P0 Day 1 — COLLAB (all three)

- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Agree repository directory structure (BLUEPRINT §13 layout) — inherited; extend with `pipeline/`, `calibration/`, `models/srcorrnet/`, `models/lora.py`
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Agree config schema: `base_checkpoint.yaml` (locked path + SHA), `adapters/*.yaml`, `gate.yaml`, `band_recovery.yaml`, `eval.yaml`; **sample_rate = 8000**
- ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] Agree extended `SeparationResult` contract: streams `[K,T]` @16 kHz (band-recovered), `p_k`, per-stream confidence, completeness prob, per-chunk gate values, OOD flag
- ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] Log decisions in `docs/decisions.md`

## ⛓ SEQUENTIAL — Dev B: checkpoint & wrapper (P0, BLOCKING)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P0-B1 | Download & verify frozen checkpoint (`export.py --download --variant SS --config 1ch_WSJ_var_2_5spk.yaml`); confirm YAML constants (sr 8000, max_n_spks 5, N_Enc 2, N_Dec 4, d_model 128) | none | Verified `model.pt` + SHA in `base_checkpoint.yaml` | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P0-B2 | **Patch A** — expose `p_k` through `_single_pass_session` / `process_waveform` / `process_stft` | P0-B1 | `models/srcorrnet/` wrapper returns `pres["probs"]` `(1,7)` | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P0-B3 | **Patch B** — forward hook on `model.encoder` capturing `E(0)` `(1,T,65,128)` | P0-B1 | Pooled `E(0)` available to Level-2 analyzer | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P0-B4 | **Patch C** — hooks on each `dec_block[i]` capturing decoder-stage features for inter-stage consistency | P0-B1 | Stage features `(B,K,T,65,128)` | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P0-B5 | Expose `prob_thres` (attractor 0.5) as a configurable parameter (monkey-patch `spk_split.forward`) | P0-B2 | Configurable count threshold | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P0-B6 | **`attractor_test.py`** — assert `p_k` shape `(1,7)` and active-slot count == true N at N=2,3,4,5 | P0-B2 | BLOCKING test green | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P0-B7 | Corpus-transfer baseline: `process_waveform` on 20 LibriSpeech `dev-clean` 2-spk mixtures; record mean SI-SDRi (the floor every adapter must beat) | P0-B1, P0-A2 | Baseline number logged | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

## 🔄 PARALLEL — Dev A: synthesis & eval sets (P0)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P0-A1 | Dynamic on-the-fly mixer at **8 kHz**, N∈{2,3,4,5}, per-speaker level offsets, clean-stem ground truth | none | `data/mixer.py` (retarget 16k→8k) + tests | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — mixer reused from CA-MoSE; retarget sample rate to 8 kHz |
| P0-A2 | LibriSpeech source at 8 kHz (`train-clean-100`, `dev-clean`, `test-clean`), keep 16 kHz copies for band-recovery targets | none | Speaker pool + prep script | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — `data/prepare_librimix.py` reused; add 8 kHz downsample + 16 kHz retention |
| P0-A3 | RIR bank: 10k RIRs via `pyroomacoustics`, 1k per T60 interval 0.1s (0.2–1.0s), cached to `data/rirs/` | none | Cached RIR bank | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — pyroomacoustics reverb stage exists in `data/augmentation.py`; bank generation new |
| P0-A4 | Noise staging: WHAM! (~17 GB) + DNS-4 stratified 20 GB subset | none | `data/prepare_wham.py` + DNS-4 fetch | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — WHAM! prep reused; DNS-4 stratified fetch new |
| P0-A5 | Codec transforms: ffmpeg Opus 6–24k / AAC 16–48k / AMR-NB/WB on mixtures | none | `data/codec_augmentation.py` (reused) | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — reused from CA-MoSE; codec is a label-free transform |
| P0-A6 | **Fixed evaluation matrix** — generate once, seed, hash: clean 2–3, sparse overlap, primary reverb-noisy (N=2, n=500), high-count, real-RIR (BUT ReverbDB SLR17), codec, held-out combos, LibriCSS | P0-A1..A5 | `data/fixed_eval/` + manifests + hashes | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — prep scripts exist per-tier; unified hashed matrix new |
| P0-A7 | Reverb reference policy: **wet source**, truncated at `n_peak+512` (BLUEPRINT §7.6) | P0-A3 | Reference generator | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

## 🔄 PARALLEL — Dev C: eval & condition-input tooling (P0)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P0-C1 | Cardinality-aware SI-SDR/SI-SDRi + PIT (Hungarian; missed speaker = 0 dB; −1 dB per hallucinated stream) | none | `eval/metrics.py` (reused) | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — reused; verify cardinality penalty matches §9.2 |
| P0-C2 | Shared 8 kHz STFT preprocessing (window 128 / hop 64) + parallel 16 kHz mixture STFT for band recovery | none | `models/preprocess.py` (retarget) | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — preprocess reused; retarget STFT to 128/64 @8k + add 16k branch |
| P0-C3 | SileroVAD (8 kHz native) voiced-frame-density proxy; validate discriminative power on LibriCSS overlap subsets; fallback = voiced-energy fraction | none | Level-1 VAD feature + validation note | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P0-C4 | DNSMOS ONNX activation on 16 kHz band-recovered output (download `sig_bak_ovrl.onnx`) | none | `eval/dnsmos.py` live (was stub) | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — interface frozen; ONNX file pending |
| P0-C5 | Config loader + logging (reused) | none | `utils/config.py` | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — reused |

## 🚧 GATE M0 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] `attractor_test.py` passes: `p_k` shape `(1,7)`, active slots == true N at N=2,3,4,5
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Patches A/B/C load cleanly (`e0_hook_test.py` confirms `E(0)` shape `(1,T,65,128)`)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] All evaluation sets generated, seeded, hashed; manifests committed
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Frozen-base corpus-transfer SI-SDRi recorded (the adapter floor)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] SileroVAD proxy validated (or fallback selected)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Zero GPU hours spent** — inference only
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed

---

# 🧩 PHASE P1 — Adapter library (Stage 1)

**Milestone:** Three LoRA adapters (reverb, noise, codec) trained individually with co-activation warm-up; cross-interference matrix measured.
**🚧 GATE M1:** Each adapter shows statistically significant SI-SDRi gain on its matched condition (Wilcoxon p<0.05) and no degradation on clean Libri2Mix; off-diagonal cross-interference harm < 0.3 dB.

**Ownership:** Dev B leads (critical path); Dev A supplies matched-condition data; Dev C runs interference matrix.

## ⛓ SEQUENTIAL — Dev B critical path (P1)

| ID | Task | Depends on | Deliverable | Status |
|----|------|------------|-------------|--------|
| P1-B1 | `models/lora.py` — parallel-branch LoRA wrapper (`y = W0 x + Σ g·B(Ax)`), co-activation warm-up sampler, 17 target Linear layers per adapter (§5.3) | M0 | LoRA library + unit tests | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P1-B2 | Freeze-and-attach harness: register LoRA on targets, freeze all base params, only adapter params in optimizer (`strict=False` load order) | P1-B1 | Training scaffold | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P1-B3 | Reuse sr_corrnet engine losses: `PIT_SISNR_time` + `0.5·PIT_SISNR_mag` + `BCEWithLogitsLoss` on `pres["logits"]` (§8.2 / §15.6) | P1-B2 | Loss wiring | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P1-B4 | Train **`adapter_noise`** first (widest data; best LoRA plumbing debug) with co-activation warm-up U(0.0,0.2) | P1-B3, P1-A1 | Adapter weights + val curve | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P1-B5 | Train **`adapter_reverb`** (RIR/wet references, T60 0.2–1.0s) | P1-B4, P1-A2 | Adapter weights | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P1-B6 | Train **`adapter_codec`** (Opus/AAC/AMR) | P1-B5, P1-A3 | Adapter weights | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

## 🔄 PARALLEL — Dev A / Dev C (P1)

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P1-A1 | Noisy training mixtures (WHAM!+DNS-4, SNR −6..+10 dB) at 8 kHz, label-free | P0-A4 | A | Noise-condition data | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — noise augmentation stage reused |
| P1-A2 | Reverb training mixtures (RIR bank convolution, wet refs) | P0-A3, P0-A7 | A | Reverb-condition data | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — RIR reverb stage reused |
| P1-A3 | Codec training mixtures (ffmpeg transforms) | P0-A5 | A | Codec-condition data | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — codec transform reused |
| P1-C1 | **Cross-interference matrix**: each adapter alone on every condition; off-diagonal harm threshold 0.3 dB | P1-B4..B6 | C | Interference table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P1-C2 | O-LoRA orthogonality penalty (escalation only, if harm > 0.3 dB) | P1-C1 | C | Optional penalty term | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

## 🚧 GATE M1 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Each adapter significant on matched condition (Wilcoxon p<0.05)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] No degradation on clean Libri2Mix for any adapter
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Cross-interference matrix measured; off-diagonal harm < 0.3 dB (else O-LoRA applied)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed

---

# ⚖️ PHASE P1b — Universal-adapter calibration gate (Stage 2)

**Milestone:** One universal adapter (full library budget ~2.5M, trained on the union of all conditions) evaluated against the eventual routing target on the primary benchmark.
**🚧 GATE M1b:** Verdict **logged before the gate network is built**. If the universal adapter matches learned gating within 0.5 dB SI-SDRi on the primary benchmark and within CIs on degraded cells → adopt the simpler system and report it as the honest headline. Otherwise proceed to P2. **This decision is irreversible.**

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P1b-B1 | Train universal adapter on union of single-condition datasets | M1 | B | Universal adapter weights | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P1b-C1 | Evaluate on primary benchmark (reverb-noisy N=2) + ≥2 multi-condition cells; **commit verdict** | P1b-B1 | C | Verdict logged in `docs/decisions.md` | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

---

# 🎛️ PHASE P2 — Condition analyzer & gate (Stage 3)

**Milestone:** Two-level condition analyzer + gate MLP trained; learned gating beats the best single adapter on co-occurring conditions; never-worse-than-base holds on clean.
**🚧 GATE M2:** Learned gating > best single adapter on co-occurring cells; **Principle-2 smoke test** passes (full system ≥ frozen base − 0.1 dB on clean Libri2Mix); held-out combination cells do not collapse.

## 🔄 PARALLEL / ⛓ SEQUENTIAL — P2 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P2-B1 | `models/condition.py` **Level 1** — raw-STFT DSP features (SNR, codec family+bitrate, voiced density via SileroVAD); deterministic, no training | M1b, P0-C3 | B | Level-1 analyzer | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P2-B2 | `models/condition.py` **Level 2** — E(0) heads: reverberation strength (T60, attention-pooled CNN) + speaker-count prior (MLP) | P2-B1, P0-B3 | B | Level-2 analyzer | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P2-B3 | `models/gate.py` — gate MLP (2×256 GELU, sigmoid×1.5), per-layer gates, L1 sparsity 1e-3, EMA smoothing 0.7 | P2-B1 | B | Gate network | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P2-B4 | Supervised condition-head losses (L1 regressions + CE classifications) + separation loss through gates + gate sparsity | P2-B2, P2-B3 | B | Stage-3 training | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P2-C1 | Per-layer vs per-adapter gate ablation (simpler wins if no gain) | P2-B3 | C | Ablation result | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P2-C2 | **Principle-2 smoke test** (`principle2_test.py`) — full system vs frozen base on clean Libri2Mix; raise sparsity 2× until it passes | P2-B4 | C | Never-worse proof | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

## 🚧 GATE M2 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Learned gating beats best single adapter on co-occurring cells
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Principle-2 smoke test passes (≥ base − 0.1 dB on clean)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Held-out combination cells (reverb+codec, noise+codec) do not collapse
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Condition heads inspectable (each supervised dimension traceable)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed

---

# 🔧 PHASE P3 — Joint polish, band recovery, calibration (Stage 4)

**Milestone:** Mandatory joint fine-tune of all adapters + gate on compound conditions; band-recovery head trained and dual-metric-guarded; all probabilities calibrated.
**🚧 GATE M3:** Dual-metric guard validated on held-out; count ECE < 0.05; dropped-speaker recall > 90% at 10% false-alarm rate.

## ⛓ SEQUENTIAL — P3 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P3-B1 | **Joint polish (mandatory)** — unlock 3 adapters + gate, train 15–20 epochs at 0.1× Stage-1 LR on compound-condition data; base frozen; O-LoRA if harm >0.3 dB | M2 | B | Polished adapters | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P3-B2 | `models/counting.py` — attractor readout (Vote 1 `p_k`) + count prior (Vote 2) + **bounded residual sweep** (Vote 3, max 3 candidates {mode−1,mode,mode+1} clipped [2,5], decoder-only) + logistic fusion | M2, P0-B4 | B | Counting subsystem | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P3-C1 | `models/band_recovery.py` — 2-conv high-band (4–8 kHz) mask head; input low-band 8 kHz STFT + mixture 16 kHz high-band; SI-SNR loss on 16 kHz recon | P3-B1 | C | Band-recovery head | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P3-C2 | **Dual-metric guard** — per-chunk bypass unless both SI-SDRi and DNSMOS improve; worst case = 8 kHz pass-through zero-padded to 16 kHz | P3-C1, P0-C4 | C | Guarded quality stage | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P3-B3 | `models/confidence.py` — per-stream confidence (p_k + inter-stage consistency + blind DNSMOS) + **completeness** (residual energy + SileroVAD-on-residual + attractor mass) + OOD Mahalanobis discount | P3-B2 | B | Confidence/completeness | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P3-C3 | **Calibration** (`calibration/`) — temperature scaling for count posterior; per-stream confidence logistic; completeness logistic (manufactured N−1 failures); counting fusion logistic; band-recovery guard thresholds. All on held-out. | P3-B3, P3-C2 | C | Fitted+hashed calibrators | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

## 🚧 GATE M3 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint polish complete; adapters composable at realistic co-activation
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] **Confusion matrix + calibration curve** produced; count ECE < 0.05
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Dropped-speaker recall > 90% at 10% FAR
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Dual-metric band-recovery guard validated (ships disabled if it cannot pass)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Residual-sweep trigger frequency measured (< 30% or threshold raised)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed

---

# 🚀 PHASE P4 — Demo, CLI, efficiency (Weeks —)

**Milestone:** End-to-end CLI + Gradio demo with condition-routing visualization; RTF measured including worst-case residual sweep and 16 kHz band-recovery STFT.
**🚧 GATE M4:** Demo runs end to end on a held-out real recording; RTF documented.

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P4-C1 | `pipeline/chunker.py` — 2.4 s chunks / 0.8 s step @8k + parallel 16 kHz mixture STFT | M3 | C | Chunker | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P4-C2 | `pipeline/stitcher.py` — max-correlation continuity + ECAPA tie-break + crossfade; global count via ECAPA clustering | M3 | C | Stitcher | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — `align/chunking.py` + `align/integration.py` + ECAPA reused; rewire to single-backbone streams |
| P4-C3 | `pipeline/infer.py` — §6.2 per-chunk order (Level-1 → Pass 1 → Level-2 → gate → Pass 2 → counting → band recovery → guarded quality) | P4-C1, P4-C2 | C | Inference pipeline | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P4-A1 | CLI entry point + reproducibility bundle (configs, hashed artifacts) | P4-C3 | A | CLI + bundle | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P4-C4 | Gradio demo: upload → count, N waveforms, spectrograms, condition/gate visualization, optional Whisper | P4-C3 | C | `demo/app.py` (rewire) | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — MockEngine skeleton reused; wire real CALM-Sep engine |
| P4-B1 | Efficiency report — RTF at average and worst-case residual sweep (0.9 extra passes/uncertain chunk) + 16 kHz STFT | P4-C3 | B | RTF table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

## 🚧 GATE M4 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Demo runs end to end on a held-out real recording
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] RTF documented (average + worst case)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Any post-processor failing its guard ships disabled
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Joint integration session completed

---

# 🏆 PHASE P5 — Full evaluation & report (Weeks —)

**Milestone:** Full measurement matrix, all mandatory baselines, all headline analyses, reproducibility bundle.
**🚧 GATE M5:** Every claim carries a bootstrap interval; universal-adapter verdict stated plainly. The only failure mode is dishonesty.

## 🔄 PARALLEL — P5 tasks

| ID | Task | Depends on | Owner | Deliverable | Status |
|----|------|------------|-------|-------------|--------|
| P5-C1 | `eval/matrix.py` + `eval/stats.py` — full matrix (SI-SDRi, DNSMOS, PESQ, count acc, ECE) with bootstrap CIs (10k) + Wilcoxon | M4 | C | Results matrix | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-C2 | Primary-benchmark headline: reverb-noisy LibriMix N=2 SI-SDRi over mixture | P5-C1 | C | Headline number | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-A1 | Real-RIR eval (BUT ReverbDB SLR17) — sim-to-real gap (mandatory) | P5-C1 | A | Real-RIR table | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-A2 | Real recordings (team-recorded + LibriCSS) — DNSMOS + Whisper WER | M4 | A | Real-audio results | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-B1 | Break-point curve (every metric vs N=2→5) + band-recovery contribution (matched pairs) | P5-C1 | B | Curves | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| P5-ALL1 | Report: each dev writes their section; reproducibility bundle (content-addressed) | P5-C1 | All | Final report | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

## Mandatory baselines (BLUEPRINT §9.6)

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Frozen base alone (8 kHz; zero-padded to 16 kHz for DNSMOS) — quality floor
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Universal adapter (P1b) — whether routing is needed
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Uniform blend, no gate — whether the gate earns its complexity
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Oracle gating — upper bound on routing
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Frozen base + band recovery, no adapters — isolates band recovery

## 🚧 GATE M5 — Acceptance criteria

- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Full matrix filled; every cell has a bootstrap interval
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] All 5 mandatory baselines reported
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] All 8 headline analyses complete (interference matrix, composition, compositional generalization, break-point, calibration, risk-coverage, band-recovery contribution, efficiency)
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Universal-adapter verdict stated plainly
- ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] Reproducibility bundle reproduces key numbers

---

# Evaluation matrix (BLUEPRINT §7.4 / §9.4)

Conditions × N ∈ {2,3,4,5}. Separation scored at 8 kHz; DNSMOS on 16 kHz band-recovered output.

| Tier | Source | Measures | N | n/cell |
|------|--------|----------|---|--------|
| Clean 2–3 | Libri2Mix / Libri3Mix (8k) | SI-SDRi, PESQ | 2,3 | 500 |
| Sparse overlap | SparseLibriMix (8k) | quality vs overlap 0–100% | 2 | 200 |
| Sparse overlap 3-spk | custom | extends SparseLibriMix to N=3 | 3 | 200 |
| **Primary: noise+reverb** | custom WHAMR!-style (8k) | **headline SI-SDRi + DNSMOS** | 2 | **500** |
| Reverb-noisy high count | same | count acc + quality | 3,4,5 | 200 |
| Reverb only | clean-reverb LibriMix | isolates `adapter_reverb` | 2,3 | 200 |
| **Real-RIR (mandatory)** | **BUT ReverbDB SLR17** | sim-to-real gap | 2 | 200 |
| Codec only | LibriMix + ffmpeg | isolates `adapter_codec` | 2 | 200 |
| Reverb+codec (held-out) | never in gate training | compositional generalization | 2,4 | 200 |
| Noise+codec (held-out) | never in gate training | compositional generalization | 2,4 | 200 |
| High count clean | Libri4Mix / Libri5Mix | break-point N=4–5 | 4,5 | 200 |
| High count degraded | + reverb-noisy | count under degradation | 4,5 | 200 |
| Real recordings | LibriCSS 1ch | DNSMOS + Whisper WER | 2+ | full |
| Band-recovery gain | matched 8k vs 16k pairs | isolates band recovery | 2 | 500 |

---

# Novelty ledger (BLUEPRINT §1.3 / §9.5)

| ID | Contribution | Proof artifact | Target phase | Status |
|----|-------------|----------------|--------------|--------|
| C1 | Condition-aware LoRA mixture on a frozen backbone (weight-space composition) | Composition analysis + learned-vs-oracle gating | P2, P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| C2 | Supervised, inspectable two-level condition analyzer | Per-dimension trace + gate ablation | P2 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| C3 | Never-worse-than-base, empirically verified | Principle-2 smoke test | P2 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| C4 | Attractor `p_k` counting + bounded residual sweep | Confusion matrix + ECE + risk-coverage | P3, P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| C5 | Residual-energy completeness detector (missed-speaker guard) | Dropped-speaker recall > 90% @10% FAR | P3 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| C6 | Dual-metric-guarded band recovery (8→16 kHz) | Matched-pair SI-SDRi + DNSMOS delta | P3, P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| C7 | Universal-adapter honesty gate | Pre-committed verdict (P1b) | P1b, P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

---

# Dataset acquisition tracker

| Dataset | Role | Owner phase | Status |
|---------|------|-------------|--------|
| LibriSpeech (8k + 16k copies) | Source speech (train-clean-100 pool; dev/test held out) | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep reused; 8 kHz downsample + 16 kHz retention pending |
| RIR bank (pyroomacoustics, 10k) | `adapter_reverb` + reverb eval | P0-A | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| WHAM! (~17 GB) | `adapter_noise` | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep reused |
| DNS-4 (stratified 20 GB) | `adapter_noise` variety | P0-A | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| Libri2/3Mix | Clean + primary eval | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep reused |
| Libri4/5Mix | High-count eval | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — `prepare_librimix_highn.py` reused |
| SparseLibriMix | Overlap eval (test only) | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — prep reused |
| **BUT ReverbDB (SLR17)** | Mandatory real-RIR eval | P0-A / P5-A | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — **note: SLR28 is AISHELL-2, NOT a RIR set** |
| LibriCSS | Real-room DNSMOS + WER | P5-A | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| Real-room set | Team-recorded flagship | P4/P5 | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| VCTK | Optional extended pool | P0-A | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — `prepare_vctk.py` reused (optional) |
| WSJ0-mix / WHAMR! (LDC) | Literature comparison only | Optional | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — excluded from training (license) |

---

# Pretrained tools (used at inference, never retrained)

| Tool | Source | Purpose | Status |
|------|--------|---------|--------|
| SR-CorrNet var-2-5 | HF `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` | Frozen base (8k, K0=5) | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — download + patch (P0-B1) |
| SileroVAD | silero-vad (8k native) | Level-1 voiced density | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| ECAPA-TDNN | SpeechBrain VoxCeleb | Stitching + global count clustering | ![done](https://img.shields.io/badge/done-brightgreen?style=flat-square) [x] — wrapper reused (`models/experts/embeddings.py`, `align/embeddings.py`) |
| DNSMOS ONNX | Microsoft DNS Challenge | Quality on 16 kHz output | ![in progress](https://img.shields.io/badge/in_progress-yellow?style=flat-square) [~] — stub reused; ONNX activation pending |
| PESQ | `pesq` pip | Reference-based quality | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| Whisper (optional) | OpenAI | Demo transcripts / LibriCSS WER | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

---

# Risk register & fallback triggers (BLUEPRINT §11)

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|------------|--------|
| `p_k` not exposed by wrapper | Low-Med | Blocker (counting) | First P0 task; patch before any other code | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| LoRA composition interference > 0.3 dB | High (expected) | Medium | Co-activation warm-up (P1) + mandatory joint polish (P3) + O-LoRA (escalation) | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Universal adapter matches full routing | Medium | Reframes headline | Trained first; pre-commit to adopt if within 0.5 dB | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Gate collapse to uniform activation | Medium | Quality plateau | Supervised heads + sparsity + oracle-gap analysis | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| SileroVAD proxy uninformative | Medium | Minor | Validate in P0; fallback voiced-energy fraction | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Never-worse fails on clean | Medium | Credibility | Principle-2 smoke test; raise sparsity until it passes | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Sim-to-real reverb gap | Medium | Real-RIR cells | BUT ReverbDB tier mandatory | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Count degrades under combined degradation | Medium | Half the grade | Residual sweep; degraded-val count from start | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Band recovery hurts SI-SDRi | Medium | Quality regression | Dual-metric guard; per-chunk bypass | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |
| Residual sweep triggers too often | Unknown | RTF budget | Measure; raise threshold or drop to 2 candidates if >30% | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] — monitoring |

---

# Compute & parameter budget tracker

| Item | Target | Actual | Status |
|------|--------|--------|--------|
| Frozen base params | 13.6M | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| New trainable params | ~3–4M | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| Per-adapter params | ~0.4–0.6M | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| Total GPU-hours | < 150 (T4) | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| RTF (avg, incl. residual sweep + 16k STFT) | measured | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |
| Inference memory | ≤ 16 GB T4 | — | ![not done yet](https://img.shields.io/badge/not_done_yet-red?style=flat-square) [ ] |

**New trainable component sizes (target):** condition analyzer + gate + counting fusion + calibration ≈ 1–1.5M; 3 LoRA adapters < 2M; band recovery ~0.1M.

---

# Phase timeline summary

| Phase | Gate | Parallel? | Critical owner |
|-------|------|-----------|----------------|
| **P0** Verify checkpoint + synthesis | M0 | 🔄 Full parallel (B patches / A data) | B |
| **P1** Adapter library (Stage 1) | M1 | Adapters sequential; A/C parallel | B |
| **P1b** Universal-adapter gate (Stage 2) | M1b | ⛓ Irreversible verdict | B |
| **P2** Condition analyzer + gate (Stage 3) | M2 | 🔄 Mostly parallel | B |
| **P3** Joint polish + band recovery + calibration (Stage 4) | M3 | ⛓ Sequential | B/C |
| **P4** Demo, CLI, efficiency | M4 | 🔄 Parallel | C |
| **P5** Full eval + report | M5 | 🔄 Parallel | All |

---

# New source files CALM-Sep must add (gap list)

Not present in the repo yet; each is a Phase deliverable above.

- `models/srcorrnet/` — wrapper exposing `p_k`, `E(0)`, decoder-stage features (Patches A/B/C)
- `models/lora.py` — parallel-branch LoRA + co-activation sampler
- `models/condition.py` — two-level condition analyzer
- `models/gate.py` — gate MLP + EMA + sparsity
- `models/counting.py` — attractor readout + residual sweep + fusion
- `models/confidence.py` — per-stream confidence + completeness + OOD
- `models/band_recovery.py` — high-band head + dual-metric guard
- `pipeline/chunker.py`, `pipeline/stitcher.py`, `pipeline/infer.py`
- `eval/matrix.py`, `eval/stats.py`
- `calibration/` — fitted temperature scalars + logistic models (hashed)
- `data/synthesis/`, `data/fixed_eval/`, `data/rirs/`
- `tests/attractor_test.py`, `tests/e0_hook_test.py`, `tests/principle2_test.py`, `tests/smoke_test.py`
- configs: `base_checkpoint.yaml`, `adapters/{reverb,noise,codec}.yaml`, `gate.yaml`, `band_recovery.yaml`, `eval.yaml`

---

*End of PROJECT_TODO.md (CALM-Sep) — edit as the project progresses. When implementation and `BLUEPRINT` disagree, one is wrong; record the resolution in `BLUEPRINT` §15.*
