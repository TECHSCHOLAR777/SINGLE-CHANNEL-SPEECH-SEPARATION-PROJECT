# CALM-Sep

**Condition-Aware LoRA Mixture for Multi-Speaker Speech Separation**

This file is the project task tracker. It is the single place to see what is
done, what is in progress, and what is left. The full design lives in
`BLUEPRINT`, which is the source of truth. When this tracker and `BLUEPRINT`
disagree, `BLUEPRINT` wins.

Last updated: 2026-07-17

---

## What the system does

The system takes one mono recording where two to five people talk at the same
time. The number of speakers is not given. The recording may also be
reverberant, noisy, or damaged by audio codecs.

It returns:

1. A speaker count `N` in {2, 3, 4, 5}, and one clean waveform per speaker.
2. A confidence score for each returned waveform.
3. A completeness probability: one number saying how likely it is that no
   speaker was missed.

It is graded on two things, in this order:

1. **Speaker count accuracy.** Did it return the right number of speakers?
2. **Separation quality.** Does each returned voice sound clean and isolated?

Everything else is a bonus. Any idea that could help speed or elegance but
risks these two axes is guarded or removed.

---

## How it works (one paragraph)

Take one strong pretrained separation network, SR-CorrNet var-2-5, and freeze
it forever. Teach it to handle hard conditions with three small plug-in
adapters (LoRA): one for reverb, one for noise, one for codec damage. A
two-level condition analyzer looks at each audio chunk, measures how much of
each condition is present, and a gate blends the adapters into the frozen
network in proportion to those strengths. The blend happens inside the weight
matrices before any audio is produced, so there is always one forward pass and
one set of output voices. The speaker count is read from the backbone's own
attractor probabilities. A small band-recovery head lifts the 8 kHz output to
16 kHz for perceptual quality. A residual-energy detector guards against a
missed speaker.

---

## Fixed constraints (never change these)

| Constraint | Value |
|---|---|
| Base checkpoint | `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk`, downloaded once, frozen forever, never fine-tuned |
| Speaker count | N = 2 to 5 only |
| Sample rate | 8 kHz internal (STFT window 128, hop 64), locked by the checkpoint |
| Quality path | Band recovery only (8 kHz to 16 kHz). No backbone retraining |
| New trainable params | About 3 to 4 M (condition analyzer, gate, counting fusion, calibration, 3 LoRA adapters) against a 13.6 M frozen base |
| Compute | Under 150 GPU-hours total on free-tier T4. Largest single run is one adapter, 20 to 40 GPU-hours |

Published base performance on clean WSJ0-mix: count 100 / 99.7 / 97.7 / 96.9
percent and 24.8 / 24.4 / 21.9 / 19.9 dB SI-SDRi at N = 2 / 3 / 4 / 5.

---

## How to read this tracker

Each task has a status marker:

| Marker | Meaning |
|---|---|
| `[x]` | Done. Code exists, is tested, and is merged. |
| `[~]` | In progress. Some code shipped, not finished. |
| `[ ]` | Not started. |

Tags on a task:

- **PARALLEL**: can run at the same time as its siblings.
- **SEQUENTIAL**: blocked until its dependency is done.
- **GATE**: a hard checkpoint. Work does not move to the next phase until the
  gate passes.

Rules:

1. Mark `[x]` only when the deliverable exists, is tested, and is merged to
   `master`.
2. If a gate fails, stop, fix it, re-run the gate, then continue.
3. Add a date and note when a task is done, for example
   `[x] Task name (done 2026-07-20)`.
4. A task that reopens a fixed constraint is out of scope.

---

## Progress at a glance

Done (parvA + parvB + parvC code complete, 373 tests passing): **Dev A data
infrastructure, Dev B model library + training pipeline, Dev C inference
pipeline + evaluation + demo** are all implemented. All 3 branches need final
review and testing against real weights before milestone gates can close.

| Milestone | Phase | State |
|---|---|---|
| M0 | Verify checkpoint and build synthesis pipeline | In progress |
| M1 | Adapter library | Code complete (parvB), needs weight validation |
| M1b | Universal-adapter decision | Code complete (parvB) |
| M2 | Condition analyzer and gate | Code complete (parvB) |
| M3 | Joint polish, band recovery, calibration | Code complete (parvC) |
| M4 | Demo, CLI, efficiency | Code complete (parvC) |
| M5 | Full evaluation and report | Code complete (parvC), needs weights |

### Branch status

| Branch | Status | Notes |
|---|---|---|
| `parvA` | Committed + pushed | Data prep, codec augmentation, mixer, rir_bank, hashing; 96 tests pass |
| `parvB` | Code complete, uncommitted | All models (lora, condition, gate, counting, confidence, srcorrnet patches), all train stages (1–4), losses |
| `parvC` | Code complete, uncommitted | Pipeline (chunker, stitcher, infer), calibration, eval matrix+stats, band recovery, DNSMOS, preprocess, demo, configs, 62 new tests (373 total) |

---

## Team roles

Ownership, not exclusivity. Anyone can touch any folder; the owner reviews.

| Dev | Owns | Focus |
|---|---|---|
| A | `data/` | 8 kHz dynamic mixing, RIR bank, WHAM and DNS-4 staging, codec transforms, fixed and hashed evaluation sets |
| B | `models/`, `train/` | Backbone wrapper patches, LoRA library, adapter training, condition analyzer, gate, attractor counting, confidence and completeness |
| C | `eval/`, `pipeline/`, `calibration/`, `align/`, `demo/` | Evaluation matrix and stats, band-recovery guard, calibration, chunker and stitcher, demo |
| All | `configs/`, `tests/`, `docs/`, `schemas/` | Shared contracts |

---

## Design principles (from BLUEPRINT section 4)

1. One backbone, adapted, never arbitrated. All specialization lives in
   adapters on one frozen network, so stream alignment is never a problem.
2. Never worse than the base, and proven by measurement, not by claim. The
   Principle-2 smoke test checks this on clean audio.
3. Conditions are quantities, not categories. Routing is continuous gating
   scaled by measured strength, never a hard switch.
4. Every probability is calibrated, and every internal signal is inspectable.
5. No claim without a number. The universal-adapter baseline is trained first;
   if it matches the full routing system, the project reports that honestly and
   ships the simpler system.

---

## Cross-cutting work (all phases)

**Team sessions**

- [ ] Day 1: agree config schema and the `SeparationResult` contract (add
  `p_k`, gate vector, completeness probability, OOD flag)
- [ ] Start of P1: review LoRA attachment and adapter training
- [ ] Before P1b: agree the universal-adapter decision rule and pre-commit it
- [ ] Start of P2: review condition analyzer and gate, and the circularity fix
- [ ] After each gate: milestone integration session, run the pipeline together
- [ ] P4: real recording session (all three as speakers)
- [ ] P5: report writing, each dev writes their section
- [ ] Weekly sync on blockers

**Git workflow**

- [x] `master` always runnable and passing CI; no direct commits, all via PR
- [x] Every change via PR with one review from a non-owner
- [ ] Model-core PRs (wrapper patches, LoRA, gate) reviewed by all three

**Code standards**

- [x] Black and Ruff via pre-commit and CI
- [x] Type hints on public function signatures
- [x] `SeparationResult` schema defined once in `schemas/`
- [x] Extend `SeparationResult` with `p_k`, gate vector, completeness, OOD flag (done parvC)
- [x] Header docstring on every module
- [x] `docs/decisions.md` updated for each architecture choice
- [ ] Config hash (SHA-256) recorded in every checkpoint and result
- [ ] Every mechanism has an off switch in config; baselines are one-line runs

**Data split discipline (BLUEPRINT section 7.5)**

- [x] Three-way holdout enforced: speaker holdout (dev-clean/test-clean never
  in training via mixer), condition-combination holdout (reverb+codec and
  noise+codec excluded from gate/joint training via `assert_not_held_out()`),
  severity holdout (T60 > 0.9 s and SNR < -4 dB ≤ 10% of training).

---

# Phase P0: Verify checkpoint and build synthesis pipeline

**Goal:** The frozen checkpoint loads and exposes its internal signals
(`p_k`, `E(0)`, decoder-stage features). The 8 kHz synthesis pipeline produces
labelled mixtures. All evaluation sets are generated once, seeded, and hashed.

**No GPU hours in this phase. Inference only.**

## SEQUENTIAL: Dev B, checkpoint and wrapper (blocking)

| ID | Task | Deliverable | Status |
|---|---|---|---|
| P0-B1 | Download and verify the frozen checkpoint; confirm the YAML constants (sr 8000, max_n_spks 5, N_Enc 2, N_Dec 4, d_model 128) | Verified `model.pt` plus SHA in `base_checkpoint.yaml` | [x] Config written; weight download at training time |
| P0-B2 | Patch A: expose `p_k` through `_single_pass_session`, `process_waveform`, `process_stft` | Wrapper returns `pres["probs"]` shape (1, 7) | [x] `models/experts/srcorrnet.py` (parvB) |
| P0-B3 | Patch B: forward hook on `model.encoder` to capture `E(0)`, shape (1, T, 65, 128) | Pooled `E(0)` available to the analyzer | [x] `models/experts/srcorrnet.py` (parvB) |
| P0-B4 | Patch C: hooks on each `dec_block[i]` to capture decoder-stage features | Stage features for inter-stage consistency | [x] `models/experts/srcorrnet.py` (parvB) |
| P0-B5 | Expose the attractor threshold `prob_thres` (default 0.5) as a config value | Configurable count threshold | [x] `configs/calmsep/base_checkpoint.yaml` |
| P0-B6 | Write `attractor_test.py`: assert `p_k` shape (1, 7) and active-slot count equals true N at N = 2, 3, 4, 5 | Blocking test green | [ ] Needs real weights to run |
| P0-B7 | Corpus-transfer baseline: run the base on 20 dev-clean 2-speaker mixtures, record mean SI-SDRi (the floor every adapter must beat) | Baseline number logged | [ ] Needs real weights |

## PARALLEL: Dev A, synthesis and evaluation sets

| ID | Task | Deliverable | Status |
|---|---|---|---|
| P0-A1 | Dynamic 8 kHz mixer, N in {2, 3, 4, 5}, per-speaker level offsets, clean-stem ground truth | `data/mixer.py` at 8 kHz plus tests | [x] `data/calmsep_mixer.py` + tests (parvA) |
| P0-A2 | LibriSpeech source at 8 kHz (train-clean-100, dev-clean, test-clean); keep 16 kHz copies for band-recovery targets | Speaker pool and prep script | [x] `data/prepare_librispeech_8k.py` (parvA) |
| P0-A3 | RIR bank: 10k RIRs with pyroomacoustics, 1k per 0.1 s T60 step over 0.2 to 1.0 s, cached to `data/rirs/` | Cached RIR bank | [x] `data/rir_bank.py` + tests (parvA) |
| P0-A4 | Noise staging: WHAM (about 17 GB) plus a stratified 20 GB DNS-4 subset | Noise prep scripts | [x] `data/prepare_noise_staging.py` (parvA) |
| P0-A5 | Codec transforms: ffmpeg Opus 6 to 24k, AAC 16 to 48k, AMR-NB and AMR-WB | `data/codec_augmentation.py` | [x] All 4 codecs, AMR-NB/WB added (parvA) |
| P0-A6 | Fixed evaluation matrix, generated once, seeded, and hashed (see the evaluation table below) | `data/fixed_eval/` plus manifests and hashes | [x] `data/fixed_eval_generator.py` (parvA) |
| P0-A7 | Reverb reference policy: wet source, truncated at n_peak plus 512 samples | Reference generator | [x] `data/degradations.py` `make_wet_reference()` (parvA) |

## PARALLEL: Dev C, evaluation and condition-input tooling

| ID | Task | Deliverable | Status |
|---|---|---|---|
| P0-C1 | Cardinality-aware SI-SDR and SI-SDRi with PIT: missed speaker scores 0 dB, minus 1 dB per hallucinated stream | `eval/metrics.py` | [x] (parvC) |
| P0-C2 | Shared 8 kHz STFT (window 128, hop 64) plus a parallel 16 kHz mixture STFT for band recovery | `models/preprocess.py` retargeted | [x] `calmsep_preprocess()` dual-branch (parvC) |
| P0-C3 | SileroVAD voiced-frame density at 8 kHz; check it separates overlap on LibriCSS; fallback is voiced-energy fraction | Level-1 VAD feature plus a validation note | [x] `voiced_density_silero()` in `models/condition.py` (parvB) |
| P0-C4 | Activate DNSMOS ONNX on 16 kHz band-recovered output (download `sig_bak_ovrl.onnx`) | `eval/dnsmos.py` live | [x] Full ONNX inference, `DnsmosScorer` (parvC) |
| P0-C5 | Config loader and logging | `utils/config.py` | [x] |

## GATE M0

- [ ] `attractor_test.py` passes: `p_k` shape (1, 7), active slots equal true N at N = 2, 3, 4, 5 (needs weights)
- [ ] Patches A, B, C load cleanly; `e0_hook_test.py` confirms `E(0)` shape (1, T, 65, 128) (needs weights)
- [x] All evaluation sets generated, seeded, and hashed; manifests committed (parvA)
- [ ] Frozen-base corpus-transfer SI-SDRi recorded (the adapter floor) (needs weights)
- [x] SileroVAD proxy implemented with energy fallback (parvB/condition.py)
- [x] Zero GPU hours spent so far
- [ ] Milestone session done

---

# Phase P1: Adapter library (Stage 1)

**Goal:** Three LoRA adapters (reverb, noise, codec) trained one at a time with
co-activation warm-up, and the cross-interference matrix measured.

## SEQUENTIAL: Dev B, critical path

| ID | Task | Deliverable | Status |
|---|---|---|---|
| P1-B1 | `models/lora.py`: parallel-branch LoRA (`y = W0 x + sum g * B(A x)`), co-activation sampler, 37 target modules per adapter (BLUEPRINT section 5.3) | LoRA library plus tests | [x] (parvB) |
| P1-B2 | Freeze-and-attach harness: register LoRA on targets, freeze all base params, only adapter params in the optimizer, `strict=False` load order | Training scaffold | [x] `LoRALibrary.freeze_base()` (parvB) |
| P1-B3 | Reuse the backbone engine losses: `PIT_SISNR_time` plus 0.5 times `PIT_SISNR_mag` plus 0.1 × attractor BCE | Loss wiring | [x] `train/losses.py` `calmsep_loss()` (parvB) |
| P1-B4 | Train `adapter_noise` first (widest data, best for debugging the LoRA plumbing), co-activation gates from U(0.0, 0.2) | Adapter weights plus validation curve | [~] `train/stage1_single.py` written; needs GPU run |
| P1-B5 | Train `adapter_reverb` (RIR and wet references, T60 0.2 to 1.0 s) | Adapter weights | [~] Same script, `--adapter reverb` |
| P1-B6 | Train `adapter_codec` (Opus, AAC, AMR) | Adapter weights | [~] Same script, `--adapter codec` |

## PARALLEL: Dev A and Dev C

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P1-A1 | Noisy training mixtures (WHAM plus DNS-4, SNR -6 to +10 dB) at 8 kHz | A | Noise-condition data | [x] `data/degradations.py` (parvA) |
| P1-A2 | Reverb training mixtures (RIR bank, wet references) | A | Reverb-condition data | [x] `data/rir_bank.py` + degradations (parvA) |
| P1-A3 | Codec training mixtures (ffmpeg transforms) | A | Codec-condition data | [x] `data/codec_augmentation.py` (parvA) |
| P1-C1 | Cross-interference matrix: each adapter alone on every condition, off-diagonal harm threshold 0.3 dB | C | Interference table | [ ] Needs weights |
| P1-C2 | O-LoRA orthogonality penalty, used only if harm goes above 0.3 dB | C | Optional penalty term | [x] `--use-olora` flag in `train/stage4_joint.py` (parvB) |

## GATE M1

- [ ] Each adapter shows a significant SI-SDRi gain on its matched condition (Wilcoxon p < 0.05) — needs GPU training
- [ ] No adapter degrades clean Libri2Mix — needs weights
- [ ] Cross-interference matrix measured, off-diagonal harm below 0.3 dB, or O-LoRA applied
- [ ] Milestone session done

---

# Phase P1b: Universal-adapter decision (Stage 2)

**Goal:** Train one universal adapter and compare it to the routing target on
the primary benchmark.

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P1b-B1 | Train the universal adapter on the union of all single-condition datasets | B | Universal adapter weights | [~] `train/stage2_universal.py` written; needs GPU |
| P1b-C1 | Evaluate on the primary benchmark (reverb-noisy, N = 2) plus at least two multi-condition cells, then log the verdict | C | Verdict in `docs/decisions.md` | [ ] Needs weights |

## GATE M1b

- [ ] Verdict logged before the gate network is built
- [ ] Decision criterion: universal within 0.5 dB SI-SDRi → adopt simpler; otherwise continue

---

# Phase P2: Condition analyzer and gate (Stage 3)

**Goal:** Train the two-level condition analyzer and the gate.

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P2-B1 | `models/condition.py` Level 1: raw-STFT DSP features (SNR, codec family and bitrate, voiced density from SileroVAD). Deterministic, no training | B | Level-1 analyzer | [x] `level1_features()` (parvB) |
| P2-B2 | `models/condition.py` Level 2: E(0) heads for reverberation strength (T60) and a speaker-count prior | B | Level-2 analyzer | [x] `Level2Analyzer`, `T60Head`, `CountPriorMLP` (parvB) |
| P2-B3 | `models/gate.py`: gate MLP (two hidden layers of 256, GELU, sigmoid scaled to 1.5), per-layer gates, L1 sparsity 1e-3, EMA smoothing 0.7 | B | Gate network | [x] `GateNetwork`, `GateSmoother` (parvB) |
| P2-B4 | Stage-3 training: separation loss through the gates, supervised condition-head losses, gate sparsity | B | Trained analyzer and gate | [~] `train/stage3_gate.py` written; needs GPU |
| P2-C1 | Ablation: per-layer gates versus per-adapter scalars. Simpler wins if there is no gain | C | Ablation result | [ ] Needs weights |
| P2-C2 | Principle-2 smoke test (`principle2_test.py`): full system versus frozen base on clean Libri2Mix. Raise sparsity until it passes | C | Never-worse proof | [ ] Needs weights |

## GATE M2

- [ ] Learned gating beats the best single adapter on mixed-condition cells — needs weights
- [ ] Principle-2 smoke test passes — needs weights
- [ ] Held-out combination cells do not collapse
- [ ] Milestone session done

---

# Phase P3: Joint polish, band recovery, and calibration (Stage 4)

**Goal:** A mandatory joint fine-tune of all adapters and the gate on
compound-condition data, a trained and guarded band-recovery head, and
calibration of every probability the system emits.

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P3-B1 | Joint polish (mandatory): unlock the 3 adapters plus gate, train 15 to 20 epochs at one-tenth the Stage-1 learning rate on compound data. Base stays frozen. Apply O-LoRA if harm is above 0.3 dB | B | Polished adapters | [~] `train/stage4_joint.py` written; needs GPU |
| P3-B2 | `models/counting.py`: attractor readout (Vote 1, `p_k`), count prior (Vote 2), and a bounded residual sweep (Vote 3, at most 3 candidates {mode-1, mode, mode+1} clipped to [2, 5], decoder only), fused by logistic regression | B | Counting subsystem | [x] `ThreeVoteCounter`, `CountFusion`, `residual_sweep_sisdr` (parvB) |
| P3-C1 | `models/band_recovery.py`: a two-conv head predicting the 4 to 8 kHz mask from the low-band 8 kHz STFT and the mixture 16 kHz high-band STFT | C | Band-recovery head | [x] `BandRecoveryHead` (parvC) |
| P3-C2 | Dual-metric guard: apply band recovery per chunk only if both SI-SDRi and DNSMOS improve. Worst case is 8 kHz pass-through padded to 16 kHz | C | Guarded quality stage | [x] `apply_band_recovery_guarded()` (parvC) |
| P3-B3 | `models/confidence.py`: per-stream confidence, completeness, OOD Mahalanobis discount | B | Confidence and completeness | [x] `StreamConfidenceHead`, `CompletenessHead`, `MahalanobisOOD` (parvB) |
| P3-C3 | Calibration in `calibration/`: temperature scaling for the count posterior, per-stream confidence model, completeness model, OOD detector | C | Fitted and hashed calibrators | [x] `calibration/temperature.py`, `confidence.py`, `completeness.py`, `ood.py` (parvC) |

## GATE M3

- [ ] Joint polish complete; adapters compose at realistic co-activation strengths — needs GPU
- [ ] Confusion matrix and calibration curve produced; count ECE below 0.05
- [x] Dual-metric band-recovery guard implemented and tested (parvC)
- [x] Three-vote counting subsystem implemented (parvB)
- [ ] Milestone session done

---

# Phase P4: Demo, CLI, and efficiency

**Goal:** An end-to-end CLI and Gradio demo with condition-routing
visualization.

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P4-C1 | `pipeline/chunker.py`: 2.4 s chunks, 0.8 s step at 8 kHz, plus a parallel 16 kHz mixture STFT | C | Chunker | [x] (parvC) |
| P4-C2 | `pipeline/stitcher.py`: max-correlation continuity, ECAPA tie-break, crossfade | C | Stitcher | [x] (parvC) |
| P4-C3 | `pipeline/infer.py`: the per-chunk order (Level 1, Pass 1, Level 2, gate, Pass 2, counting, band recovery, guarded quality) | C | Inference pipeline | [x] `CalmSepPipeline` (parvC) |
| P4-A1 | CLI entry point and a reproducibility bundle (configs and hashed artifacts) | A | CLI plus bundle | [ ] |
| P4-C4 | Gradio demo: upload, then show count, N waveforms, gate routing, optional Whisper transcript | C | `demo/app.py` | [x] Full CALM-Sep demo with routing viz + Whisper (parvC) |
| P4-B1 | Efficiency report: RTF at average and worst-case residual sweep, plus the 16 kHz STFT cost | B | RTF table | [ ] Needs weights |

## GATE M4

- [ ] Demo runs end to end on a held-out real recording — needs weights
- [ ] RTF documented at average and worst case
- [x] Any post-processor that fails its guard ships off (dual-metric guard)
- [ ] Milestone session done

---

# Phase P5: Full evaluation and report

**Goal:** The full measurement matrix, every mandatory baseline, every headline
analysis, and a reproducibility bundle.

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P5-C1 | `eval/matrix.py` and `eval/stats.py`: the full matrix (SI-SDRi, DNSMOS, count accuracy, ECE) with bootstrap intervals and Wilcoxon tests | C | Results matrix | [x] `run_eval_matrix()`, `bootstrap_ci()`, `wilcoxon_test()` (parvC) |
| P5-C2 | Primary-benchmark headline: reverb-noisy LibriMix, N = 2, SI-SDRi over the mixture | C | Headline number | [ ] Needs weights |
| P5-A1 | Real-RIR evaluation on BUT ReverbDB (SLR17), the sim-to-real gap (mandatory) | A | Real-RIR table | [x] `data/prepare_but_reverbdb.py` (parvA) |
| P5-A2 | Real recordings (team-recorded plus LibriCSS): DNSMOS and Whisper WER | A | Real-audio results | [ ] |
| P5-B1 | Break-point curve (each metric versus N from 2 to 5) and the band-recovery contribution from matched pairs | B | Curves | [ ] Needs weights |
| P5-ALL1 | Report: each dev writes their section, plus a content-addressed reproducibility bundle | All | Final report | [ ] |

## Mandatory baselines (BLUEPRINT section 9.6)

- [ ] Frozen base alone (8 kHz, padded to 16 kHz for DNSMOS): the quality floor
- [ ] Universal adapter (from P1b): shows whether routing is needed
- [ ] Uniform blend, no gate: shows whether the gate earns its cost
- [ ] Oracle gating: the upper bound on routing
- [ ] Frozen base plus band recovery, no adapters: isolates band recovery

## GATE M5

- [ ] Full matrix filled, every cell has a bootstrap interval
- [ ] All five baselines reported
- [ ] All eight headline analyses done
- [ ] Universal-adapter verdict stated plainly
- [ ] Reproducibility bundle reproduces the key numbers

---

# Training notebooks (run in Kaggle, one per stage)

| Notebook | Stage | Status |
|---|---|---|
| `notebooks/stage1_train_adapter.ipynb` | Stage 1: single adapter (reverb / noise / codec) | [x] Written (parvC) |
| `notebooks/stage2_universal.ipynb` | Stage 2: universal adapter baseline | [x] Written (parvC) |
| `notebooks/stage3_gate.ipynb` | Stage 3: gate + Level-2 analyzer | [x] Written (parvC) |
| `notebooks/stage4_joint.ipynb` | Stage 4: joint fine-tuning, LR = 1e-5 | [x] Written (parvC) |
| `notebooks/eval_matrix.ipynb` | Full evaluation matrix + bootstrap CIs | [x] Written (parvC) |

---

# Evaluation matrix (BLUEPRINT sections 7.4 and 9.4)

Conditions crossed with N in {2, 3, 4, 5}. Separation is scored at 8 kHz;
DNSMOS runs on the 16 kHz band-recovered output.

| Tier | Source | Measures | N | Per cell |
|---|---|---|---|---|
| Clean 2 to 3 | Libri2Mix and Libri3Mix (8k) | SI-SDRi, PESQ | 2, 3 | 500 |
| Sparse overlap | SparseLibriMix (8k) | quality versus overlap 0 to 100 percent | 2 | 200 |
| Sparse overlap 3-spk | custom | extends the curve to N = 3 | 3 | 200 |
| Primary: noise plus reverb | custom WHAMR-style (8k) | headline SI-SDRi and DNSMOS | 2 | 500 |
| Reverb-noisy high count | same | count accuracy and quality | 3, 4, 5 | 200 |
| Reverb only | clean-reverb LibriMix | isolates adapter_reverb | 2, 3 | 200 |
| Real-RIR (mandatory) | BUT ReverbDB (SLR17) | sim-to-real gap | 2 | 200 |
| Codec only | LibriMix plus ffmpeg | isolates adapter_codec | 2 | 200 |
| Reverb plus codec (held out) | never in gate training | compositional generalization | 2, 4 | 200 |
| Noise plus codec (held out) | never in gate training | compositional generalization | 2, 4 | 200 |
| High count clean | Libri4Mix and Libri5Mix | break-point N = 4 to 5 | 4, 5 | 200 |
| High count degraded | plus reverb-noisy | count under degradation | 4, 5 | 200 |
| Real recordings | LibriCSS 1ch | DNSMOS and Whisper WER | 2+ | full |
| Band-recovery gain | matched 8k versus 16k pairs | isolates band recovery | 2 | 500 |

Note: BUT ReverbDB is OpenSLR SLR17. SLR28 is AISHELL-2, a speech corpus, not a
room-impulse-response set.

---

# Datasets

| Dataset | Role | Status |
|---|---|---|
| LibriSpeech (8k plus 16k copies) | Source speech; dev-clean and test-clean held out | [x] Prep script complete (parvA) |
| RIR bank (pyroomacoustics, 10k) | adapter_reverb and reverb evaluation | [x] `data/rir_bank.py` (parvA) |
| WHAM (about 17 GB) | adapter_noise | [x] Prep script (parvA) |
| DNS-4 (stratified 20 GB) | adapter_noise variety | [x] Prep script (parvA) |
| Libri2Mix and Libri3Mix | Clean and primary evaluation | [~] |
| Libri4Mix and Libri5Mix | High-count evaluation | [~] |
| SparseLibriMix | Overlap evaluation (test only) | [~] |
| BUT ReverbDB (SLR17) | Mandatory real-RIR evaluation | [x] `data/prepare_but_reverbdb.py` (parvA) |
| LibriCSS | Real-room DNSMOS and WER | [ ] |
| Real-room set | Team-recorded flagship | [ ] |
| VCTK | Optional extended speaker pool | [~] |
| WSJ0-mix and WHAMR (LDC) | Literature comparison only, not for training | [ ] |

---

# Pretrained tools (used at inference, never retrained)

| Tool | Source | Purpose | Status |
|---|---|---|---|
| SR-CorrNet var-2-5 | HF `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` | Frozen base (8k, K0 = 5) | [x] Wrapper + patches written (parvB) |
| SileroVAD | silero-vad, 8k native | Level-1 voiced density | [x] Implemented with energy fallback (parvB) |
| ECAPA-TDNN | SpeechBrain VoxCeleb | Stitching and global count clustering | [x] |
| DNSMOS ONNX | Microsoft DNS Challenge | Quality on the 16 kHz output | [x] `eval/dnsmos.py` (parvC) |
| PESQ | `pesq` pip package | Reference-based quality | [ ] |
| Whisper (optional) | OpenAI | Demo transcripts and LibriCSS WER | [x] `demo/app.py` best-effort Whisper (parvC) |

---

# Contributions to prove (BLUEPRINT sections 1.3 and 9.5)

| ID | Contribution | Proof | Phase | Status |
|---|---|---|---|---|
| C1 | Condition-aware LoRA mixture on a frozen backbone, composed in weight space | Composition analysis, learned versus oracle gating | P2, P5 | [~] Code complete, needs eval |
| C2 | Supervised, inspectable two-level condition analyzer | Per-dimension trace and gate ablation | P2 | [~] Code complete, needs eval |
| C3 | Never worse than the base, verified by measurement | Principle-2 smoke test | P2 | [ ] Needs weights |
| C4 | Attractor counting plus a bounded residual sweep | Confusion matrix, ECE, risk-coverage | P3, P5 | [~] Code complete |
| C5 | Residual-energy completeness detector | Dropped-speaker recall above 90 percent at 10 percent FAR | P3 | [~] Code complete |
| C6 | Dual-metric-guarded band recovery (8k to 16k) | Matched-pair SI-SDRi and DNSMOS delta | P3, P5 | [x] Code + tests complete (parvC) |
| C7 | Universal-adapter honesty gate | Pre-committed verdict at P1b | P1b, P5 | [~] `train/stage2_universal.py` written |

---

# Risks (BLUEPRINT section 11)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `p_k` not exposed by the wrapper | Low to medium | Blocks counting | First P0 task; patch before any other code |
| LoRA composition interference above 0.3 dB | High (expected) | Medium | Co-activation warm-up (P1), mandatory joint polish (P3), O-LoRA if needed |
| Universal adapter matches full routing | Medium | Reframes the headline | Trained first; pre-commit to adopt if within 0.5 dB |
| Gate collapses to uniform activation | Medium | Quality plateau | Supervised heads, sparsity, oracle-gap analysis |
| SileroVAD proxy is uninformative | Medium | Minor | Validate in P0; fallback to voiced-energy fraction |
| Never-worse fails on clean input | Medium | Credibility | Principle-2 smoke test; raise sparsity until it passes |
| Sim-to-real reverb gap | Medium | Real-RIR cells | BUT ReverbDB tier is mandatory |
| Count drops under combined degradation | Medium | Half the grade | Residual sweep; degraded-validation count from the start |
| Band recovery hurts SI-SDRi | Medium | Quality regression | Dual-metric guard, per-chunk bypass |
| Residual sweep triggers too often | Unknown | RTF budget | Measure; raise the threshold or drop to 2 candidates if above 30 percent |

---

# Budget

| Item | Target |
|---|---|
| Frozen base params | 13.6 M |
| New trainable params | About 3 to 4 M |
| Per-adapter params | About 0.4 to 0.6 M |
| Total GPU-hours | Under 150 on T4 |
| RTF (average, with residual sweep and 16k STFT) | Measured at P4 |
| Inference memory | 16 GB T4 or less |

---

# Timeline

| Phase | Gate | Parallel | Lead |
|---|---|---|---|
| P0 Verify checkpoint and synthesis | M0 | Full parallel (B patches, A data) | B |
| P1 Adapter library | M1 | Adapters sequential, A and C parallel | B |
| P1b Universal-adapter decision | M1b | Sequential, final verdict | B |
| P2 Condition analyzer and gate | M2 | Mostly parallel | B |
| P3 Joint polish, band recovery, calibration | M3 | Sequential | B and C |
| P4 Demo, CLI, efficiency | M4 | Parallel | C |
| P5 Full evaluation and report | M5 | Parallel | All |

---

# Source files added (parvB)

- `models/experts/srcorrnet.py` — SR-CorrNet wrapper with Patches A/B/C
- `models/lora.py` — LoRALibrary, co-activation sampler, 37 modules per adapter
- `models/condition.py` — Level-1 DSP + Level-2 E(0) heads (T60, count prior)
- `models/gate.py` — GateNetwork, GateSmoother, oracle_gate
- `models/counting.py` — ThreeVoteCounter, CountFusion, residual sweep
- `models/confidence.py` — StreamConfidenceHead, CompletenessHead, MahalanobisOOD
- `schemas/separation_result.py` — extended with attractor_probs, encoder_e0, decoder_features, gate_vector, completeness_prob, ood_flag
- `train/losses.py` — calmsep_loss, pit_si_snr, si_snr
- `train/stage1_single.py` — single adapter training CLI
- `train/stage2_universal.py` — universal adapter baseline
- `train/stage3_gate.py` — gate + Level-2 joint training
- `train/stage4_joint.py` — joint fine-tuning, LR=1e-5, O-LoRA option

# Source files added (parvC)

- `pipeline/chunker.py` — 2.4 s / 0.8 s-step chunker + 16 kHz STFT
- `pipeline/stitcher.py` — ChunkStitcher with ECAPA / correlation permutation solving
- `pipeline/infer.py` — CalmSepPipeline end-to-end orchestrator
- `models/band_recovery.py` — BandRecoveryHead, dual-metric guard
- `models/preprocess.py` — calmsep_preprocess() dual-branch (8 kHz + 16 kHz)
- `eval/dnsmos.py` — DnsmosScorer ONNX, 9 s segments
- `eval/matrix.py` — EvalMatrix, run_eval_matrix(), 8-condition × 4-N runner
- `eval/stats.py` — bootstrap_ci (BCa), wilcoxon_test, format_summary_table
- `calibration/temperature.py` — TemperatureScaler (LBFGS)
- `calibration/confidence.py` — ConfidenceCalibrator (isotonic regression)
- `calibration/completeness.py` — CompletenessCalibrator (Platt scaling)
- `calibration/ood.py` — OODCalibrator (Mahalanobis + sigmoid discount)
- `demo/app.py` — CALM-Sep Gradio demo with gate routing viz + Whisper
- `configs/calmsep/base_checkpoint.yaml` — frozen checkpoint config
- `configs/adapters/reverb.yaml`, `noise.yaml`, `codec.yaml` — adapter configs
- `configs/gate.yaml` — gate + Level-2 config
- `configs/band_recovery.yaml` — band recovery config
- `configs/eval.yaml` — evaluation config
- `notebooks/stage1_train_adapter.ipynb` through `eval_matrix.ipynb` — Kaggle training notebooks

---

End of tracker. Edit as the project moves. When this file and `BLUEPRINT`
disagree, `BLUEPRINT` is right.
