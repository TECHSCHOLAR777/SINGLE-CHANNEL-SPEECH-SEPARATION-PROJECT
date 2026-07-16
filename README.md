# CALM-Sep

**Condition-Aware LoRA Mixture for Multi-Speaker Speech Separation**

This file is the project task tracker. It is the single place to see what is
done, what is in progress, and what is left. The full design lives in
`BLUEPRINT`, which is the source of truth. When this tracker and `BLUEPRINT`
disagree, `BLUEPRINT` wins.

Last updated: 2026-07-17 (branch `suryansh` — audited end-to-end)

**Branch:** `suryansh` · **PR:** https://github.com/TECHSCHOLAR777/SINGLE-CHANNEL-SPEECH-SEPARATION-PROJECT/pull/new/suryansh

---

## Introspection verdict (2026-07-17)

| Layer | Status |
|---|---|
| **Codebase (P0–P5 modules, configs, CLI, demo, tests)** | Complete on `suryansh` |
| **CPU smoke tests** | Green for CALM-Sep modules (mock / no-checkpoint path) |
| **Training weights (adapters, gate, band recovery, calibrators)** | Not produced — notebooks/scripts ready; **you train on GPU** |
| **Live M0–M5 measurement gates** | Pending checkpoint download + your training + real eval audio |
| **Large corpora on disk (LibriSpeech, WHAM, DNS-4, RIR wavs)** | Prep scripts ready; downloads are local/user steps |

Honest rule used below: `[x]` = code + CPU tests exist on this branch. Training-weight
and live-measurement tasks stay `[~]` or `[ ]` until you run them.

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

| Marker | Meaning |
|---|---|
| `[x]` | Code deliverable exists, has CPU tests/smoke, on `suryansh` |
| `[~]` | Code scaffold ready; needs GPU training, data download, or live measurement |
| `[ ]` | Not started / blocked on prior training numbers |

- **GATE** rows need measured numbers (Wilcoxon, ECE, SI-SDRi tables). They stay
  open until you train and evaluate — that is intentional.

---

## Progress at a glance

| Milestone | Phase | Code | Weights / live gate |
|---|---|---|---|
| M0 | Checkpoint + synthesis | Done | Live attractor/corpus numbers need downloaded checkpoint |
| M1 | Adapter library | Done | Untrained — run `notebooks/P1_*.ipynb` |
| M1b | Universal adapter | Done | Untrained — log verdict in `docs/decisions.md` |
| M2 | Condition + gate | Done | Untrained — then Principle-2 live test |
| M3 | Polish + band recovery + calibration | Done | Untrained |
| M4 | Demo, CLI, RTF | Done | Demo smoke via `--mock`; real audio after weights |
| M5 | Full eval + report | Runner code done | Matrix numbers after training |

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
- [x] Extend `SeparationResult` with `p_k`, gate vector, completeness, OOD flag
- [x] Header docstring on every module
- [x] `docs/decisions.md` updated for each architecture choice
- [x] Config hash helpers (`utils/hashing.py`) wired into train/CLI artifacts
- [x] Every mechanism has an off switch on `CalmSepEngine`; baselines via `eval/baselines.py`

**Data split discipline (BLUEPRINT section 7.5)**

- [~] Three-way holdout: (1) speaker holdout, dev-clean and test-clean never in
  training (enforced by the mixer); (2) condition-combination holdout, reverb
  plus codec and noise plus codec never in gate or joint training; (3) severity
  holdout, T60 above 0.9 s and SNR below -4 dB kept rare in training. Only (1)
  is enforced so far.

---

# Phase P0: Verify checkpoint and build synthesis pipeline

**Goal:** The frozen checkpoint loads and exposes its internal signals
(`p_k`, `E(0)`, decoder-stage features). The 8 kHz synthesis pipeline produces
labelled mixtures. All evaluation sets are generated once, seeded, and hashed.

**No GPU hours in this phase. Inference only.**

## SEQUENTIAL: Dev B, checkpoint and wrapper (blocking)

| ID | Task | Deliverable | Status |
|---|---|---|---|
| P0-B1 | Download and verify the frozen checkpoint; confirm the YAML constants (sr 8000, max_n_spks 5, N_Enc 2, N_Dec 4, d_model 128) | `scripts/download_checkpoint.py` + `configs/base_checkpoint.yaml` | [x] code; [~] SHA fill after download |
| P0-B2 | Patch A: expose `p_k` | `models/srcorrnet/` wrapper | [x] |
| P0-B3 | Patch B: E(0) hook | encoder forward hook | [x] |
| P0-B4 | Patch C: decoder-stage hooks | `dec_block` hooks | [x] |
| P0-B5 | Expose `prob_thres` | config + wrapper | [x] |
| P0-B6 | `attractor_test.py` | mock contract green; live N=2..5 skips without checkpoint | [x] code; [~] live gate |
| P0-B7 | Corpus-transfer baseline | `scripts/corpus_transfer_baseline.py` + `models/baseline_runner.py` | [x] code; [~] log number |

## PARALLEL: Dev A, synthesis and evaluation sets

| ID | Task | Deliverable | Status |
|---|---|---|---|
| P0-A1 | Dynamic 8 kHz mixer | `data/calmsep_mixer.py` + tests | [x] |
| P0-A2 | LibriSpeech 8k + 16k copies | `data/prepare_librispeech_8k.py` | [x] script; [~] download |
| P0-A3 | RIR bank 10k | `data/rir_bank.py` → `data/rirs/` | [x] code; [~] generate cache |
| P0-A4 | WHAM + DNS-4 staging | `prepare_wham.py`, `prepare_dns4.py` | [x] scripts; [~] download |
| P0-A5 | Codec transforms | `data/codec_augmentation.py` | [x] |
| P0-A6 | Fixed evaluation matrix manifests + hashes | `data/fixed_eval/*.jsonl` + `.sha256` | [x] |
| P0-A7 | Wet reverb references | `data/degradations.make_wet_reference` | [x] |

## PARALLEL: Dev C, evaluation and condition-input tooling

| ID | Task | Deliverable | Status |
|---|---|---|---|
| P0-C1 | Cardinality-aware SI-SDR / SI-SDRi + hallucination −1 dB | `eval/metrics.py` | [x] |
| P0-C2 | Dual-rate STFT (8k 128/64 + 16k 256/128) | `preprocess_calmsep` | [x] |
| P0-C3 | SileroVAD + STFT fallback | `data/vad_features.py`, `docs/vad_validation.md` | [x] fallback selected |
| P0-C4 | DNSMOS ONNX path | `eval/dnsmos.py` | [x] code; [~] onnx file |
| P0-C5 | Config loader and logging | `utils/config.py`, `utils/logging.py` | [x] |

## GATE M0

- [~] `attractor_test.py` live N-accuracy (needs checkpoint)
- [x] Patches A/B/C + `e0_hook_test.py` (mock/live skip path)
- [x] Eval manifests seeded + hashed under `data/fixed_eval/`
- [~] Corpus-transfer SI-SDRi number (run script after download)
- [x] VAD fallback selected (STFT); Silero opt-in
- [x] Zero GPU hours spent on codebase work
- [ ] Milestone session done

---

# Phase P1: Adapter library (Stage 1)

**Goal:** Three LoRA adapters (reverb, noise, codec) trained one at a time with
co-activation warm-up, and the cross-interference matrix measured.

## SEQUENTIAL: Dev B, critical path

| ID | Task | Deliverable | Status |
|---|---|---|---|
| P1-B1 | `models/lora.py` parallel-branch LoRA, 17 targets, co-activation | library + `tests/test_lora.py` | [x] |
| P1-B2 | Freeze-and-attach harness | `train/lora_harness.py` | [x] |
| P1-B3 | Loss wiring PIT SI-SDR + BCE (+ engine PIT when installed) | `train/lora_harness.py` | [x] |
| P1-B4 | Train `adapter_noise` | `notebooks/P1_train_adapter_noise.ipynb` + CLI | [~] untrained |
| P1-B5 | Train `adapter_reverb` | notebook + CLI | [~] untrained |
| P1-B6 | Train `adapter_codec` | notebook + CLI | [~] untrained |

## PARALLEL: Dev A and Dev C

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P1-A1 | Noisy training mixtures | A | degradations + WHAM/DNS prep | [x] code; [~] data |
| P1-A2 | Reverb training mixtures | A | RIR + wet refs | [x] code; [~] data |
| P1-A3 | Codec training mixtures | A | codec transforms | [x] |
| P1-C1 | Cross-interference matrix | C | `eval/interference.py` | [x] code; [~] numbers |
| P1-C2 | O-LoRA penalty | C | `orthogonal_penalty` in `models/lora.py` | [x] |

## GATE M1

- [ ] Adapter SI-SDRi gains (needs training)
- [ ] Clean Libri2Mix no-degrade (needs training)
- [ ] Interference matrix numbers (needs training)
- [ ] Milestone session done

---

# Phase P1b: Universal-adapter decision (Stage 2)

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P1b-B1 | Train universal adapter | B | notebook + `configs/adapters/universal.yaml` | [~] untrained |
| P1b-C1 | Evaluate + log verdict | C | slot in `docs/decisions.md` (PENDING TRAINING) | [~] |

## GATE M1b

- [ ] Verdict logged before gate training (pre-committed rule documented)
- [ ] Adopt/continue decision after measured comparison

---

# Phase P2: Condition analyzer and gate (Stage 3)

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P2-B1 | Level-1 DSP analyzer | B | `models/condition.py` | [x] |
| P2-B2 | Level-2 E(0) heads | B | T60 + count prior | [x] |
| P2-B3 | Gate MLP | B | `models/gate.py` | [x] |
| P2-B4 | Stage-3 training | B | `train/train_gate.py` + notebook | [~] untrained |
| P2-C1 | Per-layer vs per-adapter ablation | C | `eval/ablation_gate.py` | [x] code; [~] result |
| P2-C2 | Principle-2 smoke test | C | `tests/principle2_test.py` | [x] API; [~] live |

## GATE M2

- [ ] Learned gating beats best single adapter (needs training)
- [ ] Principle-2 live pass (needs training)
- [ ] Holdout cells do not collapse (needs training)
- [x] Condition heads inspectable (`ConditionVector`)
- [ ] Milestone session done

---

# Phase P3: Joint polish, band recovery, and calibration (Stage 4)

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P3-B1 | Joint polish | B | `train/train_joint_polish.py` + notebook | [~] untrained |
| P3-B2 | Counting subsystem | B | `models/counting.py` | [x] |
| P3-C1 | Band-recovery head | C | `models/band_recovery.py` + `train/train_band_recovery.py` | [x] code; [~] weights |
| P3-C2 | Dual-metric guard | C | `apply_band_recovery` | [x] |
| P3-B3 | Confidence + completeness + OOD | B | `models/confidence.py` | [x] |
| P3-C3 | Calibration package | C | `calibration/` + `train/calibrate.py` | [x] code; [~] fit on held-out |

## GATE M3

- [ ] Joint polish complete (needs training)
- [ ] Confusion / ECE numbers (needs training)
- [ ] Dropped-speaker recall (needs training)
- [ ] Dual-metric guard validated on real val (needs training)
- [ ] Residual-sweep trigger frequency (needs training)
- [ ] Milestone session done

---

# Phase P4: Demo, CLI, and efficiency

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P4-C1 | Chunker 2.4 / 0.8 @ 8 kHz + 16 kHz STFT | C | `pipeline/chunker.py` | [x] |
| P4-C2 | Stitcher xcorr + ECAPA tie-break + clustering, max 5 tracks | C | `pipeline/stitcher.py` | [x] |
| P4-C3 | Infer order §6.2 | C | `pipeline/infer.py` | [x] |
| P4-A1 | CLI + reproducibility bundle | A | `scripts/calmsep_infer.py` | [x] |
| P4-C4 | Gradio demo + specs + gates + Whisper optional | C | `demo/app.py` | [x] |
| P4-B1 | RTF report script | B | `scripts/rtf_report.py` | [x] code; [~] real hardware table |

## GATE M4

- [~] Demo E2E on mock (`python -m demo.app --mock`); real recording after weights
- [~] RTF structure written by script; fill on target GPU
- [x] Post-processors have off switches in engine config
- [ ] Milestone session done

---

# Phase P5: Full evaluation and report

| ID | Task | Owner | Deliverable | Status |
|---|---|---|---|---|
| P5-C1 | Matrix + stats (bootstrap, Wilcoxon, ECE) + PESQ helper | C | `eval/matrix.py`, `stats.py`, `pesq_metric.py` | [x] code; [~] fill |
| P5-C2 | Primary-benchmark headline | C | runner ready | [~] number |
| P5-A1 | BUT ReverbDB real-RIR | A | `prepare_but_reverbdb.py` + eval tier | [x] script; [~] table |
| P5-A2 | Real recordings / LibriCSS | A | prep + matrix tier | [~] |
| P5-B1 | Break-point + band-recovery curves | B | `eval/curves.py` | [x] code; [~] plots |
| P5-ALL1 | Final report + bundle | All | `reports/` + CLI bundle | [~] |

## Mandatory baselines (BLUEPRINT section 9.6)

Code: `eval/baselines.py` (mock dry-run). Numbers after training:

- [~] Frozen base alone
- [~] Universal adapter
- [~] Uniform blend, no gate
- [~] Oracle gating
- [~] Frozen base + band recovery

## GATE M5

- [ ] Full matrix filled with intervals
- [ ] All five baselines reported
- [ ] Eight headline analyses done
- [ ] Universal-adapter verdict stated
- [ ] Reproducibility bundle reproduces key numbers

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
| LibriSpeech (8k plus 16k copies) | Source speech; held out speakers | [x] prep script; [~] download |
| RIR bank (pyroomacoustics, 10k) | adapter_reverb | [x] generator; [~] cache |
| WHAM (about 17 GB) | adapter_noise | [x] prep; [~] download |
| DNS-4 (stratified 20 GB) | adapter_noise variety | [x] prep; [~] download |
| Libri2Mix / Libri3Mix | Clean / primary eval | [x] prep; [~] download |
| Libri4Mix / Libri5Mix | High-count eval | [x] prep; [~] download |
| SparseLibriMix | Overlap eval | [x] prep; [~] download |
| BUT ReverbDB (SLR17) | Real-RIR eval | [x] prep; [~] download |
| LibriCSS | Real-room DNSMOS / WER | [~] |
| Real-room set | Team-recorded | [ ] |
| VCTK | Optional pool | [x] prep; [~] download |
| WSJ0-mix / WHAMR (LDC) | Literature only | [ ] out of scope for training |

---

# Pretrained tools (used at inference, never retrained)

| Tool | Source | Purpose | Status |
|---|---|---|---|
| SR-CorrNet var-2-5 | HF hub | Frozen base | [x] wrapper; [~] download |
| SileroVAD | silero-vad | Level-1 voiced density | [x] opt-in; STFT fallback default |
| ECAPA-TDNN | SpeechBrain | Stitching / clustering | [x] |
| DNSMOS ONNX | Microsoft | 16 kHz quality | [x] code; [~] onnx file |
| PESQ | `pesq` | Reference quality | [x] `eval/pesq_metric.py` |
| Whisper (optional) | OpenAI | Demo / WER | [x] optional import in demo |

---

# Contributions to prove (BLUEPRINT sections 1.3 and 9.5)

| ID | Contribution | Proof | Phase | Status |
|---|---|---|---|---|
| C1 | Condition-aware LoRA mixture on a frozen backbone, composed in weight space | Composition analysis, learned versus oracle gating | P2, P5 | [ ] |
| C2 | Supervised, inspectable two-level condition analyzer | Per-dimension trace and gate ablation | P2 | [ ] |
| C3 | Never worse than the base, verified by measurement | Principle-2 smoke test | P2 | [ ] |
| C4 | Attractor counting plus a bounded residual sweep | Confusion matrix, ECE, risk-coverage | P3, P5 | [ ] |
| C5 | Residual-energy completeness detector | Dropped-speaker recall above 90 percent at 10 percent FAR | P3 | [ ] |
| C6 | Dual-metric-guarded band recovery (8k to 16k) | Matched-pair SI-SDRi and DNSMOS delta | P3, P5 | [ ] |
| C7 | Universal-adapter honesty gate | Pre-committed verdict at P1b | P1b, P5 | [ ] |

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

# Source files (now present on `suryansh`)

- [x] `models/srcorrnet/` — patches A/B/C
- [x] `models/lora.py`, `condition.py`, `gate.py`, `counting.py`, `confidence.py`, `band_recovery.py`
- [x] `pipeline/chunker.py`, `stitcher.py`, `infer.py`
- [x] `eval/matrix.py`, `stats.py`, `interference.py`, `baselines.py`, `curves.py`, `pesq_metric.py`, `ablation_gate.py`
- [x] `calibration/` (+ fit/load/hash)
- [x] `data/synthesis/`, `data/fixed_eval/` (manifests + hashes), `data/rirs/` (dir)
- [x] `tests/attractor_test.py`, `e0_hook_test.py`, `principle2_test.py`, `smoke_test.py` + module tests
- [x] configs: `base_checkpoint.yaml`, `adapters/*`, `gate.yaml`, `band_recovery.yaml`, `eval.yaml`
- [x] notebooks: `P1_*`, `P1b_*`, `P2_*`, `P3_*` (untrained)
- [x] CLI: `scripts/calmsep_infer.py`, `rtf_report.py`, `generate_fixed_eval.py`, `corpus_transfer_baseline.py`

## Your next commands (training — not run by the agent)

```bash
python scripts/download_checkpoint.py
python scripts/generate_fixed_eval.py   # already committed; re-run if needed
python -m train.train_adapter --adapter noise --config configs/adapters/noise.yaml
# then reverb, codec → P1b universal → train_gate → joint_polish → train_band_recovery → calibrate
python -m demo.app --mock
python -m scripts.calmsep_infer separate mix.wav --out out/ --mock
```

---

End of tracker. When this file and `BLUEPRINT` disagree, `BLUEPRINT` is right.
