# CoRAL-Sep: Condition-Routed Adapter Library for Speech Separation

## Master Project Blueprint and Source of Truth

This document is the complete and self-contained description of the CoRAL-Sep project. It defines the problem, the scientific and engineering ideas behind the solution, the full system architecture, the data plan, the training plan, the evaluation plan, the development roadmap, the risks, and the alternatives that were considered and rejected. A reader who has never seen any earlier material on this project should be able to understand it fully, and a developer or an autonomous coding agent should be able to build the system from this document alone. Every mechanism in this document carries either a concrete default value or an explicit procedure for choosing one.

> **Fixed constraints (never revisited):**
> - **Base checkpoint:** `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk`, downloaded once, frozen forever. The backbone is never fine-tuned, never modified, and no Stage 0 training is performed.
> - **Speaker count:** N = 2–5 only. There is no 6-or-7-speaker regime in this project.
> - **Sample rate:** 8 kHz, locked by the checkpoint (128-sample STFT window, 64-sample hop). The system operates entirely at 8 kHz internally.
> - **Quality patch:** Band recovery (Option B) is the fixed quality path for bandwidth extension to 16 kHz output. There is no Option A (backbone retraining) and no Option C.
> - **New trainable parameters:** Condition analyzer, gate MLP, counting fusion, calibration heads, and the three LoRA adapter libraries, approximately 3–4 M parameters total against a 13.6 M parameter frozen base.
> - **Compute:** Not a problem. No base training. The largest single training run is one adapter (~0.4–0.8 M parameters) for 20–40 GPU-hours on free-tier hardware.

---

## Table of Contents

1. Project Overview and Objective
2. Background Concepts, Explained Simply
3. The Foundation Model: SR-CorrNet
4. Design Principles
5. System Architecture
6. Inference Pipeline
7. Data Plan
8. Training Plan
9. Evaluation Plan
10. Development Roadmap
11. Risk Register
12. Alternatives Considered and Inspirations
13. Repository Structure and Engineering Practices
14. Glossary
15. Implementation Reference (Repo Audit)

---

## 1. Project Overview and Objective

### 1.1 The task

The system receives a single-channel (mono) audio recording in which two to five people speak at the same time. The number of concurrent speakers is not given to the system. The recording may additionally be reverberant, noisy, or codec-degraded.

The system must return:

1. An estimated number of speakers `N_hat` ∈ {2, 3, 4, 5}, and one clean waveform per estimated speaker.
2. A calibrated confidence score for each returned waveform.
3. A completeness probability: a single calibrated number expressing how likely it is that no speaker was missed.

### 1.2 What the project is graded on

The evaluation brief grades exactly two axes, at multiple difficulty levels:

1. **Speaker count accuracy.** Did the system return the correct number of speakers?
2. **Separation quality.** Does each returned voice sound clean and isolated?

Everything else is a bonus. This ordering governs every design decision: whenever a mechanism could improve efficiency or elegance at any risk to count accuracy or separation quality, the mechanism is guarded or removed.

### 1.3 The central idea in one paragraph

The project takes one strong pretrained speech separation network, SR-CorrNet var-2-5, freezes it completely, and teaches it to handle adverse conditions through a library of three small plug-in adapters called LoRA modules. Each adapter specializes in one condition: reverberation, background noise, or codec artifacts. A lightweight two-level condition analyzer inspects each chunk using raw signal statistics and neural features, estimates how much of each condition is present, and blends the adapters into the frozen network in proportion to those strengths. The blending happens inside the weight matrices before any audio is produced, so the system always runs one forward pass and always emits one coherent set of output voices. Adapters are trained with co-activation warm-up so that they compose cleanly, and they are jointly fine-tuned at Stage 4 on compound-condition data, this joint stage is mandatory. A band recovery head extends the 0–4 kHz 8 kHz output to 0–8 kHz audio at 16 kHz sample rate for perceptual quality. A dedicated residual-energy completeness detector, bounded to three sweep candidates, guards against missed speakers.

### 1.4 Why this shape of solution

Real recordings rarely contain a single, isolated difficulty. Two families of solutions were considered and rejected:

- A bank of separate specialist models with a switch fails because conditions co-occur; combining audio outputs across models is ill-posed.
- A single model fine-tuned on everything tends to average its behavior across conditions.

The adapter-mixture design takes the good part of both: specialist capacity per condition, and a single shared backbone so there is never any output-merging problem. The backbone never changes, all routing decisions produce streams from the same split, with the same speaker identities.

### 1.5 Project deliverables

1. **A working system**: documented codebase, CLI entry point, Gradio demo.
2. **A trained model bundle**: frozen base checkpoint, adapter library (3 adapters), condition analyzer, gate network, band recovery head, calibration parameters, all versioned and hashed.
3. **An evaluation report**: the full measurement matrix in Section 9 with statistical error bars.
4. **A demonstration**: Gradio page with optional Whisper transcripts and condition routing visualization.

---

## 2. Background Concepts, Explained Simply

### 2.1 Speech separation and why unknown speaker count is the hard part

Speech separation turns one mixed recording into several clean recordings, one per speaker. When the speaker count is unknown, two failure modes appear: too few streams (missed speaker, invisible if returned streams individually sound clean) or too many (hallucinated speaker). A missed speaker is the most dangerous failure because nothing about the returned audio reveals it. Section 5.8 builds a dedicated detector for this failure.

### 2.2 The STFT and time-frequency processing

The short-time Fourier transform (STFT) cuts audio into overlapping frames and computes the frequency content of each frame. Reverberation appears as energy smearing across neighboring time frames; codec compression appears as missing or quantized frequency regions. Both are recognizable in this domain, which is why the condition analyzer taps the STFT domain.

### 2.3 SI-SDR, the primary quality metric

Scale-invariant signal-to-distortion ratio (SI-SDR), measured in decibels (dB), compares an estimated waveform against the true clean waveform, scale-invariantly. SI-SDRi is the improvement over the unprocessed mixture. Typical strong systems achieve 15–24 dB SI-SDRi on standard benchmarks.

### 2.4 Perceptual metrics: DNSMOS and PESQ

DNSMOS is a neural network predicting mean opinion score; it needs no reference and expects 16 kHz input. PESQ requires a reference. Since the system operates at 8 kHz internally, the band recovery output at 16 kHz is used for DNSMOS scoring. Reporting SI-SDRi, DNSMOS, and PESQ together protects against grading on a metric the system did not optimize.

### 2.5 Permutation invariant training (PIT) and cardinality-aware scoring

PIT tries all assignments of outputs to references during training and keeps the best one. Cardinality-aware scoring handles the case where estimated and true counts differ (Section 9.2).

### 2.6 LoRA: low-rank adaptation, composability, and its actual limitations

LoRA adds a correction `g * B(Ax)` to a frozen weight matrix `W0`. Multiple LoRA corrections on the same layer add linearly: `y = W0 x + g1 B1 A1 x + g2 B2 A2 x`. This arithmetic is correct and makes composition possible.

**What this does not guarantee:** three adapters independently trained on separate conditions will not automatically compose cleanly when co-activated. Each was trained with the others absent; their corrections interact in a direction that was never optimized. This is the expected case during inference on real recordings, not an edge case.

Three structural properties remain genuine:

1. **Identity fallback at g=0.** When all gate values are exactly zero, the network is mathematically identical to the frozen base. This is structural.
2. **Linear composition.** Blending happens in weight space before any forward pass, so there is never an output-level merging problem.
3. **Tiny parameter cost.** All three adapters cost under 2.5 M parameters against a 13.6 M parameter frozen base.

**The composition problem is solved by design, not by hope:** Section 8.2 (Stage 1) requires co-activation warm-up during each adapter's individual training, and Section 8.5 (Stage 4) requires mandatory joint fine-tuning on compound-condition data.

### 2.7 Mixture of experts and condition-aware gating

This project routes continuously: the router outputs a strength in [0, 1.5] per adapter, because conditions are quantities not categories. A room has a specific T60; noise sits at a specific SNR. The gate values scale each adapter's correction to the measured strength of its condition.

### 2.8 Disentanglement and why the condition embedding must be supervised

The condition analyzer produces a vector where each dimension is supervised against a known target (SNR, T60, codec class, voiced-frame density). This supervision is mandatory because an end-to-end-only condition vector would find degenerate shortcuts, most commonly activating every adapter at medium strength for every input. Supervised dimensions are individually inspectable: a routing failure can be traced to a wrong T60 estimate.

### 2.9 Calibration, ECE, and temperature scaling

A confidence score is calibrated when it matches reality. Expected calibration error (ECE) measures the mismatch between stated confidence and actual accuracy. Temperature scaling is applied to every probability this system emits. Every calibration fit uses held-out validation data that never overlaps the evaluation set.

### 2.10 Chunked processing and stitching

Long recordings are processed in 2.4-second chunks stepped by 0.8 seconds. Chunking creates two obligations: routing decisions must not flip abruptly between adjacent chunks (handled by EMA smoothing in Section 5.5), and the global speaker count must be assembled from per-chunk counts (Section 6.4).

### 2.11 The 8 kHz constraint and band recovery

The frozen checkpoint operates at 8 kHz, which limits the audio band to 0–4 kHz. Fricative consonants and many speaker-discriminating formants live above 4 kHz. DNSMOS also expects 16 kHz input. The band recovery head (Section 5.9) extends each separated stream to 0–8 kHz audio at 16 kHz sample rate. This is a small convolutional head, not a second full model. The 8 kHz constraint is accepted and fixed; no retraining of the base is needed or planned.

---

## 3. The Foundation Model: SR-CorrNet

### 3.1 The correlation-to-filter idea

SR-CorrNet structures both input and output around signal physics:

- **Input: correlations.** For every time-frequency point, the model computes the correlation between that point and its neighborhood: surrounding time frames (2L+1) and surrounding frequency bins (2I+1). These spatio-spectro-temporal correlations give the network a direct, structured view of reverberation and spatial coherence. Raw correlations are normalized with SCOT-β (β=0.5) to suppress power-scale variation. A small convolutional module embeds these into a C-channel latent map `E(0)`.
- **Output: filters.** The network estimates a small complex-valued filter over the same neighborhood for each speaker, and the separated signal is obtained by applying that filter to the observed mixture.

The neighborhood size is set by the frozen checkpoint's config: `taps_freq=[1,1]`, `taps_frame=[1,1]`. The STFT is 128-sample window, 64-sample hop, at 8 kHz (65 frequency bins). These are immutable.

### 3.2 The SepRe architecture

```
Mixture → Unfold → Correlation module → E(0)
E(0) → TF-Encoder (B_E=2 blocks) → coarse separation features
→ Split module (attractor-based, decides K ∈ {2,3,4,5}) → K speaker streams
→ TF-Decoder (B_D=4 blocks, weight-shared + cross-speaker interaction)
→ Filter module → K complex filters → K separated signals
```

**TF-Encoder (`B_E=2` blocks):** Alternates processing along frequency and time axes. Each block: multi-head self-attention with RoPE, sandwiched between convolutional feed-forward networks with SwiGLU. Pre-norm residual connections.

**Split module (dynamic attractor, Section 3.3):** Divides encoder output into K speaker streams, determining K in the process.

**TF-Decoder (`B_D=4` blocks):** Reconstructs per-speaker features stage by stage. Weight-shared across speakers. After each shared stage, a speaker interaction module lets streams attend to each other. Intermediate decoder outputs are also the free confidence signals used by Section 5.8.

**Filter module:** Three parallel pointwise convolutions produce real part, imaginary part, and magnitude mask of the per-speaker filter. Applied to observed neighborhoods to produce each output.

### 3.3 The attractor-based dynamic split: K0=5

The dynamic split module determines speaker count inside the model. A stack of Transformer decoder blocks holds `K0+1` learnable query vectors. These cross-attend to the encoder output and produce `K0+1` attractor vectors. Each attractor yields an existence probability `p_k` via a linear layer and sigmoid.

**K0 specification:** `K0=5` is set in the frozen checkpoint (`max_n_spks: 5` in the YAML config). The learnable `spk_query` parameter has shape `(1, 7, 128)`, 7 slots where slot 0 is a residual/bookkeeping slot, slots 1–5 are the actual speaker existence slots, and slot 6 is an overflow slot. This shape is baked into the checkpoint weights and is never changed by any component of this project.

**`p_k` index convention:** `pres["probs"]` has shape `(1, 7)`. The active speaker existence probabilities are at indices 1 through 5 (`probs[0, 1:6]`). A speaker is considered active when its probability exceeds `prob_thres=0.5`. This threshold is **hardcoded** in `AttractorSplit.forward`, expose it as a configurable parameter before the calibration stage (Section 8.5).

**Batch size constraint:** `AttractorSplit.forward` contains `assert x.size(0) == 1` when `n_spks=None`. Unknown-speaker inference is strictly single-sample only. This cannot be changed without modifying the checkpoint's `AttractorSplit` logic.

**`p_k` is NOT returned by default inference APIs.** `process_waveform` and `process_stft` in `SSInference` silently drop `pres` inside `_single_pass_session`. Only `process_stft_chunk` currently returns `"pres"`. Section 15.1 documents the exact patch.

**Consequence:** Any integration that feeds a hardcoded count into the model bypasses `p_k` and must be repaired before anything else is built. The first engineering task of Phase 0 is a unit test asserting that `p_k` varies with true count on a fixture set with 2, 3, 4, and 5 speakers.

Published performance on clean benchmarks: 100.0 / 99.7 / 97.7 / 96.9 percent count accuracy; 24.8 / 24.4 / 21.9 / 19.9 dB SI-SDRi at N=2/3/4/5 respectively.

### 3.4 Why SR-CorrNet is the right sole foundation

| Benchmark | SR-CorrNet result | Strongest published competitor |
|---|---|---|
| WSJ0-2mix | 24.1 dB SI-SNRi (B, 13.6M params) | 24.2 dB (TF-Locoformer-L, 22.5M params) |
| WSJ0-2/3/4/5mix, unknown count | 24.8/24.4/21.9/19.9 dB; 100/99.7/97.7/96.9% count | SepTDA: 23.6/22.1/19.5/16.9 dB; count down to 82% at 5 spk |
| WHAMR! 1ch (noise+reverb) | 19.7 dB SI-SNRi | 18.5 dB (TF-Locoformer-M) |

### 3.5 Fixed checkpoint configuration

| Property | Value |
|---|---|
| Checkpoint ID | `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` |
| Architecture | SR-CorrNet-B (C=128, C_H=384, B_E=2, B_D=4) |
| Parameters | 13.6 M (frozen) |
| K0 (attractor slots) | 5 (`spk_query` shape: `(1, 7, 128)`) |
| Sampling rate | **8 kHz (locked)** |
| STFT window / hop | 128 / 64 samples (F=65 freq bins) |
| Max speakers | 5 |
| Freq bins (d_freq) | 65 (win/2 + 1) |
| Attention dim | 128 (d_model); fused qkv: `Linear(128, 384)`, out: `Linear(128, 128)` |
| Training data | WSJ0-{2,3,4,5}Mix |
| Status | Never modified |
| Local path after download | `sr_corrnet/checkpoints/SS/1ch_WSJ_var_2_5spk/model.pt` |
| Download command | `python sr_corrnet/export.py --download --variant SS --config 1ch_WSJ_var_2_5spk.yaml` |
| Checkpoint file format | Bare `OrderedDict` (no `model_state_dict` wrapper key), `from_pretrained` handles this |
| `load_state_dict` mode | `strict=False`, adding new LoRA params before loading is safe; they initialize from scratch |

---

## 4. Design Principles

Five principles govern every decision.

**1. One backbone, adapted, never arbitrated.** All specialization lives in adapters on a single frozen network. A shared backbone makes the stream-alignment problem structurally impossible.

**2. Never worse than the base, empirically verified, not just asserted.** When all gate values are exactly zero, the network equals the frozen base. This is structural. However, "gates are near-zero on clean input" is an empirical outcome that requires: (a) including clean audio in condition-analyzer training with zero-gate targets, (b) gate sparsity regularization (L1, weight 1e-3), and (c) a mandatory smoke test at end of Stage 3 comparing the full system on clean Libri2Mix against the frozen base alone. If the gap is negative, the sparsity weight is increased until it closes. Verified by measurement, not declared by design.

**3. Conditions are quantities, not categories.** All routing is continuous gating scaled by estimated strength, never a discrete expert switch.

**4. Every emitted probability is calibrated, and every internal representation is inspectable.** Count estimate, per-stream confidences, and completeness score all pass through temperature scaling fitted on held-out data. The condition representation is supervised dimension by dimension so that failures can be traced to a named misestimate.

**5. No claim without a measured number; the system's own headline ideas must survive their killer baselines.** The universal-adapter baseline is trained first (before the full routing system) as a calibration gate. If it matches learned gating within confidence intervals on the primary benchmark, the project adopts the simpler system and reports this honestly.

---

## 5. System Architecture

### 5.1 Overview and two-pass architecture (with circularity acknowledged)

```
Input audio (8 kHz) → Preprocessing → [Pass 1: frozen correlation module → E(0)]
                                               ↓
                               [Level-1 condition analysis: raw STFT DSP features
                                SNR estimate, codec estimate, voiced-frame density]
                                               ↓
                               [Level-2 condition analysis: pooled E(0)
                                Reverberation strength, speaker count prior]
                                               ↓
                               [Gate network → gate matrix g, EMA-smoothed]
                                               ↓
                    [Pass 2: full forward pass with adapters scaled by g]
                                               ↓
                    [Counting fusion reads p_k from attractor split]
                                               ↓
                    [Residual sweep if uncertain (max 3 candidates, clipped to [2,5])]
                                               ↓
                    [Band recovery: 0–4 kHz → 0–8 kHz at 16 kHz sample rate]
                                               ↓
                    [Guarded quality stage → confidence + completeness → outputs]
```

**Acknowledged circularity and its resolution:** The condition analyzer taps E(0), the output of the correlation module. The reverb, noise, and codec adapters all attach to the correlation module itself. This means Pass 1 reads pre-adaptation features to decide how to adapt. Under heavy codec degradation, pre-adaptation E(0) may be the least reliable input for condition estimation.

**Resolution:** The condition analyzer uses a two-level design. Level 1 operates exclusively on raw STFT statistics and never touches E(0); it provides the SNR estimate, codec estimate, and voiced-frame density, the three signal conditions where corruption is worst at the point of E(0) tapping. Level 2 refines the reverberation estimate and speaker count prior using pooled E(0), where reverberation structure survives SCOT-β normalization better than absolute power. Gate values for the reverb, noise, and codec adapters are driven primarily by Level 1. This breaks the circularity at the signal-condition adapters where it matters most.

### 5.2 Preprocessing

- Decode to mono, resample to **8 kHz** (the system's operating rate; band recovery runs after separation).
- RMS-normalize to a fixed target.
- Compute one shared STFT (window 128 samples, hop 64, Hann) per chunk. Reused by every component: base model, Level-1 DSP features, completeness residual.
- Compute one 16 kHz STFT of the mixture in parallel (for band recovery input only).

### 5.3 The LoRA adapter library

Three adapters covering the three signal degradation conditions. All three attach to the same module group: the attention projections in encoder and decoder blocks, and the filter head linear. Co-activation interference within this axis is expected and handled structurally by co-activation warm-up in Stage 1 and mandatory joint fine-tuning in Stage 4.

| Adapter | Attach points | Rank r | Trained on | What it learns |
|---|---|---|---|---|
| `adapter_reverb` | Encoder + decoder attn projections, filter head | 8 (attn), 4 (filter head) | Libri2–5Mix (8 kHz) + simulated RIRs, T60 0.2–1.0 s, wet references | Reading smeared inter-frame correlations; estimating longer effective filters |
| `adapter_noise` | Same as reverb | 8 / 4 | LibriMix (8 kHz) + WHAM! + DNS-4 (stratified 20 GB), SNR −6 to +10 dB | Separating speech correlation structure from noise floor |
| `adapter_codec` | Same as reverb | 8 / 4 | LibriMix (8 kHz) → ffmpeg: Opus 6–24 kbps, AAC 16–48 kbps, AMR-NB/WB | Compensating band loss and quantization artifacts at 8 kHz |

**Exact LoRA attachment points (verified from model code):**

All attention layers use a fused QKV projection `Linear(128, 384, bias=False)` and a separate output projection `Linear(128, 128, bias=False)`. These are the LoRA targets. The ConvFFN layers inside TF_Block use `Conv1d`, skip them or implement a Conv LoRA variant; they are not on the critical path.

```
# Encoder blocks (N_Enc=2):
model.enc_block[i].freq_block.block.sa.block.qkv               # Linear(128, 384), LoRA rank 8
model.enc_block[i].freq_block.block.sa.block.aggregate_heads[0] # Linear(128, 128), LoRA rank 8
model.enc_block[i].time_block.block.sa.block.qkv               # Linear(128, 384), LoRA rank 8
model.enc_block[i].time_block.block.sa.block.aggregate_heads[0] # Linear(128, 128), LoRA rank 8
# (i = 0, 1)

# Decoder blocks (N_Dec=4):
model.dec_block[i].freq_block.block.sa.block.qkv               # Linear(128, 384), LoRA rank 8
model.dec_block[i].freq_block.block.sa.block.aggregate_heads[0] # Linear(128, 128), LoRA rank 8
model.dec_block[i].time_block.block.sa.block.qkv               # Linear(128, 384), LoRA rank 8
model.dec_block[i].time_block.block.sa.block.aggregate_heads[0] # Linear(128, 128), LoRA rank 8
# (i = 0, 1, 2, 3)

# Cross-speaker blocks (N_Dec=4):
model.dec_cs[i].block.block['sa'].block.qkv                    # Linear(128, 384), LoRA rank 8
model.dec_cs[i].block.block['sa'].block.aggregate_heads[0]     # Linear(128, 128), LoRA rank 8
# (i = 0, 1, 2, 3)
# Note: dec_cs uses ModuleDict, so access via .block['sa'] not .block.sa

# Filter estimator head:
model.filter_estim.mask.net                                     # Linear(128, 27), LoRA rank 4
model.filter_estim_aux[i].mask.net                              # Linear(128, 27), LoRA rank 4, i=0..3
```

**Total LoRA target layers per adapter:** 4 (enc) + 8 (dec) + 4 (dec_cs) + 1 (filter head) = 17 Linear layers. Each rank-8 LoRA on `Linear(128, 384)` adds `128×8 + 8×384 = 4,096` parameters; each rank-8 on `Linear(128, 128)` adds `128×8 + 8×128 = 2,048` parameters. Approximately 0.4–0.6 M parameters per adapter.

**`get_correlation` is always `@torch.no_grad()`:** Gradients through the correlation computation step are always blocked by the model. However, LoRA corrections in the encoder's attention layers downstream of the encoder are still reachable by gradients, the no-grad boundary applies only to the correlation arithmetic itself inside `Encoder.forward`.

**Co-activation warm-up (required in Stage 1):** Each adapter trains with the other two randomly activated, gates sampled from Uniform(0.0, 0.2).

**Interference control:** Orthogonality penalty (O-LoRA style) if cross-interference matrix shows off-diagonal harm > 0.3 dB. Escalation path, not a default.

**Parameter budget:** ~0.4–0.6 M per adapter, under 2 M total. Full system (adapters + condition analyzer + gate + band recovery + calibration) under 4 M new parameters.

**Deliberate absences:** No clean adapter (frozen base is the clean specialist). No high-N adapter (base handles 2–5 well; no 6+ speaker regime).

### 5.4 The condition analyzer (two-level design)

#### Level 1: Raw STFT DSP features (no E(0), no neural network)

Computed from the shared 8 kHz STFT before any neural processing.

| Feature | Computation | Target |
|---|---|---|
| SNR estimate | Ratio of voiced-frame mean energy (frames flagged active by SileroVAD at 8 kHz) to noise-floor mean energy | SNR in dB; regression target from synthesis recipe |
| Codec estimate | Spectral bandwidth above which energy falls below a rolling percentile; hard cutoffs indicate codec family | Codec family classification + bitrate regression |
| Voiced-frame density | Fraction of frames flagged active by SileroVAD (pre-trained, not re-trained, runs natively at 8 kHz) | Proxy for overlap density; gate input and count prior cross-check |

**Validation requirement (Phase 0):** Confirm SileroVAD voiced-frame density has discriminative power for the gate on LibriCSS overlap subsets. Fallback: raw voiced-energy-fraction from the STFT (ratio of voiced-frame energy to total energy).

#### Level 2: E(0) neural features (reverberation and count prior only)

Taps temporal-averaged E(0) after Pass 1.

| Head | Architecture | Target |
|---|---|---|
| Reverberation strength | Attention-pooled 1-D CNN over time-averaged E(0) | T60 in seconds; regression target from RIR simulation |
| Speaker count prior | Two-layer MLP over pooled E(0) + Level-1 SNR + voiced density | Soft classification over 2–5; target from synthesis recipe |

#### Condition vector `c`

```
c = [SNR_hat, T60_hat, codec_class, codec_bitrate_hat, voiced_density, count_prior_dist]
```

All targets free from the synthesis recipe. Level-1 features available before Pass 1; Level-2 available after Pass 1 and before Pass 2.

### 5.5 The gate network

**Architecture:** Two-hidden-layer perceptron (256 units each, GELU activations), sigmoid output scaled to [0, 1.5]. Upper bound above 1.0 permits mild amplification for extreme conditions while preventing instability.

**Per-layer gates, not per-adapter scalars:** If evaluation shows per-layer gates add nothing over per-adapter scalars, the simpler variant wins (explicit ablation in Section 9.4).

**Sparsity regularization:** L1 penalty on all gate values, weight 1e-3 (increased if Principle 2 smoke test fails).

**Temporal smoothing:** EMA coefficient 0.7 across consecutive chunks. Without smoothing, gate flips produce audible texture changes at stitch seams.

### 5.6 Composition mechanics

```
y = W0 x + sum_i ( g_i_layer * B_i (A_i x) )
```

Corrections applied as parallel branches, never merged into weight matrices. One forward pass. One split decision. One speaker identity assignment.

### 5.7 The counting subsystem

**Vote 1 (primary): attractor probabilities `p_k`** (`pres["probs"]` shape `(1, 7)`; active speaker slots are indices 1–5; threshold `prob_thres=0.5` must be exposed as a parameter before calibration). Requires the §15.1 patch before this is accessible. Verified by `attractor_test.py` in Phase 0.

**Vote 2 (prior): condition analyzer count prior.** From Level-2 E(0) + Level-1 features. Not circular (reads pre-split features).

**Vote 3 (verification): residual sweep, bounded cost.** Runs only when top-2 posterior margin < 0.2. Sweeps at most 3 candidates: {mode−1, mode, mode+1}, clipped to [2, 5]. Runs decoder-only (encoder cached). Worst-case cost: 3 × 0.3 = 0.9 extra equivalent forward passes per uncertain chunk.

**Fusion:** Logistic regression over vote features, trained on validation data spanning all degradation conditions, temperature-calibrated.

**Count posterior targets:** ≥95% accuracy N=2–3; ≥85% accuracy N=4–5, on degraded validation mixtures; count ECE < 0.05 after calibration.

### 5.8 Confidence and completeness

**Per-stream confidence:** Calibrated logistic model over: (1) attractor probability `p_k`, (2) inter-stage consistency (correlation of magnitude spectrograms between last two decoder stages), (3) DNSMOS-like blind quality estimate on the band-recovered 16 kHz stream (zero-masked for N>3 if estimator proves unreliable at high counts).

**Completeness probability:** Calibrated logistic model over: (a) residual energy fraction (a missed speaker's energy has nowhere else to go), (b) SileroVAD activity score on the residual signal (speech in residual = missed speaker), (c) total attractor probability mass above the chosen count. Ground truth manufactured by forcing split to N−1 on synthetic validation mixtures.

**OOD discount:** When condition vector `c` falls beyond the 99th-percentile Mahalanobis distance from training distribution, all confidence scores are multiplied by a fixed discount factor and the output is flagged OOD. The system always returns its best attempt; it flags, not refuses.

### 5.9 Band recovery and the quality stage

**Context:** The frozen checkpoint produces separated waveforms at 8 kHz (0–4 kHz bandwidth). Insufficient for perceptual quality and DNSMOS scoring. Band recovery (Option B) is the accepted, fixed solution.

**Band recovery head:**

A small convolutional network takes each separated stream's complex STFT at 8 kHz and the mixture's complex STFT at 16 kHz and predicts the high-band (4–8 kHz) spectral mask per speaker. The mask is applied to the mixture's high-band STFT to produce the speaker's high-band signal. Low-band (8 kHz output) and predicted high-band are concatenated in the STFT domain and converted to a 16 kHz waveform.

Architecture: 2 convolutional layers per frequency stream (under 0.1 M parameters). Input: separated low-band STFT + mixture high-band STFT. Output: per-speaker high-band magnitude and phase masks.

**Dual-metric guard:** The band recovery head alters the output only if its effect on a chunk is positive by both SI-SDRi (measured against 16 kHz reference, zero-padded lower half) and DNSMOS. Per-chunk: if either metric decreases, the head is bypassed and the 8 kHz output is zero-padded to 16 kHz. The worst case is always pass-through. A single-metric guard would sacrifice one graded axis for the other.

---

## 6. Inference Pipeline

### 6.1 Chunking

2.4-second chunks, 0.8-second step. All chunking at 8 kHz internally. The 16 kHz mixture STFT (for band recovery) is computed once per chunk in parallel.

### 6.2 Per-chunk processing order

```
1. Shared 8 kHz STFT + 16 kHz mixture STFT (parallel)
2. Level-1 DSP condition features (raw 8 kHz STFT + SileroVAD)
3. Pass 1: frozen correlation module → E(0)
4. Level-2 condition features (pooled E(0): T60, count prior)
5. Gate network: [c from Level-1 + Level-2] → gate matrix g (EMA-smoothed)
6. Pass 2: full forward pass with adapters scaled by g → separated 8 kHz streams
7. Counting fusion: p_k from attractor split + Level-2 count prior + residual sweep if uncertain
8. Band recovery: per-speaker 8 kHz stream + 16 kHz mixture high-band → 16 kHz stream
9. Chunk outputs: 16 kHz streams, p_k, stage-consistency features, residual
```

### 6.3 Stitching

Adjacent chunks overlap by 1.6 seconds. Stream continuity decided by maximum correlation of overlapping separated waveforms, with ECAPA-TDNN speaker-embedding similarity as tie-breaker. Linear crossfade over overlap. A stream present in one chunk and absent in the next is faded out, not cut.

### 6.4 Global speaker count for long recordings

ECAPA-TDNN embeddings per stream per chunk, clustered across the whole recording (agglomerative, validation-tuned threshold). `N_hat_global` = number of clusters with total speech duration exceeding 1.0 second. Per-cluster confidence aggregates member streams' confidences weighted by duration.

### 6.5 Outputs

One WAV per estimated speaker at **16 kHz** (band-recovered). JSON report: global count + posterior distribution, per-stream confidences, completeness probability, per-chunk condition estimates and gate values, OOD flags.

---

## 7. Data Plan

All training data synthesized from free public corpora. Every synthesis label is free because the project controls the recipe. Evaluation sets are fixed, seeded, generated once, and hashed before any model training begins.

### 7.1 Source speech corpus: LibriSpeech (at 8 kHz for training)

All source audio is downsampled to **8 kHz** before mixing, matching the frozen checkpoint's operating rate. Keep 16 kHz copies for band recovery training targets and DNSMOS evaluation.

- **Training pool:** `train-clean-100` (100 h, 251 speakers, ~3 GB at 8 kHz). Covers all adapter and condition-analyzer training via dynamic mixing.
- **Extended pool (optional):** `train-clean-360` (360 h, 921 speakers). Useful for 4-to-5-speaker synthesis.
- **Reserved:** `dev-clean` and `test-clean` speakers strictly held out.

**Corpus honesty note:** The frozen checkpoint was trained on WSJ0-mix. Adapters and condition analyzer are trained on LibriMix-based data at 8 kHz. State this corpus difference when comparing to published numbers.

### 7.2 Training data, per component

| Component | Corpus / recipe | Key details |
|---|---|---|
| Frozen base checkpoint | Pretrained, downloaded from HF Hub | Never modified |
| `adapter_reverb` | Libri2–5Mix (8 kHz) + simulated RIRs | T60 0.2–1.0 s via pyroomacoustics; wet references; T60 label free |
| `adapter_noise` | LibriMix (8 kHz) + WHAM! (~17 GB) + DNS-4 (stratified 20 GB subset) | SNR −6 to +10 dB; label free |
| `adapter_codec` | LibriMix (8 kHz) → ffmpeg: Opus 6–24 kbps, AAC 16–48 kbps, AMR-NB/WB | Codec is a transform on existing mixtures; labels free |
| Condition analyzer + gate | Co-occurring degradations from full synthesis matrix | All labels (SNR, T60, codec, voiced density, N) free from recipe log |
| Band recovery head | 16 kHz source audio mixed at 16 kHz (target: dry sources at 16 kHz) | Also used for DNSMOS evaluation; keep regardless |
| Completeness calibration | Manufactured: force split to N−1 on `dev-clean` mixtures | No external data; only reliable source of labeled missed-speaker examples |

**Storage note:** `train-clean-100` at 8 kHz (~3 GB) + WHAM! (~17 GB) + cached RIR bank (~1 GB) = ~21 GB. Stage as two Kaggle datasets (speech + noise). Dynamic mixing means no pre-rendered mixture files needed.

**RIR bank:** Generate 10 k RIRs (1 k per T60 interval of 0.1 s) using `pyroomacoustics` (pip-installable, CPU). Cache to disk before training.

### 7.3 Noise datasets

- **WHAM! noise** (~17 GB): urban ambient, field standard.
- **DNS-4 / INTERSPEECH 2022 DNS Challenge noise**: stratified 20 GB subset. Stratify to avoid overrepresentation of speech-like noise (babble, crowd).

### 7.4 Evaluation data: fixed, seeded, generated once, hashed

All evaluation audio at 8 kHz for separation scoring; band-recovered 16 kHz versions for DNSMOS.

| Tier | Source | What it measures | N | n per cell |
|---|---|---|---|---|
| Clean 2–3 spk | Libri2Mix / Libri3Mix test sets (8 kHz) | Literature-comparable baseline; SI-SDRi, PESQ | 2, 3 | 500 |
| Sparse overlap | SparseLibriMix test (8 kHz) | Quality vs. overlap ratio 0–100% | 2 | 200 |
| Sparse overlap, 3 spk | Custom 3-spk sparse sets (synthesis pipeline) | Extends SparseLibriMix to N=3 (SparseLibriMix is 2-spk only) | 3 | 200 |
| **Primary benchmark: noise + reverb** | Custom reverb-noisy LibriMix (8 kHz; WHAMR!-style recipe; actual WHAMR! excluded: requires WSJ0 LDC license) | **Headline; SI-SDRi, DNSMOS on band-recovered output** | 2 | **500** |
| Reverb-noisy, high count | Same pipeline | Degraded count accuracy + quality | 3, 4, 5 | 200 |
| Reverb only | Clean-reverb LibriMix (8 kHz) | Isolates adapter_reverb | 2, 3 | 200 |
| Real-RIR reverb (mandatory) | **BUT ReverbDB (OpenSLR SLR17, 1,244 measured RIRs, free)** convolved with LibriSpeech test at 8 kHz | Sim-to-real gap; mandatory, not optional | 2 | 200 |
| Codec only | LibriMix (8 kHz) + ffmpeg (Opus 6 kbps; AAC 16 kbps; AMR-NB) | adapter_codec isolated | 2 | 200 |
| Codec + reverb (held-out combo) | LibriMix + codec + RIR at 8 kHz | Compositional generalization; **never in gate training** | 2, 4 | 200 |
| Noise + codec (held-out combo) | LibriMix + noise + codec at 8 kHz | Compositional generalization; **never in gate training** | 2, 4 | 200 |
| High count, clean | Libri4Mix / Libri5Mix test at 8 kHz | Count break-point curve N=4–5 | 4, 5 | 200 |
| High count, degraded | Same + reverb-noisy at 8 kHz | Count accuracy under degradation | 4, 5 | 200 |
| Real recordings | LibriCSS (1ch downmix); separation at 8 kHz, band-recover for DNSMOS | DNSMOS + Whisper WER; no clean references | 2+ | Full test set |
| Band recovery gain | Matched pairs: 8 kHz pass-through vs. band-recovered 16 kHz on primary benchmark | Isolates band recovery contribution | 2 | 500 |

**Note on OpenSLR SLR28:** SLR28 is the AISHELL-2 Mandarin ASR corpus, not a RIR database. The correct source for real room impulse responses is **BUT ReverbDB (OpenSLR SLR17)**, used above.

**Real-RIR evaluation is mandatory.** Simulated rooms produce cleaner RIRs than real measured rooms. A system scoring well on simulated reverb but poorly on real reverb has an engineering gap that must be diagnosed.

### 7.5 Three-way holdout discipline

1. **Speaker holdout:** `dev-clean`, `test-clean` speakers never in training.
2. **Condition-combination holdout:** reverb+codec and noise+codec held out of all gate and joint-training data; appear only in the evaluation matrix.
3. **Severity holdout:** T60 > 0.9 s and SNR < −4 dB underrepresented in training (10% of reverb/noise samples); probed in evaluation.

### 7.6 Reference policy for reverberant data

Training target and evaluation reference: **wet source** (individual speaker convolved with the RIR, truncated at `n_peak + 512` samples). The system separates speakers, it does not dereverberate them.

### 7.7 Pretrained tools used at inference, never re-trained

| Tool | Source | Purpose |
|---|---|---|
| SR-CorrNet pretrained checkpoint | HF `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` | Frozen base (8 kHz, K0=5) |
| SileroVAD | GitHub silero-team/silero-vad, pip (8 kHz native) | Voiced-frame density for Level-1 condition analysis |
| ECAPA-TDNN | SpeechBrain, VoxCeleb-trained | Stream stitching; global count clustering |
| DNSMOS ONNX | Microsoft DNS Challenge GitHub | Quality evaluation on 16 kHz band-recovered output |
| PESQ | `pesq` pip package | Reference-based perceptual quality |
| Whisper (optional) | OpenAI, `whisper` pip | Demo transcripts; LibriCSS WER |

---

## 8. Training Plan

The frozen base checkpoint is never touched. Everything trained: 3 LoRA adapters, condition analyzer Level-2 heads, gate MLP, band recovery head, counting fusion logistic regression, calibration parameters. Approximately 3–4 M new parameters total.

```
Stage 0: Download and verify the frozen checkpoint (no training, no GPU)
Stage 1: 3 adapters individually with co-activation warm-up
Stage 2: Universal-adapter baseline (train first, before gate)
Stage 3: Condition analyzer + gate
Stage 4: Joint adapter polish (mandatory) + band recovery + calibration
```

**Total estimated compute:** Under 150 GPU-hours on Kaggle T4 / Google Colab T4. Feasible on free-tier hardware.

### 8.1 Stage 0: Obtain and verify the frozen checkpoint

**Step 1, Download:**
```bash
python sr_corrnet/export.py --download --variant SS --config 1ch_WSJ_var_2_5spk.yaml
# Saves to: sr_corrnet/checkpoints/SS/1ch_WSJ_var_2_5spk/model.pt
# File is a bare OrderedDict; from_pretrained handles both bare and wrapped formats
```

**Step 2, Apply the three required patches (Section 15.1) before any further work:**

Patch A: Expose `p_k` through `_single_pass_session` in `engine_infer.py`.
Patch B: Register `register_forward_hook` on `model.encoder` to capture E(0) = `(B, T, F, 128)` before enc_block.
Patch C: Register hooks on each `dec_block[i]` (input side) to capture decoder stage features `(B, K, T, F, 128)` for inter-stage consistency.

All three patches are non-destructive (hooks and a one-line passthrough); the checkpoint loads cleanly after them.

**Step 3, Run `attractor_test.py`:**
```python
# Assertions that must pass before any adapter code is written:
assert pres["probs"].shape == (1, 7)           # slot layout confirmed
assert pres["probs"][0, 1:6].min() < 0.5       # at least one non-active slot at N<5
# Run at N=2,3,4,5: count active slots (probs > 0.5) == true N for each fixture
```
If this test fails, i.e. `pres` is not returned or `probs` shape is wrong, stop and fix the wrapper before touching anything else.

**Step 4, Confirm YAML constants:**
`sampling_rate: 8000`, `max_n_spks: 5`, `frame_length: 128`, `frame_shift: 64`, `N_Enc: 2`, `N_Dec: 4`, `d_model: 128`, `n_head: 4`

**Step 5, Establish corpus-transfer baseline:**
Run `process_waveform` on 20 LibriSpeech-based 2-speaker mixtures (synthesized from `dev-clean`). Record mean SI-SDRi. This is the floor every adapter and gate must beat.

**No GPU hours spent in Stage 0.** Inference only. No training, no data generation.

### 8.2 Stage 1: Adapters individually with co-activation warm-up

Each adapter trains with the base frozen. The existing `Engine` training loop is reused with the following changes: (a) freeze all base parameters, (b) register LoRA branches on the target Linear layers (Section 5.3), (c) only adapter parameters appear in the optimizer.

**Loss reuse from the existing engine:**
```python
# These are already implemented in sr_corrnet/models/SR_CorrNet_SS/loss.py and engine.py:
loss_main   = PIT_SISNR_time(scale_inv=True)          # time-domain SI-SNR under PIT
loss_mag    = PIT_SISNR_mag(scale_inv=True)            # magnitude SI-SNR, aux stages
bce_loss    = nn.BCEWithLogitsLoss()                   # attractor existence (pres["logits"])

# Combined (copied from engine._train):
loss = loss_main + 0.5 * loss_mag(aux_outputs, targets, prior_idx=perm) + bce_loss(pres["logits"], presence_target)
# presence_target: (1, 7) binary, slots 1..N → 1, rest → 0
```

**`model.forward` call for adapter training:**
```python
out, out_aux, pres = model(model_input, aux_loss=True, n_spks=None)
# model_input: (1, 2, 65, T), (B=1, 2*num_mics, F, T) real/imag stacked
# out: list of K tensors, each (1, 1, 65, T, 2), (B, M_o, F, T, real/imag)
# out_aux: list of N_Dec=4 elements, each a list of K tensors same shape
# pres: {"logits": (1,7), "probs": (1,7), "split_res": ...}
```

**Co-activation warm-up (required):** During each adapter's Stage 1 run, the other two adapters are activated with gates sampled from Uniform(0.0, 0.2). Prevents composition failure at inference.

**Training details:** AdamW, lr 3e-4, weight decay 0.01, cosine decay, 20–30 epochs, batch size 1, segment 4 s (= 32,000 samples at 8 kHz), gradient clip norm 5, mixed precision where available.

**Training order:** `adapter_noise` first (largest data variety, best for debugging LoRA plumbing), then `adapter_reverb`, then `adapter_codec`.

**Exit criterion per adapter:** Statistically significant SI-SDRi improvement on matched-condition validation (Wilcoxon, p < 0.05); no degradation on clean Libri2Mix.

**Cost per adapter:** 20–40 GPU-hours on T4. Three adapters: 60–120 GPU-hours total.

### 8.3 Stage 2: Universal adapter baseline (calibration gate, train first)

Before building the condition analyzer and gate, train one universal adapter: a single LoRA of the full adapter library's parameter budget (~2.5 M), trained on the union of all single-condition datasets.

**Decision rule:** Evaluate on the primary benchmark (reverb-noisy LibriMix, N=2) and at least two multi-condition cells. If the universal adapter matches learned gating within 0.5 dB SI-SDRi on the primary benchmark and within confidence intervals on the degraded cells, adopt the simpler system and document this as the honest headline finding. If there is a measured gap, proceed to Stage 3.

This decision is irreversible: commit the verdict before building the gate network.

**Cost:** 30–50 GPU-hours.

### 8.4 Stage 3: Condition analyzer and gate

Adapters frozen. Data: co-occurring degradations from the full synthesis matrix (excluding held-out cells). Loss: separation loss (gradients through gates only) + supervised condition-head losses (L1 for regressions, cross-entropy for classifications) + gate sparsity L1.

Level-1 DSP features are deterministic (no training). Level-2 reverb head and count-prior MLP are trained here along with the gate MLP.

**Principle 2 smoke test (mandatory at end of Stage 3):** Clean Libri2Mix, full system vs. frozen base. If system is worse, increase sparsity weight by 2× and retrain gate MLP only. Repeat until smoke test passes.

**Cost:** 15–30 GPU-hours (small new components only).

### 8.5 Stage 4: Joint adapter polish (mandatory) + band recovery + calibration

**Joint polish (mandatory, not contingent):** Unlock all three adapters + gate simultaneously. Train 15–20 epochs at one-tenth Stage 1 learning rate on compound-condition data. Base remains frozen. Adapters adjust corrections in each other's presence at realistic co-activation strengths. Add orthogonality penalty (O-LoRA) if cross-interference harm > 0.3 dB.

**Band recovery head:** Train after joint polish. Input: per-speaker separated 8 kHz STFT + mixture 16 kHz high-band STFT. Target: per-speaker high-band mask. Loss: SI-SNR on reconstructed 16 kHz waveform. Validate dual-metric guard thresholds on held-out validation.

**Calibration (everything frozen after this):**
1. Temperature scaling for count posterior.
2. Per-stream confidence logistic model.
3. Completeness logistic model (manufactured-failure validation set).
4. Counting fusion logistic regression.
5. Band recovery guard thresholds (per-chunk improvement thresholds for SI-SDRi and DNSMOS).

### 8.6 Hyperparameter defaults

| Component | Setting |
|---|---|
| LoRA rank / alpha | 8 / 8 on attention; 4 / 4 on convolutions |
| Co-activation warm-up gate range | Uniform[0.0, 0.2] for the other two adapters |
| Adapter optimizer | AdamW, lr 3e-4, wd 0.01, cosine decay, 20–30 epochs |
| Gate MLP | 2 × 256 GELU, sigmoid × 1.5 |
| Gate sparsity | L1, weight 1e-3; increased if Principle 2 smoke test fails |
| Gate EMA smoothing | coefficient 0.7 |
| Uncertainty trigger | Top-2 posterior margin < 0.2 |
| Residual sweep | Max 3 candidates clipped to [2, 5] |
| Band recovery | 2 conv layers, ~0.1 M params |
| Chunking | 2.4 s window, 0.8 s step, at 8 kHz |
| OOD threshold | Mahalanobis distance at 99th percentile of training c vectors |

---

## 9. Evaluation Plan

### 9.1 Statistical rules

Fixed seeded evaluation sets. Bootstrap 95% confidence intervals (10,000 resamples, utterance-level). Wilcoxon signed-rank tests on per-utterance deltas. A difference is claimed only when p < 0.05 and the interval excludes zero.

### 9.2 Cardinality-aware scoring

Assign by Hungarian algorithm maximizing total SI-SDR. Each matched pair contributes its SI-SDRi. Each unmatched reference (missed speaker) contributes 0 dB. Combined penalized score: mean SI-SDRi minus 1 dB per hallucinated stream. Count metrics always reported separately.

### 9.3 Primary benchmark

**Noisy-reverberant LibriMix test set, 2-speaker, SI-SDRi over the unprocessed mixture.**

This is the number that appears in the abstract, the poster headline, and the first results table. All other numbers are secondary. Literature comparison uses WHAMR! (WSJ0-based); state the corpus difference when citing.

### 9.4 The evaluation matrix

Conditions: {clean, reverb, noise, codec, reverb+noise*, reverb+codec†, noise+codec†, all-three} × N ∈ {2, 3, 4, 5}.

\* Primary benchmark.  
† Held-out combination cells.

Per cell: SI-SDRi with bootstrap interval, DNSMOS on band-recovered 16 kHz output, PESQ where applicable, count accuracy, count ECE.

### 9.5 Headline analyses

1. **Cross-interference matrix.** Every adapter alone on every condition. Off-diagonal harm > 0.3 dB triggers orthogonality penalty in Stage 4.
2. **Composition analysis.** Frozen base / each single adapter / universal adapter / uniform blend / learned gating / oracle gating.
3. **Compositional generalization.** Held-out combination cells vs. comparable trained cells.
4. **Break-point curve.** Every metric vs. N from 2 to 5.
5. **Count and confidence calibration.** Confusion matrices, reliability diagrams, ECE. Dropped-speaker recall: target >90% at 10% false-alarm rate.
6. **Risk-coverage curve.** Quality of the accepted subset as confidence threshold sweeps.
7. **Band recovery contribution.** Matched pairs: 8 kHz pass-through vs. band-recovered 16 kHz. Delta SI-SDRi and DNSMOS; fraction of chunks where guard activates vs. bypasses.
8. **Efficiency report.** RTF including worst-case residual sweep (0.9 extra equivalent passes per uncertain chunk) and 16 kHz STFT for band recovery.

### 9.6 Mandatory baselines

| Baseline | What it tests | Commitment |
|---|---|---|
| Frozen base alone (8 kHz; zero-padded to 16 kHz for DNSMOS) | Quality floor | Always reported |
| Universal adapter (Stage 2) | Whether routing is needed | Trained first; adopted if within 0.5 dB of full routing on primary benchmark |
| Uniform blend, no gate | Whether gate earns its complexity | Always reported |
| Oracle gating | Upper bound on routing | Always reported |
| Frozen base + band recovery, no adapters | Isolates band recovery from adapter contribution | Always reported |

---

## 10. Development Roadmap

| Phase | Goal | Exit gate | Contingency |
|---|---|---|---|
| **P0: Verify checkpoint and build synthesis pipeline** | Download checkpoint. Unit-test `p_k` attractor exposure (N=2,3,4,5). Validate SileroVAD voiced-density proxy. Build 8 kHz synthesis pipeline. Generate and hash all evaluation sets. | `attractor_test.py` passes; eval sets hashed; frozen base corpus-transfer SI-SDRi confirmed. Zero GPU hours spent. | If `p_k` not exposed, modify wrapper before any other work. If SileroVAD proxy fails, substitute voiced-energy-fraction from STFT. |
| **P1: Adapter library** | Stage 1: 3 adapters with co-activation warm-up. Cross-interference matrix measured. | Each adapter significant on matched condition; harmless on clean. Off-diagonal harm < 0.3 dB. | Drop failing adapters; apply orthogonality penalty if interference exceeds bar. |
| **P1b: Universal adapter calibration gate** | Stage 2: universal adapter trained and evaluated. Decision locked. | Verdict logged before gate is built. | Irreversible. If adopted: report as headline finding; analyzer, counting, confidence stacks remain as independent contributions. |
| **P2: Condition analyzer and gate** | Stage 3. Principle 2 smoke test passed. | Learned gating beats best single adapter on co-occurring cells; smoke test passes; held-out combination cells do not collapse. | Rule-based gates from supervised heads if MLP collapses. Increase sparsity weight if smoke test fails. |
| **P3: Joint polish, band recovery, calibration** | Stage 4 (mandatory joint fine-tune) + band recovery + all calibrations. | Dual-metric guard validated; ECE < 0.05; dropped-speaker recall > 90% at 10% FAR. | If recall short, tune threshold. Band recovery ships disabled if dual-metric guard cannot pass on validation. |
| **P4: Demo, CLI, efficiency report** | CLI + Gradio demo with routing visualization; RTF measured. | Demo runs end to end on held-out real recording; RTF documented. | Any post-processor failing its guard ships disabled. |
| **P5: Full evaluation and report** | Full matrix, all baselines, all analyses, reproducibility bundle. | Every claim carries an interval; universal-adapter verdict stated plainly. | The only failure mode is dishonesty. |

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `p_k` not exposed by checkpoint wrapper | Low-Medium | Blocker for counting subsystem | First P0 engineering task; modify wrapper before any other code |
| LoRA composition interference > 0.3 dB | High (expected) | Medium | Co-activation warm-up (Stage 1); Stage 4 joint polish (mandatory); O-LoRA penalty (escalation) |
| Universal adapter matches full routing system | Medium | Reframes headline | Trained first; pre-committed to adopt if within 0.5 dB |
| Gate collapse to uniform activation | Medium | Quality plateau | Supervised condition heads; sparsity regularization; oracle-gap analysis makes collapse visible |
| SileroVAD proxy uninformative for overlap | Medium | Minor | Validated in P0; fallback: voiced-energy-fraction from STFT |
| Never-worse guarantee fails on clean input | Medium | Credibility | Principle 2 smoke test at end of Stage 3; sparsity weight increased until test passes |
| Sim-to-real gap for reverb | Medium | BUT ReverbDB cells | BUT ReverbDB eval tier mandatory; gap identified in P0 |
| Count accuracy degrades under combined degradations | Medium | Half the grade | Residual sweep (max 3 candidates); count accuracy on degraded validation from start |
| Band recovery hurts SI-SDRi (oversmoothing) | Medium | Quality regression | Dual-metric guard; per-chunk bypass; worst case is 8 kHz pass-through |
| Compositional holdout collapses | Medium | Limits claims | Broaden Stage 4 sampling if collapse detected at P2 gate |
| Residual sweep triggers too frequently | Unknown | RTF budget | Measure trigger frequency; if >30%, raise uncertainty threshold or lower sweep to 2 candidates |

---

## 12. Alternatives Considered and Inspirations

### 12.1 Alternatives rejected

- **Bank of full models with a selector:** stream alignment is ill-posed; single backbone removes the problem structurally.
- **Hard one-of-N expert routing:** co-occurring conditions cannot be represented; audible switching artifacts.
- **Output-space ensembling:** permutation mismatch; all composition in this system happens in weight space.
- **Test-time adaptation:** unbounded worst case on graded output.
- **Full fine-tune per condition:** loses composability; multiplies storage.
- **Option A (16 kHz backbone retraining) or Option C (bandwidth adapter):** accepted the checkpoint as-is; band recovery (Option B) achieves the DNSMOS requirement at negligible compute cost.
- **6-7 speaker extension:** out of scope; base handles 2–5 with high accuracy; extending K0 requires backbone modification.

### 12.2 Inspirations

SR-CorrNet and its transformer-decoder attractor mechanism; LoRA, LoRA-Hub, O-LoRA; continuous MoE routing; SileroVAD; ECAPA-TDNN; DNSMOS; temperature scaling and ECE calibration; manufactured-failure calibration from reliability engineering.

---

## 13. Repository Structure and Engineering Practices

```
coral-sep/
  configs/
    base_checkpoint.yaml   # locked checkpoint path + SHA; no training settings
    adapters/              # one YAML per adapter (reverb, noise, codec)
    gate.yaml
    band_recovery.yaml
    eval.yaml
  data/
    synthesis/             # mixture generation, degradation transforms, recipe logging
    fixed_eval/            # seeded, hashed evaluation sets and manifests
    rirs/                  # cached RIR bank (pyroomacoustics output)
  models/
    srcorrnet/             # base network wrapper; exposes p_k, E(0), stage outputs
    lora.py                # parallel-branch LoRA wrapper; co-activation warm-up sampler
    condition.py           # two-level analyzer: Level-1 DSP + Level-2 E(0) heads
    gate.py                # gate MLP, EMA smoothing, sparsity
    counting.py            # attractor readout, residual sweep (max 3, clipped [2,5]), fusion
    confidence.py          # per-stream confidence, completeness, OOD discount
    band_recovery.py       # high-band prediction head + dual-metric guard
  pipeline/
    chunker.py             # 8 kHz chunks + parallel 16 kHz STFT for band recovery
    stitcher.py            # ECAPA-TDNN stream stitching
    infer.py               # Section 6.2 processing order
  eval/
    metrics.py             # SI-SDR cardinality-aware, DNSMOS on 16 kHz, PESQ
    matrix.py              # Section 9 matrix and analyses
    stats.py               # bootstrap, Wilcoxon, reliability diagrams
  calibration/             # fitted temperature scalars and logistic models, hashed
  demo/                    # CLI + Gradio web demo
  reports/                 # result tables, one directory per system version hash
  tests/
    smoke_test.py          # 60-s fixture end-to-end; count ∈ {2,3,4,5}, file validity, JSON schema
    principle2_test.py     # clean Libri2Mix: all adapters active vs. frozen base
    attractor_test.py      # p_k vary with true count at N=2,3,4,5
```

**Engineering practices:**

- **Config-driven, config hash in every artifact.** Every checkpoint and result records the SHA-256 of the config that produced it.
- **Frozen artifacts are immutable and hashed.** Base checkpoint, adapter weights, calibration files, evaluation sets: content-addressed.
- **The wrapper exposes internals by contract.** Returns waveforms, p_k, stage outputs, pooled E(0). `attractor_test.py` is the gate for everything downstream.
- **Every mechanism ships with its off switch.** Adapters, gate, residual sweep, band recovery: each disabled by config. Baselines are one-line runs, not code forks.
- **Seeds fixed and logged everywhere.**
- **Continuous smoke test on every merge.**

---

## 14. Glossary

| Term | Meaning |
|---|---|
| 8 kHz constraint | The locked operating rate of the frozen checkpoint; band recovery extends output to 16 kHz after separation |
| Attractor | Learned vector representing one candidate speaker slot (K0=5); existence probability is the model's belief the slot is occupied |
| Band recovery (Option B) | Small convolutional head predicting 4–8 kHz spectral content from low-band separated output + 16 kHz mixture; guarded by dual-metric test; the only quality-extension mechanism used |
| BUT ReverbDB | Real room impulse response database (OpenSLR SLR17); used for mandatory sim-to-real reverb evaluation. SLR28 is AISHELL-2 (Mandarin ASR corpus), not a RIR database |
| Calibration / ECE | Agreement between stated confidence and actual accuracy; ECE = average disagreement across confidence bins |
| Cardinality-aware scoring | Evaluation rule for N_hat ≠ N; missed speakers contribute 0 dB, not excluded |
| Co-activation warm-up | Training each adapter with the other two randomly active at low strength [0.0, 0.2] |
| Completeness probability | Calibrated belief that no speaker was missed |
| Condition analyzer | Two-level: Level 1 = raw STFT DSP (SNR, codec, voiced density via SileroVAD); Level 2 = E(0) neural (T60, count prior) |
| Dual-metric guard | Band recovery activates per-chunk only if both SI-SDRi and DNSMOS improve vs. pass-through |
| E(0) | Output of the frozen correlation module; tapped by Level-2 condition analysis |
| Gate | Learned scalar in [0, 1.5] scaling one adapter's correction at one layer |
| K0 | Number of attractor query slots in the frozen checkpoint; 5 in var-2-5; never changed |
| LoRA | Low-rank adaptation; trainable correction B·A added to a frozen weight matrix; corrections add linearly |
| Manufactured failure | Synthetic example created by forcing the split to N−1, generating labeled missed-speaker cases |
| Option B | The fixed band recovery approach; there is no Option A or Option C in this project |
| Oracle gating | Setting gates from the true synthesis recipe; upper bound on routing quality |
| PIT | Permutation invariant training; optimizing over the best assignment of outputs to references |
| Principle 2 smoke test | Required at end of Stage 3: full system with all adapters active vs. frozen base on clean Libri2Mix |
| Residual energy | Energy of mixture minus sum of separated streams; physical trace of a missed speaker |
| Residual sweep | Vote 3 in counting; max 3 candidates clipped to [2,5]; worst-case cost 0.9 extra equivalent forward passes |
| SCOT-β | Smooth coherence transform normalization (β=0.5) applied to correlation features |
| SileroVAD | Pre-trained VAD running natively at 8 kHz; provides voiced-frame density for Level-1 condition analysis; not re-trained |
| SI-SDR / SI-SDRi | Scale-invariant signal-to-distortion ratio; SI-SDRi = improvement over the unprocessed mixture |
| Stage 4 joint polish | Mandatory joint fine-tuning of all adapters + gate on compound conditions; resolves LoRA composition interference |
| Universal adapter | Single LoRA of full adapter budget trained on all conditions; Stage 2 calibration gate, trained before the routing system |
| Wet reference | Training/evaluation target including room reverberation, truncated at n_peak + 512 samples |

---

---

## 15. Implementation Reference (Repo Audit)

This section documents every finding from the full code audit of the SR-CorrNet repository. It is the implementation ground truth: exact file paths, class names, method signatures, tensor shapes, attribute paths, and required patches. A developer starting implementation should read this section before touching any code.

The audit was performed on the extracted codebase at commit matching `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk`. All line references are approximate; the attribute paths are exact.

---

### 15.1 Three Required Patches (Must Apply in Phase 0, Before Anything Else)

These three patches expose internal model state that CoRAL-Sep needs but the default inference API silently drops. They are non-destructive: no weights change, no behavior changes for existing users.

---

#### Patch A, Expose `p_k` through `_single_pass_session`

**File:** `sr_corrnet/models/SR_CorrNet_SS/engine_infer.py`

**Problem:** `infer_chunk()` returns a dict containing `"pres"` (which holds `{"logits": (1,7), "probs": (1,7), "split_res": ...}`). However, `_single_pass_session()` only forwards `"stft_out"` and discards `pres`.

**Call chain:**
```
SSInference.process_waveform / process_stft
  → EngineInfer._single_pass_session(mixture_stft, n_spks)
      → self.infer_chunk(mixture_stft, n_spks)       ← returns dict WITH pres
      → chunk_result["stft_out"] extracted             ← pres DROPPED HERE
      → return {"stft_out": stft_out, "vad": None, "doa": None}
```

**Fix:** In `_single_pass_session`, change the return statement to:
```python
return {
    "stft_out": stft_out,
    "vad": None,
    "doa": None,
    "pres": chunk_result.get("pres"),   # ADD THIS LINE
}
```

Then in `inference.py`, `process_stft` and `process_waveform` must forward `pres` into their returned dicts:
```python
# In process_stft and process_waveform, after calling _single_pass_session:
result["pres"] = session_result.get("pres")
```

After this patch, `SSInference.process_stft(stft)["pres"]["probs"]` returns `(1, 7)` tensor.

---

#### Patch B, Expose E(0) via forward hook on `model.encoder`

**Problem:** In `Model.forward` (`model.py`), `x_enc` is computed by `self.encoder(x[:, self.ref_ch], x_mf)`, this is E(0), shape `(B, T, F, 128)`. It is never stored or returned.

**Fix (non-invasive, register once after loading checkpoint):**
```python
_e0_cache = {}

def _e0_hook(module, input, output):
    _e0_cache["e0"] = output  # shape: (B, T, F, 128)

model.encoder.register_forward_hook(_e0_hook)

# After model(model_input):
e0 = _e0_cache["e0"]                       # (1, T, F, 128)
e0_pooled = e0.mean(dim=(1, 2))            # (1, 128), input to Level-2 condition heads
```

This hook fires every forward pass. The E(0) captured here is pre-positional-encoding, pre-enc_block, exactly what the condition analyzer Level-2 heads should read. Do not tap E(N_Enc) (after enc_block) as it is entangled with the split decision.

---

#### Patch C, Expose decoder stage features via hooks on `dec_block[i]`

**Problem:** `decoder_forward` builds `x_dec_h = []` (list of per-stage inputs, shape `(B, K, T, F, 128)`) but only uses it internally for auxiliary filter heads. These features are the inter-stage consistency signal needed by the per-stream confidence head (Section 5.8).

**Fix (register hooks on each dec_block module):**
```python
_dec_stage_cache = {}

for i, block in enumerate(model.dec_block):
    def _make_hook(idx):
        def _hook(module, input, output):
            _dec_stage_cache[idx] = input[0]  # input[0]: (B*K, T, F, 128), reshaped inside decoder_forward
        return _hook
    block.register_forward_hook(_make_hook(i))

# After model(model_input):
# _dec_stage_cache[i] contains the input to dec_block[i], shape (B*K, T, F, 128)
# Reshape: _dec_stage_cache[i].view(B, K, T, F, 128) for use in confidence head
```

Note: `decoder_forward` reshapes `x` to `(B*K, T, F, C)` before passing to each `dec_block`. Account for this in the hook: reshape back to `(B, K, T, F, 128)` using the known K from `pres["probs"]`.

---

### 15.2 Model Forward Pass: Full Signature and Return Values

**File:** `sr_corrnet/models/SR_CorrNet_SS/model.py`

```python
class Model(nn.Module):
    def forward(self, x, aux_loss=False, n_spks=None):
        # x: (B, 2*M, F, T) real/imag stacked, or (2*M, F, T), batch dim auto-unsqueezed
        # For 1ch config: x shape is (1, 2, 65, T)
        # aux_loss: True → compute out_aux (slower); False → out_aux = None
        # n_spks: None → attractor infers count (batch_size must be 1); int → known count
        #
        # Returns: (out, out_aux, pres)
        #   out:     list of K tensors, each (1, 1, 65, T, 2), (B, M_o, F, T, real/imag)
        #   out_aux: list of N_Dec=4 elements [list of K tensors same shape] if aux_loss else None
        #   pres:    {"logits": (1,7), "probs": (1,7), "split_res": ...} if is_var_spks else None
```

**Tensor shape table (1ch, 8kHz, T time frames):**

| Tensor | Shape | Notes |
|---|---|---|
| Input `x` | `(1, 2, 65, T)` | real/imag stacked, single channel |
| `x_mf` (multi-frame) | `(1, 1, 9, 65, T)` | 1 mic, L=3×3=9 neighbors |
| E(0) = `x_enc` before enc_block | `(1, T, 65, 128)` | from Patch B hook |
| `x_enc` after enc_block | `(1, T, 65, 128)` | feeds AttractorSplit |
| `pres["probs"]` | `(1, 7)` | slots 1–5 are speaker probs |
| `x_sep` (after split) | `(1, K, T, 65, 128)` | K = active speakers |
| dec_block input at stage i | `(K, T, 65, 128)` | from Patch C hook (B=1, squeezed) |
| `out[n]` (per speaker) | `(1, 1, 65, T, 2)` | (B, M_o, F, T, real/imag) |
| `out_aux[i][n]` | `(1, 1, 65, T, 2)` | same shape, from stage i |

---

### 15.3 `SSInference` Public API: What It Returns

**File:** `sr_corrnet/inference.py`

| Method | Input | Returns | Notes |
|---|---|---|---|
| `from_pretrained(config, checkpoint_path, device)` | HF repo ID or local path | `SSInference` instance | `strict=False` on load, new params safe |
| `process_file(path, output_dir, n_spks)` | path string | `{"waveforms": list, "vad": None, "doa": None}` | After Patch A: also `"pres"` |
| `process_waveform(waveform, n_spks)` | `(M, L)` or `(L,)` float tensor | same as above | Normalizes internally |
| `process_stft(stft_input, n_spks)` | `(M, F, T)` complex | `{"stft_out": (N,F,T) complex, ...}` | After Patch A: also `"pres"` |
| `process_stft_chunk(stft_chunk, n_spks)` | `(M, F, T)` complex | `{"stft_out": (N,M_o,F,T), "pres": dict}` | Already returns `pres` without patch |
| `model.stft(waveform, cplx=True)` | `(M, L)` | `(M, 65, T)` complex | 128-sample window, 64-sample hop |
| `model.istft(stft, cplx=True, squeeze=True)` | `(F, T)` or `(M, F, T)` complex | `(L,)` or `(M, L)` | Inverse STFT |

**Important:** `process_waveform` and `process_stft` apply std-normalization before STFT. `process_stft_chunk` does not normalize, if you call it directly, normalize the waveform first or the STFT magnitudes will be inconsistent.

---

### 15.4 `AttractorSplit`: Key Implementation Details

**File:** `sr_corrnet/models/SR_CorrNet_SS/modules/module.py`

```python
class AttractorSplit(nn.Module):
    def __init__(self, d_model, d_freq, n_head, max_n_spks, dropout_rate):
        # spk_query: nn.Parameter, shape (1, max_n_spks+2, d_model) = (1, 7, 128)
        # dec: AttractorDecoder (2 x TransDecoderBlock, cross-attn + self-attn + FFN)
        # pres_linear: nn.Linear(128, 1), maps each attractor → existence logit
        # split: SplitModule, Linear(2*C, 4*C) → SiLU → Linear(4*C, C), fuses attractor+encoder
        # pe_tf: buffer (5000, 1500, 128), 2D sinusoidal PE, pre-allocated

    def forward(self, x, n_spks=None, prob_thres=0.5):
        # x: (B, T, F, C), output of enc_block
        # HARD CONSTRAINT: assert x.size(0) == 1 when n_spks=None
        # Returns: (x_sep, pres_dict, speaker_mask)
        # x_sep: (B, K, T, F, C) where K = (probs[0,1:] > prob_thres).sum()
        # pres_dict: {"logits": (B,7), "probs": (B,7), "split_res": list}
```

**`prob_thres=0.5` must be exposed as a parameter** before the calibration stage. Current implementation has it hardcoded in the function default. Patch: in the `Engine` or `EngineInfer`, pass a configurable threshold when calling `model.spk_split.forward`. The simplest approach: wrap the model and monkey-patch `model.spk_split.forward` with `functools.partial(original_forward, prob_thres=config_value)`.

**`pe_tf` buffer size `(5000, 1500, 128)`:** Indexed at runtime as `pe_tf[:T, :F, :]`. For 8 kHz audio with 4-second segments: T = 4000/64 = 62.5 → 63 frames; F = 65. Both are well within the pre-allocated limits. No issue for standard segments.

---

### 15.5 `STFT` Utility: Exact Parameters

**File:** `sr_corrnet/utils/util_stft.py`

```python
class STFT(STFTBase):
    # For 1ch_WSJ_var_2_5spk: frame_length=128, frame_shift=64, normalize=True
    # Hann window, sqrt-scaled; K = non-trainable nn.Parameter, shape (N+2, 1, N)
    # num_bins = N//2 + 1 = 65

    def forward(self, x, cplx=False):
        # x: (N,C,S) or (N,S)
        # cplx=True  → returns complex (N,C,65,T) or (N,65,T)
        # cplx=False → returns (magnitude, phase) each (N,C,65,T) or (N,65,T)
        # normalize=True: skips the x * N**0.5 scaling (already applied to K)
```

**The STFT module is a `nn.Module` with frozen parameters, it is part of the checkpoint.** When building the band recovery head's separate 16 kHz STFT, instantiate a fresh `STFT(frame_length=256, frame_shift=128)`, do not reuse the model's 8 kHz STFT.

---

### 15.6 Training Engine: Loss Functions and Signatures

**File:** `sr_corrnet/models/SR_CorrNet_SS/loss.py` and `engine.py`

```python
class PIT_SISNR_time(nn.Module):
    def forward(self, estims, targets, eps=1e-10, return_perm_idx=False):
        # estims:  list of N tensors, each (B, L), time-domain waveforms
        # targets: list of N tensors, each (B, L)
        # Iterates all permutations of range(N), picks best SI-SNR assignment
        # return_perm_idx=True: returns (scalar_loss, list_of_perm_indices)
        # PIT is exponential in N: N=5 → 120 permutations. Fine for N≤5.

class PIT_SISNR_mag(nn.Module):
    def forward(self, estims, targets, eps=1e-10, prior_idx=None):
        # estims: list of N_Dec lists, each containing N tensors (magnitude STFT)
        # prior_idx: permutation from the time-domain PIT, reuse for consistent alignment
        # Used only when aux_loss=True

# Auxiliary loss weight schedule (from engine.py):
# w_aux = 0.5 * (0.95 ** (epoch - 100)) if epoch > 100 else 0.5

# Attractor BCE (from engine.py):
# presence_target = build from synthesis recipe: slots 1..N → 1, rest → 0, shape (1, 7)
# cur_loss_pres = nn.BCEWithLogitsLoss()(pres["logits"], presence_target)

# Full adapter training loss:
# loss = loss_main + 0.5 * loss_mag(out_aux, targets, prior_idx=perm) + loss_pres
```

**Converting `out` (complex STFT) to waveform for loss computation:**
```python
# model returns out[n]: (B, M_o, F, T, 2), last dim is real/imag
# Engine's existing helper (engine.py _train):
estim_stft = torch.complex(out[n][..., 0], out[n][..., 1])  # (B, M_o, F, T)
estim_wav = engine.istft(estim_stft[:, ref_ch], cplx=True)  # (B, L)
```

**Target preparation for PIT:**
```python
# target_stft: (B, N, F, T) complex, synthesized mixture's per-speaker STFTs
# Reuse existing dataset.py for loading or build equivalently for dynamic mixing
```

---

### 15.7 Inference Wrapper: Loading Pattern for CoRAL-Sep

```python
from sr_corrnet import SSInference

# Step 1: Load base model
model = SSInference.from_pretrained(
    "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk",
    device="mps",  # or "cuda:0" or "cpu"
)
# model.engine.model is the nn.Module (Model instance)
base_nn = model.engine.model

# Step 2: Register LoRA branches on target Linear layers (before freezing)
# (your lora.py wraps each target Linear in-place)
register_lora(base_nn, adapter_name="reverb", rank=8, target_modules=LORA_TARGET_PATHS)

# Step 3: Freeze all base parameters; only LoRA params trainable
for name, param in base_nn.named_parameters():
    if "lora_" not in name:
        param.requires_grad_(False)

# Step 4: Apply Patch A, B, C (hooks)
base_nn.encoder.register_forward_hook(_e0_hook)
for i, blk in enumerate(base_nn.dec_block):
    blk.register_forward_hook(_make_dec_hook(i))

# Step 5: Verify
out, out_aux, pres = base_nn(fixture_stft, aux_loss=True, n_spks=None)
assert pres["probs"].shape == (1, 7)
assert _e0_cache["e0"].shape[2] == 65  # F bins
```

**`strict=False` means:** if you call `from_pretrained` after adding LoRA params to the model, the base weights load correctly and the new LoRA params keep their initialization. This is the safe loading order for all stages.

---

### 15.8 Hard Constraints Checklist

These are non-negotiable properties of the frozen checkpoint. Violating any of them corrupts inference without an error message.

| Constraint | Value | Source | Consequence if violated |
|---|---|---|---|
| Batch size for unknown-speaker inference | **1** | `assert x.size(0) == 1` in `AttractorSplit.forward` | AssertionError at runtime |
| Operating sample rate | **8 kHz** | Dataset YAML `sampling_rate: 8000` | Wrong STFT interpretation; no error |
| STFT window / hop / bins | **128 / 64 / 65** | Baked into checkpoint STFT parameters | Dimension mismatch if changed |
| `spk_query` shape | **(1, 7, 128)** | Checkpoint weight | Must not reshape or extend |
| `pe_tf` buffer max T | **5000** | Pre-allocated in `AttractorSplit.__init__` | IndexError if input > 5000 frames (~5.3 min at 8kHz/64-hop) |
| `get_correlation` is no-grad | **Always** | `@torch.no_grad()` on `Encoder.get_correlation` | LoRA in encoder attn CAN still receive gradients; this only blocks correlation arithmetic |
| `prob_thres` | **0.5** (hardcoded default) | `AttractorSplit.forward` signature | Expose as configurable before calibration; do not assume it can be changed without a patch |
| `filter_estim_aux` count | **4** (= N_Dec) | Initialized as `nn.ModuleList` of length N_Dec | Must match config; baked into checkpoint |

---

### 15.9 Module Hierarchy Reference

Complete path map for navigating the model (use with `base_nn.` prefix):

```
base_nn
├── encoder                          # Encoder: correlation → E(0), shape (B,T,F,128)
│   ├── embed[0]                     # Conv2d(18, 512, 3), 18 = 2*1*3*3 (1ch, 3×3 neighborhood)
│   └── embed[2]                     # Conv2d(512, 128, 3)
├── enc_block                        # nn.ModuleList of N_Enc=2 TF_Blocks
│   └── [i]                          # TF_Block
│       ├── freq_block.block         # TransBlock (frequency axis)
│       │   ├── sa.block.qkv         # Linear(128, 384) ← LoRA target
│       │   └── sa.block.aggregate_heads[0]  # Linear(128,128) ← LoRA target
│       └── time_block.block         # TransBlock (time axis), same structure
├── spk_split                        # AttractorSplit: encoder output → speaker streams
│   ├── spk_query                    # nn.Parameter (1, 7, 128), K0+2 slots
│   ├── dec.net1                     # TransDecoderBlock 1 (cross-attn + self-attn + FFN)
│   │   ├── ca.block.kv              # Linear(128, 256), cross-attn key/value
│   │   ├── ca.block.q               # Linear(128, 128), cross-attn query
│   │   └── sa.block.qkv             # Linear(128, 384), self-attn
│   ├── dec.net2                     # TransDecoderBlock 2, same structure
│   └── pres_linear                  # Linear(128, 1), existence logit per attractor
├── dec_block                        # nn.ModuleList of N_Dec=4 TF_Blocks (same structure as enc_block)
├── dec_cs                           # nn.ModuleList of N_Dec=4 CrossSpkBlocks
│   └── [i].block.block['sa']        # CS_TransBlock MHSA, note: ModuleDict key 'sa'
│       ├── block.qkv                # Linear(128, 384) ← LoRA target
│       └── block.aggregate_heads[0] # Linear(128, 128) ← LoRA target
├── filter_estim                     # FilterEstimator: dec output → complex filters
│   └── mask.net                     # Linear(128, 27) ← LoRA target (27 = 3*1*1*3*3)
└── filter_estim_aux                 # nn.ModuleList of N_Dec=4 FilterEstimators (aux losses)
    └── [i].mask.net                 # Linear(128, 27) ← LoRA target
```

---

### 15.10 Existing Tests and Coverage Gaps

**File:** `tests/smoke_test_future_work.py`, 6 tests, all covering `future_work/` only.

| Existing test | What it covers |
|---|---|
| `test_import_public_api` | future_work `__all__` imports |
| `test_geometry_variable_mics` | FutureSRCorrNet forward shapes |
| `test_geometry_mic_mask_and_freeze` | freeze_backbone, trainable_parameter_groups |
| `test_speaker_memory_and_diarization` | MemoryBank, IdentityAttractorSplit, 3 losses |
| `test_long_context_session` | LongContextSession.process_longform |
| `test_geometry_relative_and_miso_guard` | Translation invariance, MIMO NotImplementedError |

**Zero tests exist for:** `Model`, `Engine`, `EngineInfer`, `SSInference`, `AttractorSplit`, `FilterEstimator`, or any core separation pipeline component.

**Tests CoRAL-Sep must add (§13 references these):**

```python
# attractor_test.py, BLOCKING: nothing else can start until this passes
def test_pk_exposed_and_varies():
    model = SSInference.from_pretrained("shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk", device="cpu")
    for n_true in [2, 3, 4, 5]:
        mix = synthesize_fixture(n_spks=n_true, sr=8000, duration=4.0)
        stft = model.stft(mix.unsqueeze(0), cplx=True)
        result = model.process_stft(stft, n_spks=None)
        pres = result["pres"]
        assert pres is not None, "pres not returned, apply Patch A"
        assert pres["probs"].shape == (1, 7), f"unexpected shape {pres['probs'].shape}"
        n_active = (pres["probs"][0, 1:6] > 0.5).sum().item()
        assert n_active == n_true, f"N={n_true}: model counted {n_active}"

# principle2_test.py, Required at end of Stage 3
def test_never_worse_on_clean():
    # Frozen base vs. full system with all adapters active on 20 clean Libri2Mix segments
    # Assert: mean SI-SDRi(system) >= mean SI-SDRi(base) - 0.1 dB (allow 0.1 dB noise)

# e0_hook_test.py, Verify Patch B works
def test_e0_hook_shape():
    # Confirm _e0_cache["e0"].shape == (1, T, 65, 128) after a forward pass
```

---

### 15.11 `future_work/` Compatibility Note

The `future_work/` files (`FutureSRCorrNet`, `LongContextSession`, `SpeakerMemoryBank`) are **a separate model**, they do not share weights with the published `Model` checkpoint. `load_backbone_from_published_state_dict` copies matching-shape parameters by key name. CoRAL-Sep does not use `future_work/` at all; it builds directly on the published `Model` class. The `future_work/` code can coexist in the repo without conflict.

---

*End of blueprint. Fixed constraints at the top are never revisited: frozen checkpoint (`shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk`), 8 kHz operating rate, K0=5, N ∈ {2–5}, Option B band recovery. When implementation and document disagree, one of them is wrong, and the resolution is recorded here.*
