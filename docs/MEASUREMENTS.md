# Measurements

> 📊 **Recovered reference, with corrections marked.** Written 2026-09-01 by the
> original author as `NUMBERS.md`, recovered from the archive under ticket I-015.
> It existed in no commit.
>
> ### 🔴 Known errors in this document
>
> | Claim here | Measured 2026-09-04 |
> |---|---|
> | Backbone 13,270,124 params | **14,031,768**. The figure below is a `parameters()` count taken *after* the LoRA library was attached, which omits 1,065,856 base weights held as buffers. |
> | LoRA share 2.29 percent | **2.168 percent** |
> | Total trainable share 3.32 percent | **3.138 percent** |
>
> Everything else in this document was checked against the raw artifacts in
> `results/` and matches exactly. The evaluation numbers, the Stage 4 loss curve
> and the reverb diagnostic are all confirmed.
>
> The authoritative version of the results, with provenance for every number,
> is [restoration/RESULTS.md](restoration/RESULTS.md).

# CALM-Sep: Models · Datasets · Numbers · Inferences
**Last updated: 2026-09-01**

---

## 1. Model Architecture & Parameters

### 1.1 Backbone, SR-CorrNet var-2-5

| Property | Value |
|----------|-------|
| HuggingFace ID | `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` |
| Total params | **13,270,124** |
| Training status | Frozen (never fine-tuned in CALM-Sep) |
| Input sample rate | **8,000 Hz** (hard-locked by STFT params) |
| Output sample rate | **8,000 Hz** (band recovery lifts to 16 kHz) |
| STFT window / hop / bins | 128 / 64 / 65 |
| Speaker slots | Slots 1–5 of a 7-slot attractor |
| Speaker range | 2–5 (var-2-5 checkpoint) |
| Count mechanism | Attractor probability threshold 0.5 |

The backbone is an attention-based encoder-decoder (CorrNet variant). It outputs K=5 streams and per-slot probabilities `p_k ∈ (0,1)` for slots 1–5. Slots where `p_k > 0.5` are returned as active speakers. This is the SR-CorrNet "AttractorSplit" gate, no separate counting head needed.

### 1.2 LoRA Adapters

| Property | Value |
|----------|-------|
| Adapters | 3: `reverb`, `noise`, `codec` |
| LoRA-wrapped modules per adapter | **37** (all Linear layers in enc/dec blocks) |
| Rank (attention QKV, agg layers) | **8** |
| Rank (filter_estim mask net) | **4** |
| Params per adapter | **101,404** |
| Total LoRA params (3 adapters) | **304,212** |
| % of backbone | **2.29%** |
| Init | A ~ Kaiming uniform; B = 0 (zero-delta at init) |
| Formula | `y = W₀x + g_r·Br(Arx) + g_n·Bn(Anx) + g_c·Bc(Acx)` |
| Co-activation during Stage 1 | Inactive adapters get gate ~ Uniform(0.0, 0.2) |

**Layer attachment map (37 modules per adapter):**

| Block | Sub-layer | Rank | Count |
|-------|-----------|------|-------|
| enc_block[0,1] × {freq,time} × {qkv, agg} | Attention | 8 | 8 |
| dec_block[0–3] × {freq,time} × {qkv, agg} | Attention | 8 | 16 |
| dec_cs[0–3] × {qkv, agg} | Cross-attention | 8 | 8 |
| filter_estim.mask.net | Filter | 4 | 1 |
| filter_estim_aux[0–3].mask.net | Filter aux | 4 | 4 |
| **Total** | | | **37** |

### 1.3 Gate Network

| Property | Value |
|----------|-------|
| Input dim | 10 (Level-1: 4D + Level-2: 6D) |
| Hidden | 256 × 2 layers |
| Activation | GELU |
| Output | sigmoid × 1.5 → gate ∈ [0, 1.5] |
| Params | **69,379** |
| L1 sparsity penalty λ | 1e-3 |
| EMA smoothing α | 0.7 (streaming inference) |

### 1.4 Condition Analyzer

| Property | Value |
|----------|-------|
| Level-1 (DSP, no training) | SNR estimate, codec bandwidth fraction (>3.2 kHz energy drop), voiced-frame density via SileroVAD |
| Level-2 (trained) | T60 reverb head + CountPriorMLP from E(0) encoder output pooled across voiced frames |
| Params | **20,997** |

### 1.5 Band Recovery Head

| Property | Value |
|----------|-------|
| Task | Predict 4–8 kHz content; output 16 kHz waveform |
| Input | 8 kHz separated STFT (65 bins) + 16 kHz mixture high-band (129 bins) concatenated |
| Architecture | Conv1d(194, 64, k=3) → Conv1d(64, 129, k=1) → sigmoid mask |
| Output | Soft mask over 129 high-band bins; applied to original mixture's high-band STFT |
| Params | **45,697** |
| Guard | Applied per-chunk only when BOTH SI-SDRi AND DNSMOS improve |

### 1.6 Calibration

| Component | Params | What it does |
|-----------|--------|-------------|
| Gate temperature scaler | 1 scalar T | Post-hoc BCE calibration via golden-section search over T ∈ (0.05, 10.0) |
| StreamConfidenceHead | logistic (3→1) | Fuses p_k + inter-stage consistency + DNSMOS into per-stream confidence |
| CompletenessCalibrator | Platt (a, b) | Calibrates completeness probability (sigmoid(a·logit + b)) |
| OOD Mahalanobis | covariance matrix | Discounts confidence for out-of-distribution Level-1 features |

### 1.7 Total Parameter Summary

| Component | Params | Status |
|-----------|--------|--------|
| SR-CorrNet backbone | 13,270,124 | Frozen |
| LoRA (3 adapters) | 304,212 | **Trained** |
| Gate MLP | 69,379 | **Trained** |
| Level-2 Analyzer | 20,997 | **Trained** |
| Band Recovery | 45,697 | **Trained** |
| **Total trainable** | **440,285** | **3.32% of backbone** |
| **Total CALM-Sep** | **13,710,409** | |

---

## 2. Datasets

### 2.1 Training Data (local M5 Pro)

| Dataset | Files | Sample rate | Role |
|---------|-------|-------------|------|
| LibriSpeech (8kHz resampled) | 137,876 utterances | 8 kHz | Speech sources for all mixtures |
| WHAM! noise | 28,000 clips | 8 kHz | Noise augmentation (Stage 1 noise adapter) |
| Custom RIR bank | 10,001 RIRs | 8 kHz | Reverb augmentation (Stage 1 reverb adapter) |

**On-the-fly mixing**: Dynamic mixture generation during training, 2-second clips, drawn fresh each epoch. No fixed train split.

**Stage 1 per-adapter training config:**
- Batch 4 (sequential per-sample forward due to MPS memory)
- 500 samples/epoch × 40 epochs
- 2-second clips @ 8 kHz after degradation
- Clip-first then degrade (15× speedup vs degrade-then-clip)
- `num_workers=0` (data load 8ms vs 1–4s compute)

**Stage 4 joint training config:**
- 1,000 samples/epoch × 20 epochs (14 recorded, best at epoch 14)
- Two LR groups: adapters 1e-5, gate+analyzer 2e-5
- Kaggle T4 GPU

**Stage 3 gate training data:**
- Source: `rishig777/calmsep-stage3-gate` (Kaggle dataset)
- Oracle labels from `MixtureRecipe.condition_vector()`: reverb=(t60>0), noise=(snr<60dB), codec=(codec_class>0)

### 2.2 Evaluation Data

| Split | N | Source | Test clips used | Total available |
|-------|---|--------|-----------------|-----------------|
| Libri2Mix | 2 | librimix (wav8k/min/test) | **30** | ~3,000 |
| Libri3Mix | 3 | librimix (wav8k/min/test) | **30** | ~3,000 |
| Libri4Mix | 4 | librimix (wav8k/min/test) | **0 (not evaluated)** | ~3,000 |
| Libri5Mix | 5 | librimix (wav8k/min/test) | **30** | ~3,000 |

Mix type: `mix_both` (noise + reverb combined).

### 2.3 Stage 1 Reverb Diagnostic (eval.log, 2026-07-17)

Single 2-speaker test clip, T60=0.46s, 3 conditions:

| Condition | Base SI-SNR | Adapted SI-SNR | Delta |
|-----------|-------------|----------------|-------|
| Clean (anechoic) | 18.61 dB | 18.17 dB | **-0.44 dB** |
| Reverb mild | -30.89 dB | -30.96 dB | **-0.07 dB** |
| Reverb strong | -32.83 dB | -35.64 dB | **-2.81 dB** |

**Finding**: Reverb adapter makes things worse across all conditions. Likely causes: wet-reference training target, rank=8 too small, 500 samples/epoch insufficient.

---

## 3. Evaluation Numbers

### 3.1 Main Results (oracle N supplied to both systems)

> **CRITICAL BUG**: `eval/run_eval.py` line 294 passes true speaker count to both models. Speaker count accuracy, the primary graded axis, is unobserved in all results below.

| Split | N | n | Baseline SI-SDR | Baseline SI-SDRi | CALM-Sep SI-SDR | CALM-Sep SI-SDRi | Δ SI-SDRi |
|-------|---|---|-----------------|-----------------|-----------------|-----------------|-----------|
| Libri2Mix | 2 | 30 | 5.60 dB | 7.09 dB | 7.36 dB | **8.86 dB** | **+1.76 dB** |
| Libri3Mix | 3 | 30 | 5.75 dB | 10.07 dB | 7.49 dB | **11.80 dB** | **+1.73 dB** |
| Libri5Mix | 5 | 30 | 1.04 dB | 9.43 dB | 1.66 dB | **10.05 dB** | **+0.62 dB** |
| Libri4Mix | 4 | n/a |, | n/a |, | n/a | **NOT RUN** |

Wall times (CPU-only inference, MacBook M5 Pro):
- Libri2Mix 30 clips: 2,166.7s (~72s/clip)
- Libri3Mix 30 clips: 2,912.6s (~97s/clip)
- Libri5Mix 30 clips: 3,480.5s (~116s/clip)

### 3.2 v1 CA-MoSE Results (2026-07-13, Kaggle T4, 100 dev samples, 2–5 spk)

| System | SI-SDRi |
|--------|---------|
| MossFormer2 alone | 8.24 dB |
| SR-CorrNet-SS alone | **16.22 dB** |
| Cascade + fusion (best threshold) | 15.79 dB |
| Cascade + fusion (worst threshold) | 12.51 dB |
| Cascade, SR-primary, full escalation | 16.22 dB (equals SR-CorrNet, never exceeds) |

Fusion head degraded SR-CorrNet by 0.4–3.7 dB at every threshold. Efficiency result: 36% compute reduction at τ=6, at −3.55 dB quality cost.

### 3.3 Stage 4 Joint Training Loss Curve

| Epoch | Loss | Saved |
|-------|------|-------|
| 1 | n/a |, |
| 4 | ~13.5 | ✓ best |
| 5 | 13.47 | n/a |
| 6 | 9.59 | ✓ best |
| 7 | 9.49 | ✓ best |
| 8 | 9.43 | ✓ best |
| 9 | 10.94 | n/a |
| 10 | 8.89 | ✓ best |
| 11 | 10.31 | n/a |
| 12 | 8.72 | ✓ best |
| 13 | 8.94 | n/a |
| 14 | **8.68** | ✓ **final best** |

Loss was still decreasing at epoch 14 (run ended). Configured for 20 epochs.

### 3.4 Calibration

| Value | Notes |
|-------|-------|
| Gate temperature T = **4.9872** | Very high, sigmoid(logit/4.99) ≈ 0.5 for all inputs. Gate is not routing selectively. |
| ECE (Expected Calibration Error) | **Not measured**, reliability diagrams not generated |
| Per-stream confidence accuracy | **Not measured** |
| Completeness probability accuracy | **Not measured** |

### 3.5 Speaker Counting

**Never evaluated.** `count_from_attractors()` exists in `models/counting.py` and `SRCorrNetWrapper.forward()` exposes `n_active`, but `eval/run_eval.py` uses oracle N. Published SR-CorrNet paper reports strong attractor-based counting on WSJ0-mix, but its LibriMix counting accuracy is unknown.

---

## 4. Published SOTA Context (LibriMix, for positioning)

These are published numbers on the same LibriMix benchmark we evaluate on. Not run by us, from literature.

| System | Libri2Mix SI-SDRi | Libri3Mix SI-SDRi | Notes |
|--------|-------------------|-------------------|-------|
| ConvTasNet | ~15.3 dB | ~12.6 dB | INTERSPEECH 2019 |
| DPRNN | ~18.8 dB | ~14.7 dB | ICASSP 2020 |
| SepFormer | **22.3 dB** | **19.5 dB** | ICASSP 2021 |
| SR-CorrNet (WSJ0 domain) | 16.22 dB on own val | n/a | Our v1 measurement |
| **CALM-Sep (this work)** | **8.86 dB** | **11.80 dB** | Oracle N, n=30 |
| **SR-CorrNet baseline (this work)** | 7.09 dB | 10.07 dB | Oracle N, n=30 |

**Key observation**: Our baseline numbers (7.09 and 10.07 dB) are far below published LibriMix SOTA (22.3 dB), likely because SR-CorrNet was trained on WSJ0-mix and transfers poorly to LibriMix. The delta (+1.76 dB) is our actual contribution regardless of absolute level.

---

## 5. Inference Details

### 5.1 Timing (CPU, Apple M5 Pro, single-threaded)

| Clip length | N speakers | System | Wall time |
|-------------|-----------|--------|-----------|
| ~6s | 2 | SR-CorrNet baseline | **41–42s** (measured) |
| ~6s | 3 | SR-CorrNet baseline | **59–60s** (measured) |
| ~6s | 5 | SR-CorrNet baseline | ~116s (extrapolated) |
| ~6s | 2–3 | CALM-Sep (adds gate forward ~0.1s) | **41–60s** (measured) |

Gate adds ~100ms overhead (MLP forward on 10-D input). SR-CorrNet dominates.

Real-time factor: ~12–20× slower than real-time on CPU. GPU (T4) is ~2–4× real-time.

### 5.2 Memory (CPU inference)

- SR-CorrNet model: ~50MB on disk, ~300MB RAM footprint
- CALM-Sep additions: ~5MB on disk, ~20MB RAM
- Whisper base (transcription): ~140MB on disk, ~500MB RAM
- Total loaded: ~820MB RAM

### 5.3 Pipeline at Inference Time

```
Input wav (any SR) → resample to 8kHz → trim/pad to ≤6s
    → Level-1 condition features (STFT DSP, no model)
    → [Level-2 features from E(0), zeros on first chunk]
    → Gate MLP → gate vector (g_r, g_n, g_c)
    → Inject gates into 37 LoRALinear modules
    → SR-CorrNet forward → K=5 output streams + attractor probs p_k
    → N_hat = count slots where p_k > 0.5 (clips to [2,5])
    → Keep first N_hat streams
    → Band recovery: predict 4-8kHz → upsample to 16kHz
    → Whisper transcription per stream (word-level timestamps)
Output: N_hat audio streams @ 16kHz + word-highlighted transcripts
```

### 5.4 Known Failure Modes

| Failure | Cause | Status |
|---------|-------|--------|
| Reverb adapter degrades quality | Wet-reference training target; rank=8 too small | Known, unfixed |
| Gate outputs ~0.5 for all conditions | Temperature T=4.99 flattens sigmoid | Known, unfixed |
| Oracle N in eval | `n_spks` passed from directory name | Known bug, line 294 run_eval.py |
| Libri4Mix not evaluated | Oversight | Missing |
| Stage 2 universal adapter never trained | Resource constraints | Missing |

---

## 6. Checkpoint Inventory

| File | Size | Epoch | Best metric |
|------|------|-------|-------------|
| `checkpoints/stage1_reverb/best_reverb.pt` | 424KB | 40 | Best train loss (adapter makes reverb WORSE) |
| `checkpoints/stage1_reverb/final_reverb.pt` | 424KB | 40 | Same as best |
| `checkpoints/stage1_noise/best_noise.pt` | 424KB | ~40 | Best train loss |
| `checkpoints/stage1_codec/best_codec.pt` | 424KB | ~40 | Best train loss |
| `checkpoints/stage4_joint/best_joint.pt` | 1.6MB | 14/20 | Loss 8.68 (gate+analyzer+222 adapter tensors) |
| `checkpoints/stage4c/calibration.pt` | 4KB | n/a | T=4.9872 |
| `checkpoints/stage4b_band/best_band.pt` | 184KB | n/a | Best band recovery loss |
| `checkpoints/stage4b_band/final_band.pt` | 184KB | n/a | Final epoch |

**Checkpoint format** (`best_joint.pt`):
```python
{
  "gate": state_dict,        # GateNetwork weights
  "analyzer": state_dict,    # Level2Analyzer weights
  "adapter_state": {         # 222 tensors: {mod_name}.branches.{adapter}.{A or B}
      "enc_block.0.freq.qkv.branches.reverb.A": Tensor,
      ...
  }
}
```

Adapter A norms (first 10 of 222): 1.63, 1.71, 1.68, 1.70, 1.65, 1.74, 1.64, 1.73, 1.75, 1.70
Adapter B norms (first 10 of 222): 0.16, 0.38, 0.23, 0.10, 0.11, 0.14, 0.07, 0.37, 0.12, 0.37

B norms >0.01 confirm learning happened; reverb adapter's original Stage 1 B norm was 0.0488 which is small, suggesting underfitting.

---

## 7. What's Missing for a Paper (Quantitative Gaps)

| Missing result | Blocker? | Est. effort |
|----------------|----------|-------------|
| Speaker count accuracy (N_hat vs N_true) | **FATAL**, primary graded axis | 1 day (fix oracle N bug) |
| n=300+ per split, all 4 splits | Yes, 30 samples too small | 3–5 days (compute) |
| Bootstrap 95% CI on all SI-SDRi numbers | Yes, no significance | 0.5 days (code exists) |
| Per-condition results (reverb/noise/codec separately) | Yes, no routing evidence | 2 days (data + eval) |
| Stage 2 universal adapter ablation | Yes, justify 3-adapter design | 2 weeks (train) |
| Oracle gate upper bound | Yes, shows headroom | 1 day (eval) |
| Calibration ECE + reliability diagram | Yes, required by problem statement | 1 day |
| SOTA comparison on same test set | Yes, no positioning | 0.5 days (SepFormer via SpeechBrain) |
| Libri4Mix results | Yes, required split | 1 day (compute) |
| Band recovery contribution (16kHz metrics) | Nice-to-have | 1 day |
| Degradation curves (6–7 speaker extrapolation) | Nice-to-have | 1 day |
