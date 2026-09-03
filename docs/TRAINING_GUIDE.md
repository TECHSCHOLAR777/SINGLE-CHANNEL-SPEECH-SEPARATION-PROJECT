# CoRAL-Sep Training Guide

End-to-end training sequence from raw data to a calibrated, deployable pipeline.
All notebooks are in `notebooks/` and target Kaggle T4/P100 GPUs.

---

## Prerequisites

| Item | Note |
|------|------|
| GPU | T4 (16 GB) minimum; P100 preferred for Stage 4 |
| Python | 3.10+ |
| Key deps | `torch`, `soundfile`, `scipy`, `huggingface_hub`, `transformers` |
| HF checkpoint | `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` (auto-downloaded) |
| DNSMOS model | `sig_bak_ovrl.onnx` from DNS-Challenge repo (optional, for eval only) |

Set the following path constants at the top of every notebook before running:

```python
DATA_ROOT  = "/kaggle/input/calmsep-8k"   # staged dataset root (see Phase 0)
REPO_PATH  = "/kaggle/input/calmsep-code" # this codebase
```

---

## Phase 0, Data preparation (run once, CPU job)

These scripts must complete **before** any training notebook.
Run them locally or in a CPU-only Kaggle session. They do not require GPU.

### Step 0.1, LibriSpeech → 8 kHz

```bash
python data/prepare_librispeech_8k.py \
    --input-dir  /data/LibriSpeech \
    --output-dir /data/LibriSpeech_8k \
    --splits train-clean-100 train-clean-360 dev-clean test-clean
```

Writes `manifest.json` used by the mixer and eval generator.

### Step 0.2, Stage noise (WHAM! + DNS-4)

```bash
# WHAM! + DNS-4 at 8 kHz and 16 kHz side-by-side
python data/prepare_noise_staging.py \
    --wham-dir   /data/wham_noise \
    --dns4-dir   /data/dns4 \
    --output-dir /data/calmsep-8k/noise

# DNS-4 stratified subset (optional cap at 20 GB)
python data/prepare_dns4.py \
    --out-dir /data/calmsep-8k/dns4_subset \
    --target-gb 20
```

### Step 0.3, BUT ReverbDB (measured RIRs)

```bash
python data/prepare_but_reverbdb.py \
    --output-dir /data/calmsep-8k/but_reverbdb
```

Downloads SLR17 from OpenSLR, resamples to 8 kHz, writes `but_bank.json`.

### Step 0.4, Simulated RIR bank

```bash
python -c "
from data.rir_bank import build_rir_bank
build_rir_bank(n_rooms=2000, out_path='data/rirs/bank.json')
"
```

### Step 0.5, Fixed evaluation manifests

Pre-built JSONL manifests are already committed in `data/fixed_eval/` (16 files).
If you need to regenerate them:

```bash
python data/fixed_eval_generator.py \
    --librispeech-8k /data/LibriSpeech_8k \
    --noise-dir      /data/calmsep-8k/noise \
    --rir-bank       data/rirs/bank.json \
    --out-dir        data/eval
```

---

## Phase 1, Stage 1: Single-adapter training

**Notebook:** `notebooks/stage1_train_adapter.ipynb`
**Runs:** 3 times (once per adapter)
**GPU time per run:** ~6–8 hours on T4
**Config files:** `configs/adapters/{reverb,noise,codec}.yaml`

Train each LoRA adapter independently on its own degradation type.
The base SR-CorrNet weights are **frozen throughout**.

### Run order

| Run | `ADAPTER_NAME` | Config | Output |
|-----|---------------|--------|--------|
| 1 | `reverb` | `configs/adapters/reverb.yaml` | `adapters/reverb_adapter.pt` |
| 2 | `noise` | `configs/adapters/noise.yaml` | `adapters/noise_adapter.pt` |
| 3 | `codec` | `configs/adapters/codec.yaml` | `adapters/codec_adapter.pt` |

Change `ADAPTER_NAME` and `CONFIG_PATH` at the top of the notebook for each run.

**Key hyperparameters (from config):**
- LR = 1e-4, epochs = 40, batch = 8
- Co-activation gate for inactive adapters: U(0.0, 0.2), do not change
- Severity holdout: T60 > 0.9 s for reverb; SNR < −4 dB for noise (10% allowed)
- Codec held-out combos: `reverb+codec`, `noise+codec` never seen here

**What to check after each run:**
```
Adapter keys: [...]   # should list A and B matrices for each target module
```

---

## Phase 1b, Stage 2: Universal adapter baseline (decision gate)

**Notebook:** `notebooks/stage2_universal.ipynb`  
  *(also available as `notebooks/P1b_train_universal_adapter.ipynb` for reference)*
**GPU time:** ~8–10 hours on T4
**Prerequisite:** All 3 Stage 1 adapters saved

Train a single adapter on all conditions combined.

**Decision rule (irreversible before Stage 3):**
> If the universal adapter is within **0.5 dB SI-SDRi** of the best per-adapter
> on its primary condition → adopt the universal adapter (simpler system).
> Otherwise, keep per-adapter routing.

The training script prints the comparison table. Read it before moving to Stage 3.
Record the verdict in `docs/decisions.md`.

---

## Phase 2, Stage 3: Gate + Level-2 analyzer

**Notebook:** `notebooks/stage3_gate.ipynb`
**GPU time:** ~4–6 hours on T4
**Prerequisites:** Stage 1 adapters (all 3) in `STAGE1_DIR`
**Config:** `configs/gate.yaml`

Trains `GateNetwork` (10-D → 3 gate values) and `Level2Analyzer` (T60 + count
prior from pooled encoder E(0)) jointly. Adapter and base weights stay frozen.

```python
STAGE1_DIR     = "/kaggle/working/adapters"
CHECKPOINT_DIR = "/kaggle/working/gate"
```

**Key hyperparameters:**
- LR = 5e-5 (half of Stage 1), epochs = 30, batch = 8
- Held-out combos: `reverb+codec` and `noise+codec` withheld from gate training
- Gate output: sigmoid × 1.5, L1 sparsity α = 0.001

**What to check after run:**
```
gate_net.pt:        OK
level2_analyzer.pt: OK
```

---

## Phase 3, Stage 4: Joint fine-tuning

**Notebook:** `notebooks/stage4_joint.ipynb`
**GPU time:** ~10–14 hours on T4 / 8–10 hours on P100
**Prerequisites:** Stage 1 adapters + Stage 3 gate/analyzer checkpoints
**Config:** `configs/default.yaml`

Fine-tunes all components together: adapters + gate + Level-2 analyzer.

```python
STAGE1_DIR     = "/kaggle/working/adapters"
STAGE3_DIR     = "/kaggle/working/gate"
CHECKPOINT_DIR = "/kaggle/working/joint"
LR             = 1e-5   # MUST be exactly 1/10 of Stage 1 LR, do not change
```

**Key hyperparameters:**
- LR = **1e-5** (1/10 of Stage 1, hardcoded by blueprint)
- Epochs = 20, batch = 8
- `--use-olora` flag: adds O-LoRA cross-adapter interference penalty (recommended)
- Held-out combos still withheld: `reverb+codec`, `noise+codec`

**What to check after run:**
```
joint_reverb_adapter.pt:  OK
joint_noise_adapter.pt:   OK
joint_codec_adapter.pt:   OK
joint_gate_net.pt:        OK
joint_level2_analyzer.pt: OK
```

---

## Phase 4, Calibration fitting

**Script:** `train/calibrate.py` (CLI, no notebook)
**Prerequisites:** Stage 4 joint checkpoints
**Purpose:** Fit temperature scaling, isotonic confidence calibration, Platt completeness
  calibration, and Mahalanobis OOD detector on held-out logits/labels.

```bash
# Dry-run on synthetic data (smoke test)
python -m train.calibrate --dry-run --out-dir calibration/artifacts

# Real run, replace synthetic data in calibrate.py with held-out dumps from Stage 4
python -m train.calibrate --out-dir calibration/artifacts
```

Writes artifacts to `calibration/artifacts/`:
- `temperature.pt`, temperature scalar
- `confidence_isotonic.pkl`, isotonic regressor
- `completeness_platt.pkl`, Platt sigmoid
- `mahalanobis_ood.pkl`, Mahalanobis OOD detector

---

## Phase 5, Evaluation

**Notebook:** `notebooks/eval_matrix.ipynb`
**GPU time:** ~2–4 hours on T4
**Prerequisites:** Stage 4 joint checkpoints; calibration artifacts (optional but
  recommended for calibrated confidence metrics)

Runs the full **8-condition × 4-N evaluation matrix**:

| Conditions | Speaker counts |
|-----------|---------------|
| clean, reverb, noise, codec, reverb+noise, reverb+codec, noise+codec, reverb+noise+codec | N ∈ {2, 3, 4, 5} |

```python
CHECKPOINT_DIR = "/kaggle/working/joint"
EVAL_MANIFEST  = "data/eval/fixed_eval_manifest.jsonl"
OUTPUT_DIR     = "/kaggle/working/results"
DNSMOS_MODEL   = None  # path to sig_bak_ovrl.onnx for DNSMOS, else None
```

**Outputs:**
- `results/eval_matrix.jsonl`, per-mixture records
- `results/summary.csv`, aggregated by (condition, N)
- `results/bootstrap_ci.json`, 95% BCa confidence intervals (BLUEPRINT §9.1)

**Metrics computed:** SI-SDRi, SDRi, DNSMOS (if model available), count accuracy,
completeness probability, OOD flag rate, cardinality-aware score (§9.2).

---

## Complete sequence at a glance

```
Phase 0  (CPU, once)
  0.1  prepare_librispeech_8k.py
  0.2  prepare_noise_staging.py  +  prepare_dns4.py
  0.3  prepare_but_reverbdb.py
  0.4  rir_bank.build_rir_bank()
  0.5  fixed_eval_generator.py   (or use committed data/fixed_eval/ manifests)

Phase 1  (GPU × 3)
  stage1_train_adapter.ipynb  [ADAPTER_NAME=reverb]   → adapters/reverb_adapter.pt
  stage1_train_adapter.ipynb  [ADAPTER_NAME=noise]    → adapters/noise_adapter.pt
  stage1_train_adapter.ipynb  [ADAPTER_NAME=codec]    → adapters/codec_adapter.pt

Phase 1b (GPU × 1, decision gate)
  stage2_universal.ipynb  →  verdict in docs/decisions.md

Phase 2  (GPU × 1)
  stage3_gate.ipynb  →  gate_net.pt + level2_analyzer.pt

Phase 3  (GPU × 1)
  stage4_joint.ipynb  →  joint_*.pt (5 files)

Phase 4  (CPU)
  python -m train.calibrate  →  calibration/artifacts/

Phase 5  (GPU × 1)
  eval_matrix.ipynb  →  results/
```

**Total GPU time estimate:** 30–45 hours across all stages on T4.

---

## Checkpoint directory layout

After all phases complete, your working directory should look like:

```
/kaggle/working/
├── adapters/
│   ├── reverb_adapter.pt
│   ├── noise_adapter.pt
│   ├── codec_adapter.pt
│   └── universal_adapter.pt       # Stage 2 (kept for comparison)
├── gate/
│   ├── gate_net.pt
│   └── level2_analyzer.pt
├── joint/
│   ├── joint_reverb_adapter.pt
│   ├── joint_noise_adapter.pt
│   ├── joint_codec_adapter.pt
│   ├── joint_gate_net.pt
│   └── joint_level2_analyzer.pt
├── calibration/artifacts/
│   ├── temperature.pt
│   ├── confidence_isotonic.pkl
│   ├── completeness_platt.pkl
│   └── mahalanobis_ood.pkl
└── results/
    ├── eval_matrix.jsonl
    ├── summary.csv
    └── bootstrap_ci.json
```

---

## Running the demo after training

```bash
python demo/app.py \
    --joint-dir  /kaggle/working/joint \
    --calib-dir  calibration/artifacts
```

The Gradio UI shows gate routing bars, per-stream transcripts (Whisper), and a
diagnostics JSON panel with count estimate, completeness probability, and OOD flag.

---

## Common issues

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` in Stage 4 | Reduce `BATCH_SIZE` to 4; gradient accumulation × 2 |
| `FileNotFoundError` on RIR bank | Run Phase 0.3–0.4 first; check `but_bank.json` exists |
| `AttributeError: ood_flag` in demo | Ensure `schemas/separation_result.py` has CoRAL-Sep fields (already in integration branch) |
| Stage 3 gate output always `[0,0,0]` | Check `in_dim=10` in `configs/gate.yaml`; Level-1 + Level-2 must both feed the gate |
| BCa bootstrap `nan` CIs | Need ≥ 50 samples per bucket; increase `n_per_bucket` in `configs/eval.yaml` |
| `SSInference.from_pretrained` crash | Ensure `checkpoint_path=` kwarg is used, not `config=` (known upstream quirk) |
