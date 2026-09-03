# Scripts

Operational entry points. Everything reusable lives in `src/coralsep/`; these are the things you run once, in order, from the project root.

They stay flat rather than nested. Ten files with a table beside them is easier to scan than four directories holding two files each, and it keeps the import paths short.

---

## Getting data

| Script | What it does | Run when |
|---|---|---|
| `download_step1.py` | Downloads the raw corpora with resume support and a live ETA | 1️⃣ first |
| `prepare_all_data.sh` | Drives the full preparation chain, resumable, survives a dropped SSH session | 2️⃣ after download |
| `preflight_data.py` | Verifies the LibriMix layout **before** generation, so a metadata mismatch fails in seconds rather than hours | 3️⃣ before generating |
| `slice_for_kaggle.py` | Cuts a Kaggle-uploadable subset, roughly 900 MB, with a recorded seed | 4️⃣ before training on Kaggle |

`preflight_data.py` exists because both data-generation failures in this project had the same shape: a metadata mismatch that only surfaced after hours of work.

## Getting the model

| Script | What it does |
|---|---|
| `download_checkpoint.py` | Downloads the frozen backbone, verifies its SHA-256, writes hash and path into `configs/base_checkpoint.yaml` |
| `upload_stage3_gate.sh` | Publishes a trained Stage 3 gate checkpoint to Kaggle |

## Training

| Script | What it does | Status |
|---|---|:--:|
| `train_local_mps.sh` | Trains all three Stage 1 adapters in sequence on Apple Silicon | 🟢 |
| `retrain_all_stage1_fixed.sh` | Re-runs Stage 1 after the four data-pipeline bugs were fixed | 📜 historical |

## Checking

| Script | What it does | Status |
|---|---|:--:|
| `run_baseline.py` | Scores the frozen backbone alone on one LibriMix split. The number every adapter configuration must beat. | 🟢 |
| `validate_alignment.py` | Identity-lock check: counts how often a stitched track changes which speaker it maps to | 🟠 partial |

`validate_alignment.py` runs only with `--skip-pair`. Its paired cheap-versus-expensive mode belonged to the retired v1 cascade and exits with an explanation. The identity-switch metric it implements is worth reading: the first version of that metric was unsound because it counted silent tracks and compared every window against window zero instead of its predecessor, and the corrected version is documented at the top of the file.

---

## Console entry points

Installed by `pip install -e .`, so these work from anywhere:

```bash
coralsep-baseline --data-root <librimix>/Libri2Mix --max-samples 30
coralsep-infer    --input mixture.wav --checkpoints checkpoints/
coralsep-demo
```
