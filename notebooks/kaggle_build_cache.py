# CA-MoSE — Kaggle Cache Builder
#
# Paste each block below into its own Kaggle notebook cell.
#
# Settings you MUST set in the Kaggle notebook sidebar before running:
#   Accelerator : GPU T4 x2  (or P100)
#   Internet    : ON  (requires phone verification on your Kaggle account)
#
# What this does, and why:
#   The experts are FROZEN. Running them every epoch is the dominant cost of
#   training and it recomputes the identical tensors every time. This notebook
#   runs them ONCE over a clean 3-speaker subset, caches the outputs, and saves
#   the cache as a Kaggle Dataset. Every training session after this attaches the
#   cache read-only and trains the ~2M trainable params with no expert loaded.
#
#   We generate mix_clean, NOT mix_both. mix_both needs WHAM! noise (50 GB during
#   generation) and Kaggle cannot hold it. mix_clean needs zero WHAM and is
#   exactly what the L1 tier calls for (3 speakers, clean, anechoic), which is
#   what you are training on first anyway. Noise comes back for L2/L3 later, on
#   real hardware.
#
# Budget: roughly 2-3 hours end to end. Well inside the 12-hour session cap.


# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — repo + deps
# ══════════════════════════════════════════════════════════════════════════════
!git clone -q https://github.com/TECHSCHOLAR777/SINGLE-CHANNEL-SPEECH-SEPARATION-PROJECT.git /kaggle/working/camose
%cd /kaggle/working/camose
!pip install -q -e ".[experts]" 2>&1 | tail -2
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!df -h /kaggle/working /kaggle/temp | sed -n '1,3p'


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — LibriSpeech (scratch disk, NOT /kaggle/working: it is not the output)
# ══════════════════════════════════════════════════════════════════════════════
# train-clean-100 is 6.3 GB. dev/test are ~350 MB each. No WHAM: mix_clean only.
import os
os.makedirs("/kaggle/temp/ls", exist_ok=True)
%cd /kaggle/temp/ls
!wget -q --show-progress https://www.openslr.org/resources/12/train-clean-100.tar.gz
!wget -q --show-progress https://www.openslr.org/resources/12/dev-clean.tar.gz
!wget -q --show-progress https://www.openslr.org/resources/12/test-clean.tar.gz
!for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done
!du -sh /kaggle/temp/ls/LibriSpeech


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — generate a Libri3Mix mix_clean SUBSET
# ══════════════════════════════════════════════════════════════════════════════
# Full Libri3Mix is 332 GB. We take a subset: enough to train 2M params, not
# enough to blow the disk. Tune N_TRAIN / N_DEV if you have headroom.
N_TRAIN, N_DEV, N_TEST = 1500, 200, 200

!git clone -q https://github.com/JorisCos/LibriMix.git /kaggle/temp/LibriMix

import pandas as pd, pathlib
src = pathlib.Path("/kaggle/temp/LibriMix/metadata/Libri3Mix")
dst = pathlib.Path("/kaggle/temp/meta/Libri3Mix"); dst.mkdir(parents=True, exist_ok=True)

for name, n in [("train-100", N_TRAIN), ("dev", N_DEV), ("test", N_TEST)]:
    f = src / f"mixture_{name}_mix_clean.csv"
    df = pd.read_csv(f).head(n)
    df.to_csv(dst / f.name, index=False)
    print(f"{f.name}: {len(df)} rows")

# preflight: does every file these CSVs reference actually exist?
# Both previous data runs died three hours in on exactly this. Takes seconds.
!python /kaggle/working/camose/scripts/preflight_data.py \
    --metadata-dir /kaggle/temp/meta/Libri3Mix \
    --librispeech-dir /kaggle/temp/ls/LibriSpeech

# generate. mix_clean => --types mix_clean, no --wham_dir needed.
%cd /kaggle/temp/LibriMix
!python scripts/create_librimix_from_metadata.py \
    --librispeech_dir /kaggle/temp/ls/LibriSpeech \
    --wham_dir /kaggle/temp/ls \
    --metadata_dir /kaggle/temp/meta \
    --librimix_outdir /kaggle/temp/data \
    --n_src 3 --freqs 16k --modes max --types mix_clean
!du -sh /kaggle/temp/data/Libri3Mix


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — run the frozen experts ONCE, cache the tensors
# ══════════════════════════════════════════════════════════════════════════════
# MossFormer2 is a 2-speaker checkpoint, so target_speakers=3 residual-pads it.
# The expensive expert is Hungarian-aligned onto MossFormer2's speaker order at
# build time; the fusion head assumes that order and cannot check it.
%cd /kaggle/working/camose
os.environ["HF_HOME"] = "/kaggle/temp/hf"

for split in ["train", "dev"]:
    !python scripts/build_train_cache.py \
        --librimix-root /kaggle/temp/data/Libri3Mix \
        --subset {split} \
        --output /kaggle/working/cache/{split} \
        --num-speakers 3 \
        --crop-sec 4.0 --crops-per-utterance 2 \
        --device cuda

!du -sh /kaggle/working/cache/*
!cat /kaggle/working/cache/train/manifest.json


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — sanity: the cache actually drives a training step
# ══════════════════════════════════════════════════════════════════════════════
import torch
from train.cached_dataset import cached_train_loader
from train.trainer import CAMoSETrainer, CAMoSETrainable
from train.losses import CompositeLoss
from models.cascade_gate import CascadeGate

loader = cached_train_loader("/kaggle/working/cache/train", batch_size=4)
trainer = CAMoSETrainer(
    model=CAMoSETrainable(),
    gate=CascadeGate(tau=12.0, signal="min"),
    loss_fn=CompositeLoss(),
    device="cuda" if torch.cuda.is_available() else "cpu",
)
batch = next(iter(loader))
breakdown, n_escalated = trainer.train_step(batch)
print(f"loss={breakdown.total.item():.4f}  escalated={n_escalated}/{batch.mixture.shape[0]}")
assert torch.isfinite(breakdown.total), "loss is not finite; stop and investigate"
print("cache is live. Save this notebook's output as a Kaggle Dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — shrink the output so Kaggle will save it
# ══════════════════════════════════════════════════════════════════════════════
# Only /kaggle/working is saved (20 GB cap). Drop the cloned repo from the
# output; keep the cache. Then: Save Version -> Save & Run All, and afterwards
# "New Dataset" from the notebook output.
!rm -rf /kaggle/working/camose/.git
!du -sh /kaggle/working
!echo "cache is the deliverable:" && du -sh /kaggle/working/cache
