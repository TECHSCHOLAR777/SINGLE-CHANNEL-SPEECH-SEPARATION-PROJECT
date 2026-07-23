#!/usr/bin/env bash
# retrain_all_stage1_fixed.sh
# Complete Stage 1 re-training on Lightning AI GPU studio.
# Run from: /teamspace/studios/this_studio (the studio root, NOT calmsep/)
#
# Trains all three adapters in sequence:
#   1. reverb  — retrain from scratch (previous run had RNG-collision bug)
#   2. noise   — retrain from scratch (previous run aborted at epoch 2)
#   3. codec   — first-time train (requires ffmpeg — installed below)
#
# Bugs fixed vs original continue_stage1.sh:
#   BUG1: worker RNG collision  → worker_init_fn, each worker now gets unique seed
#   BUG2: noise glob per sample → pre-computed once at dataset init (28k → 1 stat call)
#   BUG3: silent clean fallback → raises FileNotFoundError when noise files missing
#   BUG4: codec without ffmpeg  → installs ffmpeg, then errors if still missing

set -euo pipefail

STUDIO=/teamspace/studios/this_studio
REPO=$STUDIO/calmsep
PYTHON=$STUDIO/venv/bin/python
DATA=$STUDIO/data/calmsep-8k
CKPT=$STUDIO/checkpoints
LOG=$STUDIO/logs

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$CKPT/stage1_reverb" "$CKPT/stage1_noise" "$CKPT/stage1_codec" "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/pipeline_fixed.log"; }

# ── shared args ───────────────────────────────────────────────────────────────
LIBRI="$DATA/librispeech-8k"
RIR="$DATA/rirs/bank.json"
NOISE="$DATA/noise"

# 2000 samples/epoch × 2 workers with unique seeds = full diversity every epoch
SPE=2000
WORKERS=2

# ─────────────────────────────────────────────────────────────────────────────
# 1. REVERB ADAPTER — 60 epochs
#    Increased from 40 because the previous run had only 250 unique samples/epoch
#    (BUG1 RNG collision halved effective training data). 60 epochs × 2000 samples
#    = 4× more unique training signal than the original buggy run.
# ─────────────────────────────────────────────────────────────────────────────
log "=== [1/3] REVERB adapter — 60 epochs from scratch ==="

PYTHONUNBUFFERED=1 $PYTHON -u -m train.stage1_single \
    --adapter reverb \
    --data-root   "$LIBRI" \
    --rir-bank    "$RIR" \
    --checkpoint-dir "$CKPT/stage1_reverb" \
    --device cuda \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 60 \
    --samples-per-epoch "$SPE" \
    --num-workers "$WORKERS" \
    --bf16 \
    2>&1 | tee "$LOG/stage1_reverb_fixed.log"

log "Reverb done → $CKPT/stage1_reverb/best_reverb.pt"

# ─────────────────────────────────────────────────────────────────────────────
# 2. NOISE ADAPTER — 40 epochs from scratch
#    Previous run reached only epoch 2 before the GPU studio was paused.
#    Fixed: noise_files are pre-computed once, not re-globbed per sample.
# ─────────────────────────────────────────────────────────────────────────────
log "=== [2/3] NOISE adapter — 40 epochs from scratch ==="

PYTHONUNBUFFERED=1 $PYTHON -u -m train.stage1_single \
    --adapter noise \
    --data-root   "$LIBRI" \
    --rir-bank    "$RIR" \
    --noise-dir   "$NOISE" \
    --checkpoint-dir "$CKPT/stage1_noise" \
    --device cuda \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 40 \
    --samples-per-epoch "$SPE" \
    --num-workers "$WORKERS" \
    --bf16 \
    2>&1 | tee "$LOG/stage1_noise_fixed.log"

log "Noise done → $CKPT/stage1_noise/best_noise.pt"

# ─────────────────────────────────────────────────────────────────────────────
# 3. CODEC ADAPTER — install ffmpeg, then train 40 epochs
#    ffmpeg is required for real Opus/AAC/AMR-NB roundtrip.
#    Without it, every sample falls back to mu-law (G.711) which is a different
#    degradation and the adapter learns the wrong artifacts.
# ─────────────────────────────────────────────────────────────────────────────
log "=== [3/3] Installing ffmpeg ==="

if command -v ffmpeg &>/dev/null; then
    log "ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
elif command -v conda &>/dev/null; then
    log "Installing ffmpeg via conda..."
    conda install -y -c conda-forge ffmpeg 2>&1 | tail -3
    log "ffmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
elif command -v apt-get &>/dev/null; then
    log "Installing ffmpeg via apt-get..."
    apt-get install -y ffmpeg 2>&1 | tail -3
    log "ffmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
else
    log "ERROR: Cannot install ffmpeg automatically. Install it manually, then re-run."
    exit 1
fi

log "=== [3/3] CODEC adapter — 40 epochs ==="

PYTHONUNBUFFERED=1 $PYTHON -u -m train.stage1_single \
    --adapter codec \
    --data-root   "$LIBRI" \
    --rir-bank    "$RIR" \
    --checkpoint-dir "$CKPT/stage1_codec" \
    --device cuda \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 40 \
    --samples-per-epoch "$SPE" \
    --num-workers "$WORKERS" \
    --bf16 \
    2>&1 | tee "$LOG/stage1_codec_fixed.log"

log "Codec done → $CKPT/stage1_codec/best_codec.pt"

# ─────────────────────────────────────────────────────────────────────────────
log "========================================================"
log "All Stage 1 adapters complete. Checkpoints:"
ls -lh "$CKPT"/stage1_*/best_*.pt 2>/dev/null | tee -a "$LOG/pipeline_fixed.log"
log "========================================================"
