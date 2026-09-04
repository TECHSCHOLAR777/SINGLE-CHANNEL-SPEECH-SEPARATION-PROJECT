#!/usr/bin/env bash
# train_local_mps.sh, Stage 1 adapter training on Apple M5 Pro MPS
#
# Estimated time (M5 Pro, 2s clips, 1000 samp/epoch @ ~4300 samp/hr):
#   reverb  40 epochs × ~14 min/epoch ≈  9.3 h
#   noise   40 epochs × ~14 min/epoch ≈  9.3 h
#   codec   40 epochs × ~14 min/epoch ≈  9.3 h
#   Total all adapters                ≈ 28 h
#
# Usage:
#   bash train_local_mps.sh              # all three adapters
#   bash train_local_mps.sh reverb       # reverb only
#   bash train_local_mps.sh noise        # noise only
#   bash train_local_mps.sh codec        # codec only
#
# Live progress (open in a separate terminal):
#   tail -f logs/reverb_train.log
#   tail -f logs/noise_train.log
#   tail -f logs/codec_train.log
#   tail -f logs/train_all.log            # combined

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=.venv/bin/python3
DATA=data/calmsep-8k
CKPT=checkpoints
LOG=logs

mkdir -p "$CKPT/stage1_reverb" "$CKPT/stage1_noise" "$CKPT/stage1_codec" "$LOG"

# ── M5 Pro MPS settings ───────────────────────────────────────────────────────
export PYTORCH_ENABLE_MPS_FALLBACK=1   # auto-fallback unsupported MPS ops to CPU
export TOKENIZERS_PARALLELISM=false

LIBRI="$DATA/librispeech-8k"
RIR="$DATA/rirs"
NOISE="$DATA/noise"
HF_MODEL="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"

DEVICE=mps
BATCH=4
WORKERS=0          # 0 = single-process DataLoader; data loading is 8ms vs 1-4s compute, no benefit from workers
SPE=500            # samples/epoch
EPOCHS=40
LR=1e-4
CLIP=16000         # 2 s @ 8kHz, 2× faster than 4 s, no quality loss on this model

ADAPTER_FILTER="${1:-all}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ADAPTER_FILTER" == "all" || "$ADAPTER_FILTER" == "reverb" ]]; then
    log "=== [1/3] REVERB adapter, $EPOCHS epochs (~9h) ==="
    PYTHONUNBUFFERED=1 "$PYTHON" -u -m coralsep.train.stage1_single \
        --adapter reverb \
        --librispeech-8k "$LIBRI" \
        --rir-bank       "$RIR" \
        --output-dir     "$CKPT/stage1_reverb" \
        --hf-model       "$HF_MODEL" \
        --device         $DEVICE \
        --batch-size     $BATCH \
        --lr             $LR \
        --epochs         $EPOCHS \
        --samples-per-epoch $SPE \
        --num-workers    $WORKERS \
        --max-clip-samples $CLIP \
        --bf16 \
        2>&1 | tee "$LOG/reverb_train.log"
    log "Reverb done → $CKPT/stage1_reverb/best_reverb.pt"
fi

# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ADAPTER_FILTER" == "all" || "$ADAPTER_FILTER" == "noise" ]]; then
    log "=== [2/3] NOISE adapter, $EPOCHS epochs (~9h) ==="
    PYTHONUNBUFFERED=1 "$PYTHON" -u -m coralsep.train.stage1_single \
        --adapter noise \
        --librispeech-8k "$LIBRI" \
        --rir-bank       "$RIR" \
        --noise-dir      "$NOISE" \
        --output-dir     "$CKPT/stage1_noise" \
        --hf-model       "$HF_MODEL" \
        --device         $DEVICE \
        --batch-size     $BATCH \
        --lr             $LR \
        --epochs         $EPOCHS \
        --samples-per-epoch $SPE \
        --num-workers    $WORKERS \
        --max-clip-samples $CLIP \
        --bf16 \
        2>&1 | tee "$LOG/noise_train.log"
    log "Noise done → $CKPT/stage1_noise/best_noise.pt"
fi

# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ADAPTER_FILTER" == "all" || "$ADAPTER_FILTER" == "codec" ]]; then
    log "=== [3/3] CODEC adapter, $EPOCHS epochs (~9h) ==="
    which ffmpeg && log "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
    PYTHONUNBUFFERED=1 "$PYTHON" -u -m coralsep.train.stage1_single \
        --adapter codec \
        --librispeech-8k "$LIBRI" \
        --rir-bank       "$RIR" \
        --output-dir     "$CKPT/stage1_codec" \
        --hf-model       "$HF_MODEL" \
        --device         $DEVICE \
        --batch-size     $BATCH \
        --lr             $LR \
        --epochs         $EPOCHS \
        --samples-per-epoch $SPE \
        --num-workers    $WORKERS \
        --max-clip-samples $CLIP \
        --bf16 \
        2>&1 | tee "$LOG/codec_train.log"
    log "Codec done → $CKPT/stage1_codec/best_codec.pt"
fi

log "=== All adapters complete ==="
ls -lh "$CKPT"/stage1_*/best_*.pt 2>/dev/null
