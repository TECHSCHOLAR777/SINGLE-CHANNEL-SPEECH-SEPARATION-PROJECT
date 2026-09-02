#!/usr/bin/env bash
# CoRAL-Sep full data generation. One command, resumable, survives a dropped SSH.
#
#   bash scripts/prepare_all_data.sh --root /workspace/data
#   bash scripts/prepare_all_data.sh --root /workspace/data --check   # dry run
#
# Every stage is idempotent: it writes a marker to $ROOT/.done/ on success and
# skips itself on re-run. If a stage dies, fix the cause and re-run the same
# command; completed stages are skipped.
#
# Preflight gates run BEFORE each generation stage. Both previous data runs died
# mid-generation on files that were never on disk. Never again.

set -euo pipefail

ROOT=""
CHECK_ONLY=0
SKIP_HIGHN=0
TRAIN_SPLIT="train-100"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)        ROOT="$2"; shift 2 ;;
    --check)       CHECK_ONLY=1; shift ;;
    --skip-highn)  SKIP_HIGHN=1; shift ;;
    --train-split) TRAIN_SPLIT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo "usage: $0 --root /workspace/data [--check] [--skip-highn]" >&2; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DONE="$ROOT/.done"
LOGS="$ROOT/logs"
LS_DIR="$ROOT/librispeech"
LS_ROOT="$LS_DIR/LibriSpeech"
WHAM_DIR="$ROOT/wham_noise"
mkdir -p "$DONE" "$LOGS"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
skip() { printf '    [skip] %s (already done)\n' "$*"; }

stage() {  # stage <name> <command...>
  local name="$1"; shift
  if [[ -f "$DONE/$name" ]]; then skip "$name"; return 0; fi
  say "$name"
  if [[ $CHECK_ONLY -eq 1 ]]; then printf '    [dry-run] would run: %s\n' "$*"; return 0; fi
  "$@" 2>&1 | tee "$LOGS/$name.log"
  touch "$DONE/$name"
}

# ── disk budget ───────────────────────────────────────────────────────────────
# Sized for: LibriSpeech (train-100 + dev + test), WHAM! noise, Libri3Mix and
# Libri2Mix (train-100 + dev + test, 16k/max, mix_both), Libri4/5Mix (dev+test),
# reverb eval. Tarballs are deleted after extraction where possible.
NEED_GB=190
[[ $SKIP_HIGHN -eq 1 ]] && NEED_GB=150

say "disk check"
AVAIL_GB=$(df -BG --output=avail "$(dirname "$ROOT")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
printf '    need ~%s GB   available %s GB   at %s\n' "$NEED_GB" "$AVAIL_GB" "$ROOT"
if [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
  echo "    INSUFFICIENT DISK. Provision a bigger volume before starting;" >&2
  echo "    running out 3 hours into generation costs you the whole run." >&2
  [[ $CHECK_ONLY -eq 1 ]] || exit 1
fi

# ── stage 1: source corpora ───────────────────────────────────────────────────
stage librispeech_and_wham \
  python -m data.prepare_wham --output-dir "$ROOT"

# prepare_librimix downloads LibriSpeech itself and generates Libri3Mix.
# Run it with --include-train so the train split lands in the same pass.
stage libri3mix \
  python -m data.prepare_librimix \
    --output-dir "$ROOT" \
    --librispeech-dir "$LS_DIR" \
    --include-train \
    --train-split "$TRAIN_SPLIT"

# ── preflight before every remaining generation stage ─────────────────────────
say "preflight: Libri3Mix metadata vs files on disk"
if [[ $CHECK_ONLY -eq 0 ]]; then
  python "$REPO/scripts/preflight_data.py" \
    --metadata-dir "$ROOT/tools/LibriMix/metadata/Libri3Mix" \
    --librispeech-dir "$LS_ROOT" \
    --wham-dir "$WHAM_DIR" \
    --limit 500 | tee "$LOGS/preflight_libri3mix.log"
fi

# ── stage 2: high-N mixtures for L3 and the stop classifier ───────────────────
if [[ $SKIP_HIGHN -eq 0 ]]; then
  stage libri45mix \
    python -m data.prepare_librimix_highn \
      --output-dir "$ROOT" \
      --librispeech-dir "$LS_DIR" \
      --n-src 4,5
fi

# ── stage 3: reverberant eval set (the only P4 work unblocked today) ──────────
stage reverb_eval \
  python -m data.make_reverb_eval \
    --librimix-root "$ROOT/Libri3Mix" \
    --output-dir "$ROOT/Libri3Mix-reverb" \
    --subset test

say "done"
printf '    data root : %s\n' "$ROOT"
printf '    logs      : %s\n' "$LOGS"
printf '    stages    : %s\n' "$(ls "$DONE" 2>/dev/null | tr '\n' ' ')"
echo
echo "Next: export LIBRIMIX_ROOT=$ROOT/Libri3Mix"
