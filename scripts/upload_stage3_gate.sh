#!/usr/bin/env bash
# Run AFTER placing a fresh ~/.kaggle/kaggle.json
# Downloads best_gate.pt from the Stage 3 kernel output and uploads as a Kaggle dataset.
set -euo pipefail

KERNEL_SLUG="${1:-}"          # e.g. rishig777/calmsep-stage3-gate-training
DATASET_NAME="calmsep-stage3-gate"
TMP_DIR="$(mktemp -d)"

echo "=== Listing your recent Kaggle kernels ==="
kaggle kernels list --user rishig777 --page-size 20 --sort-by dateRun

if [[ -z "$KERNEL_SLUG" ]]; then
  echo ""
  echo "Paste the kernel slug for your Stage 3 run (from the list above):"
  read -r KERNEL_SLUG
fi

echo ""
echo "=== Downloading output from: $KERNEL_SLUG ==="
kaggle kernels output "$KERNEL_SLUG" -p "$TMP_DIR"

echo "Downloaded files:"
ls -lh "$TMP_DIR"

# Find the gate checkpoint
GATE_FILE=$(find "$TMP_DIR" -name "best_gate.pt" | head -1)
if [[ -z "$GATE_FILE" ]]; then
  echo "ERROR: best_gate.pt not found in kernel output."
  echo "Files available:"
  find "$TMP_DIR" -name "*.pt" | sort
  exit 1
fi
echo "Found: $GATE_FILE  ($(du -h "$GATE_FILE" | cut -f1))"

# Prepare dataset directory
DS_DIR="$(mktemp -d)"
cp "$GATE_FILE" "$DS_DIR/best_gate.pt"

# Also grab final_gate.pt if available
FINAL=$(find "$TMP_DIR" -name "final_gate.pt" | head -1)
[[ -n "$FINAL" ]] && cp "$FINAL" "$DS_DIR/final_gate.pt" && echo "Also copied: final_gate.pt"

# Write dataset metadata
cat > "$DS_DIR/dataset-metadata.json" << JSON
{
  "title": "CALM-Sep Stage 3 Gate Checkpoint",
  "id": "rishig777/${DATASET_NAME}",
  "licenses": [{"name": "unknown"}]
}
JSON

echo ""
echo "=== Creating Kaggle dataset: rishig777/${DATASET_NAME} ==="
kaggle datasets create -p "$DS_DIR" --dir-mode zip

echo ""
echo "Done! Dataset live at: https://www.kaggle.com/datasets/rishig777/${DATASET_NAME}"
echo ""
echo "Add it to the Stage 4 notebook as input dataset: rishig777/${DATASET_NAME}"

# Cleanup
rm -rf "$TMP_DIR" "$DS_DIR"
