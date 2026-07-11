# Dev C real-data validation on RunPod

This guide separates the checks that are already complete in ordinary CI from the one
remaining Dev C acceptance run that needs real speech, pretrained models, and a CUDA host.

## Status before opening a pod

- **P0-INT2 is already complete.** `scripts/run_baseline.py` loads configuration through
  `utils.config.load_config`.
- **P0-INT4 is complete in code and CI.** `tests/test_p0_e2e.py` exercises YAML loading,
  baseline execution, PIT SI-SDRi, and result artifact creation with a deterministic fake
  expert. It needs no LibriMix files or GPU.
- The data-dependent item is **P1-INT2**, not P0-INT2: verify cross-chunk speaker identity on
  one real Libri3Mix test utterance longer than four seconds.

Run the local checks first:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/test_p0_e2e.py tests/test_align_integration.py
```

## 1. Prepare the pod

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,experts]"
```

Use the least expensive CUDA pod that can load MossFormer2 and ECAPA without an
out-of-memory error. The validation uses one utterance, so a long training session is not
required. Keep model caches and the dataset on persistent storage if the pod may be stopped.

## 2. Make one Libri3Mix test split available

Use an existing Libri3Mix directory when possible. Its required layout is:

```text
$LIBRIMIX_ROOT/wav16k/max/test/mix_both/*.wav
$LIBRIMIX_ROOT/wav16k/max/test/s1/*.wav
$LIBRIMIX_ROOT/wav16k/max/test/s2/*.wav
$LIBRIMIX_ROOT/wav16k/max/test/s3/*.wav
```

When no prepared copy is available, the repository can create the development and test
splits:

```bash
python data/prepare_librimix.py --output-dir /workspace/datasets
export LIBRIMIX_ROOT=/workspace/datasets/Libri3Mix
```

The preparation script downloads LibriSpeech data and generates audio on disk. Do this only
once on persistent storage. P1-INT2 needs the test split only; do not pass `--include-train`.

## 3. Run P1-INT2 only

This is the smallest real-data run and avoids loading the expensive second expert:

```bash
python scripts/validate_alignment.py \
  --librimix-root "$LIBRIMIX_ROOT" \
  --device cuda \
  --output-dir outputs/p1_alignment \
  --skip-pair \
  --strict
```

The command writes `outputs/p1_alignment/alignment_validation.json` and aligned WAV files.
P1-INT2 passes only when all of the following are true:

- `p1_int2.passed` is `true`;
- at least two analysis chunks were evaluated;
- the number of persistent tracks equals the number of reference speakers;
- `identity_switches` is `0`.

Inspect the compact result without printing the long assignment trace:

```bash
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("outputs/p1_alignment/alignment_validation.json").read_text())
print(json.dumps({
    "utterance_id": report["utterance_id"],
    "duration_sec": report["duration_sec"],
    "p1_int2": {
        key: report["p1_int2"][key]
        for key in (
            "num_chunks",
            "num_persistent_tracks",
            "num_reference_speakers",
            "identity_switches",
            "passed",
        )
    },
}, indent=2))
PY
```

## 4. Optional P1-INT1 rerun

To also rerun two-expert alignment, omit `--skip-pair`:

```bash
python scripts/validate_alignment.py \
  --librimix-root "$LIBRIMIX_ROOT" \
  --device cuda \
  --output-dir outputs/p1_alignment_full \
  --strict
```

Without a configured SR-CorrNet checkout/checkpoint, the repository uses its configured
TF-GridNet/SepFormer fallback. To force SR-CorrNet, add:

```bash
  --srcorrnet-repo /workspace/SR_CorrNet \
  --srcorrnet-checkpoint /workspace/checkpoints/srcorrnet.pt
```

The same checks can be run through pytest:

```bash
RUN_REAL_EXPERTS=1 \
LIBRIMIX_ROOT="$LIBRIMIX_ROOT" \
DEVICE=cuda \
python -m pytest -q -s tests/test_m1_real_experts.py
```

## 5. What to commit

Do not commit model caches, checkpoints, generated WAVs, or the `outputs/` directory. They
are ignored intentionally. Record the compact JSON summary in the pull-request description.
After `p1_int2.passed=true`, update P1-INT2 and the corresponding M1 acceptance line in
`README.md` to green, including the utterance ID and run date.

If the strict run fails, keep P1-INT2 yellow. Save the JSON, try another valid test utterance
with `--sample-index`, and compare the assignment trace before changing
`--match-threshold` or `--ema`; do not tune on the final test utterance repeatedly.
