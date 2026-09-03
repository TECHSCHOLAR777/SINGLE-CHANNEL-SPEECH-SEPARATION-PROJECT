# Datasets

Manifests and generated banks. **No audio corpora live here**, and none of them ever will: `.gitignore` excludes `*.wav` and `*.flac`, and the training corpora run to roughly 176,000 files.

---

## `fixed_eval/`

25 seeded evaluation tiers, each a JSONL manifest with a SHA-256 sidecar, plus an index.

This is the strongest reproducibility asset in the project. Each manifest pins exactly which mixtures make up an evaluation tier and what they contained when it was generated, so a later regeneration can be *proven* identical or *proven* different rather than assumed either way. `tests/test_fixed_eval_manifest.py` verifies the hashes.

| Tier family | Covers |
|---|---|
| `clean_2_3_n{2,3}` | Clean anechoic mixtures, 2 and 3 speakers |
| `codec_only_n2` | Codec damage in isolation |
| `high_count_clean_n{4,5}` | Clean 4 and 5 speaker mixtures |
| `high_count_degraded_n{4,5}` | Degraded 4 and 5 speaker mixtures |
| `libricss_real_n{2,3}` | Real recorded meeting audio |
| `band_recovery_gain_n2` | Isolating the band recovery contribution |

🟠 **No recorded experiment has used any of them.** Every published number came from LibriMix instead. Regenerate with:

```bash
python -m coralsep.data.fixed_eval_generator --out-dir datasets/fixed_eval
```

## `rirs/`

Destination for the generated room impulse response bank, roughly 10,001 responses at about 12 KB each. Empty in a fresh clone; the `.gitkeep` holds the directory.

```bash
python -m coralsep.data.rir_bank --output datasets/rirs --per-bucket 1250
```

---

## Corpora you have to fetch yourself

| Dataset | Role | Scale | How |
|---|---|---|---|
| LibriSpeech 8 kHz | training speech | 137,876 utterances | `python -m coralsep.data.prepare_librispeech_8k` |
| WHAM! noise | noise augmentation | 28,000 clips | `python -m coralsep.data.prepare_wham` |
| LibriMix `wav8k/min` | evaluation | ~3,000 clips per split | LibriMix generator, see `docs/restoration/REPRODUCTION.md` |

The training corpora were staged locally under `data/calmsep-8k/` on the original author's machine and sliced for Kaggle with `scripts/slice_for_kaggle.py`. That slice is the Kaggle dataset `rishig777/calmsep-8k-slice`, which the Stage 4 training log names as its audio input.

⚠️ **Training mixtures cannot be regenerated exactly.** Training uses on-the-fly dynamic mixing with a fresh draw each epoch, and no run recorded its mixer seed. The slicing script now prints and accepts a seed, so future runs can be reproduced; past ones cannot.
