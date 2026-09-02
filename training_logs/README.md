# Training logs

Raw logs from training runs that actually happened. These are evidence, not
documentation: nothing here is edited, and a number in `docs/restoration/RESULTS.md`
is only allowed to exist if a file here or in `eval/eval_outputs/` produced it.

| File | Run | Hardware | Recorded |
|---|---|---|---|
| `stage4_joint_kaggle.log` | Stage 4 joint polish | Kaggle, Tesla T4, 15.6 GB, BF16 | 2026-07-21 |

## `stage4_joint_kaggle.log`

Kaggle notebook stream capture, one JSON object per output line. Confirms
independently of any documentation:

- Epochs 1 to 14 completed of 20 configured, roughly 2,930 s per epoch.
- Best loss 8.6809 at epoch 14, the lowest value recorded and still falling
  when the run ended.
- `best_joint.pt` written with 222 adapter tensors.
- Inputs by Kaggle dataset name: `rishig777/calmsep-8k-slice` for audio,
  `rishig777/calmsep-model/calmsep-tiny` for the backbone source,
  `rishig777/calmsep-stage1-adapters`, and `rishig777/calmsep-stage3-gate`.
- Stage 1 adapter sizes as copied: reverb 432.3 KB, noise 433.1 KB,
  codec 433.1 KB. Gate checkpoints 367.9 KB each.
- The byte size of every source file baked into the training image. Those sizes
  match this repository exactly, which is how we know the code on `master` is
  the code that produced this result.

The run ended before its configured length. Loss was still decreasing, so the
joint stage is unfinished rather than converged.

**Provenance.** Recovered from `calm-sep-context-2026-09-01-v2.zip` under
ticket I-015. It existed in no commit on any of the thirteen branches. SHA-256
`50e7b7193160bdfee70af56487599826e4481e2c9982b48f5044671b5052b7c9`, identical
to the archive copy.
