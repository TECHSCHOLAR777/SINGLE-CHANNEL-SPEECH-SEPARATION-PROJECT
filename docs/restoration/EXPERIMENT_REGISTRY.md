# Experiment Registry

**Purpose:** one record per experiment that was actually executed, with everything needed to judge whether it can be trusted or repeated.

**Status:** [AMBER] Seven executed experiments identified. None is fully reproducible, because no run recorded its seed, its configuration file or its checkpoint hash.

**Last verified:** 2026-09-02

**Inclusion rule:** an entry exists here only if there is evidence the run happened. Planned runs are listed in section 9 and are not given experiment IDs.

---

## Index

| ID | Objective | Date | Hardware | Evidence | State |
|---|---|---|---|---|---|
| EXP-001 | v1 cascade versus single expert | 2026-07-13 | Kaggle T4 | narrative only | [SUPERSEDED] |
| EXP-002 | Stage 1 reverb adapter training | 2026-07-17 to 07-18 | Apple M5 Pro, MPS | checkpoint size, narrative | [PARTIALLY_VERIFIED] |
| EXP-003 | Stage 1 reverb adapter diagnostic | 2026-07-17 | CPU, Lightning AI | full log | [VERIFIED] |
| EXP-004 | Stage 1 noise and codec adapter training | 2026-07-18 onward | Apple M5 Pro, MPS | checkpoint sizes | [CLAIMED] |
| EXP-005 | Stage 3 gate training | before 2026-07-21 | Kaggle | checkpoint copy in the Stage 4 log | [PARTIALLY_VERIFIED] |
| EXP-006 | Stage 4 joint training | 2026-07-21 | Kaggle T4 | full epoch log | [VERIFIED] |
| EXP-007 | LibriMix evaluation, 3 splits | after 2026-07-21 | CPU, Apple M5 Pro | two result JSONs | [PARTIALLY_VERIFIED] |

---

## EXP-001: v1 CA-MoSE cascade versus single expert

**State:** [SUPERSEDED]

**Objective.** Determine whether a compute-adaptive cascade of a cheap and an expensive separator, joined by a learned fusion head, beats the expensive separator alone.

**Hypothesis.** The fusion head would exceed both experts while a routing gate cut compute.

**Dataset and split.** 100 development samples, mixed 2 to 5 speakers, LibriMix-derived.

**Model.** MossFormer2 as cheap expert, SR-CorrNet-SS as expensive expert, plus SceneAnalyzer, TwoLevelRouter and CRRRFusionHead, roughly 2.66M trainable parameters.

**Hardware.** Kaggle T4. **Training:** 30 epochs on the heads.

**Result.** The fusion head degraded the strong expert by 0.4 to 3.7 dB at every threshold. At full escalation the cascade exactly equalled SR-CorrNet at 16.22 dB and never exceeded it. Compute fell 36 percent at threshold 6, at a cost of 3.55 dB. The speaker-count stop-classifier reached 61.4 percent validation accuracy and 10 percent count accuracy at inference.

**Reproducibility.** None. The code was deliberately not carried into v2 and the trained checkpoint was recorded as sitting unbacked in a local Downloads folder.

**Why it stays in the registry.** It is the measurement that produced the current architecture. See `APPROACH_EVOLUTION.md`.

**Evidence.** `docs/PROJECT_HISTORY.md` part 1, `NUMBERS.md` section 3.2.

---

## EXP-002: Stage 1 reverb adapter training

**State:** [PARTIALLY_VERIFIED]

**Objective.** Train a rank-8 LoRA adapter on reverberant mixtures so the frozen backbone survives reverberation.

**Dataset.** Dynamic on-the-fly mixtures from DATA-001 speech and DATA-003 RIRs. 2-second clips at 8 kHz, clipped before degradation.

**Configuration.** Batch 4 with per-sample sequential forward, 500 samples per epoch, 40 epochs, learning rate 1e-4, BF16 autocast with float32 STFT islands, gradient clip 5.0, `num_workers=0`.

**Seed.** Not recorded. **Hardware.** Apple M5 Pro, 24 GB, MPS backend.

**Artifact.** `checkpoints/stage1_reverb/best_reverb.pt`, 424 KB.

**Result.** A checkpoint was produced. Its quality was measured separately in EXP-003 and it is harmful.

**Reproducibility.** Blocked. No seed, no config file, and dynamic mixing means the training data cannot be regenerated exactly.

**Related tickets.** I-025, I-022.

---

## EXP-003: Stage 1 reverb adapter diagnostic

**State:** [VERIFIED]

**Objective.** Determine whether the trained reverb adapter improves the backbone on reverberant input.

**Design.** One 2-speaker mixture, T60 = 0.46 s, evaluated in three conditions: clean anechoic, mild reverberation, strong reverberation. Base model versus adapted model.

**Controls.** Two, both passing. At gate zero the adapted model reproduced the base output to a maximum difference of 0.000000 through two independent code paths. LoRA A matrices had a mean norm of 1.5813 and B matrices were non-zero, confirming weights were learned.

**Result.**

| Condition | Base SI-SNR | Adapted SI-SNR | Delta |
|---|---:|---:|---:|
| Clean | 18.61 dB | 18.17 dB | -0.44 dB |
| Reverb mild | -30.89 dB | -30.96 dB | -0.07 dB |
| Reverb strong | -32.83 dB | -35.64 dB | -2.81 dB |

**Interpretation.** The adapter is harmful. Because both controls passed, the fault is in the training objective and not in the injection mechanism. The leading hypothesis is a wet reference target.

**Hardware and environment.** CPU, Lightning AI workspace, 2026-07-17 19:10 UTC. The platform was banned the following day for unrelated reasons.

**Reproducibility.** Partially. `eval/eval_reverb_adapter.py` is the script, but it hard-codes paths on a platform no longer in use (I-033) and needs the checkpoint and the RIR bank.

**Command, as recorded in the module docstring.**
```
python eval/eval_reverb_adapter.py \
  --checkpoint <path>/checkpoints/stage1_reverb/best_reverb.pt \
  --librispeech-8k <path>/data/calmsep-8k/librispeech-8k \
  --rir-bank <path>/data/calmsep-8k/rirs \
  --output-dir <path>/eval_outputs
```

**Evidence.** `eval/eval_outputs/eval.log`, 4,745 bytes, tracked.

**Related tickets.** I-025, I-033.

---

## EXP-004: Stage 1 noise and codec adapter training

**State:** [CLAIMED]

**Objective.** Train the noise and codec adapters under the same recipe as EXP-002.

**Evidence.** Only indirect: the Stage 4 Kaggle log records copying `best_noise.pt` at 433.1 KB and `best_codec.pt` at 433.1 KB. No training log for either survives.

**Open question.** `NUMBERS.md` records roughly 40 epochs for the noise adapter. A project memory note dated 2026-07-18 records the then-current `best_noise.pt` as an epoch-2 artifact requiring retraining. Both can be true if retraining happened between the two dates, but no log confirms it. See I-022.

**Reproducibility.** Blocked, same reasons as EXP-002.

---

## EXP-005: Stage 3 gate training

**State:** [PARTIALLY_VERIFIED]

**Objective.** Train the condition analyser and the gate network with supervised oracle condition labels.

**Labels.** Oracle vectors from `MixtureRecipe.condition_vector()`: reverb where T60 > 0, noise where SNR < 60 dB, codec where the codec class is above zero.

**Dataset.** Kaggle dataset `rishig777/calmsep-stage3-gate`.

**Evidence.** The Stage 4 log records copying `best_gate.pt` and `final_gate.pt` at 367.9 KB each, so the run happened. No training log, no epoch count, no loss curve survives.

**Result.** Unknown. The gate that came out of this stage, after Stage 4c calibration, does not route (I-003). Whether the failure originates here or in the calibration stage is undetermined.

**Reproducibility.** Blocked. Needs the Kaggle dataset.

**Related tickets.** I-003.

---

## EXP-006: Stage 4 joint training

**State:** [VERIFIED]

**Objective.** Jointly polish the adapters, the gate and the analyser end to end.

**Configuration.** 1,000 samples per epoch, 20 epochs configured, two learning-rate groups, adapters at 1e-5 and gate plus analyser at 2e-5. BF16 enabled.

**Hardware.** Kaggle, Tesla T4, 15.6 GB VRAM. Roughly 2,930 s per epoch, 41,474 s total wall time.

**Inputs, as logged.** Audio from `rishig777/calmsep-8k-slice`, model source from `rishig777/calmsep-model/calmsep-tiny` including a copy of the `sr_corrnet` source tree, Stage 1 adapters from `rishig777/calmsep-stage1-adapters`, gate data from `rishig777/calmsep-stage3-gate`.

**Result.** Completed 14 of 20 epochs. Best loss 8.6809 at epoch 14, still decreasing. Checkpoint written with 222 adapter tensors.

Full curve in `RESULTS.md` section 2.

**Reproducibility.** Partially. The log records exact dataset identifiers, the hardware and the per-epoch losses, which is more than any other run in this project. It does not record a seed or a config hash.

**Notable detail from the log.** The Kaggle run bakes source files into `/tmp/calmsep_project` and prints each file's byte size. Those sizes match the versions in this repository exactly, which independently confirms that the code on `master` is the code that produced this result.

**Evidence.** `training_logs/joint_stage4_kaggle.log`, recovered under I-015.

---

## EXP-007: LibriMix evaluation, three splits

**State:** [PARTIALLY_VERIFIED]

**Objective.** Compare the full system against the frozen backbone on LibriMix.

**Protocol defect.** The true speaker count was supplied to both systems from the directory name. See I-002. This is disclosed in the project's own documentation and is the reason this experiment cannot support its headline claim.

**Dataset.** LibriMix `wav8k/min/test`, `mix_both`. 30 clips per split for Libri2Mix, Libri3Mix and Libri5Mix. Libri4Mix not run.

**Hardware.** CPU only, Apple M5 Pro, single threaded.

**Result.** See `RESULTS.md` section 1.

**Artifacts.** `eval/eval_outputs/calmsep_eval.json` and `eval/eval_outputs/calmsep_eval_5.json`. The second was recovered from the archive and exists in no commit.

**Two runs, not one.** The two JSON files have different schemas: the 5-speaker file carries a `delta_si_sdr` field that the 2 and 3 speaker file lacks. That field appears only in the archive version of `run_eval.py` (I-012). So Libri5Mix was evaluated with a later version of the harness than Libri2Mix and Libri3Mix were. The three rows of the results table were not produced by the same code.

**Reproducibility.** Blocked. Requires the LibriMix test set, the checkpoints and the backbone.

**Related tickets.** I-002, I-012, I-015, I-023, I-026.

---

## 8. Cross-cutting reproducibility gaps

Every experiment above shares the same four gaps. They are recorded once here rather than repeated in each entry.

| Gap | Consequence | Ticket |
|---|---|---|
| No seed recorded in any run | no run can be repeated bit for bit | I-020 |
| No configuration file version or hash tied to any result | the exact settings behind a number are inferred, not known | I-020 |
| No checkpoint hash in any result artifact | cannot prove which weights produced which number | see `DATA_AND_MODEL_INVENTORY.md` section 3 |
| Dynamic mixing with no recorded seed | training data cannot be regenerated | I-020 |

Closing these is cheap for future runs and impossible for past ones. That asymmetry is the argument for closing them before the next run rather than after.

---

## 9. Runs that were planned and never executed

No experiment ID is assigned, because nothing happened.

| Planned run | Why it matters | Ticket |
|---|---|---|
| Stage 2 universal adapter training | the ablation that justifies three adapters instead of one | I-024 |
| Libri4Mix evaluation | fills the hole in the speaker-count sweep | I-023 |
| Evaluation without oracle speaker count | the primary graded axis | I-002 |
| Per-condition evaluation, reverb, noise and codec separately | the only evidence that could demonstrate routing | I-003 |
| Oracle gate upper bound | quantifies the headroom the gate is leaving on the table | I-003 |
| Calibration ECE and reliability diagram | required by the problem statement | I-034 |
| Bootstrap confidence intervals and Wilcoxon tests on existing results | code exists in `eval/stats.py` and has never been run | I-026 |
| A published baseline, SepFormer or ConvTasNet, on the same clips | external positioning | I-023 |

---

## Related documents

`RESULTS.md` · `DATA_AND_MODEL_INVENTORY.md` · `APPROACH_EVOLUTION.md` · `REPRODUCTION.md` · `ISSUE_LEDGER.md`
