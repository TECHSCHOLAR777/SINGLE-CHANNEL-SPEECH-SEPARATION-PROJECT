> 📜 **Historical narrative, preserved as written.** Composed 2026-07-18 by the
> original author while a training run was in progress, and recovered from the
> 2026-09-01 archive under ticket I-015. It existed in no commit.
>
> It uses the project names in force at the time, CA-MoSE and CALM-Sep. The
> project is now CoRAL-Sep. Nothing here has been rewritten, because the value of
> this document is that it records what people believed and measured at the time,
> including the parts that turned out to be wrong.
>
> For the current state see [restoration/PROJECT_STATUS.md](restoration/PROJECT_STATUS.md).

# The Whole Story: CA-MoSE to CALM-Sep

Written 2026-07-18, while the second reverb adapter run is training in the background.
This document exists because the last complete record we wrote (PROJECT_CRUX.md, in the
old CA-MoSE repo) stops on 2026-07-13, and a lot has happened since. This is the full
account: what we set out to build, what actually happened, what the numbers said, where
we trusted the wrong signals, and where we are right now. Nothing here is polished up.
Where we do not have a number yet, it says so.

---

## Part 1. CA-MoSE: the first attempt

### What we believed

Somewhere in the first week of July, three of us started building CA-MoSE, a
compute-adaptive cascade for separating 2 to 5 overlapping speakers from a single
microphone. The idea felt elegant: run a cheap separator first, check the output with a
blind quality estimator, and only wake up the expensive model when the cheap one fails.
On top of that, a learned fusion head would merge the two streams into something better
than either alone. Three trainable heads (SceneAnalyzer, TwoLevelRouter, CRRRFusionHead,
about 2.66M params total) sitting on two frozen experts.

The first dated milestones in the commit trail are from 2026-07-11: the full
SceneAnalyzer landed (PR #16), plus escalation logging and the end-to-end test
scaffolding. The foundation before that (DynamicMixer, LibriMix prep, augmentation,
ECAPA wrappers, the eval harness) was built in parallel across three branches.

### July 11 and 12: integration pain

Two days of gluing parts together surfaced the first real bugs:

The identity lock test (P1-INT2) reported 2 speaker switches on a 3-speaker file. It
took three separate root causes to close: the switch metric itself was unsound (it
counted silent tracks and compared against window 0 instead of consecutive windows),
`max_tracks` was never forwarded into the ChunkStitcher, and the test itself was invalid
because MossFormer2 is a 2-speaker model being asked about 3 speakers. After the fixes,
real speech on a Kaggle T4 gave 0 switches. That part of v1 genuinely worked, and we
kept it.

The stream-count mismatch (cheap expert returns 2 stems, expensive returns 3) was
patched with a residual-pad hack: fill the missing stream with mixture minus the sum of
emitted stems. The comment in the code said, from day one, that this is not a real third
speaker. Remember this. It matters later.

### July 13: the big day, and the honest numbers

Everything ran on a free Kaggle T4. We fixed ten bugs in one session, most of them
device mismatches and shape errors: token wiring, silent cache death without resume,
REAL-M crashing on 3-speaker input, CPU tensors gathered on GPU, an embedding dim of 64
where ECAPA actually gives 192, NaN losses on variable-speaker batches, a factorial
permutation blowup at 5 speakers replaced with Hungarian assignment, and a temperature
scalar living on the wrong device. Ten bugs in one day means the code had never truly
run end-to-end before. That is a lesson, not an excuse.

Then we measured (100 dev samples, mixed 2 to 5 speakers):

| System | SI-SDRi |
|---|---|
| MossFormer2 alone | 8.24 dB |
| SR-CorrNet-SS alone | **16.22 dB** |
| Cascade, fusion mode, best threshold | 15.79 dB |
| Cascade, fusion mode, worst threshold | 12.51 dB |
| Cascade, sr-primary, full escalation | 16.22 dB (equals, never exceeds) |

The fusion head, the thing we trained, made the strong expert worse by 0.4 to 3.7 dB at
every setting we tried. The efficiency story was real though: at threshold tau=6, half
the utterances went through the cheap path only, expected real-time factor 0.20 against
0.31 for always-expensive. A 36% compute reduction, at a measured 3.55 dB quality cost.

The stop-classifier we trained for speaker counting reached 61.4% validation accuracy
and then scored 10% count accuracy at inference. Two bugs: `min_count=1` allowed the
peel-off to stop at one speaker when mixtures always have at least two, and a fitted
temperature of 8.54 flattened every prediction toward 0.5 so the classifier could never
commit.

### Where we fell for the green flag

The M2 gate criterion said "beats best single expert on mixed-condition validation." We
wrote that criterion before we had any numbers. It was aspiration dressed up as a
milestone. When the cascade at full escalation exactly matched SR-CorrNet, that looked
like success for a moment. It was not. A cascade that equals its own component has added
nothing but latency and parameters. We also presented the P2-INT4 comparison without
saying loudly enough that the cheap expert was structurally broken on 3+ speakers (the
residual-pad hack), which made the cascade look worse than a fair version might have
been, and made our conclusion cheaper than it should have felt.

### Where we ignored the red flags

The residual-pad comment warned us from day one. Ten CUDA bugs in one session told us
the pipeline had never been dry-run locally. The README pulse counter drifted from the
actual task table more than once, and it was caught by a human, not by us, both times.
Each of these was visible early and each was deferred.

The one genuinely honest thing we did on July 13: we did not delete the negative result.
We reframed M2 around what we had actually measured (the compute reduction), marked
"beats the expert" as a confirmed-negative stretch goal, and wrote PROJECT_CRUX.md with
every number in it, including the ugly ones.

---

## Part 2. The misreading, and the reset

For a few days afterward the failure was summarized inside the team as "BEATs failed."
This was wrong, and correcting it mattered more than it sounds. There was never a BEATs
model anywhere in the project. The thing that failed was a criterion, the flag literally
named `cascade_beats_single_expert`, which came back False at every threshold. A model
did not fail us. Our own gate design did. Once that was said out loud, the path forward
stopped being "find a better model" and became "stop putting learned layers on top of a
strong frozen expert."

On the night of 2026-07-16 we wrote a 24-hour master plan for v2. Its two rules came
straight from the post-mortem: the frozen pretrained expert alone is already strong, so
ship it and spend the time on breadth and evaluation depth. And the negative result is
presentation material, not embarrassment. The plan salvaged v1's tested plumbing (data
scripts, metrics, the chunk stitcher that scored 0 switches) and explicitly banned
copying any of the failed model code: no fusion, no cascade gate, no scene analyzer.

---

## Part 3. CALM-Sep: the current build

### The design

CALM-Sep stands for Condition-Aware LoRA Mixture. One strong pretrained network,
SR-CorrNet var-2-5 (7.4M params, 8 kHz, `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk`),
frozen forever. It is never fine-tuned. Instead, three small LoRA adapters (101,404
params each, patched into 37 linear modules) teach it to survive hard conditions: one
for reverb, one for noise, one for codec damage. A condition analyzer measures how much
of each condition is present in a chunk, and a gate blends the adapters into the frozen
weights in proportion, before any audio is produced. One forward pass, one set of output
voices. Speaker count is read from the backbone's own attractor probabilities, the thing
that was never broken in v1. A band-recovery head lifts 8 kHz output to 16 kHz, and a
residual-energy detector guards against a missed speaker.

The full design lives in BLUEPRINT, which is the source of truth. The grading order is
fixed: speaker count accuracy first, separation quality second, everything else is
bonus. Training goes in four stages: Stage 1 trains each adapter alone on its condition,
Stage 2 a universal adapter, Stage 3 the gate network, Stage 4 a joint polish.

### The ground rules that got written in stone this week

Two of them came from friction, and both are permanent. First: all compute is local, on
the M5 Pro's Apple silicon GPU. The training data was staged through a Lightning AI
workspace early on, and the moment that risked burning paid credits the rule became
absolute: Lightning AI is never used again, for anything. Second: the blueprint does not
get quietly compromised to make a training run easier. If the plan and the code
disagree, the plan wins or the plan gets amended in writing.

### July 17 and 18: making a Mac train

Getting Stage 1 running locally was its own debugging saga, and it is worth recording
because every one of these will bite anyone who tries this setup again.

**Slow data, fast fix.** Reverb, noise, and codec degradations were being applied to
full 15 to 30 second utterances, after which we cropped 2 seconds for training. Applying
the reverb convolution to audio we were about to throw away made data prep roughly 15
times slower than needed. The fix clips to 2 seconds first, then degrades. A small
wrinkle inside that fix: the mixture object is a dataclass whose `mixture` field is a
read-only property, so the clip has to rebuild the inner sample and use
`dataclasses.replace` rather than assigning in place.

**The exponential slowdown.** Batches went 1 second, then 11, then 86. It looked like a
memory leak. It was not. On Apple's MPS backend, the first forward pass for each unique
speaker count triggers Metal shader compilation, which can take from a few seconds to a
minute and a half, per count. Since our mixtures have 2 to 5 speakers, the first epoch
was paying compilation four separate times at random moments. The fix is a warm-up loop
before training that runs one forward and backward for each speaker count. It costs
about 8 seconds on a warm shader cache and up to 90 on a cold one, and after it, batch
times are stable.

**The five-hour stall.** A forgotten benchmark process from an earlier experiment sat on
the MPS device and starved the real training run for about five hours. Killing it
required force. The cleanup command then accidentally killed a DataLoader worker process
belonging to the live training run, which crashed it. Two rules came out of that
afternoon: data loading takes 8 milliseconds per sample against 1 to 4 seconds of
compute, so worker processes buy nothing here, and `num_workers=0` means there are no
child processes for a stray kill command to hit. Single process, single owner.

**Precision on MPS.** BF16 autocast works on this chip, but `torch.complex` does not
support half precision, so the STFT and iSTFT run inside explicit float32 islands while
the transformer body runs in BF16. `torch.mps.empty_cache()` after every optimizer step
keeps Metal command buffers from accumulating.

**One silent death.** One launch completed its warm-up and then produced no batch output
at all before dying. It was never fully root-caused; the strongest suspect is the same
blocked-device condition from the stale benchmark. The relaunch under a clean device ran
normally. Recorded here as an unknown, not a solved case.

**The batching attempt, and its honest ending.** The training loop pushes samples
through the model one at a time, four sequential forward passes per batch of four. We
built a true batched path: group each batch by speaker count, pack the STFT with real
and imaginary parts stacked on a channel dimension, pass the speaker count as a vector
so the model takes its batch-aware branch, one forward per group. It passed a smoke
test. Then, minutes into the real run, it died: holding four full activation graphs at
once needs more than the roughly 30 GiB unified-memory ceiling this 24 GB machine can
give the GPU. Out of memory, run dead. The per-sample loop exists precisely because each
sample's graph is freed by its own backward call before the next forward starts. We
reverted to it, left the batched code in place with a comment explaining why it is off,
and accepted the sequential speed. A crashed overnight run costs more hours than
per-sample forward ever will.

### Where things stand right now, 2026-07-18

Training config: batch 4, 2-second clips at 8 kHz, 500 samples per epoch, 40 epochs per
adapter, learning rate 1e-4, BF16, gradient clip 5.0. Adapters train in sequence:
reverb, then noise, then codec.

The first full epoch of today's earlier run finished with an average loss of 27.28,
down from mid-30s on the first batches, and saved a best checkpoint. After the batched
experiment ran out of memory, the run was relaunched on the per-sample path and is
currently in progress from epoch 1, holding a steady 5 to 6 seconds per batch, which
puts one adapter at roughly 7 to 8 hours and all three near a full day of wall clock.

To be plain about scores: **CALM-Sep has no separation quality numbers yet.** No
SI-SDRi, no count accuracy, no comparison against the bare frozen backbone. Those come
after Stage 1 finishes and the eval harness runs. Nothing in this section should be read
as a result. The only honest numbers in this whole document are v1's, and v1's numbers
are mostly about what did not work.

---

## Part 4. The timeline, in one place

| Date | What happened |
|---|---|
| First week of July 2026 | CA-MoSE foundation: data pipeline, mixers, expert wrappers, eval harness, model stubs. Three devs, parallel branches. |
| 2026-07-11 | Full SceneAnalyzer lands (PR #16). Escalation logging, count coordinator, E2E test. |
| 2026-07-11 to 07-12 | Integration. Identity lock fixed (three root causes). Residual-pad hack added for the 2-vs-3 stream mismatch. LibriMix prep defaults corrected to 8 kHz min. |
| 2026-07-13 | Kaggle T4 day. Ten bugs fixed live. Cache built, heads trained 30 epochs. Verdict: cascade never beats the frozen expert, fusion head costs 0.4 to 3.7 dB. Efficiency result is real: 36% compute cut at tau=6. Stop-classifier count accuracy 10%, two root causes found. M2 reframed around efficiency and closed. PROJECT_CRUX.md written. |
| 2026-07-14 to 07-15 | Digestion. The "BEATs failed" misreading gets corrected: the failure was the gate criterion, not any model. |
| 2026-07-16, night | 24-hour v2 master plan written: all-frozen routing system, no learned fusion, salvage v1's tested plumbing only. |
| 2026-07-17 | CALM-Sep takes shape: frozen SR-CorrNet backbone plus three LoRA adapters plus condition gate. BLUEPRINT becomes the source of truth. Data staged at 8 kHz. First reverb adapter run produces a checkpoint by night. Lightning AI banned permanently; all compute moves to the local M5 Pro. |
| 2026-07-18 | The MPS debugging day: pre-clip fix (15x data speedup), Metal shader warm-up (kills the exponential slowdown), stale benchmark killed after a 5-hour stall, worker crash leads to `num_workers=0`, samples per epoch set to 500. Epoch 1 completes at loss 27.28. A grouped-batch forward is built, passes a smoke test, then dies of GPU memory exhaustion minutes into the real run. Reverted to the memory-safe per-sample loop, relaunched, and the full 3-adapter run is training as this document is written. |

---

## Part 5. What is pending

1. Finish Stage 1: reverb, then noise, then codec adapters, 40 epochs each on the M5
   Pro. This is running now and needs nothing but time and an undisturbed GPU.
2. Stage 2 (universal adapter), Stage 3 (gate network), Stage 4 (joint polish), in that
   order, per BLUEPRINT.
3. The evaluation that v1 never earned: per-condition SI-SDRi with the adapters on and
   off, count accuracy from the backbone attractors, and an honest table against the
   bare frozen expert. If an adapter does not beat the frozen backbone on its own
   condition, that gets written down, not massaged.
4. From the v1 side, still open: the trained cascade checkpoint sits only in a local
   Downloads folder, unbacked. It is a dead architecture, but it is also the evidence
   behind the post-mortem numbers, and it should be archived somewhere deliberate.

---

## Part 6. What this project has actually taught us

Written as reminders to ourselves, because every one of these was learned the expensive
way.

A criterion written before the first measurement is a wish, not a gate. Reframe wishes
the moment real numbers arrive, in the open, instead of letting a red badge sit for
days and then quietly moving the goalposts.

When something fails, name the exact thing that failed. "BEATs failed" cost us days of
aiming at the wrong problem. The correct sentence was one flag name long.

A known hack documented in a comment is still a known hack in the results table. If a
comparison depends on it, the disclosure belongs next to the number.

Run everything end to end, at toy scale, on a laptop, before renting a GPU. Nine of the
ten Kaggle bugs would have surfaced in a single local CPU dry-run with one batch.

The strong frozen model is not the enemy. Both times we tried to put trainable layers
on top of SR-CorrNet's output, we made it worse. The v2 bet is that the right place to
help a frozen model is inside its weights, gently, with small adapters, and only under
conditions it was never trained for. Whether that bet pays off is exactly what the
current training run exists to find out.

And the simplest one: progress trackers, pulse counters, and status badges are part of
the work. If the tracker says something the table does not, someone we respect will
notice, and they will be right to.
