# Dev C: Evaluation, Alignment, Counting, Demo

Owner: Dev C. Status: complete vertical, all modules land with known-answer tests, full suite green (64 passed), ruff and black clean under the repo's own CI commands.

## What this covers

Everything in the Dev C lane from the development plan, delivered as one vertical rather than phase-gated, so integration can pull any piece the moment Dev A data or Dev B experts are ready.

| Module | Purpose | Key guarantee |
|--------|---------|---------------|
| `utils/config.py` | Layered YAML loader, deep-merge, dot-path access | Nothing downstream hardcodes a tunable; `load_config(a, b, overrides=...)` with last-wins precedence |
| `eval/metrics.py` | SI-SDR, SI-SDRi, PIT, counting | Framework-free numpy; mismatched-count (unknown-N) handling is configurable and reported, never silent |
| `eval/reporting.py` | Run log + every report query | Each table/figure is a query over one JSONL log, not a hand-built artifact |
| `align/hungarian.py` | Stream alignment across experts/chunks | Embedding cost with waveform-xcorr fallback; the method actually used is always returned |
| `align/chunking.py` | Long-form identity lock + stitch | Matches against full track history so a speaker silent for a chunk keeps their track |
| `models/router.py` | Two-level adaptive router | Sigmoid gating (experts co-activate), load-balance + null-sparsity losses, ~0.5M params |
| `models/stop_classifier.py` | Learned + calibrated stop decision | Temperature scaling; frozen feature-order contract; `weights_only=True` load |
| `train/train_stop_classifier.py` | Trains the above | `--self-test` proves training mechanics with zero data or GPU |
| `demo/app.py` | Gradio demo | Engine injected as a callable; `MockEngine` works today, real experts drop in unchanged |

## The M0 contract: the ruler is straight

`eval/metrics.py` is verified by known-answer tests, not tolerance guesses:

- An oracle estimate scores above 60 dB SI-SDR.
- Using the mixture itself as the estimate scores exactly 0.0 dB SI-SDRi (`abs=1e-9`).
- Noise orthogonalized to the reference recovers the requested SNR within 0.05 dB.
- A permuted oracle recovers the exact permutation via Hungarian PIT.

Finding worth knowing: SI-SDR's epsilon guard makes the *perfect-match* case amplitude-dependent (the residual is zero, so EPS dominates). This is correct behavior for the metric; only a naive test of it is wrong. Scale-invariance is therefore tested in the finite-residual regime, where the property actually holds, and the reason is documented in the test.

## Unknown-N handling (the eval's whole point)

`pit_si_sdr` handles both mismatched-count cases explicitly:

- **Over-separation** (more estimates than references): the best subset is matched; extra streams are reported in `unassigned_estimates` and do not affect the score. Stem Hygiene prunes them upstream; the harness still records that they existed.
- **Under-separation** (missing a speaker): scored by `missing_policy`. `mixture_fallback` gives that speaker exactly 0 dB SI-SDRi (the listener is left with the mixture); `silence_floor` assigns a configured floor. Missing speakers are always listed in `missing_references`.

## SR-CorrNet weights: read before relying on the escalation path

The baseline config ships `srcorrnet_repo: null` and `srcorrnet_checkpoint: null`, and no checkpoint is committed to the repo. The wrapper's `is_available` returns False in that state and the runner skips the expensive expert gracefully. Before any M1/M2 number that depends on the escalation path is trusted, the SR-CorrNet checkpoint must be present locally and its provenance verified: the upstream GitHub repo `dmlguq456/SR_CorrNet` returned 404 as of July 8, so any weights in hand came from elsewhere and their source should be recorded in `docs/decisions.md`.

Security note for whoever wires the checkpoint: `srcorrnet.py` currently loads with `torch.load(weights_only=False)`, which executes arbitrary code embedded in a pickle. For a checkpoint from an unverified source that is a real risk. Switch to `weights_only=True` once the expected tensor layout is known, matching how `stop_classifier.py` already loads.

## Running it

```bash
pip install -e ".[dev]"
pytest tests/ -v                                   # full suite, no GPU or weights needed
python -m train.train_stop_classifier --self-test  # proves training mechanics
python -m demo.app --mock                           # demo on the weight-free mock engine
```
