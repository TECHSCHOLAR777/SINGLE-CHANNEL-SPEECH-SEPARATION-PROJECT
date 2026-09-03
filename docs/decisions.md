# Architecture decision log

> 🕰️ **Historical record, not edited.** Entries run from 2026-07-09 to 2026-07-17 and
> use the project names in force at the time: CA-MoSE for the abandoned cascade,
> then CALM-Sep. The project is now CoRAL-Sep. The old names are left in place
> because rewriting a dated decision log falsifies the record.
>
> Decisions taken from 2026-09-02 onward, with full context and consequences,
> live in [restoration/DECISIONS.md](restoration/DECISIONS.md).
>
> One entry is known to be wrong: the 2026-07-17 line stating 17 LoRA attachment
> points. The measured count is 37. Tracked as I-039; a dated correction will be
> appended rather than the original entry edited.

| Date | Decision | Reason |
|------|----------|--------|
| 2026-07-09 | `SeparationResult` dataclass in `schemas/separation_result.py` is the single shared output type | Prevents ad-hoc result objects across experts, fusion, eval, and demo |
| 2026-07-09 | SepFormer used as Phase 0 control baseline (SpeechBrain `sepformer-wsj03mix`) | Published 19.8 dB SI-SDRi reference; MossFormer2 wrapper added in Phase 1 |
| 2026-07-09 | SR-CorrNet loaded from cloned repo path, not pip | Upstream repo is not packaged; graceful skip when not configured |
| 2026-07-09 | `data/mixer_stub.py` loads pre-mixed Libri3Mix from disk | Unblocks Dev B baseline before Dev A delivers on-the-fly dynamic mixer |
| 2026-07-09 | Black + Ruff via pre-commit | Agreed coding standards from DEVELOPMENT_PLAN Section 7 |
| 2026-07-09 | SI-SDRi via Asteroid PIT in baseline runner | Matches eval harness plan; numpy fallback for CI without full deps |
| 2026-07-09 | Dev C ships full vertical (eval + align + router + stop-classifier + demo), not phase-gated | Integration can pull any piece the moment Dev A/B deliver; every module lands with known-answer tests |
| 2026-07-09 | `eval/metrics.py` is the canonical SI-SDR/PIT source; baseline_runner's `compute_sisdri` is a P0 stopgap | One ruler for training, CI, and demo; runner should import from eval later |
| 2026-07-09 | Unknown-N mismatched-count handling is configurable via `missing_policy`, never silent | Over/under-separation is the evaluated axis; scores must be honest and reported (unassigned/missing lists) |
| 2026-07-09 | Stop-classifier calibrated by temperature scaling, loaded with `weights_only=True` | Honest confidence badge; safe checkpoint loading (contrast srcorrnet.py `weights_only=False`) |
| 2026-07-09 | Demo engine injected as a callable; `MockEngine` is weight-free | Demo UI buildable before any model is ready; real experts drop in without touching demo code |
| 2026-07-09 | `utils` added to pyproject packages include | Was missing; `pip install -e .` would not have shipped the config loader |
| 2026-07-10 | MossFormer2 via optional `clearvoice` package; ECAPA embeddings attached in-wrapper | M1 gate requires streams + embeddings; avoids blocking on Dev C ECAPA wrapper |
| 2026-07-10 | Baseline runner imports SI-SDRi from `eval/metrics.py` | P0-INT1: single canonical metric implementation |
| 2026-07-10 | TF-GridNet fallback with SepFormer last resort | P1-B3: graceful degradation when SR-CorrNet repo unavailable |
| 2026-07-10 | Preprocessing: -26 dBFS peak, STFT 512/128 | MASTER §4.2 Stage 0 spec |
| 2026-07-11 | Cascade gate uses REAL-M `min_sisnr_db` vs tau (strict `<`) | Conservative escalation per MASTER §4.3; borderline inputs escalate |
| 2026-07-11 | CRRR fusion: `s_fused = s_SR + alpha * R_theta` with ~1M conv residual net | MASTER §4.2 fusion formula; alpha from confidence, entropy, SI-SDR proxy, scene weights |
| 2026-07-11 | Composite loss in `train/losses.py` with MASTER §7.2 lambdas | Single assembly point for all seven training terms |
| 2026-07-11 | `models/scene_analyzer.py` interim stub until Dev A P2-A1 | Unblocks P2 trainer; smaller than full 1.5M spec |
| 2026-07-11 | Trainer self-test mode with synthetic batches (no expert downloads) | CI verifies training mechanics before real LibriMix runs |
| 2026-07-11 | Stop-classifier features in `models/counting_features.py` (P3-B1) | Dev B owns extractors; energy VAD default (CI-safe), optional Silero; ECAPA via existing embedder |
| 2026-07-11 | `align/integration.py` owns expert-pair and long-form alignment orchestration behind small protocols | Keeps real wrappers swappable while deterministic fake engines cover the integration seam in CI |
| 2026-07-11 | Silent, near-zero, extreme, or non-finite rows map to neutral finite alignment cost | Prevents BLAS-specific divide/overflow warnings and keeps NaN/Inf out of the Hungarian solver |
| 2026-07-11 | P0 end-to-end coverage uses a deterministic injected fake expert | Proves config, runner, PIT SI-SDRi, and reporting compose without downloading weights or datasets |
| 2026-07-11 | Counting report artifacts are generated from canonical `RunLog` records | One source produces confusion/calibration JSON, CSV, Markdown, and SVG without notebook-only state |
| 2026-07-11 | Real M1 acceptance tests are opt-in with `RUN_REAL_EXPERTS=1` | Default CI stays reproducible; RunPod/Kaggle can execute the same assertions with real speech and weights |
| 2026-07-17 | Re-baseline to CALM-Sep; patches A/B/C via `SRCorrNetWrapper` hooks | BLUEPRINT §15.1 without editing vendored engine_infer |
| 2026-07-17 | Primary LoRA attach set = 17 QKV+filter layers | Matches BLUEPRINT §5.3 layer count; aggregate/aux optional |
| 2026-07-17 | P1b universal-adapter verdict PENDING until training | Pre-committed: adopt if within 0.5 dB on primary benchmark; log here before gate training |
