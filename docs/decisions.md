# Architecture decision log

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
