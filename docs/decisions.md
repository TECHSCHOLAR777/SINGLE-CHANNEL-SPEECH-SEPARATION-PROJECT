# Architecture decision log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-07-09 | `SeparationResult` dataclass in `schemas/separation_result.py` is the single shared output type | Prevents ad-hoc result objects across experts, fusion, eval, and demo |
| 2026-07-09 | SepFormer used as Phase 0 control baseline (SpeechBrain `sepformer-wsj03mix`) | Published 19.8 dB SI-SDRi reference; MossFormer2 wrapper added in Phase 1 |
| 2026-07-09 | SR-CorrNet loaded from cloned repo path, not pip | Upstream repo is not packaged; graceful skip when not configured |
| 2026-07-09 | `data/mixer_stub.py` loads pre-mixed Libri3Mix from disk | Unblocks Dev B baseline before Dev A delivers on-the-fly dynamic mixer |
| 2026-07-09 | Black + Ruff via pre-commit | Agreed coding standards from DEVELOPMENT_PLAN Section 7 |
| 2026-07-09 | SI-SDRi via Asteroid PIT in baseline runner | Matches eval harness plan; numpy fallback for CI without full deps |
