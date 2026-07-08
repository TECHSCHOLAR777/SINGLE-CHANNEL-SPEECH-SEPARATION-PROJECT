# Person B (Dev B) — Phase 0 Work Summary

> **Role:** Expert integration, cascade gate, fusion head (primary); speaker counting (secondary)  
> **Phase:** P0 Foundation (weeks 1–2)  
> **Milestone gate:** M0 — all three developers reproduce the same SI-SDRi baseline on Libri3Mix

---

## Scope

Person B owns the **model vertical** (`models/`, `train/`) and **leads the repository skeleton** in Phase 0. The tasks below were implemented without depending on Dev A or Dev C deliverables.

| Task (from DEVELOPMENT_PLAN.md) | Status | Deliverable |
|--------------------------------|--------|-------------|
| Repository skeleton, environment, dependency lockfile, pre-commit hooks | Done | Full repo layout, `pyproject.toml`, `requirements.txt`, `.pre-commit-config.yaml`, CI workflow |
| Shared separation result interface | Done | `schemas/separation_result.py` |
| Mixer stub (baseline dependency) | Done | `data/mixer_stub.py` |
| Baseline runner: SepFormer + SR-CorrNet on Libri3Mix | Done | `models/baseline_runner.py`, `scripts/run_baseline.py` |
| Expert wrappers (SepFormer, SR-CorrNet) | Done | `models/experts/sepformer.py`, `models/experts/srcorrnet.py` |
| Unit tests for Person B modules | Done | `tests/test_*.py` (12 tests, all passing) |
| Design documentation | Done | `docs/models.md`, `docs/decisions.md` |

---

## Repository structure created

```
summer-project-2/
├── .github/workflows/ci.yml      # Lint + pytest on push/PR
├── .pre-commit-config.yaml       # Black + Ruff
├── pyproject.toml                # Package definition and tool config
├── requirements.txt              # Core dependencies
├── requirements-dev.txt          # Dev/test dependencies
├── README.md                     # Quick start and layout
├── configs/
│   ├── baseline.yaml             # Baseline run configuration
│   └── default.yaml              # Shared config placeholder (Dev C owns loader)
├── data/
│   └── mixer_stub.py             # Libri3Mix disk loader (stub until Dev A mixer)
├── models/
│   ├── baseline_runner.py        # SI-SDRi baseline evaluation
│   └── experts/
│       ├── sepformer.py          # SpeechBrain SepFormer wrapper
│       └── srcorrnet.py          # SR-CorrNet wrapper (cloned repo)
├── schemas/
│   └── separation_result.py      # Shared SeparationResult contract
├── scripts/
│   └── run_baseline.py           # CLI for Phase 0 baseline
├── tests/                        # 12 unit tests
└── docs/
    ├── decisions.md              # Architecture decision log
    ├── models.md                 # Models area design note
    └── PERSON_B_PHASE0_SUMMARY.md
```

Placeholder directories for other owners: `eval/` (Dev C), `align/` (Dev C), `demo/` (Dev C), `train/` (Dev B, Phase 2+).

---

## Key deliverables

### 1. Repository skeleton and tooling

- **Package:** `ca-mose` installable via `pip install -e ".[dev]"`
- **Lint/format:** Black and Ruff enforced by pre-commit hooks
- **CI:** GitHub Actions runs `ruff check`, `black --check`, and `pytest` on Python 3.10 and 3.11
- **Dependencies:** PyTorch, SpeechBrain, Asteroid, soundfile, PyYAML, tqdm

### 2. Shared interface — `SeparationResult`

All expert wrappers and the future pipeline return a single type:

```python
@dataclass
class SeparationResult:
    streams: np.ndarray       # [K, T] separated waveforms
    sample_rate: int          # 16000 Hz project standard
    speaker_count: int        # K_hat
    metadata: list[StreamMetadata]  # per-stream confidence, expert source, etc.
    mixture: np.ndarray | None
    escalated: bool
    expert_used: str
```

Defined once in `schemas/separation_result.py` — no ad-hoc result types elsewhere.

### 3. Mixer stub — `data/mixer_stub.py`

Loads pre-mixed Libri3Mix from disk until Dev A delivers the on-the-fly dynamic mixer:

- Expected layout: `{data_root}/wav16k/max/{subset}/mix_both/`, `s1/`, `s2/`, `s3/`
- Returns `MixtureSample(mixture, references, sample_rate, utterance_id)`

### 4. Expert wrappers

| Expert | File | Source | Notes |
|--------|------|--------|-------|
| SepFormer | `models/experts/sepformer.py` | `speechbrain/sepformer-wsj03mix` | Phase 0 control baseline; lazy model load |
| SR-CorrNet | `models/experts/srcorrnet.py` | Cloned `github.com/dmlguq456/SR_CorrNet` | Skipped gracefully if repo/checkpoint not configured |

Both return `SeparationResult` with frozen pretrained weights (inference only).

### 5. Baseline runner

**Entry point:**

```bash
python scripts/run_baseline.py --config configs/baseline.yaml
```

**Behavior:**

1. Discovers Libri3Mix test clips via mixer stub
2. Runs SepFormer (and SR-CorrNet if configured) on each sample
3. Computes permutation-invariant **SI-SDRi** (Asteroid PIT when available; numpy fallback otherwise)
4. Writes results to `outputs/baseline/baseline_results.json` and `baseline_results.md`

**Config (`configs/baseline.yaml`):**

| Key | Purpose |
|-----|---------|
| `data_root` | Path to Libri3Mix (set after Dev A download) |
| `subset` | `test` / `dev` / `train` |
| `max_samples` | Cap for quick runs |
| `device` | `cuda` or `cpu` |
| `srcorrnet_repo` | Path to cloned SR-CorrNet repo |
| `srcorrnet_checkpoint` | Path to SR-CorrNet weights |

---

## Tests

| File | What it covers |
|------|----------------|
| `tests/test_separation_result.py` | Schema validation, torch conversion |
| `tests/test_mixer_stub.py` | Libri3Mix discovery with synthetic fixture |
| `tests/test_sepformer_wrapper.py` | SepFormer output shape (mocked, no download) |
| `tests/test_srcorrnet_wrapper.py` | SR-CorrNet availability checks |
| `tests/test_baseline_runner.py` | SI-SDRi computation (perfect vs poor separation) |

```bash
pytest tests/ -v   # 12 passed
```

---

## Architecture decisions logged

Recorded in `docs/decisions.md`:

- Single `SeparationResult` schema for all modules
- SepFormer as Phase 0 baseline (MossFormer2 in Phase 1)
- SR-CorrNet loaded from cloned repo, not pip
- Mixer stub unblocks baseline before Dev A’s dynamic mixer
- Black + Ruff via pre-commit
- SI-SDRi via Asteroid PIT with numpy fallback for CI

---

## Blocked on teammates (not Person B work)

| Item | Owner | Action needed |
|------|-------|---------------|
| Libri3Mix download | Dev A | Set `data_root` in `configs/baseline.yaml` |
| Full dynamic mixer | Dev A | Will replace `data/mixer_stub.py` |
| Eval harness (`eval/metrics.py`) | Dev C | M0 gate requires identical SI-SDRi numbers from shared harness |
| Config loader | Dev C | Baseline uses direct YAML load for now |

---

## How to reproduce locally

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pre-commit install

# 2. Unit tests (no GPU or data)
pytest tests/ -v

# 3. Baseline (GPU + Libri3Mix required)
#    Edit configs/baseline.yaml → set data_root
python scripts/run_baseline.py --config configs/baseline.yaml --max-samples 50
```

---

## Next up — Phase 1 (Person B)

| Task | Deliverable |
|------|-------------|
| MossFormer2 inference wrapper | `models/experts/mossformer2.py` |
| SR-CorrNet wrapper (full integration) | Attractor + confidence outputs |
| REAL-M blind quality estimator | Cascade gate quality signal |
| Pair with Dev C on alignment interface | Expert output → Hungarian alignment |

---

## References

- [MASTER_PROJECT.md](../MASTER_PROJECT.md) — architecture and component specs
- [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md) — team roles and phase breakdown
- [docs/models.md](models.md) — models area design note
- [docs/decisions.md](decisions.md) — decision log
