# CA-MoSE: Condition-Aware Mixture-of-Separation-Experts

Multi-speaker blind speech separation for three or more concurrent speakers using conditional cascade routing between MossFormer2 and SR-CorrNet.

See [MASTER_PROJECT.md](MASTER_PROJECT.md) for architecture and [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for team workflow.

## Quick start

```bash
# Create environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"
pre-commit install

# Run unit tests (no GPU or pretrained weights required)
pytest tests/ -v

# Run Phase 0 baseline (requires GPU + downloaded Libri3Mix data)
python scripts/run_baseline.py --config configs/baseline.yaml
```

## Repository layout

| Directory | Owner | Purpose |
|-----------|-------|---------|
| `data/` | Dev A | Mixer, augmentation, dataset prep |
| `models/` | Dev B | Experts, router, fusion, cascade |
| `train/` | Dev B | Training loops |
| `eval/` | Dev C | Metrics and evaluation harness |
| `align/` | Dev C | Stream alignment |
| `demo/` | Dev C | Gradio demo |
| `schemas/` | Shared | Interface contracts (e.g. `SeparationResult`) |
| `configs/` | Shared | YAML configuration files |
| `tests/` | Shared | Unit and integration tests |
| `docs/` | Shared | Design notes and decision log |

## Phase 0 baseline (Dev B)

The baseline runner loads pretrained SepFormer and SR-CorrNet on Libri3Mix test clips and reports SI-SDRi. All three developers must independently reproduce the same numbers at milestone M0.

```bash
python scripts/run_baseline.py --config configs/baseline.yaml --max-samples 50
```

Set `data_root` in `configs/baseline.yaml` to your Libri3Mix test directory before running.
