# Models area design note (Dev B)

## Purpose

The `models/` directory owns frozen expert wrappers, the cascade gate, fusion head, router integration, and the Phase 0 baseline runner. Training loops live in `train/`.

## Phase 0 deliverables

1. **Expert wrappers** (`models/experts/`)
   - `SepFormerExpert`: SpeechBrain SepFormer, 3-speaker, frozen.
   - `SRCorrNetExpert`: SR-CorrNet-B from cloned repo, frozen, with attractor confidence in metadata.

2. **Baseline runner** (`models/baseline_runner.py`, `scripts/run_baseline.py`)
   - Loads Libri3Mix test clips via `data/mixer_stub.py`.
   - Runs each expert, computes permutation-invariant SI-SDRi.
   - Writes `outputs/baseline/baseline_results.json` and `.md`.

3. **Shared schema** (`schemas/separation_result.py`)
   - All experts return `SeparationResult` with `streams [K, T]`, `speaker_count`, and per-stream `StreamMetadata`.

## Interface contract

```python
result: SeparationResult = expert.separate(mixture, sample_rate)
# result.streams.shape == (K, T)
# result.speaker_count == K
# len(result.metadata) == K
```

## Phase 1 additions (planned)

- `MossFormer2Expert` wrapper (`models/experts/mossformer2.py`) — ClearVoice / MossFormer2_SS_16K
- `REALMQualityEstimator` for cascade gate (`models/realm_quality.py`)
- `preprocess.py` — resample 16 kHz, -26 dBFS peak norm, STFT branch
- `TFGridNetExpert` fallback when SR-CorrNet unavailable (`models/experts/tfgridnet.py`)
- Hungarian alignment consumes expert outputs (owned by Dev C in `align/`)

## Phase 2 additions (Dev B)

- `CascadeGate` (`models/cascade_gate.py`) — REAL-M score vs tau, escalate if below
- `CRRRFusionHead` (`models/fusion.py`) — Confidence-Routed Residual Refinement (~1M params)
- `SceneAnalyzer` stub (`models/scene_analyzer.py`) — interim until Dev A full analyzer
- `CompositeLoss` (`train/losses.py`) — all seven MASTER §7.2 terms
- `CAMoSETrainer` (`train/trainer.py`) — trains scene/router/fusion heads; experts frozen

## Phase 3 additions (Dev B, P3-B1)

- `CountingFeatureExtractor` (`models/counting_features.py`) — four stop-classifier signals:
  - Residual energy ratio and mixture-consistency error from waveform math
  - VAD speech probability on the residual (`VADAdapter`, energy default, Silero optional)
  - Minimum ECAPA cosine distance to prior stems (reuses `ECAPAEmbedder`)
- Feature vector order frozen in `FEATURE_NAMES`; consumed by Dev C's `StopClassifier`
- `compute_stop_features()` accepts precomputed VAD/embedding or computes them end-to-end

## Hardware notes

- Target: Kaggle T4 16 GB.
- SepFormer inference fits comfortably on T4.
- SR-CorrNet RTF ~0.31; baseline runs are batch-offline, not real-time.
