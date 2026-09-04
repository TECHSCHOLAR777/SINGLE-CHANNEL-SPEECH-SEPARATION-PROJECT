# SileroVAD voiced-density proxy (P0-C3)

## Decision

**Fallback selected for default CI path:** STFT voiced-energy fraction
(`src/coralsep/data/vad_features.py`), because Silero requires a torch hub
download and is opt-in via `CALMSEP_ENABLE_SILERO=1`. The flag keeps the
pre-rename spelling in the source (`_silero_enabled` in `vad_features.py`);
it was not renamed to `CORALSEP_ENABLE_SILERO` alongside the rest of the
project and nothing else in the tree references it.

## Validation procedure (run once with LibriCSS locally)

```bash
set CALMSEP_ENABLE_SILERO=1
python -c "from coralsep.data.vad_features import validate_vad_proxy; ..."
```

`validate_vad_proxy(waveforms, overlap_labels)` checks that voiced-frame density
spreads with overlap ratio. If Silero spread is uninformative, keep the STFT
fallback (already the default).

## Status

- Code: done
- Live LibriCSS validation number: pending local data (not blocking codebase)
