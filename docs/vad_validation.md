# SileroVAD voiced-density proxy (P0-C3)

## Decision

**Fallback selected for default CI path:** STFT voiced-energy fraction
(`data/vad_features.py`), because Silero requires a torch hub download and is
opt-in via `CORALSEP_ENABLE_SILERO=1`.

## Validation procedure (run once with LibriCSS locally)

```bash
set CORALSEP_ENABLE_SILERO=1
python -c "from data.vad_features import validate_vad_proxy; ..."
```

`validate_vad_proxy(waveforms, overlap_labels)` checks that voiced-frame density
spreads with overlap ratio. If Silero spread is uninformative, keep the STFT
fallback (already the default).

## Status

- Code: done
- Live LibriCSS validation number: pending local data (not blocking codebase)
