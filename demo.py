"""
CALM-Sep Demo — Condition-Aware LoRA Mixture for speech separation.

Usage:
    PYTHONPATH=/path/to/sr_corrnet_src:. .venv/bin/python3 demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

_SR_CORRNET_SRC = Path.home() / "Downloads/SR_CorrNet_local_mixboth/future_work"
if _SR_CORRNET_SRC.exists() and str(_SR_CORRNET_SRC) not in sys.path:
    sys.path.insert(0, str(_SR_CORRNET_SRC))

CKPT_DIR = Path("checkpoints")
_SR = 8_000
_SR_OUT = 16_000
_MAX_SECS = 6
_HF_MODEL = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"
MAX_SPKS = 5

ADAPTER_LABELS = {"reverb": "Reverb", "noise": "Noise", "codec": "Codec"}

# ─────────────────────────────────────────────────────────────────────────────
# Model state
# ─────────────────────────────────────────────────────────────────────────────

_state: dict = {}


def _ensure_loaded():
    if _state:
        return
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    from models.condition import level1_tensor
    from models.gate import GateNetwork
    from models.lora import ADAPTER_NAMES, LoRALibrary, LoRALinear
    from train.stage1_single import _get_inner_module, _load_model

    device = torch.device("cpu")

    ss_base = _load_model(_HF_MODEL, device)
    _get_inner_module(ss_base).eval()
    _state["ss_base"] = ss_base

    ss_calm = _load_model(_HF_MODEL, device)
    inner = _get_inner_module(ss_calm)
    lib = LoRALibrary(inner)
    lib.freeze_base()

    gate_net = GateNetwork().to(device)
    temperature = 1.0

    joint = CKPT_DIR / "stage4_joint" / "best_joint.pt"
    if joint.exists():
        ckpt = torch.load(str(joint), map_location="cpu", weights_only=False)
        gate_net.load_state_dict(ckpt["gate"])
        adapter_state = ckpt.get("adapter_state", {})
        for mod_name, mod in inner.named_modules():
            if not isinstance(mod, LoRALinear):
                continue
            for adapter_name, branch in mod.branches.items():
                for param_name, param in branch.named_parameters():
                    key = f"{mod_name}.branches.{adapter_name}.{param_name}"
                    if key in adapter_state:
                        param.data.copy_(adapter_state[key])

    calib = CKPT_DIR / "stage4c" / "calibration.pt"
    if calib.exists():
        c = torch.load(str(calib), map_location="cpu", weights_only=False)
        temperature = float(c["temperature"].item())

    inner.to(device).eval()
    gate_net.eval()

    _state.update(
        {
            "ss_calm": ss_calm,
            "inner": inner,
            "lib": lib,
            "gate_net": gate_net,
            "temperature": temperature,
            "level1": level1_tensor,
            "ADAPTER_NAMES": ADAPTER_NAMES,
            "device": device,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resample_to_8k(wav: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == _SR:
        return wav.astype(np.float32)
    try:
        import torchaudio

        t = torch.from_numpy(wav).float().unsqueeze(0)
        return torchaudio.functional.resample(t, sr_in, _SR).squeeze(0).numpy()
    except Exception:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_in, _SR)
        return resample_poly(wav, _SR // g, sr_in // g).astype(np.float32)


def _to_16k(wav_8k: np.ndarray) -> np.ndarray:
    try:
        import torchaudio

        t = torch.from_numpy(wav_8k).float().unsqueeze(0)
        return torchaudio.functional.resample(t, _SR, _SR_OUT).squeeze(0).numpy()
    except Exception:
        out = np.zeros(len(wav_8k) * 2, dtype=np.float32)
        out[::2] = wav_8k
        return out


def _extract_waves(out_dict: dict) -> list[np.ndarray]:
    result = []
    for w in out_dict.get("waveforms", []):
        arr = w.squeeze().cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w).squeeze()
        result.append(arr.astype(np.float32))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Separation runners
# ─────────────────────────────────────────────────────────────────────────────


def _run_baseline(wav_8k: np.ndarray) -> tuple[list[np.ndarray], float]:
    ss = _state["ss_base"]
    dev = _state["device"]
    t0 = time.time()
    wav_t = torch.from_numpy(wav_8k).float().to(dev).unsqueeze(0)
    with torch.no_grad():
        out = ss.process_waveform(wav_t, n_spks=None)
    return _extract_waves(out) or [wav_8k], time.time() - t0


def _run_calmsep(wav_8k: np.ndarray) -> tuple[list[np.ndarray], float, dict[str, float]]:
    dev = _state["device"]
    ss, lib = _state["ss_calm"], _state["lib"]
    gate_net = _state["gate_net"]
    temperature = _state["temperature"]
    l1fn = _state["level1"]
    NAMES = _state["ADAPTER_NAMES"]

    wav_t = torch.from_numpy(wav_8k).float().to(dev)
    l1 = l1fn(wav_t).to(dev)
    cond = torch.cat([l1, torch.zeros(6, device=dev)]).unsqueeze(0)
    with torch.no_grad():
        gate_prob = gate_net(cond).squeeze(0).float().clamp(1e-6, 1 - 1e-6)
    if temperature != 1.0:
        logit = torch.log(gate_prob / (1 - gate_prob))
        gate_prob = torch.sigmoid(logit / temperature)
    gate_vals = {NAMES[i]: float(gate_prob[i].item()) for i in range(len(NAMES))}

    lib.set_gates(gate_vals)
    lib.inject_gates()
    t0 = time.time()
    with torch.no_grad():
        out = ss.process_waveform(wav_t.unsqueeze(0), n_spks=None)
    elapsed = time.time() - t0
    lib.set_gates({n: 0.0 for n in NAMES})
    lib.inject_gates()
    return _extract_waves(out) or [wav_8k], elapsed, gate_vals


# ─────────────────────────────────────────────────────────────────────────────
# Waveform figure
# ─────────────────────────────────────────────────────────────────────────────


def _waveform_fig(wav: np.ndarray, color: str) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(7, 1.6))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    step = max(1, len(wav) // 4000)
    t = np.arange(0, len(wav), step) / _SR
    w = wav[::step]

    # Glow fill + line
    ax.fill_between(t, w, alpha=0.12, color=color, linewidth=0)
    ax.fill_between(t, np.clip(w, 0, None), alpha=0.45, color=color, linewidth=0)
    ax.fill_between(t, np.clip(w, None, 0), alpha=0.45, color=color, linewidth=0)
    ax.plot(t, w, color=color, linewidth=0.7, alpha=0.95)
    ax.axhline(0, color="#222", linewidth=0.5, zorder=0)

    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(t[0], t[-1])
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic HTML fragments
# ─────────────────────────────────────────────────────────────────────────────


def _gate_html(gate_vals: dict[str, float]) -> str:
    rows = ""
    for key, label in ADAPTER_LABELS.items():
        pct = gate_vals.get(key, 0.0) * 100
        color = "#00D4B8" if pct >= 60 else ("#818cf8" if pct >= 30 else "#333")
        rows += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
          <span style="font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;
                       color:#555;width:46px;flex-shrink:0;font-family:inherit;">{label}</span>
          <div style="flex:1;height:3px;background:#1a1a1a;border-radius:2px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:2px;
                        box-shadow:0 0 8px {color}80;
                        transition:width 0.7s cubic-bezier(.4,0,.2,1);"></div>
          </div>
          <span style="font-size:0.75rem;color:{color};width:34px;text-align:right;
                       font-variant-numeric:tabular-nums;font-weight:600;">{pct:.0f}%</span>
        </div>"""
    return f'<div style="padding:6px 0;">{rows}</div>'


def _status_html(base_time: float, calm_time: float, temperature: float, n_det: int) -> str:
    return f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;padding:12px 0 4px;">
      <span style="font-size:0.72rem;padding:4px 12px;border-radius:100px;
                   border:1px solid rgba(129,140,248,0.25);color:#818cf8;background:rgba(129,140,248,0.06);">
        SR-CorrNet &nbsp;<strong>{base_time:.1f}s</strong>
      </span>
      <span style="font-size:0.72rem;padding:4px 12px;border-radius:100px;
                   border:1px solid rgba(0,212,184,0.25);color:#00D4B8;background:rgba(0,212,184,0.06);">
        CALM-Sep &nbsp;<strong>{calm_time:.1f}s</strong>
      </span>
      <span style="font-size:0.72rem;padding:4px 12px;border-radius:100px;
                   border:1px solid rgba(255,255,255,0.08);color:#555;background:rgba(255,255,255,0.03);">
        T = <strong style="color:#888;">{temperature:.3f}</strong>
      </span>
      <span style="font-size:0.72rem;padding:4px 12px;border-radius:100px;
                   border:1px solid rgba(255,255,255,0.08);color:#555;background:rgba(255,255,255,0.03);">
        {n_det} speaker{"s" if n_det != 1 else ""} detected
      </span>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main process fn
# ─────────────────────────────────────────────────────────────────────────────


def process(audio_input, progress=gr.Progress(track_tqdm=False)):
    if audio_input is None:
        raise gr.Error("Upload an audio file first.")

    sr_in, wav_np = audio_input
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    wav_np = wav_np.astype(np.float32)
    peak = np.abs(wav_np).max()
    if peak > 0:
        wav_np /= peak

    max_samples = int(_MAX_SECS * sr_in)
    if len(wav_np) > max_samples:
        wav_np = wav_np[:max_samples]

    progress(0.05, desc="Resampling …")
    wav_8k = _resample_to_8k(wav_np, sr_in)
    peak8 = np.abs(wav_8k).max()
    if peak8 > 0:
        wav_8k /= peak8

    progress(0.10, desc="Loading models …")
    _ensure_loaded()

    progress(0.15, desc="Running SR-CorrNet baseline …")
    base_waves, base_time = _run_baseline(wav_8k)

    progress(0.55, desc="Running CALM-Sep …")
    calm_waves, calm_time, gate_vals = _run_calmsep(wav_8k)

    progress(0.92, desc="Rendering …")

    n_det = max(len(base_waves), len(calm_waves))
    while len(base_waves) < n_det:
        base_waves.append(np.zeros(len(wav_8k), dtype=np.float32))
    while len(calm_waves) < n_det:
        calm_waves.append(np.zeros(len(wav_8k), dtype=np.float32))

    base_audio = [(_SR_OUT, _to_16k(w)) for w in base_waves[:MAX_SPKS]]
    calm_audio = [(_SR_OUT, _to_16k(w)) for w in calm_waves[:MAX_SPKS]]

    gate_h = _gate_html(gate_vals)
    stat_h = _status_html(base_time, calm_time, _state["temperature"], n_det)
    mix_fig = _waveform_fig(wav_8k, "#555555")
    base_fig = _waveform_fig(base_waves[0], "#818cf8")
    calm_fig = _waveform_fig(calm_waves[0], "#00D4B8")

    outputs = [gate_h, stat_h, mix_fig, base_fig, calm_fig]
    for i in range(MAX_SPKS):
        outputs.append(gr.update(visible=(i < n_det)))
        outputs.append(base_audio[i] if i < len(base_audio) else None)
    for i in range(MAX_SPKS):
        outputs.append(gr.update(visible=(i < n_det)))
        outputs.append(calm_audio[i] if i < len(calm_audio) else None)
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body, .gradio-container {
  background: #000 !important;
  color: #e8e8e8 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif !important;
  font-size: 16px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
footer { display: none !important; }

/* ── Layout container ── */
.cs-container {
  width: 100%;
  max-width: 920px;
  margin: 0 auto;
  padding: 0 2rem;
}

/* ── Sticky nav ── */
.cs-nav {
  position: sticky;
  top: 0;
  z-index: 100;
  padding: 0 2rem;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  background: rgba(0,0,0,0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}
.cs-nav-inner {
  max-width: 920px;
  margin: 0 auto;
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.cs-nav-logo {
  font-size: 0.95rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #fff;
}
.cs-nav-logo span { color: #00D4B8; }
.cs-nav-links {
  display: flex;
  align-items: center;
  gap: 2rem;
}
.cs-nav-links a {
  font-size: 0.82rem;
  color: #555;
  text-decoration: none;
  transition: color 0.2s;
}
.cs-nav-links a:hover { color: #aaa; }
.cs-nav-cta {
  font-size: 0.8rem;
  font-weight: 600;
  color: #000 !important;
  background: #00D4B8 !important;
  border-radius: 100px !important;
  padding: 0.4rem 1.1rem !important;
  border: none !important;
  cursor: pointer;
  text-decoration: none;
  font-family: inherit;
}

/* ── Gradio structural overrides ── */
.block, .form, .panel, .wrap, [class*="component-"], .gr-group {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
label span {
  font-size: 0.7rem !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  color: #3a3a3a !important;
  font-family: inherit !important;
  font-weight: 500 !important;
}
button.primary, button[variant="primary"] {
  background: #00D4B8 !important;
  color: #000 !important;
  border-radius: 100px !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.01em !important;
  border: none !important;
  padding: 0.65rem 1.75rem !important;
  font-family: inherit !important;
  transition: background 0.2s !important;
  width: 100% !important;
}
button.primary:hover { background: #00bba3 !important; }
button.secondary {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  color: #666 !important;
  border-radius: 100px !important;
  font-family: inherit !important;
}
input[type=range] { accent-color: #00D4B8; }
audio { border-radius: 8px; width: 100%; }
.waveform-container { background: #0d0d0d !important; border-radius: 8px; }
[data-testid="plot"] { background: transparent !important; }

/* ── Hero ── */
@keyframes glow-breathe {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}
@keyframes wv-drift {
  from { stroke-dashoffset: 0; }
  to   { stroke-dashoffset: -240; }
}
@keyframes stat-fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

.cs-hero {
  position: relative;
  min-height: 88vh;
  padding: 5rem 2rem 4rem;
  text-align: center;
  overflow: hidden;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.cs-hero::before {
  content: "";
  position: absolute;
  top: -15%;
  left: 50%;
  transform: translateX(-50%);
  width: 1100px;
  height: 600px;
  background: radial-gradient(ellipse at center,
    rgba(0,212,184,0.17) 0%,
    rgba(0,212,184,0.06) 38%,
    transparent 65%);
  pointer-events: none;
  animation: glow-breathe 5s ease-in-out infinite;
}
/* Decorative waveform SVG lines */
.cs-hero-wave {
  width: min(640px, 88%);
  margin: 2.75rem auto 0;
  opacity: 0.3;
}
.cs-hero-wave svg { width: 100%; overflow: visible; }
.wv {
  fill: none;
  stroke-width: 1.5;
  stroke-dasharray: 240;
  stroke-linecap: round;
}
.wv-0 { stroke: rgba(255,255,255,0.35); animation: wv-drift 9s linear infinite; }
.wv-1 { stroke: #7c83e0; animation: wv-drift 7s linear infinite reverse; }
.wv-2 { stroke: #00D4B8; animation: wv-drift 5s linear infinite; }

/* CTA row */
.cs-hero-cta {
  display: flex;
  gap: 0.85rem;
  justify-content: center;
  margin-top: 2.5rem;
  flex-wrap: wrap;
}
.cs-cta-primary {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #00D4B8;
  color: #000 !important;
  font-weight: 700;
  font-size: 0.88rem;
  padding: 0.78rem 1.9rem;
  border-radius: 100px;
  text-decoration: none;
  letter-spacing: 0.01em;
  transition: background 0.2s, transform 0.15s;
}
.cs-cta-primary:hover { background: #00bba3; transform: translateY(-1px); }
.cs-cta-ghost {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(255,255,255,0.1);
  color: #555 !important;
  font-size: 0.88rem;
  padding: 0.78rem 1.9rem;
  border-radius: 100px;
  text-decoration: none;
  transition: color 0.2s, border-color 0.2s, transform 0.15s;
}
.cs-cta-ghost:hover { color: #bbb !important; border-color: rgba(255,255,255,0.22); transform: translateY(-1px); }
.cs-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 14px;
  border-radius: 100px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.03);
  font-size: 0.75rem;
  color: #555;
  letter-spacing: 0.01em;
  margin-bottom: 2.75rem;
}
.cs-pill-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #00D4B8;
  flex-shrink: 0;
}
.cs-display {
  font-size: clamp(2.8rem, 6.5vw, 5.2rem);
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1.02;
  color: #fff;
}
.cs-display-muted {
  font-size: clamp(2.8rem, 6.5vw, 5.2rem);
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1.02;
  color: #2a2a2a;
  margin-bottom: 2rem;
}
.cs-hero-sub {
  font-size: 1rem;
  color: #4a4a4a;
  max-width: 480px;
  margin: 0 auto 0;
  line-height: 1.7;
}
.cs-hero-sub strong { color: #777; font-weight: 600; }

/* ── Stat strip ── */
.cs-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.cs-stat {
  padding: 2rem 1.75rem;
  border-right: 1px solid rgba(255,255,255,0.06);
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.cs-stat:last-child { border-right: none; }
.cs-stat-num {
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  color: #22c55e;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.cs-stat-num.teal { color: #00D4B8; }
.cs-stat-label { font-size: 0.72rem; color: #3a3a3a; letter-spacing: 0.02em; }
.cs-stat-sub { font-size: 0.65rem; color: #222; }

/* ── Section shared ── */
.cs-section {
  padding: 5rem 2rem;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.cs-section-inner { max-width: 920px; margin: 0 auto; }
.cs-eyebrow {
  font-size: 0.65rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #2a2a2a;
  margin-bottom: 0.75rem;
  display: block;
}
.cs-heading {
  font-size: clamp(1.6rem, 3vw, 2rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #fff;
  margin-bottom: 0.4rem;
  line-height: 1.2;
}
.cs-heading-sub {
  font-size: 0.85rem;
  color: #3a3a3a;
  margin-bottom: 2.5rem;
}

/* ── Benchmark cards ── */
.cs-bench-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: rgba(255,255,255,0.06);
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.06);
}
.cs-bench-card { background: #080808; }
.cs-bench-card-head {
  padding: 1.1rem 1.4rem 0.9rem;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
}
.cs-bench-ds {
  font-size: 0.92rem;
  font-weight: 700;
  color: #ccc;
  letter-spacing: -0.01em;
}
.cs-bench-ds em { color: #00D4B8; font-style: normal; }
.cs-bench-tag {
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #2a2a2a;
}
.cs-btable {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
  font-variant-numeric: tabular-nums;
}
.cs-btable th {
  padding: 0.55rem 1.4rem;
  text-align: right;
  font-size: 0.58rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #2a2a2a;
  font-weight: 500;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
.cs-btable th:first-child { text-align: left; }
.cs-btable td {
  padding: 0.7rem 1.4rem;
  text-align: right;
  color: #444;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}
.cs-btable td:first-child { text-align: left; color: #555; }
.cs-btable tr:last-child td { border-bottom: none; }
.td-base { color: #7c83e0 !important; }
.td-calm { color: #00D4B8 !important; }
.td-delta { color: #22c55e !important; font-weight: 700; }

/* ── Pipeline ── */
.cs-pipe-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 14px;
  overflow: hidden;
  margin-top: 2rem;
}
.cs-pipe-card {
  background: #080808;
  padding: 1.5rem 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.cs-pipe-n {
  font-size: 0.6rem;
  letter-spacing: 0.16em;
  color: #00D4B8;
  font-weight: 700;
  text-transform: uppercase;
}
.cs-pipe-name {
  font-size: 0.88rem;
  font-weight: 700;
  color: #ddd;
  letter-spacing: -0.01em;
  line-height: 1.3;
}
.cs-pipe-desc {
  font-size: 0.72rem;
  color: #333;
  line-height: 1.65;
}

/* ── Demo section ── */
.cs-demo-outer {
  padding: 5rem 2rem 2rem;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.cs-demo-inner { max-width: 920px; margin: 0 auto; }
.cs-demo-heading { margin-bottom: 2rem; }
.cs-demo-title {
  font-size: clamp(1.6rem, 3vw, 2rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #fff;
  margin-bottom: 0.35rem;
}
.cs-demo-sub { font-size: 0.82rem; color: #333; }

/* ── Upload styling ── */
.cs-upload-block [data-testid="audio"],
.cs-upload-block .block {
  background: #080808 !important;
  border: 1px dashed rgba(0,212,184,0.2) !important;
  border-radius: 14px !important;
  min-height: 150px !important;
}

/* ── Waveform row ── */
.cs-wave-row { padding: 1.5rem 0 0; }
.cs-wave-lbl {
  font-size: 0.6rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #2a2a2a;
  margin-bottom: 5px;
  display: block;
}

/* ── Status ── */
.cs-status { padding: 0.75rem 0 0; }

/* ── Results ── */
.cs-col { padding: 0.25rem 0; }
.cs-col-head {
  display: flex;
  align-items: center;
  gap: 9px;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  margin-bottom: 1rem;
}
.cs-col-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.cs-col-title {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.cs-spk-lbl {
  font-size: 0.58rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #2a2a2a;
  margin: 0.9rem 0 0.3rem;
  display: block;
}
.cs-gate-head {
  font-size: 0.58rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #2a2a2a;
  margin: 1rem 0 0.6rem;
}

/* ── Demo content (Gradio Group) ── */
#cs-demo-content {
  max-width: 920px !important;
  margin: 0 auto !important;
  padding: 0 2rem 5rem !important;
}
/* Button column — align to bottom of upload */
.cs-btn-col { display: flex !important; align-items: flex-end !important; padding-bottom: 4px !important; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Static HTML
# ─────────────────────────────────────────────────────────────────────────────

_NAV_HTML = """
<div class="cs-nav">
  <div class="cs-nav-inner">
    <span class="cs-nav-logo">CALM<span>·Sep</span></span>
    <nav class="cs-nav-links">
      <a href="#results">Results</a>
      <a href="#architecture">Architecture</a>
      <a href="#try" class="cs-nav-cta">Try it live</a>
    </nav>
  </div>
</div>
"""

_HERO_HTML = """
<div class="cs-hero" id="hero">
  <div class="cs-pill">
    <span class="cs-pill-dot"></span>
    Single-channel speech separation
  </div>

  <div class="cs-display">Unmix every voice,</div>
  <div class="cs-display-muted">hear each one clearly.</div>

  <p class="cs-hero-sub">
    CALM-Sep dynamically routes each mixture through condition-specific LoRA adapters —
    achieving <strong>+1.76 dB SI-SDR</strong> over the SR-CorrNet baseline
    on LibriMix benchmarks.
  </p>

  <div class="cs-hero-cta">
    <a href="#try" class="cs-cta-primary">Separate voices &rarr;</a>
    <a href="#results" class="cs-cta-ghost">View results</a>
  </div>

  <!-- Three waveform lines: mix (white) → baseline (indigo) → CALM-Sep (teal) -->
  <div class="cs-hero-wave" aria-hidden="true">
    <svg viewBox="0 0 800 64" xmlns="http://www.w3.org/2000/svg">
      <path class="wv wv-0" d="M0,32 Q25,12 50,32 Q75,52 100,32 Q125,12 150,32 Q175,52 200,32 Q225,12 250,32 Q275,52 300,32 Q325,12 350,32 Q375,52 400,32 Q425,12 450,32 Q475,52 500,32 Q525,12 550,32 Q575,52 600,32 Q625,12 650,32 Q675,52 700,32 Q725,12 750,32 Q775,52 800,32"/>
      <path class="wv wv-1" d="M0,32 Q25,4 50,32 Q75,60 100,32 Q125,4 150,32 Q175,60 200,32 Q225,4 250,32 Q275,60 300,32 Q325,4 350,32 Q375,60 400,32 Q425,4 450,32 Q475,60 500,32 Q525,4 550,32 Q575,60 600,32 Q625,4 650,32 Q675,60 700,32 Q725,4 750,32 Q775,60 800,32"/>
      <path class="wv wv-2" d="M0,32 Q20,20 40,32 Q60,44 80,32 Q100,20 120,32 Q140,44 160,32 Q180,20 200,32 Q220,44 240,32 Q260,20 280,32 Q300,44 320,32 Q340,20 360,32 Q380,44 400,32 Q420,20 440,32 Q460,44 480,32 Q500,20 520,32 Q540,44 560,32 Q580,20 600,32 Q620,44 640,32 Q660,20 680,32 Q700,44 720,32 Q740,20 760,32 Q780,44 800,32"/>
    </svg>
  </div>
</div>

<div class="cs-stats">
  <div class="cs-stat">
    <span class="cs-stat-num">+1.76</span>
    <span class="cs-stat-label">dB SI-SDR · Libri2Mix</span>
    <span class="cs-stat-sub">vs SR-CorrNet baseline</span>
  </div>
  <div class="cs-stat">
    <span class="cs-stat-num">+1.73</span>
    <span class="cs-stat-label">dB SI-SDRi · Libri3Mix</span>
    <span class="cs-stat-sub">vs SR-CorrNet baseline</span>
  </div>
  <div class="cs-stat">
    <span class="cs-stat-num teal">222</span>
    <span class="cs-stat-label">LoRA adapter tensors</span>
    <span class="cs-stat-sub">reverb · noise · codec</span>
  </div>
  <div class="cs-stat">
    <span class="cs-stat-num teal" style="font-size:1.85rem;letter-spacing:-0.03em;">T=4.987</span>
    <span class="cs-stat-label">calibrated gate temp.</span>
    <span class="cs-stat-sub">BCE 0.9747 → 0.5412</span>
  </div>
</div>
"""

_BENCH_HTML = """
<section id="results" class="cs-section">
  <div class="cs-section-inner">
    <span class="cs-eyebrow">Evaluation</span>
    <h2 class="cs-heading">Consistent gains across speaker counts.</h2>
    <p class="cs-heading-sub">LibriMix test set · 30 samples per split · PIT SI-SDR metric</p>

    <div class="cs-bench-grid">
      <div class="cs-bench-card">
        <div class="cs-bench-card-head">
          <span class="cs-bench-ds">Libri<em>2</em>Mix</span>
          <span class="cs-bench-tag">2-speaker mixtures</span>
        </div>
        <table class="cs-btable">
          <thead>
            <tr><th>Metric</th><th>SR-CorrNet</th><th>CALM-Sep</th><th>Δ</th></tr>
          </thead>
          <tbody>
            <tr><td>SI-SDR</td><td class="td-base">5.60 dB</td><td class="td-calm">7.36 dB</td><td class="td-delta">+1.76</td></tr>
            <tr><td>SI-SDRi</td><td class="td-base">7.09 dB</td><td class="td-calm">8.86 dB</td><td class="td-delta">+1.77</td></tr>
          </tbody>
        </table>
      </div>

      <div class="cs-bench-card">
        <div class="cs-bench-card-head">
          <span class="cs-bench-ds">Libri<em>3</em>Mix</span>
          <span class="cs-bench-tag">3-speaker mixtures</span>
        </div>
        <table class="cs-btable">
          <thead>
            <tr><th>Metric</th><th>SR-CorrNet</th><th>CALM-Sep</th><th>Δ</th></tr>
          </thead>
          <tbody>
            <tr><td>SI-SDR</td><td class="td-base">5.75 dB</td><td class="td-calm">7.49 dB</td><td class="td-delta">+1.74</td></tr>
            <tr><td>SI-SDRi</td><td class="td-base">10.07 dB</td><td class="td-calm">11.80 dB</td><td class="td-delta">+1.73</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</section>
"""

_PIPELINE_HTML = """
<section id="architecture" class="cs-section">
  <div class="cs-section-inner">
    <span class="cs-eyebrow">Architecture</span>
    <h2 class="cs-heading">Four stages. One coherent system.</h2>

    <div class="cs-pipe-grid">
      <div class="cs-pipe-card">
        <span class="cs-pipe-n">Stage 01</span>
        <span class="cs-pipe-name">Adapter Training</span>
        <span class="cs-pipe-desc">Three condition-specific LoRA adapters — reverb, noise, codec — trained independently on augmented LibriSpeech mixtures.</span>
      </div>
      <div class="cs-pipe-card">
        <span class="cs-pipe-n">Stage 02</span>
        <span class="cs-pipe-name">Gate Training</span>
        <span class="cs-pipe-desc">Level-1 DSP features (SNR, codec bandwidth, voiced density) + Level-2 learned E(0) heads drive a soft gating network.</span>
      </div>
      <div class="cs-pipe-card">
        <span class="cs-pipe-n">Stage 03</span>
        <span class="cs-pipe-name">Joint Fine-tune</span>
        <span class="cs-pipe-desc">End-to-end PIT SI-SDR optimization of gate + LoRA mixture. 14 epochs, temperature-calibrated (T=4.987).</span>
      </div>
      <div class="cs-pipe-card">
        <span class="cs-pipe-n">Stage 04</span>
        <span class="cs-pipe-name">Band Recovery</span>
        <span class="cs-pipe-desc">Oracle-trained BandRecoveryHead: 8→16 kHz in 58s of training. Best loss 0.012, bypasses SR-CorrNet bottleneck.</span>
      </div>
    </div>
  </div>
</section>
"""

_DEMO_HEAD_HTML = """
<div id="try" class="cs-demo-outer">
  <div class="cs-demo-inner">
    <h2 class="cs-demo-title">Try it live.</h2>
    <p class="cs-demo-sub">Upload any mixture · speaker count auto-detected · clips trimmed to 6 s</p>
  </div>
</div>
"""

_DEMO_CLOSE_HTML = ""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio theme
# ─────────────────────────────────────────────────────────────────────────────


def _make_theme():
    teal = gr.themes.Color(
        c50="#f0fdfb",
        c100="#ccfaf5",
        c200="#99f5ec",
        c300="#5ee9e3",
        c400="#2dd4c8",
        c500="#00D4B8",
        c600="#00a890",
        c700="#007d6b",
        c800="#005c4f",
        c900="#003d34",
        c950="#001f1a",
        name="teal",
    )
    slate = gr.themes.Color(
        c50="#f9fafb",
        c100="#f3f4f6",
        c200="#e5e7eb",
        c300="#d1d5db",
        c400="#9ca3af",
        c500="#6b7280",
        c600="#4b5563",
        c700="#374151",
        c800="#1f2937",
        c900="#111827",
        c950="#030712",
        name="slate",
    )
    return gr.themes.Base(primary_hue=teal, neutral_hue=slate).set(
        body_background_fill="#000",
        body_background_fill_dark="#000",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        block_border_color="transparent",
        block_border_color_dark="transparent",
        block_label_background_fill="transparent",
        block_label_text_color="#444",
        block_label_text_color_dark="#444",
        input_background_fill="#111",
        input_background_fill_dark="#111",
        input_border_color="rgba(255,255,255,0.08)",
        input_border_color_dark="rgba(255,255,255,0.08)",
        slider_color="#00D4B8",
        button_primary_background_fill="#00D4B8",
        button_primary_background_fill_dark="#00D4B8",
        button_primary_background_fill_hover="#00a890",
        button_primary_text_color="#000",
        button_primary_text_color_dark="#000",
        button_secondary_background_fill="rgba(255,255,255,0.05)",
        button_secondary_background_fill_dark="rgba(255,255,255,0.05)",
        button_secondary_text_color="#888",
        body_text_color="#fff",
        body_text_color_dark="#fff",
        body_text_color_subdued="#555",
        background_fill_primary="#000",
        background_fill_primary_dark="#000",
        background_fill_secondary="#0a0a0a",
        background_fill_secondary_dark="#0a0a0a",
        border_color_primary="rgba(255,255,255,0.07)",
        border_color_primary_dark="rgba(255,255,255,0.07)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────


def build_ui():
    with gr.Blocks(css=CSS, title="CALM-Sep", theme=_make_theme()) as demo:

        # ── Nav + landing ─────────────────────────────────────────────────
        gr.HTML(_NAV_HTML)
        gr.HTML(_HERO_HTML)
        gr.HTML(_BENCH_HTML)
        gr.HTML(_PIPELINE_HTML)
        gr.HTML(_DEMO_HEAD_HTML)

        # ── Interactive demo — constrained to 920px ────────────────────────
        with gr.Group(elem_id="cs-demo-content"):

            with gr.Row():
                with gr.Column(scale=4, elem_classes=["cs-upload-block"]):
                    audio_in = gr.Audio(
                        label="Input mixture — any format, any sample rate",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )
                with gr.Column(scale=1, min_width=140, elem_classes=["cs-btn-col"]):
                    run_btn = gr.Button("Separate", variant="primary", size="lg")

            with gr.Row():
                with gr.Column():
                    gr.HTML('<span class="cs-wave-lbl">Mixture</span>')
                    mix_wave = gr.Plot(show_label=False)
                with gr.Column():
                    gr.HTML(
                        '<span class="cs-wave-lbl" style="color:#7c83e0">Baseline — Spk 1</span>'
                    )
                    base_wave = gr.Plot(show_label=False)
                with gr.Column():
                    gr.HTML(
                        '<span class="cs-wave-lbl" style="color:#00D4B8">CALM-Sep — Spk 1</span>'
                    )
                    calm_wave = gr.Plot(show_label=False)

            status_out = gr.HTML("", elem_classes=["cs-status"])

            with gr.Row(equal_height=False):

                with gr.Column():
                    gr.HTML(
                        '<div class="cs-col-head"><div class="cs-col-dot" style="background:#7c83e0"></div><span class="cs-col-title" style="color:#7c83e0">Baseline · SR-CorrNet</span></div>'
                    )
                    base_groups, base_players = [], []
                    for i in range(MAX_SPKS):
                        with gr.Group(visible=(i < 2)) as grp:
                            gr.HTML(f'<span class="cs-spk-lbl">Speaker {i+1}</span>')
                            base_players.append(gr.Audio(label="", type="numpy", interactive=False))
                        base_groups.append(grp)

                with gr.Column():
                    gr.HTML(
                        '<div class="cs-col-head"><div class="cs-col-dot" style="background:#00D4B8"></div><span class="cs-col-title" style="color:#00D4B8">CALM-Sep · Gate + LoRA</span></div>'
                    )
                    gate_html_out = gr.HTML("")
                    gr.HTML('<div class="cs-gate-head">Adapter gate activations</div>')
                    calm_groups, calm_players = [], []
                    for i in range(MAX_SPKS):
                        with gr.Group(visible=(i < 2)) as grp:
                            gr.HTML(f'<span class="cs-spk-lbl">Speaker {i+1}</span>')
                            calm_players.append(gr.Audio(label="", type="numpy", interactive=False))
                        calm_groups.append(grp)

        outputs = [gate_html_out, status_out, mix_wave, base_wave, calm_wave]
        for i in range(MAX_SPKS):
            outputs += [base_groups[i], base_players[i]]
        for i in range(MAX_SPKS):
            outputs += [calm_groups[i], calm_players[i]]

        run_btn.click(fn=process, inputs=[audio_in], outputs=outputs)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print("Pre-loading models …")
    try:
        _ensure_loaded()
        print(f"Ready. T={_state.get('temperature', 1.0):.4f}")
    except Exception as e:
        print(f"Warning: pre-load failed ({e}). Will retry on first request.")
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
