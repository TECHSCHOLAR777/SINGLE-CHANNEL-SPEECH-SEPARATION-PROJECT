"""
Gradio demo interface (Dev C, Phase 6, mock-ready from Phase 0).

Upload a mixture, get per-speaker streams, the estimated count with its
calibrated confidence, an escalation indicator, and a diagnostics panel. The
separation engine is injected as a callable so the demo works at every
project stage: MockEngine today (bandpass splits, loudly labeled as mock),
single experts at M1, the full cascade at M2+, all without touching this file.

Run:
    python -m demo.app --mock
    python -m demo.app --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Protocol

import numpy as np

from schemas.separation_result import SeparationResult, StreamMetadata

MAX_DISPLAY_STREAMS = 5
"""UI slots rendered; engines may return fewer."""


class Engine(Protocol):
    def __call__(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult: ...


class MockEngine:
    """
    Weight-free placeholder engine for UI development.

    Splits the mixture into low/mid/high frequency bands so each output slot
    is audibly different without any model. Every stream is labeled mock and
    confidence is fixed low so nobody mistakes it for separation.
    """

    BANDS_HZ: tuple[tuple[float, float], ...] = ((0, 400), (400, 1500), (1500, 8000))

    def __call__(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        mix = np.asarray(mixture, dtype=np.float32)
        spectrum = np.fft.rfft(mix)
        freqs = np.fft.rfftfreq(mix.shape[0], d=1.0 / sample_rate)
        streams = []
        for lo, hi in self.BANDS_HZ:
            band = spectrum.copy()
            band[(freqs < lo) | (freqs >= hi)] = 0.0
            streams.append(np.fft.irfft(band, n=mix.shape[0]).astype(np.float32))
        arr = np.stack(streams, axis=0)
        metadata = [
            StreamMetadata(expert_source="mock", confidence=0.33, extra={"band_hz": list(b)})
            for b in self.BANDS_HZ
        ]
        return SeparationResult(
            streams=arr,
            sample_rate=sample_rate,
            speaker_count=arr.shape[0],
            metadata=metadata,
            mixture=mix,
            escalated=False,
            expert_used="mock",
        )


def _to_mono_float(audio: tuple[int, np.ndarray]) -> tuple[np.ndarray, int]:
    """Normalize a Gradio audio tuple to mono float32 in [-1, 1]."""
    sr, data = audio
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    return arr.astype(np.float32), int(sr)


def run_separation(
    audio: tuple[int, np.ndarray] | None,
    engine: Engine,
) -> tuple:
    """
    Demo callback: mixture in, UI payload out.

    Returns a status markdown, MAX_DISPLAY_STREAMS audio slots (unused slots
    None), and a diagnostics JSON string.
    """
    if audio is None:
        return ("Upload a mixture first.", *([None] * MAX_DISPLAY_STREAMS), "{}")

    mixture, sr = _to_mono_float(audio)
    result = engine(mixture, sr)

    mean_conf = float(np.mean([m.confidence for m in result.metadata]))
    badge = (
        f"### Estimated speakers: {result.speaker_count} "
        f"({mean_conf:.0%} confident)\n"
        f"Engine: `{result.expert_used}` | "
        f"Escalated: {'yes' if result.escalated else 'no'}"
    )
    if result.expert_used == "mock":
        badge = "**MOCK ENGINE: frequency bands, not separation.**\n" + badge

    slots: list[tuple[int, np.ndarray] | None] = [None] * MAX_DISPLAY_STREAMS
    for i in range(min(result.num_streams, MAX_DISPLAY_STREAMS)):
        slots[i] = (result.sample_rate, result.streams[i])

    diagnostics = {
        "speaker_count": result.speaker_count,
        "expert_used": result.expert_used,
        "escalated": result.escalated,
        "duration_sec": round(result.duration_sec, 2),
        "sample_rate": result.sample_rate,
        "per_stream": [
            {"confidence": m.confidence, "source": m.expert_source, **m.extra}
            for m in result.metadata
        ],
    }
    return (badge, *slots, json.dumps(diagnostics, indent=2))


def build_demo(engine: Engine):
    """Construct the Gradio Blocks app around an injected engine."""
    import gradio as gr

    with gr.Blocks(title="CA-MoSE Speech Separation") as demo:
        gr.Markdown("# CA-MoSE: Multi-Speaker Speech Separation")
        gr.Markdown("Upload overlapping speech; receive one track per detected speaker.")
        audio_in = gr.Audio(label="Mixture", type="numpy")
        run_btn = gr.Button("Separate", variant="primary")
        badge = gr.Markdown()
        outputs = [
            gr.Audio(label=f"Speaker {i + 1}", type="numpy") for i in range(MAX_DISPLAY_STREAMS)
        ]
        with gr.Accordion("Diagnostics", open=False):
            diag = gr.Code(language="json", label="Run diagnostics")

        run_btn.click(
            fn=lambda a: run_separation(a, engine),
            inputs=[audio_in],
            outputs=[badge, *outputs, diag],
        )
    return demo


def _engine_from_config(config_path: str) -> Engine:
    """Build the best available real engine from a baseline config."""
    from models.experts.sepformer import SepFormerExpert
    from utils.config import load_config

    cfg = load_config(config_path)
    expert = SepFormerExpert(device=cfg.get("device", "cpu"))

    def engine(mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        return expert.separate(mixture, sample_rate)

    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Run with the weight-free mock engine")
    parser.add_argument("--config", type=str, help="Baseline config for a real engine")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    engine: Engine | Callable
    if args.mock or not args.config:
        engine = MockEngine()
    else:
        engine = _engine_from_config(args.config)

    build_demo(engine).launch(server_port=args.port)


if __name__ == "__main__":
    main()
