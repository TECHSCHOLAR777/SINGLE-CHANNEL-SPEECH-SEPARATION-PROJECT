"""
Gradio demo interface — CoRAL-Sep edition (Dev C, P4-C4).

Upload a mixture, get per-speaker streams, estimated speaker count with
calibrated confidence, adapter gate routing visualization, Whisper transcript,
completeness / OOD quality flags, and a full diagnostics panel.

The separation engine is injected as a callable for staged rollout:
  --mock        weight-free bandpass placeholder
  --coralsep     full CoRAL-Sep pipeline (requires GPU + weights)
  --config X    legacy baseline config (SepFormer)

Run:
    python -m coralsep.demo.app --mock
    python -m coralsep.demo.app --coralsep --device cuda
    python -m coralsep.demo.app --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Protocol

import numpy as np

from coralsep.schemas.separation_result import SeparationResult, StreamMetadata

MAX_DISPLAY_STREAMS = 5
"""UI slots rendered; engines may return fewer."""

_ADAPTER_NAMES = ("reverb", "noise", "codec")


class Engine(Protocol):
    def __call__(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult: ...


class MockEngine:
    """
    Weight-free placeholder for UI development.

    Bandpass-splits the mixture so each output slot is audibly different.
    Labeled mock so nobody mistakes it for real separation.
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


class CoralSepEngine:
    """Full CoRAL-Sep inference engine wrapper for the demo."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from coralsep.models.experts.srcorrnet import SRCorrNetExpert
        from coralsep.pipeline.infer import CoralSepPipeline, InferenceCfg

        expert = SRCorrNetExpert(device=self._device)
        cfg = InferenceCfg(device=self._device)
        self._pipeline = CoralSepPipeline(expert=expert, cfg=cfg)

    def __call__(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        self._load()
        assert self._pipeline is not None
        result = self._pipeline.run(mixture, sample_rate)

        # Wrap PipelineResult → SeparationResult.
        metadata = [
            StreamMetadata(
                expert_source="coralsep",
                confidence=float(result.completeness_prob),
                extra={"gate": result.gate_vector},
            )
            for _ in range(result.speaker_count)
        ]
        sr_out = result.streams_16k.shape[1]
        return SeparationResult(
            streams=result.streams_16k,
            sample_rate=16000,
            speaker_count=result.speaker_count,
            metadata=metadata,
            expert_used="coralsep",
            gate_vector=result.gate_vector,
            completeness_prob=result.completeness_prob,
            ood_flag=result.ood_flag,
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


def _gate_routing_markdown(gate_vec: dict[str, float]) -> str:
    """Render adapter gate values as a compact Markdown bar chart."""
    if not gate_vec:
        return ""
    lines = ["**Adapter gate routing**"]
    for name in _ADAPTER_NAMES:
        g = gate_vec.get(name, 0.0)
        bar_filled = int(round(g * 20))
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        lines.append(f"`{name:6s}` [{bar}] {g:.2f}")
    return "\n".join(lines)


def _whisper_transcript(streams: np.ndarray, sample_rate: int) -> str:
    """
    Attempt a Whisper transcript of the first separated stream.

    Requires `pip install openai-whisper`. Returns empty string if unavailable.
    """
    try:
        import whisper
        import tempfile, os
        import soundfile as sf

        model = whisper.load_model("tiny")  # fast, low VRAM
        wav = streams[0]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, wav, sample_rate)
            tmp_path = tmp.name
        res = model.transcribe(tmp_path, language=None)
        os.unlink(tmp_path)
        text = str(res.get("text", "")).strip()
        return f"**Whisper (stream 1):** {text}" if text else ""
    except Exception:
        return ""


def run_separation(
    audio: tuple[int, np.ndarray] | None,
    engine: Engine,
) -> tuple:
    """
    Demo callback: mixture in, UI payload out.

    Returns:
        (status_md, *MAX_DISPLAY_STREAMS audio slots, gate_md,
         transcript_md, diagnostics_json)
    """
    if audio is None:
        empty = [None] * MAX_DISPLAY_STREAMS
        return ("Upload a mixture first.", *empty, "", "", "{}")

    mixture, sr = _to_mono_float(audio)
    result = engine(mixture, sr)

    # Header badge.
    mean_conf = float(np.mean([m.confidence for m in result.metadata]))
    flags = []
    if result.ood_flag:
        flags.append("OOD")
    if result.completeness_prob < 0.6:
        flags.append(f"completeness {result.completeness_prob:.0%}")
    flag_str = "  ⚠ " + ", ".join(flags) if flags else ""
    badge = (
        f"### Estimated speakers: {result.speaker_count} "
        f"({mean_conf:.0%} confident){flag_str}\n"
        f"Engine: `{result.expert_used}`"
    )
    if result.expert_used == "mock":
        badge = "**MOCK ENGINE — frequency bands, not speech separation.**\n" + badge

    # Per-speaker audio slots.
    slots: list[tuple[int, np.ndarray] | None] = [None] * MAX_DISPLAY_STREAMS
    for i in range(min(result.num_streams, MAX_DISPLAY_STREAMS)):
        slots[i] = (result.sample_rate, result.streams[i])

    # Gate routing visualization.
    gate_md = _gate_routing_markdown(result.gate_vector)

    # Whisper transcript (best-effort).
    transcript_md = _whisper_transcript(result.streams, result.sample_rate)

    # Full diagnostics JSON.
    diagnostics = {
        "speaker_count": result.speaker_count,
        "expert_used": result.expert_used,
        "escalated": result.escalated,
        "completeness_prob": round(result.completeness_prob, 3),
        "ood_flag": result.ood_flag,
        "gate_vector": result.gate_vector,
        "duration_sec": round(result.duration_sec, 2),
        "sample_rate": result.sample_rate,
        "per_stream": [
            {"confidence": round(m.confidence, 3), "source": m.expert_source, **m.extra}
            for m in result.metadata
        ],
    }
    return (badge, *slots, gate_md, transcript_md, json.dumps(diagnostics, indent=2))


def build_demo(engine: Engine):
    """Construct the Gradio Blocks app around the injected engine."""
    import gradio as gr

    with gr.Blocks(title="CoRAL-Sep Speech Separation") as demo:
        gr.Markdown("# CoRAL-Sep: Condition-Adaptive Multi-Speaker Speech Separation")
        gr.Markdown(
            "Upload overlapping speech (2–5 speakers). "
            "Adapter gate routing is computed from acoustic condition analysis."
        )

        with gr.Row():
            audio_in = gr.Audio(label="Mixture", type="numpy")
            run_btn = gr.Button("Separate", variant="primary", scale=0)

        badge = gr.Markdown()

        with gr.Row():
            outputs = [
                gr.Audio(label=f"Speaker {i + 1}", type="numpy")
                for i in range(MAX_DISPLAY_STREAMS)
            ]

        with gr.Row():
            gate_panel = gr.Markdown(label="Gate routing")
            transcript_panel = gr.Markdown(label="Transcript")

        with gr.Accordion("Diagnostics", open=False):
            diag = gr.Code(language="json", label="Run diagnostics")

        all_outputs = [badge, *outputs, gate_panel, transcript_panel, diag]

        run_btn.click(
            fn=lambda a: run_separation(a, engine),
            inputs=[audio_in],
            outputs=all_outputs,
        )
        audio_in.change(
            fn=lambda a: run_separation(a, engine),
            inputs=[audio_in],
            outputs=all_outputs,
        )

    return demo


def _engine_from_config(config_path: str) -> Engine:
    """Build a legacy baseline engine from a config file."""
    from coralsep.models.experts.sepformer import SepFormerExpert
    from coralsep.utils.config import load_config

    cfg = load_config(config_path)
    expert = SepFormerExpert(device=cfg.get("device", "cpu"))

    def engine(mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        return expert.separate(mixture, sample_rate)

    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Weight-free mock engine")
    parser.add_argument("--coralsep", action="store_true", help="Full CoRAL-Sep pipeline")
    parser.add_argument("--config", type=str, help="Legacy baseline config (SepFormer)")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for CoRAL-Sep")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    engine: Engine | Callable
    if args.coralsep:
        engine = CoralSepEngine(device=args.device)
    elif args.config:
        engine = _engine_from_config(args.config)
    else:
        engine = MockEngine()

    build_demo(engine).launch(server_port=args.port)


if __name__ == "__main__":
    main()
