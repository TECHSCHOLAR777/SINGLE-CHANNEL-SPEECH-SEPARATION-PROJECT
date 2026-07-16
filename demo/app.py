"""
Gradio demo for CALM-Sep (P4-C4).

Run:
    python -m demo.app --mock
    python -m demo.app --checkpoint path/to/model.pt
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Protocol

import numpy as np

from schemas.separation_result import SeparationResult, StreamMetadata

MAX_DISPLAY_STREAMS = 5


class Engine(Protocol):
    def __call__(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult: ...


class MockEngine:
    """Weight-free placeholder: band splits labeled as mock."""

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
            expert_used="mock",
            completeness=0.5,
            ood_flag=False,
            gate_vector={"reverb": 0.0, "noise": 0.0, "codec": 0.0},
            condition_estimates={"snr_db": 40.0, "t60_s": 0.0},
        )


def _to_mono_float(audio: tuple[int, np.ndarray]) -> tuple[np.ndarray, int]:
    sr, data = audio
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    return arr.astype(np.float32), int(sr)


def _optional_whisper(wav: np.ndarray, sr: int) -> str | None:
    try:
        import whisper  # type: ignore

        model = whisper.load_model("tiny")
        # whisper expects 16 kHz
        if sr != 16000:
            from models.preprocess import resample_audio

            wav = resample_audio(wav, sr, 16000)
        result = model.transcribe(wav)
        return str(result.get("text", "")).strip() or None
    except Exception:
        return None


def run_separation(
    audio: tuple[int, np.ndarray] | None,
    engine: Engine,
    with_whisper: bool = False,
) -> tuple:
    if audio is None:
        return ("Upload a mixture first.", *([None] * MAX_DISPLAY_STREAMS), "{}")

    mixture, sr = _to_mono_float(audio)
    result = engine(mixture, sr)

    mean_conf = float(np.mean([m.confidence for m in result.metadata]))
    badge = (
        f"### Speakers: {result.speaker_count} ({mean_conf:.0%} conf)\n"
        f"Engine: `{result.expert_used}` | "
        f"Completeness: {result.completeness if result.completeness is not None else 'n/a'} | "
        f"OOD: {'yes' if result.ood_flag else 'no'}"
    )
    if result.expert_used == "mock":
        badge = "**MOCK ENGINE**\n" + badge

    slots: list[tuple[int, np.ndarray] | None] = [None] * MAX_DISPLAY_STREAMS
    for i in range(min(result.num_streams, MAX_DISPLAY_STREAMS)):
        slots[i] = (result.sample_rate, result.streams[i])

    transcript = None
    if with_whisper and result.num_streams:
        transcript = _optional_whisper(result.streams[0], result.sample_rate)

    diagnostics = result.to_report_dict()
    if transcript is not None:
        diagnostics["whisper_transcript_spk1"] = transcript
    return (badge, *slots, json.dumps(diagnostics, indent=2))


def build_demo(engine: Engine, with_whisper: bool = False):
    import gradio as gr

    with gr.Blocks(title="CALM-Sep") as demo:
        gr.Markdown("# CALM-Sep")
        gr.Markdown(
            "Condition-Aware LoRA Mixture for multi-speaker separation. "
            "Shows count, gates, condition estimates, and completeness."
        )
        audio_in = gr.Audio(label="Mixture", type="numpy")
        run_btn = gr.Button("Separate", variant="primary")
        badge = gr.Markdown()
        outputs = [
            gr.Audio(label=f"Speaker {i + 1}", type="numpy") for i in range(MAX_DISPLAY_STREAMS)
        ]
        with gr.Accordion("Diagnostics (gates / conditions / p_k)", open=True):
            diag = gr.Code(language="json", label="Run diagnostics")

        run_btn.click(
            fn=lambda a: run_separation(a, engine, with_whisper=with_whisper),
            inputs=[audio_in],
            outputs=[badge, *outputs, diag],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Weight-free mock engine")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/base_checkpoint.yaml")
    parser.add_argument("--whisper", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    engine: Engine | Callable
    if args.mock or not args.checkpoint:
        if args.mock:
            engine = MockEngine()
        else:
            from pipeline.infer import CalmSepEngine, MockCalmSepWrapper

            engine = CalmSepEngine(device=args.device, wrapper=MockCalmSepWrapper(), base_only=True)
    else:
        from pipeline.infer import CalmSepEngine

        engine = CalmSepEngine(device=args.device, checkpoint_path=args.checkpoint)

    build_demo(engine, with_whisper=args.whisper).launch(server_port=args.port)


if __name__ == "__main__":
    main()
