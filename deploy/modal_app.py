"""
CoRAL-Sep · Modal deployment
────────────────────────────
Deploy:
    modal deploy modal_deploy.py

The @modal.cls pattern ensures models are loaded once per container
in @modal.enter() before any request arrives.
"""
from __future__ import annotations

from pathlib import Path

import modal

_HERE = Path(__file__).parent
_SR_SRC = (
    Path.home() / "Downloads/SR_CorrNet_local_mixboth/future_work/sr_corrnet"
)

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "numpy>=1.24,<2",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "scipy>=1.11",
        "pyyaml>=6.0",
        "soundfile>=0.12",
        "tqdm>=4.66",
        "huggingface_hub>=0.20",
        "asteroid==0.7.0",
        "speechbrain==1.0.0",
        "matplotlib>=3.7",
        "gradio==6.20.0",
        "openai-whisper>=20231117",
        "ffmpeg-python",
        "librosa",
        "loguru",
        "rotary-embedding-torch",
    )
    .run_commands(
        "python3 -c \""
        "from huggingface_hub import snapshot_download;"
        "snapshot_download('shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk')"
        "\"",
        "python3 -c \"import whisper; whisper.load_model('base')\"",
    )
    .add_local_dir(str(_HERE / "models"), "/app/models")
    .add_local_dir(str(_HERE / "train"), "/app/train")
    .add_local_dir(str(_HERE / "align"), "/app/align")
    .add_local_dir(str(_HERE / "calibration"), "/app/calibration")
    .add_local_dir(
        str(_HERE / "checkpoints" / "stage4_joint"),
        "/app/checkpoints/stage4_joint",
    )
    .add_local_dir(
        str(_HERE / "checkpoints" / "stage4c"),
        "/app/checkpoints/stage4c",
    )
    .add_local_dir(str(_SR_SRC), "/app/sr_corrnet")
    .add_local_file(str(_HERE / "demo.py"), "/app/demo.py")
)

app = modal.App("coral-sep", image=_image)


@app.cls(
    cpu=4.0,
    memory=12288,
    timeout=600,
    min_containers=1,
    max_containers=3,
)
class CoralSep:
    @modal.enter()
    def load(self):
        """Runs once when the container boots — before any request."""
        import os
        import sys

        os.chdir("/app")
        sys.path.insert(0, "/app")

        from coralsep.demo import _ensure_loaded

        _ensure_loaded()
        print("[coral-sep] Models loaded — container is warm.")

    @modal.asgi_app()
    def serve(self):
        import gradio as gr
        from fastapi import FastAPI

        from coralsep.demo import build_ui

        api = FastAPI()
        ui = build_ui()
        return gr.mount_gradio_app(api, ui.queue(), path="/")
