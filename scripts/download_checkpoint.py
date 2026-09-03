"""Download and verify the frozen SR-CorrNet var-2-5 checkpoint (task P0-B1).

Run once, before any other Phase P0 work:
    python scripts/download_checkpoint.py

What it does:
  1. Calls SSInference.from_pretrained to trigger HF Hub download.
  2. Locates the cached checkpoint file.
  3. Computes SHA-256.
  4. Verifies architecture constants against BLUEPRINT §3.5.
  5. Writes the SHA-256 and local path into configs/base_checkpoint.yaml.

Prerequisites:
    git clone https://github.com/dmlguq456/SR_CorrNet_SS.git
    cd SR_CorrNet_SS && pip install -e ".[hub]"
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "configs" / "base_checkpoint.yaml"
HF_MODEL = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"

EXPECTED = {
    "sample_rate": 8000,
    "max_n_spks": 5,
    "n_enc": 2,
    "n_dec": 4,
    "d_model": 128,
    "stft_window": 128,
    "stft_hop": 64,
    "freq_bins": 65,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_checkpoint_file(model) -> Path | None:
    """Walk common HF Hub cache locations for the checkpoint file."""
    try:
        # SSInference stores the local path after from_pretrained
        local = getattr(model.engine, "checkpoint_path", None)
        if local and Path(local).exists():
            return Path(local)
    except Exception:
        pass

    # Fallback: search HF Hub cache
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=HF_MODEL, filename="model.pt")
        return Path(path)
    except Exception:
        pass

    # Last resort: search known local path from BLUEPRINT §3.5
    local_path = ROOT / "sr_corrnet" / "checkpoints" / "SS" / "1ch_WSJ_var_2_5spk" / "model.pt"
    if local_path.exists():
        return local_path

    return None


def verify_constants(model) -> None:
    """Assert architecture constants from BLUEPRINT §3.5."""
    base_nn = model.engine.model

    cfg = model.engine.config if hasattr(model.engine, "config") else {}

    sr = getattr(cfg, "sampling_rate", None) or getattr(cfg, "sample_rate", None)
    if sr is not None:
        assert int(sr) == EXPECTED["sample_rate"], f"sample_rate={sr}, expected 8000"

    # Check spk_query shape: (1, 7, 128) = (1, max_n_spks+2, d_model)
    spk_query = base_nn.spk_split.spk_query
    assert spk_query.shape == (
        1,
        7,
        128,
    ), f"spk_query shape={spk_query.shape}, expected (1, 7, 128)"

    # Check enc_block and dec_block lengths
    assert (
        len(base_nn.enc_block) == EXPECTED["n_enc"]
    ), f"n_enc={len(base_nn.enc_block)}, expected {EXPECTED['n_enc']}"
    assert (
        len(base_nn.dec_block) == EXPECTED["n_dec"]
    ), f"n_dec={len(base_nn.dec_block)}, expected {EXPECTED['n_dec']}"

    print("Architecture constants verified.")


def main() -> None:
    try:
        from sr_corrnet import SSInference
    except ImportError:
        print("ERROR: sr_corrnet not installed.")
        print("  git clone https://github.com/dmlguq456/SR_CorrNet_SS.git")
        print('  cd SR_CorrNet_SS && pip install -e ".[hub]"')
        sys.exit(1)

    print(f"Loading {HF_MODEL} ...")
    model = SSInference.from_pretrained(checkpoint_path=HF_MODEL, device="cpu")
    print("Loaded.")

    verify_constants(model)

    ckpt_path = find_checkpoint_file(model)
    if ckpt_path is None:
        print("WARNING: Could not locate checkpoint file on disk. SHA-256 not computed.")
        sha = ""
    else:
        print(f"Checkpoint: {ckpt_path}")
        sha = sha256_file(ckpt_path)
        print(f"SHA-256: {sha}")

    with open(YAML_PATH) as f:
        config = yaml.safe_load(f)

    config["sha256"] = sha
    if ckpt_path is not None:
        config["local_path"] = str(ckpt_path.relative_to(ROOT))

    with open(YAML_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated {YAML_PATH}")


if __name__ == "__main__":
    main()
