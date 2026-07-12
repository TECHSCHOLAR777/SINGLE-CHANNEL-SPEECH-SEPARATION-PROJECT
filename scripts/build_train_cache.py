"""
Build the frozen-expert training cache (the P2 training unlock).

Runs the frozen experts ONCE over a fixed set of mixtures and writes their
outputs to sharded ``.pt`` files that ``train/cached_dataset.py`` feeds straight
into ``CAMoSETrainer`` — so training never loads an expert and every epoch is
cheap.

Per mixture it computes and caches:
  * MossFormer2 streams, padded to ``--target-speakers`` (residual/zero pad)
  * expensive-expert streams, Hungarian-aligned onto the MossFormer2 order
    (CRRRFusionHead assumes matching speaker order and cannot check it)
  * REAL-M blind min-SI-SNR on the cheap output  -> cascade-gate signal
  * ECAPA embeddings for streams and references   -> speaker-consistency loss
  * a blind mask-entropy proxy and a silence ``trivial_mask``

Mixture sources (pick one):
  --librimix-root DIR      load a generated LibriMix layout from disk
  --dynamic-source-glob G  mix on the fly from LibriSpeech clean files

Example (Kaggle smoke)::

    python -m scripts.build_train_cache \
        --librimix-root /kaggle/working/Libri3Mix \
        --subset train-100 --mix-dir mix_clean --mode min \
        --limit 500 --target-speakers 3 --device cuda \
        --srcorrnet-repo /kaggle/working/SR_CorrNet \
        --srcorrnet-checkpoint /kaggle/input/srcorrnet/best.pt \
        --out-dir /kaggle/working/cache/train
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment

from align.hungarian import xcorr_cost_matrix
from data.mixer_stub import MixtureSample
from train.cached_dataset import make_sample_dict, save_cache_shard

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Mixture sources
# --------------------------------------------------------------------------- #
def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    return audio.astype(np.float32), sr


def load_librimix_samples(
    root: str | Path,
    subset: str,
    mix_dir: str = "mix_clean",
    mode: str = "min",
    sample_rate_dir: str = "wav16k",
    limit: int | None = None,
) -> list[MixtureSample]:
    """
    Load a generated LibriMix layout, e.g.
    ``{root}/{sample_rate_dir}/{mode}/{subset}/{mix_dir|s1..sN}/<uid>.wav``.

    Supports ``mix_clean`` (the clean smoke tier) as well as ``mix_both``,
    and auto-detects speaker count from which ``sN/`` dirs exist.
    """
    subset_dir = Path(root) / sample_rate_dir / mode / subset
    mixd = subset_dir / mix_dir
    if not mixd.is_dir():
        raise FileNotFoundError(
            f"Mixture dir not found: {mixd}\n"
            "Check --librimix-root/--subset/--mix-dir/--mode against the generated layout."
        )
    n_spk = sum(1 for i in range(1, 6) if (subset_dir / f"s{i}").is_dir())
    if n_spk < 1:
        raise FileNotFoundError(f"No s1/../s5/ stem dirs under {subset_dir}")

    mix_files = sorted(mixd.glob("*.wav"))
    if limit is not None:
        mix_files = mix_files[:limit]

    samples: list[MixtureSample] = []
    for mix_path in mix_files:
        uid = mix_path.stem
        refs: list[np.ndarray] = []
        sr: int | None = None
        for k in range(1, n_spk + 1):
            rp = subset_dir / f"s{k}" / f"{uid}.wav"
            if not rp.exists():
                break
            ref, rsr = _load_wav(rp)
            sr = rsr if sr is None else sr
            refs.append(ref)
        if not refs or sr is None:
            continue
        mixture, msr = _load_wav(mix_path)
        if msr != sr:
            continue
        min_len = min(len(mixture), *(len(r) for r in refs))
        samples.append(
            MixtureSample(
                mixture=mixture[:min_len],
                references=np.stack([r[:min_len] for r in refs], axis=0),
                sample_rate=sr,
                utterance_id=uid,
            )
        )
    return samples


def load_dynamic_samples(
    source_glob: str,
    n_samples: int,
    allowed_n: list[int],
    seed: int = 0,
) -> list[MixtureSample]:
    """Mix on the fly from LibriSpeech clean files via DynamicMixer."""
    import glob as _glob

    from data.dynamic_mix_dataset import DynamicMixDataset

    files = sorted(_glob.glob(source_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No source files matched glob: {source_glob}")
    ds = DynamicMixDataset(
        source_files=files,
        n_samples=n_samples,
        allowed_n=allowed_n,
        seed=seed,
    )
    return [ds[i] for i in range(len(ds))]


# --------------------------------------------------------------------------- #
# Per-sample processing
# --------------------------------------------------------------------------- #
def _crop_or_pad(x: np.ndarray, length: int) -> np.ndarray:
    """Crop (from the front) or zero-pad the last axis to exactly ``length``."""
    t = x.shape[-1]
    if t == length:
        return x
    if t > length:
        return x[..., :length]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, length - t)]
    return np.pad(x, pad, mode="constant")


def _fit_k(streams: np.ndarray, k: int) -> np.ndarray:
    """Pad with zero streams or truncate so the first axis is exactly ``k``."""
    kc = streams.shape[0]
    if kc == k:
        return streams
    if kc > k:
        return streams[:k]
    pad = np.zeros((k - kc, streams.shape[1]), dtype=np.float32)
    return np.concatenate([streams, pad], axis=0)


def align_expensive_to_cheap(moss: np.ndarray, sr: np.ndarray) -> np.ndarray:
    """
    Reorder expensive-expert streams ``sr`` [K,T] onto the MossFormer2 speaker
    order ``moss`` [K,T] via cross-correlation Hungarian matching. Both must
    already share K rows.
    """
    cost = xcorr_cost_matrix(moss, sr)  # [K, K], lower = better match
    _, col = linear_sum_assignment(cost)
    return sr[col]


def mask_entropy_proxy(streams: np.ndarray, frame: int = 400) -> np.ndarray:
    """
    Blind per-stream mask-entropy proxy in [0, 1].

    Frames the streams, forms the per-frame energy distribution across streams,
    and assigns each stream the presence-weighted mean normalized entropy. High
    values flag frames where no stream cleanly dominates (ambiguous separation)
    — exactly what the fusion head should down-weight.
    """
    k, t = streams.shape
    if k <= 1:
        return np.zeros(k, dtype=np.float32)
    n_frames = max(t // frame, 1)
    trimmed = streams[:, : n_frames * frame].reshape(k, n_frames, frame)
    energy = (trimmed**2).sum(axis=-1) + _EPS  # [K, F]
    p = energy / energy.sum(axis=0, keepdims=True)  # [K, F]
    ent = -(p * np.log(p + _EPS)).sum(axis=0) / math.log(k)  # [F] in [0,1]
    weighted = (p * ent[None, :]).sum(axis=1) / (p.sum(axis=1) + _EPS)  # [K]
    return weighted.astype(np.float32)


def trivial_mask_proxy(
    mixture: np.ndarray, sample_rate: int, seg_seconds: float = 1.0
) -> np.ndarray:
    """Per-segment near-silence flag (scale-invariant RMS floor)."""
    seg = max(int(sample_rate * seg_seconds), 1)
    n = max(math.ceil(len(mixture) / seg), 1)
    global_rms = float(np.sqrt(np.mean(mixture**2) + _EPS))
    floor = 0.1 * global_rms
    flags = np.zeros(n, dtype=np.float32)
    for i in range(n):
        block = mixture[i * seg : (i + 1) * seg]
        if block.size == 0:
            continue
        rms = float(np.sqrt(np.mean(block**2) + _EPS))
        flags[i] = 1.0 if rms < floor else 0.0
    return flags


def _load_progress(out_dir: Path) -> dict:
    """Resume state from a prior (possibly interrupted) run, or a fresh start."""
    path = out_dir / "_progress.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"next_index": 0, "shard_idx": 0, "n_written": 0, "n_skipped": 0}


def _save_progress(
    out_dir: Path, next_index: int, shard_idx: int, n_written: int, n_skipped: int
) -> None:
    """
    Checkpoint after every shard flush so a killed/disconnected session (e.g. a
    Kaggle timeout mid cache-build) can resume from the last saved shard instead
    of reprocessing everything through the frozen experts from sample 0.
    """
    path = out_dir / "_progress.json"
    payload = {
        "next_index": next_index,
        "shard_idx": shard_idx,
        "n_written": n_written,
        "n_skipped": n_skipped,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stream_embeddings(result) -> np.ndarray | None:
    """Stack per-stream ECAPA embeddings from a SeparationResult, or None."""
    embs = [m.embedding for m in result.metadata]
    if any(e is None for e in embs):
        return None
    return np.stack([np.asarray(e, dtype=np.float32) for e in embs], axis=0)


# --------------------------------------------------------------------------- #
# Main build loop
# --------------------------------------------------------------------------- #
def build_cache(args: argparse.Namespace) -> dict:
    from models.experts.embeddings import ECAPAEmbedder
    from models.experts.mossformer2 import MossFormer2Expert
    from models.experts.srcorrnet import SRCorrNetExpert
    from models.realm_quality import REALMQualityEstimator

    device = args.device
    sr_target = args.sample_rate
    seg_len = int(sr_target * args.segment_seconds)
    k = args.target_speakers

    # Mixture source.
    if args.librimix_root:
        samples = load_librimix_samples(
            args.librimix_root, args.subset, args.mix_dir, args.mode, limit=args.limit
        )
    else:
        samples = load_dynamic_samples(
            args.dynamic_source_glob, args.limit or 500, args.allowed_n, seed=args.seed
        )
    if not samples:
        raise RuntimeError("No mixtures loaded — check the source arguments.")

    moss = MossFormer2Expert(device=device, compute_embeddings=True, target_speakers=k)
    # SR-CorrNet strictly — no SepFormer/TF-GridNet fallback.
    expensive = SRCorrNetExpert(
        device=device,
        repo_path=args.srcorrnet_repo,
        checkpoint_path=args.srcorrnet_checkpoint,
        hf_model_id=args.srcorrnet_hf_model,
        num_speakers=k,
    )
    if not expensive.is_available:
        raise RuntimeError(
            "SR-CorrNet-SS is required but not available. Clone dmlguq456/SR_CorrNet_SS "
            'and `pip install -e ".[hub]"`, or pass --srcorrnet-repo / --srcorrnet-checkpoint. '
            "SepFormer fallback is intentionally disabled."
        )
    realm = REALMQualityEstimator(device=device)
    ref_embedder = ECAPAEmbedder(device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress = {"next_index": 0, "shard_idx": 0, "n_written": 0, "n_skipped": 0}
    if not args.no_resume:
        progress = _load_progress(out_dir)
        if progress["next_index"] > 0:
            print(
                f"resuming from sample {progress['next_index']}/{len(samples)} "
                f"(shard {progress['shard_idx']}, {progress['n_written']} already written) "
                "— pass --no-resume to restart from scratch"
            )

    shard: list[dict] = []
    shard_idx = progress["shard_idx"]
    n_written = progress["n_written"]
    n_skipped = progress["n_skipped"]
    start_index = progress["next_index"]

    for i, s in enumerate(samples):
        if i < start_index:
            continue
        try:
            mixture = _crop_or_pad(s.mixture.astype(np.float32), seg_len)
            refs = _crop_or_pad(s.references.astype(np.float32), seg_len)
            n_true = refs.shape[0]

            moss_res = moss.separate(mixture, sr_target)
            moss_streams = _fit_k(moss_res.streams.astype(np.float32), k)
            moss_streams = _crop_or_pad(moss_streams, seg_len)

            exp_res = expensive.separate(mixture, sr_target)
            sr_streams = _fit_k(exp_res.streams.astype(np.float32), k)
            sr_streams = _crop_or_pad(sr_streams, seg_len)
            sr_streams = align_expensive_to_cheap(moss_streams, sr_streams)

            quality = realm.estimate(mixture, moss_streams, sr_target)
            quality_db = float(quality.min_sisnr_db)

            sr_conf = np.array(
                [float(m.confidence) for m in exp_res.metadata[:k]], dtype=np.float32
            )
            sr_conf = _fit_k(sr_conf.reshape(-1, 1), k).reshape(-1)[:k]
            if sr_conf.shape[0] < k:
                sr_conf = np.pad(sr_conf, (0, k - sr_conf.shape[0]))

            entropy = mask_entropy_proxy(moss_streams)
            trivial = trivial_mask_proxy(mixture, sr_target, args.trivial_seg_seconds)

            stream_emb = _stream_embeddings(moss_res)  # [K, D] or None
            if stream_emb is not None:
                ref_emb = np.stack(
                    [ref_embedder.embed_stream(refs[j], sr_target) for j in range(refs.shape[0])],
                    axis=0,
                )
            else:
                ref_emb = None

            sample = make_sample_dict(
                mixture=torch.from_numpy(mixture),
                references=torch.from_numpy(refs),
                moss_streams=torch.from_numpy(moss_streams),
                sr_streams=torch.from_numpy(np.ascontiguousarray(sr_streams)),
                true_count=float(n_true),
                quality_db=quality_db,
                sr_confidence=torch.from_numpy(sr_conf),
                moss_mask_entropy=torch.from_numpy(entropy),
                trivial_mask=torch.from_numpy(trivial),
                stream_embeddings=None if stream_emb is None else torch.from_numpy(stream_emb),
                reference_embeddings=None if ref_emb is None else torch.from_numpy(ref_emb),
            )
            shard.append(sample)
            n_written += 1
        except Exception as exc:  # noqa: BLE001 - log and continue on any per-sample failure
            n_skipped += 1
            print(f"[skip {i}] {type(exc).__name__}: {exc}")
            continue

        if len(shard) >= args.shard_size:
            save_cache_shard(out_dir / f"shard_{shard_idx:05d}.pt", shard, sr_target)
            print(f"wrote shard {shard_idx} ({len(shard)} samples), {n_written} total")
            shard = []
            shard_idx += 1
            _save_progress(out_dir, i + 1, shard_idx, n_written, n_skipped)

    if shard:
        save_cache_shard(out_dir / f"shard_{shard_idx:05d}.pt", shard, sr_target)
        print(f"wrote shard {shard_idx} ({len(shard)} samples), {n_written} total")
        shard_idx += 1

    _save_progress(out_dir, len(samples), shard_idx, n_written, n_skipped)

    manifest = {
        "n_written": n_written,
        "n_skipped": n_skipped,
        "target_speakers": k,
        "segment_seconds": args.segment_seconds,
        "sample_rate": sr_target,
        "expensive_expert": type(expensive).__name__,
        "source": args.librimix_root or args.dynamic_source_glob,
        "subset": args.subset,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the CA-MoSE frozen-expert training cache")
    src = p.add_argument_group("mixture source")
    src.add_argument("--librimix-root", default=None, help="Root of a generated LibriMix layout")
    src.add_argument("--subset", default="train-100", help="LibriMix subset (train-100/dev/test)")
    src.add_argument("--mix-dir", default="mix_clean", help="mix_clean or mix_both")
    src.add_argument("--mode", default="min", help="min or max")
    src.add_argument("--dynamic-source-glob", default=None, help="Glob of LibriSpeech clean files")
    src.add_argument("--allowed-n", type=int, nargs="+", default=[3])
    src.add_argument("--limit", type=int, default=None, help="Cap number of mixtures")

    exp = p.add_argument_group("experts")
    exp.add_argument("--target-speakers", type=int, default=3)
    exp.add_argument("--srcorrnet-repo", default=None, help="Local SR_CorrNet_SS clone (optional)")
    exp.add_argument("--srcorrnet-checkpoint", default=None, help="Local checkpoint (optional)")
    exp.add_argument(
        "--srcorrnet-hf-model",
        default="shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk",
        help="HF Hub model id used when no local checkpoint is given",
    )

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", required=True)
    out.add_argument("--shard-size", type=int, default=128)
    out.add_argument("--segment-seconds", type=float, default=3.0)
    out.add_argument("--trivial-seg-seconds", type=float, default=1.0)
    out.add_argument("--sample-rate", type=int, default=16000)
    out.add_argument("--device", default="cpu")
    out.add_argument("--seed", type=int, default=0)
    out.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing _progress.json in --out-dir and restart from sample 0",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
