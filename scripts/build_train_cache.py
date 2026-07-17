#!/usr/bin/env python3
"""
Precompute frozen-expert outputs into a training cache (Dev C, P2-INT3 enabler).

Why this exists
---------------
The experts (MossFormer2, SR-CorrNet / TF-GridNet) are FROZEN. Only the scene
analyzer (~1.5M), router (~0.5M) and fusion head are trained, a few million
parameters total. Re-running two frozen separation networks on every sample of
every epoch is the single dominant cost of training, and it buys nothing: the
outputs are identical every time.

So run them ONCE, cache the tensors, and train the small heads on the cache.
`TrainBatch` was already designed to accept precomputed `moss_streams`,
`sr_streams` and `quality_scores_db`, so the trainer needs no changes.

Practical effect on Kaggle: a 12-hour session cap and 30 GPU-hours a week go
from "barely one training run" to "iterate freely", because after the cache is
built the training loop touches no expert at all.

What one cache record contains
------------------------------
Exactly the per-sample fields of `train.trainer.TrainBatch`:

    mixture            [T]        fp16
    references         [K, T]     fp16
    moss_streams       [K, T]     fp16   cheap expert, residual-padded to K
    sr_streams         [K, T]     fp16   expensive expert, REORDERED to moss order
    quality_scores_db  scalar     fp32   REAL-M blind min SI-SNR
    sr_confidence      [K]        fp32
    moss_mask_entropy  [K]        fp32
    trivial_mask       [n_seg]    fp32
    true_count         scalar     fp32

Speaker-order alignment is not optional. `CRRRFusionHead` documents that
`moss_streams` and `sr_streams` must be in the SAME speaker order, and it has no
way to check. Feeding it misaligned streams trains the fusion head to correct
speaker A's waveform toward speaker B, which converges to something that looks
like a loss curve and is worthless. The cache therefore Hungarian-aligns the
expensive expert onto the cheap expert's order at build time, using the Dev C
alignment path, and stores them aligned.

Usage
-----
    python scripts/build_train_cache.py \
      --librimix-root /kaggle/working/data/Libri3Mix \
      --subset train \
      --output /kaggle/working/cache/train \
      --num-speakers 3 \
      --crop-sec 4.0 --crops-per-utterance 2 \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from align.hungarian import align_results, reorder_result  # noqa: E402
from data.mixer_stub import discover_librimix_samples  # noqa: E402
from models.experts.mossformer2 import MossFormer2Expert  # noqa: E402
from models.experts.tfgridnet import get_expensive_expert  # noqa: E402
from models.realm_quality import REALMQualityEstimator  # noqa: E402
from schemas.separation_result import SeparationResult  # noqa: E402

_EPS = 1e-8


def mask_entropy(streams: np.ndarray) -> np.ndarray:
    """
    Per-stream separation-uncertainty proxy in [0, 1].

    Build the implied soft mask m_k(t) = |s_k(t)|^2 / sum_j |s_j(t)|^2, then take
    the mean binary entropy of that mask over time. A confidently separated
    stream owns its frames (m near 1) or stays out of them (m near 0), so its
    binary entropy is low. A stream the model is ambivalent about sits near 0.5
    and scores high. This is what the fusion gate wants to know: how much should
    it trust the cheap expert here.
    """
    power = np.square(streams.astype(np.float64))
    total = power.sum(axis=0, keepdims=True) + _EPS
    m = np.clip(power / total, _EPS, 1.0 - _EPS)
    binary = -(m * np.log2(m) + (1.0 - m) * np.log2(1.0 - m))
    return binary.mean(axis=1).astype(np.float32)


def trivial_mask(references: np.ndarray, sample_rate: int, n_seg: int) -> np.ndarray:
    """
    Per-segment flag: 1 where the segment is trivially separable.

    A segment is trivial when at most one reference speaker is actually active in
    it (silence, or a single talker). Those segments carry no separation signal
    and the router should learn to take the cheap path through them.
    """
    k, t = references.shape
    seg_len = max(1, t // n_seg)
    out = np.zeros(n_seg, dtype=np.float32)
    for s in range(n_seg):
        lo, hi = s * seg_len, min((s + 1) * seg_len, t)
        if hi <= lo:
            continue
        block = references[:, lo:hi]
        rms = np.sqrt(np.mean(np.square(block.astype(np.float64)), axis=1))
        peak = float(rms.max())
        if peak <= _EPS:
            out[s] = 1.0  # pure silence
            continue
        active = int(np.sum(rms > 0.1 * peak))  # -20 dB relative to the loudest
        out[s] = 1.0 if active <= 1 else 0.0
    return out


def _crops(total: int, crop: int, n: int, rng: np.random.Generator) -> list[int]:
    if total <= crop:
        return [0]
    return sorted(int(x) for x in rng.integers(0, total - crop, size=n))


def build(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    crop = int(round(args.crop_sec * args.sample_rate))
    k = args.num_speakers

    print(f"discovering {args.subset} samples under {args.librimix_root} ...")
    samples = discover_librimix_samples(
        args.librimix_root, subset=args.subset, max_samples=args.max_utterances
    )
    if not samples:
        raise SystemExit(f"no samples found in {args.librimix_root} subset={args.subset}")
    print(f"  {len(samples)} utterances")

    print("loading frozen experts ...")
    cheap = MossFormer2Expert(device=args.device, compute_embeddings=False, target_speakers=k)
    expensive = get_expensive_expert(device=args.device, num_speakers=k)
    quality_model = REALMQualityEstimator(device=args.device)
    print(f"  cheap     : {cheap.EXPERT_NAME} (target_speakers={k})")
    print(f"  expensive : {type(expensive).__name__}")

    shard: list[dict] = []
    shard_idx = 0
    written = 0
    skipped = 0
    started = time.time()

    for u, sample in enumerate(samples):
        if sample.references.shape[0] != k:
            skipped += 1
            continue

        for start in _crops(sample.mixture.shape[0], crop, args.crops_per_utterance, rng):
            mix = sample.mixture[start : start + crop].astype(np.float32)
            refs = sample.references[:, start : start + crop].astype(np.float32)
            if mix.shape[0] < crop:  # short tail, pad
                pad = crop - mix.shape[0]
                mix = np.pad(mix, (0, pad))
                refs = np.pad(refs, ((0, 0), (0, pad)))

            try:
                moss: SeparationResult = cheap.separate(mix, sample.sample_rate)
                sr: SeparationResult = expensive.separate(mix, sample.sample_rate)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {sample.utterance_id}@{start}: {type(exc).__name__}: {exc}")
                skipped += 1
                continue

            # Speaker-order alignment. Not optional: the fusion head assumes it
            # and cannot verify it. See module docstring.
            moss.mixture = mix
            sr.mixture = mix
            alignment = align_results(moss, sr)
            sr = reorder_result(sr, alignment)

            if moss.streams.shape[0] != k or sr.streams.shape[0] != k:
                print(
                    f"  [skip] {sample.utterance_id}@{start}: stream count "
                    f"moss={moss.streams.shape[0]} sr={sr.streams.shape[0]} expected {k}"
                )
                skipped += 1
                continue

            q = quality_model.estimate_result(moss)
            sr_conf = np.array([m.confidence for m in sr.metadata] + [0.0] * k, dtype=np.float32)[
                :k
            ]

            shard.append(
                {
                    "utterance_id": sample.utterance_id,
                    "start": int(start),
                    "mixture": torch.from_numpy(mix).half(),
                    "references": torch.from_numpy(refs).half(),
                    "moss_streams": torch.from_numpy(moss.streams.astype(np.float32)).half(),
                    "sr_streams": torch.from_numpy(sr.streams.astype(np.float32)).half(),
                    "quality_scores_db": torch.tensor(float(q.min_sisnr_db)),
                    "sr_confidence": torch.from_numpy(sr_conf),
                    "moss_mask_entropy": torch.from_numpy(mask_entropy(moss.streams)),
                    "trivial_mask": torch.from_numpy(
                        trivial_mask(refs, sample.sample_rate, args.n_seg)
                    ),
                    "true_count": torch.tensor(float(k)),
                }
            )
            written += 1

            if len(shard) >= args.shard_size:
                path = out_dir / f"shard_{shard_idx:05d}.pt"
                torch.save(shard, path)
                print(f"  [shard] {path.name}  {len(shard)} records")
                shard, shard_idx = [], shard_idx + 1

        if (u + 1) % 25 == 0:
            rate = written / max(time.time() - started, 1e-6)
            print(f"  {u + 1}/{len(samples)} utterances | {written} records | {rate:.1f} rec/s")

    if shard:
        path = out_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(shard, path)
        print(f"  [shard] {path.name}  {len(shard)} records")
        shard_idx += 1

    manifest = {
        "subset": args.subset,
        "num_speakers": k,
        "sample_rate": args.sample_rate,
        "crop_samples": crop,
        "crop_sec": args.crop_sec,
        "n_seg": args.n_seg,
        "records": written,
        "skipped": skipped,
        "shards": shard_idx,
        "shard_size": args.shard_size,
        "cheap_expert": cheap.EXPERT_NAME,
        "expensive_expert": type(expensive).__name__,
        "speaker_order": "sr aligned onto moss via Hungarian (align.hungarian)",
        "dtype": "fp16 waveforms, fp32 scalars",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    mb = sum(f.stat().st_size for f in out_dir.glob("*.pt")) / 1e6
    print()
    print(f"cache written to {out_dir}")
    print(f"  {written} records, {skipped} skipped, {shard_idx} shards, {mb:.0f} MB")
    print(f"  elapsed {(time.time() - started) / 60:.1f} min")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--librimix-root", required=True)
    p.add_argument("--subset", default="train", choices=["train", "dev", "test"])
    p.add_argument("--output", required=True)
    p.add_argument("--num-speakers", type=int, default=3)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--crop-sec", type=float, default=4.0)
    p.add_argument("--crops-per-utterance", type=int, default=2)
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--n-seg", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    build(p.parse_args())


if __name__ == "__main__":
    main()
