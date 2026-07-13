"""
CA-MoSE training loop (Dev B, Phase 2).

Trains the Scene Analyzer (stub), Router, and CRRR Fusion Head while keeping
expert wrappers frozen. Supports a self-test mode with synthetic tensors so
CI can verify the training mechanics without pretrained expert weights.

Usage:
    python -m train.trainer --self-test
    python -m train.trainer --config configs/training.yaml --epochs 10
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.cascade_gate import CascadeGate
from models.fusion import CRRRFusionHead
from models.router import TwoLevelRouter
from models.scene_analyzer import SceneAnalyzer
from train.losses import CompositeLoss, LossBreakdown, LossWeights
from utils.config import cfg_get, load_config

_EPS = 1e-8


@dataclass
class TrainBatch:
    """One training batch with optional precomputed expert outputs."""

    mixture: torch.Tensor
    references: torch.Tensor
    true_count: torch.Tensor
    trivial_mask: torch.Tensor
    moss_streams: torch.Tensor
    sr_streams: torch.Tensor
    quality_scores_db: torch.Tensor
    sr_confidence: torch.Tensor
    moss_mask_entropy: torch.Tensor
    stream_embeddings: torch.Tensor | None = None
    reference_embeddings: torch.Tensor | None = None


@dataclass
class ForwardOutput:
    """Forward pass artifacts for loss computation."""

    estimates: torch.Tensor
    count_logits: torch.Tensor
    router_weights: torch.Tensor
    fusion_residual: torch.Tensor | None
    escalated_mask: torch.Tensor
    loss_breakdown: LossBreakdown | None = None


@dataclass
class TrainMetrics:
    """Aggregated metrics from one epoch."""

    loss: float
    escalation_rate: float
    term_means: dict[str, float] = field(default_factory=dict)


class CAMoSETrainable(nn.Module):
    """Trainable CA-MoSE heads: scene analyzer, router, fusion."""

    def __init__(
        self,
        feature_dim: int = 64,
        num_experts: int = 3,
        null_index: int = 2,
        fusion_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.scene_analyzer = SceneAnalyzer(feature_dim=feature_dim)
        self.router = TwoLevelRouter(
            feature_dim=feature_dim, num_experts=num_experts, null_index=null_index
        )
        assert self.scene_analyzer.feature_dim == self.router.feature_dim, (
            f"feature_dim mismatch: SceneAnalyzer={self.scene_analyzer.feature_dim}, "
            f"TwoLevelRouter={self.router.feature_dim}"
        )
        self.fusion = CRRRFusionHead(hidden_channels=fusion_hidden)
        self.null_index = null_index

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CAMoSETrainer:
    """
    End-to-end trainer for CA-MoSE trainable heads (P2-B6).

    Frozen experts are supplied via TrainBatch fields; the trainer focuses on
    scene analysis, routing, cascade gating decisions, and fusion on escalated
    inputs.
    """

    def __init__(
        self,
        model: CAMoSETrainable,
        gate: CascadeGate,
        loss_fn: CompositeLoss,
        device: str = "cpu",
        lr: float = 1e-4,
        tf_expert_index: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.gate = gate
        # CompositeLoss is an nn.Module with real params (SpeakerConsistencyLoss's
        # projection layer) — left on CPU it raises "mat2 is on cpu" the moment
        # the speaker loss runs on CUDA-moved embeddings.
        self.loss_fn = loss_fn.to(device)
        self.device = torch.device(device)
        self.tf_expert_index = tf_expert_index
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def forward_batch(self, batch: TrainBatch, compute_loss: bool = True) -> ForwardOutput:
        """Run trainable forward pass and optionally compute composite loss."""
        mixture = batch.mixture.to(self.device)
        references = batch.references.to(self.device)
        moss = batch.moss_streams.to(self.device)
        sr = batch.sr_streams.to(self.device)
        # Move these to the device BEFORE any escalated-index gather: esc_idx is
        # derived from `quality` (on device), so indexing a still-on-CPU tensor
        # with it raises "indices should be ... on the same device". Harmless on
        # CPU, fatal on CUDA — which is why it only surfaced on the Kaggle run.
        sr_confidence = batch.sr_confidence.to(self.device)
        moss_mask_entropy = batch.moss_mask_entropy.to(self.device)

        scene_out = self.model.scene_analyzer(mixture)
        router_w = self.model.router(scene_out["segment_features"])
        tf_weight = router_w[..., self.tf_expert_index].mean(dim=1, keepdim=True)
        tf_weight = tf_weight.expand(-1, sr.shape[1])

        trivial_mask = batch.trivial_mask.to(self.device)
        n_seg = router_w.shape[1]
        if trivial_mask.shape[1] != n_seg:
            if trivial_mask.shape[1] < n_seg:
                pad = torch.zeros(
                    trivial_mask.shape[0],
                    n_seg - trivial_mask.shape[1],
                    device=trivial_mask.device,
                )
                trivial_mask = torch.cat([trivial_mask, pad], dim=1)
            else:
                trivial_mask = trivial_mask[:, :n_seg]

        quality = batch.quality_scores_db.to(self.device)
        if quality.ndim == 0:
            quality = quality.unsqueeze(0).expand(mixture.shape[0])
        elif quality.shape[0] != mixture.shape[0]:
            quality = quality.expand(mixture.shape[0])
        escalated_mask = quality < self.gate.tau

        estimates = moss.clone()
        fusion_residual: torch.Tensor | None = None
        if escalated_mask.any():
            esc_idx = escalated_mask.nonzero(as_tuple=False).squeeze(-1)
            if esc_idx.ndim == 0:
                esc_idx = esc_idx.unsqueeze(0)
            fused_out = self.model.fusion(
                sr_streams=sr[esc_idx],
                moss_streams=moss[esc_idx],
                mixture=mixture[esc_idx],
                sr_confidence=sr_confidence[esc_idx],
                moss_mask_entropy=moss_mask_entropy[esc_idx],
                scene_weight_tf=tf_weight[esc_idx],
            )
            estimates[esc_idx] = fused_out.fused_streams
            fusion_residual = torch.zeros_like(sr)
            fusion_residual[esc_idx] = fused_out.residual

        breakdown: LossBreakdown | None = None
        if compute_loss:
            breakdown = self.loss_fn(
                estimates=estimates,
                references=references,
                count_logits=scene_out["count_logits"],
                true_counts=batch.true_count.to(self.device),
                router_weights=router_w,
                trivial_mask=trivial_mask,
                null_index=self.model.null_index,
                fusion_residual=fusion_residual,
                stream_embeddings=(
                    batch.stream_embeddings.to(self.device)
                    if batch.stream_embeddings is not None
                    else None
                ),
                reference_embeddings=(
                    batch.reference_embeddings.to(self.device)
                    if batch.reference_embeddings is not None
                    else None
                ),
            )

        return ForwardOutput(
            estimates=estimates,
            count_logits=scene_out["count_logits"],
            router_weights=router_w,
            fusion_residual=fusion_residual,
            escalated_mask=escalated_mask,
            loss_breakdown=breakdown,
        )

    def train_step(self, batch: TrainBatch) -> tuple[LossBreakdown, int]:
        """Single optimization step. Returns loss breakdown and escalation count."""
        self.model.train()
        self.optimizer.zero_grad()
        forward = self.forward_batch(batch, compute_loss=True)
        assert forward.loss_breakdown is not None
        forward.loss_breakdown.total.backward()
        self.optimizer.step()
        n_esc = int(forward.escalated_mask.sum().item())
        return forward.loss_breakdown, n_esc

    def train_epoch(self, loader: DataLoader) -> TrainMetrics:
        """Train one epoch over a DataLoader of TrainBatch items."""
        self.model.train()
        total_loss = 0.0
        n_esc = 0
        n_total = 0
        term_sums: dict[str, float] = {}

        for item in loader:
            batch = _collate_train_batch(item) if isinstance(item, list) else item
            breakdown, batch_esc = self.train_step(batch)
            total_loss += float(breakdown.total.detach().item())
            n_total += batch.mixture.shape[0]
            n_esc += batch_esc
            for k, v in breakdown.weighted.items():
                term_sums[k] = term_sums.get(k, 0.0) + v

        n_batches = max(len(loader), 1)
        return TrainMetrics(
            loss=total_loss / n_batches,
            escalation_rate=n_esc / max(n_total, 1),
            term_means={k: v / n_batches for k, v in term_sums.items()},
        )


def _collate_train_batch(items: list[TrainBatch]) -> TrainBatch:
    """Stack a list of single-sample TrainBatch objects."""
    return TrainBatch(
        mixture=torch.stack([x.mixture for x in items]),
        references=torch.stack([x.references for x in items]),
        true_count=torch.stack([x.true_count for x in items]),
        trivial_mask=torch.stack([x.trivial_mask for x in items]),
        moss_streams=torch.stack([x.moss_streams for x in items]),
        sr_streams=torch.stack([x.sr_streams for x in items]),
        quality_scores_db=torch.stack([x.quality_scores_db for x in items]),
        sr_confidence=torch.stack([x.sr_confidence for x in items]),
        moss_mask_entropy=torch.stack([x.moss_mask_entropy for x in items]),
        stream_embeddings=(
            torch.stack([x.stream_embeddings for x in items])
            if items[0].stream_embeddings is not None
            else None
        ),
        reference_embeddings=(
            torch.stack([x.reference_embeddings for x in items])
            if items[0].reference_embeddings is not None
            else None
        ),
    )


class SyntheticTrainDataset(Dataset):
    """Synthetic mixtures for trainer self-test (no expert weight downloads)."""

    def __init__(
        self,
        n_samples: int,
        n_speakers: int = 3,
        t: int = 4000,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.n_speakers = n_speakers
        self.t = t
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> TrainBatch:
        del idx
        sr_hz = 16000
        time = np.arange(self.t, dtype=np.float32) / sr_hz
        freqs = [300.0, 500.0, 700.0, 900.0, 1100.0][: self.n_speakers]
        refs = np.stack(
            [np.sin(2 * np.pi * f * time).astype(np.float32) for f in freqs],
            axis=0,
        )
        mixture = refs.sum(axis=0)
        noise = self.rng.normal(0, 0.02, size=refs.shape).astype(np.float32)
        moss = refs + noise
        sr_streams = refs + self.rng.normal(0, 0.01, size=refs.shape).astype(np.float32)

        quality = float(self.rng.uniform(8.0, 16.0))
        emb_dim = 64
        stream_emb = self.rng.normal(size=(self.n_speakers, emb_dim)).astype(np.float32)
        ref_emb = stream_emb + self.rng.normal(0, 0.05, size=(self.n_speakers, emb_dim)).astype(
            np.float32
        )

        return TrainBatch(
            mixture=torch.from_numpy(mixture),
            references=torch.from_numpy(refs),
            true_count=torch.tensor(float(self.n_speakers)),
            trivial_mask=torch.zeros(2),
            moss_streams=torch.from_numpy(moss),
            sr_streams=torch.from_numpy(sr_streams),
            quality_scores_db=torch.tensor(quality),
            sr_confidence=torch.full((self.n_speakers,), 0.85),
            moss_mask_entropy=torch.full((self.n_speakers,), 0.3),
            stream_embeddings=torch.from_numpy(stream_emb),
            reference_embeddings=torch.from_numpy(ref_emb),
        )


def synth_train_loader(
    n_samples: int = 32,
    batch_size: int = 4,
    seed: int = 0,
) -> DataLoader:
    """Build a DataLoader of synthetic training batches."""
    ds = SyntheticTrainDataset(n_samples=n_samples, seed=seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_train_batch,
    )


def build_trainer_from_config(cfg: dict[str, Any], device: str = "cpu") -> CAMoSETrainer:
    """Construct trainer from merged YAML config."""
    tau = float(cfg_get(cfg, "realm.quality_threshold_tau", 12.0))
    gate_signal = cfg_get(cfg, "training.gate_signal", "min")
    model = CAMoSETrainable(
        feature_dim=int(cfg_get(cfg, "training.feature_dim", 64)),
        num_experts=int(cfg_get(cfg, "training.num_experts", 3)),
        null_index=int(cfg_get(cfg, "training.null_index", 2)),
        fusion_hidden=int(cfg_get(cfg, "training.fusion_hidden", 256)),
    )
    gate = CascadeGate(tau=tau, signal=gate_signal)
    weights = LossWeights(
        si_sdr_upit=float(cfg_get(cfg, "training.loss.si_sdr_upit", 1.0)),
        multi_res_stft=float(cfg_get(cfg, "training.loss.multi_res_stft", 0.5)),
        count_bce=float(cfg_get(cfg, "training.loss.count_bce", 0.3)),
        load_balance=float(cfg_get(cfg, "training.loss.load_balance", 0.1)),
        null_sparsity=float(cfg_get(cfg, "training.loss.null_sparsity", 0.1)),
        residual_reg=float(cfg_get(cfg, "training.loss.residual_reg", 0.1)),
        speaker_consistency=float(cfg_get(cfg, "training.loss.speaker_consistency", 0.1)),
    )
    return CAMoSETrainer(
        model=model,
        gate=gate,
        loss_fn=CompositeLoss(weights),
        device=device,
        lr=float(cfg_get(cfg, "training.lr", 1e-4)),
        tf_expert_index=int(cfg_get(cfg, "training.tf_expert_index", 1)),
    )


def run_self_test(epochs: int = 2, device: str = "cpu") -> dict[str, Any]:
    """Synthetic training smoke test for CI."""
    trainer = build_trainer_from_config({"realm": {"quality_threshold_tau": 12.0}}, device=device)
    loader = synth_train_loader(n_samples=16, batch_size=4, seed=42)
    history: list[dict[str, float]] = []
    for _ in range(epochs):
        metrics = trainer.train_epoch(loader)
        history.append({"loss": metrics.loss, "escalation_rate": metrics.escalation_rate})
    final_loss = history[-1]["loss"]
    if not math.isfinite(final_loss):
        raise RuntimeError(f"non-finite loss after self-test: {final_loss}")
    return {
        "epochs": epochs,
        "history": history,
        "trainable_params": trainer.model.parameter_count(),
    }


def save_checkpoint(trainer: CAMoSETrainer, path: str | Path) -> None:
    """Save trainable head weights and optimizer state."""
    payload = {
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "gate_tau": trainer.gate.tau,
    }
    torch.save(payload, Path(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CA-MoSE trainable heads")
    parser.add_argument(
        "--config", nargs="*", default=["configs/experts.yaml", "configs/training.yaml"]
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/training/checkpoint.pt")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Train on a frozen-expert cache (build_train_cache.py) instead of synthetic data",
    )
    parser.add_argument(
        "--val-cache-dir",
        default=None,
        help="Optional cache to evaluate cascade vs single experts after training (P2-INT4)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.self_test:
        result = run_self_test(epochs=2, device=args.device)
        print(json.dumps(result, indent=2))
        return

    cfg = load_config(*args.config)
    trainer = build_trainer_from_config(cfg, device=args.device)

    if args.cache_dir:
        from train.cached_dataset import cached_train_loader

        loader = cached_train_loader(args.cache_dir, batch_size=args.batch_size, shuffle=True)
        print(f"training on cache {args.cache_dir} ({len(loader.dataset)} samples)")
    else:
        loader = synth_train_loader(
            n_samples=int(cfg_get(cfg, "training.synth_samples", 64)),
            batch_size=args.batch_size,
            seed=int(cfg_get(cfg, "seed", 42)),
        )

    history = []
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(loader)
        history.append(
            {"epoch": epoch, "loss": metrics.loss, "escalation_rate": metrics.escalation_rate}
        )
        print(f"epoch {epoch}: loss={metrics.loss:.4f} escalation={metrics.escalation_rate:.2%}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(trainer, out_path)
    meta_path = out_path.with_suffix(".json")

    eval_report = None
    if args.val_cache_dir:
        from scripts.evaluate_cascade import evaluate_cache

        eval_report = evaluate_cache(args.val_cache_dir, out_path, device=args.device)
        print("P2-INT4 evaluation:", json.dumps(eval_report, indent=2))

    meta_path.write_text(
        json.dumps({"history": history, "evaluation": eval_report}, indent=2), encoding="utf-8"
    )
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
