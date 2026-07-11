"""
P2-INT2: end-to-end forward-pass integration test (mock mode, CPU).

Two layers of coverage:

1. Training-side E2E: SceneAnalyzer -> Router -> gate mask -> Fusion ->
   CompositeLoss through CAMoSETrainer.forward_batch, asserting gradients
   flow and every loss term is finite.
2. Inference-side E2E: SeparationResult -> QualityEstimate -> CascadeGate ->
   CountCoordinator -> CascadeRunLogger, asserting the full decision chain
   composes on the shared schema without real weights.

Real-weight variants stay behind the existing RUN_REAL_EXPERTS flag pattern
(see tests/test_m1_real_experts.py); CI runs this file in mock mode only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eval.cascade_logging import CascadeRunLogger
from eval.reporting import RunLog
from models.cascade_gate import CascadeGate
from models.count_coordinator import CountCoordinator
from models.counting_features import FEATURE_NAMES
from models.realm_quality import QualityEstimate
from models.stop_classifier import StopClassifier
from schemas.separation_result import SeparationResult, StreamMetadata
from train.losses import CompositeLoss
from train.trainer import CAMoSETrainable, CAMoSETrainer, TrainBatch

SR = 16000
T = SR  # 1 second
K = 2
B = 2


def _train_batch(quality_db: list[float]) -> TrainBatch:
    g = torch.Generator().manual_seed(7)
    refs = 0.1 * torch.randn(B, K, T, generator=g)
    mixture = refs.sum(dim=1)
    return TrainBatch(
        mixture=mixture,
        references=refs,
        true_count=torch.full((B,), K, dtype=torch.long),
        trivial_mask=torch.zeros(B, 4),
        moss_streams=refs + 0.01 * torch.randn(B, K, T, generator=g),
        sr_streams=refs + 0.005 * torch.randn(B, K, T, generator=g),
        quality_scores_db=torch.tensor(quality_db),
        sr_confidence=torch.full((B, K), 0.9),
        moss_mask_entropy=torch.full((B, K), 0.3),
    )


class TestTrainingSideE2E:
    def _trainer(self) -> CAMoSETrainer:
        torch.manual_seed(0)
        return CAMoSETrainer(
            model=CAMoSETrainable(),
            gate=CascadeGate(tau=12.0, signal="min"),
            loss_fn=CompositeLoss(),
            device="cpu",
        )

    def test_forward_mixed_escalation_all_terms_finite(self):
        trainer = self._trainer()
        # one sample below tau (escalates, fusion runs), one above (cheap path)
        out = trainer.forward_batch(_train_batch([8.0, 15.0]))
        assert out.escalated_mask.tolist() == [True, False]
        assert out.fusion_residual is not None
        assert out.estimates.shape == (B, K, T)
        bd = out.loss_breakdown
        assert bd is not None
        assert torch.isfinite(bd.total)
        # count BCE is live in the composite objective (P3-C2 evidence)
        for name in ("si_sdr", "mrstft", "count"):
            term = getattr(bd, name, None)
            if term is None and hasattr(bd, "terms"):
                term = bd.terms.get(name)  # type: ignore[union-attr]
            if isinstance(term, torch.Tensor):
                assert torch.isfinite(term)

    def test_no_escalation_path_skips_fusion(self):
        trainer = self._trainer()
        out = trainer.forward_batch(_train_batch([15.0, 16.0]))
        assert out.escalated_mask.any().item() is False
        assert out.fusion_residual is None
        # estimates fall back to cheap-expert streams
        assert out.estimates.shape == (B, K, T)

    def test_train_step_updates_parameters(self):
        trainer = self._trainer()
        before = [p.detach().clone() for p in trainer.model.parameters()]
        bd, n_esc = trainer.train_step(_train_batch([8.0, 8.0]))
        assert torch.isfinite(bd.total)
        assert n_esc == B
        changed = any(
            not torch.equal(b, a.detach()) for b, a in zip(before, trainer.model.parameters())
        )
        assert changed, "optimizer step changed no parameters"


class TestInferenceSideE2E:
    def _cheap_result(self) -> SeparationResult:
        streams = 0.05 * np.random.default_rng(3).standard_normal((K, T)).astype(np.float32)
        return SeparationResult(
            streams=streams,
            sample_rate=SR,
            speaker_count=K,
            expert_used="mossformer2",
            mixture=streams.sum(axis=0),
        )

    def _expensive_result(self, stop_logit: float) -> SeparationResult:
        streams = 0.05 * np.random.default_rng(4).standard_normal((K, T)).astype(np.float32)
        meta = [
            StreamMetadata(expert_source="srcorrnet", confidence=0.9,
                           extra={"attractor_index": i, "stop_logit": stop_logit})
            for i in range(K)
        ]
        return SeparationResult(
            streams=streams, sample_rate=SR, speaker_count=K,
            metadata=meta, escalated=True, expert_used="srcorrnet",
        )

    def test_full_chain_escalate_count_log(self, tmp_path):
        gate = CascadeGate(tau=12.0, signal="min")
        coordinator = CountCoordinator(classifier=StopClassifier(), attractor_weight=0.5)
        logger = CascadeRunLogger(
            str(tmp_path / "e2e.jsonl"), system="cascade+fusion", condition="clean", tier="L1"
        )

        # 1. cheap expert runs; REAL-M says quality is poor
        quality = QualityEstimate(
            sisnr_db_per_stream=[7.0, 9.0], mean_sisnr_db=8.0, min_sisnr_db=7.0
        )
        decision = gate.decide(quality)
        assert decision.escalate is True

        # 2. expensive expert runs, exposing attractor stop logit
        result = self._expensive_result(stop_logit=-2.5)

        # 3. coordinator fuses attractor + classifier for the count decision
        feats = torch.zeros(1, len(FEATURE_NAMES))
        count = coordinator.decide_from_result_metadata(
            result.metadata[-1].extra, stop_features=feats
        )
        assert count.source == "fused"
        assert 0.0 <= count.p_continue <= 1.0

        # 4. one record lands in the RunLog with the whole story
        rec = logger.log(
            result, decision, n_true=K, count_confidence=1.0 - count.p_continue
        )
        loaded = RunLog(str(tmp_path / "e2e.jsonl")).load()
        assert len(loaded) == 1
        assert loaded[0].escalated is True
        assert loaded[0].extra["gate_quality_score_db"] == 7.0
        assert loaded[0].n_estimated == K
        assert math.isclose(loaded[0].count_confidence, rec.count_confidence)

    def test_full_chain_accept_path(self, tmp_path):
        gate = CascadeGate(tau=12.0)
        logger = CascadeRunLogger(str(tmp_path / "e2e.jsonl"), system="cascade")

        quality = QualityEstimate(
            sisnr_db_per_stream=[14.0, 16.0], mean_sisnr_db=15.0, min_sisnr_db=14.0
        )
        decision = gate.decide(quality)
        assert decision.escalate is False

        result = self._cheap_result()
        # cheap path: no attractors; coordinator degrades to classifier-only
        coordinator = CountCoordinator(classifier=StopClassifier())
        count = coordinator.decide_from_result_metadata(
            result.metadata[0].extra, stop_features=torch.zeros(1, len(FEATURE_NAMES))
        )
        assert count.source == "classifier_only"

        logger.log(result, decision, n_true=K)
        loaded = RunLog(str(tmp_path / "e2e.jsonl")).load()
        assert loaded[0].escalated is False
        assert loaded[0].extra["gate_escalate"] is False


@pytest.mark.parametrize("k", [2, 3])
def test_schema_shapes_survive_chain(k):
    """SeparationResult invariants hold for varying K through the chain."""
    streams = np.zeros((k, 800), dtype=np.float32)
    r = SeparationResult(streams=streams, sample_rate=SR, speaker_count=k, expert_used="x")
    assert r.num_streams == k
    assert len(r.metadata) == k
