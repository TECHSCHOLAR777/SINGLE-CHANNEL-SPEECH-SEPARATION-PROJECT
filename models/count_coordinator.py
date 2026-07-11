"""
Speaker-count coordinator (Dev B + C, Phase 3, P3-INT1).

Fuses two independent counting signals into one continue/stop decision per
peeling iteration (MASTER_PROJECT section 4.5):

1. SR-CorrNet's TDA attractor stop logit — the expensive expert's own
   internal "another attractor exists" signal, surfaced by the wrapper in
   ``SeparationResult.metadata[i].extra["stop_logit"]``.
2. The calibrated Dev C stop-classifier — an MLP over residual energy ratio,
   residual VAD probability, min embedding distance, and mixture-consistency
   error (``models/stop_classifier.py`` + ``models/counting_features.py``).

Fusion is a convex combination in logit space:

    z_fused = w_att * z_attractor + (1 - w_att) * z_classifier

with graceful degradation: if one signal is missing (cheap-only path has no
attractors; untrained classifier can be disabled), the other is used alone.
Temperature calibration is the classifier's own responsibility; the attractor
logit is passed through as-is since SR-CorrNet is frozen upstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from models.stop_classifier import StopClassifier

_EPS = 1e-8


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


@dataclass
class CountDecision:
    """Outcome of one coordinator evaluation at peeling iteration k.

    Attributes:
        continue_peeling: True when another speaker likely remains.
        p_continue: Fused probability that more speakers remain, [0, 1].
        p_attractor: Attractor-only probability, or None if unavailable.
        p_classifier: Classifier-only probability, or None if unavailable.
        source: Which signals produced the decision:
            "fused" | "attractor_only" | "classifier_only" | "fallback".
        threshold: Decision threshold applied to p_continue.
    """

    continue_peeling: bool
    p_continue: float
    p_attractor: float | None
    p_classifier: float | None
    source: str
    threshold: float


class CountCoordinator:
    """
    Fuses attractor stop logits with the calibrated stop-classifier.

    Args:
        classifier: Trained (or untrained, for mock mode) StopClassifier.
            Pass None to run attractor-only.
        attractor_weight: w_att in [0, 1]; weight of the attractor logit in
            the fused logit. 0.5 = equal trust. Tune on Libri2-5Mix dev
            (P3-INT2) once the classifier checkpoint exists (P3-C5).
        threshold: Decide continue when p_continue >= threshold. 0.5 default;
            raise for conservative under-counting, lower for over-counting.
        fallback_continue: Decision when NEITHER signal is available (should
            not happen in a wired pipeline; logged as source="fallback").
    """

    def __init__(
        self,
        classifier: StopClassifier | None = None,
        attractor_weight: float = 0.5,
        threshold: float = 0.5,
        fallback_continue: bool = False,
    ) -> None:
        if not 0.0 <= attractor_weight <= 1.0:
            raise ValueError(f"attractor_weight must be in [0,1], got {attractor_weight}")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0,1), got {threshold}")
        self.classifier = classifier
        self.attractor_weight = float(attractor_weight)
        self.threshold = float(threshold)
        self.fallback_continue = bool(fallback_continue)

    # ------------------------------------------------------------------
    # signal extraction
    # ------------------------------------------------------------------

    @staticmethod
    def attractor_logit_from_metadata(extra: dict) -> float | None:
        """Pull the TDA stop logit from a stream's metadata.extra, if present."""
        z = extra.get("stop_logit")
        if z is None:
            return None
        try:
            z = float(z)
        except (TypeError, ValueError):
            return None
        if math.isnan(z) or math.isinf(z):
            return None
        return z

    def classifier_logit(self, features: np.ndarray | torch.Tensor) -> float | None:
        """
        Run the stop-classifier on one feature vector, returning the
        temperature-scaled logit (so fusion happens in calibrated space).
        """
        if self.classifier is None:
            return None
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features.astype(np.float32))
        if features.ndim == 1:
            features = features.unsqueeze(0)
        self.classifier.eval()
        with torch.no_grad():
            raw = self.classifier(features)
            temp = self.classifier.temperature.clamp_min(_EPS)
            z = (raw / temp).item()
        if math.isnan(z) or math.isinf(z):
            return None
        return z

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------

    def decide(
        self,
        attractor_logit: float | None = None,
        stop_features: np.ndarray | torch.Tensor | None = None,
    ) -> CountDecision:
        """
        Fuse available signals into one continue/stop decision.

        Args:
            attractor_logit: SR-CorrNet TDA stop logit (None on cheap-only path).
            stop_features: [F] or [1, F] feature vector for the stop-classifier
                (None to skip the classifier).

        Returns:
            CountDecision with fused probability and provenance.
        """
        z_att = attractor_logit
        if z_att is not None and (math.isnan(z_att) or math.isinf(z_att)):
            z_att = None
        z_cls = self.classifier_logit(stop_features) if stop_features is not None else None

        p_att = _sigmoid(z_att) if z_att is not None else None
        p_cls = _sigmoid(z_cls) if z_cls is not None else None

        if z_att is not None and z_cls is not None:
            z = self.attractor_weight * z_att + (1.0 - self.attractor_weight) * z_cls
            p, source = _sigmoid(z), "fused"
        elif z_att is not None:
            p, source = p_att, "attractor_only"  # type: ignore[assignment]
        elif z_cls is not None:
            p, source = p_cls, "classifier_only"  # type: ignore[assignment]
        else:
            p, source = (1.0 if self.fallback_continue else 0.0), "fallback"

        return CountDecision(
            continue_peeling=p >= self.threshold,
            p_continue=float(p),
            p_attractor=p_att,
            p_classifier=p_cls,
            source=source,
            threshold=self.threshold,
        )

    def decide_from_result_metadata(
        self,
        extra: dict,
        stop_features: np.ndarray | torch.Tensor | None = None,
    ) -> CountDecision:
        """Convenience: extract the attractor logit from metadata.extra first."""
        return self.decide(
            attractor_logit=self.attractor_logit_from_metadata(extra),
            stop_features=stop_features,
        )
