"""Tests for models/count_coordinator.py (P3-INT1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from models.count_coordinator import CountCoordinator, CountDecision
from models.counting_features import FEATURE_NAMES
from models.stop_classifier import StopClassifier


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def test_attractor_only_path():
    coord = CountCoordinator(classifier=None, threshold=0.5)
    d = coord.decide(attractor_logit=2.0)
    assert d.source == "attractor_only"
    assert d.continue_peeling is True
    assert d.p_classifier is None
    assert abs(d.p_continue - _sigmoid(2.0)) < 1e-9

    d = coord.decide(attractor_logit=-2.0)
    assert d.continue_peeling is False


def test_classifier_only_path():
    clf = StopClassifier()
    coord = CountCoordinator(classifier=clf)
    feats = torch.zeros(1, len(FEATURE_NAMES))
    d = coord.decide(attractor_logit=None, stop_features=feats)
    assert d.source == "classifier_only"
    assert d.p_attractor is None
    assert 0.0 <= d.p_continue <= 1.0


def test_fused_is_convex_combination_in_logit_space():
    clf = StopClassifier()
    coord = CountCoordinator(classifier=clf, attractor_weight=0.7)
    feats = torch.randn(1, len(FEATURE_NAMES))
    z_cls = coord.classifier_logit(feats)
    z_att = 1.5
    d = coord.decide(attractor_logit=z_att, stop_features=feats)
    assert d.source == "fused"
    expected = _sigmoid(0.7 * z_att + 0.3 * z_cls)
    assert abs(d.p_continue - expected) < 1e-6


def test_fused_respects_temperature_calibration():
    clf = StopClassifier()
    feats = torch.randn(1, len(FEATURE_NAMES))
    coord = CountCoordinator(classifier=clf, attractor_weight=0.0)
    z_before = coord.classifier_logit(feats)
    with torch.no_grad():
        clf.temperature.fill_(2.0)
    z_after = coord.classifier_logit(feats)
    assert abs(z_after - z_before / 2.0) < 1e-5


def test_fallback_when_no_signals():
    coord = CountCoordinator(classifier=None, fallback_continue=False)
    d = coord.decide()
    assert d.source == "fallback"
    assert d.continue_peeling is False

    coord = CountCoordinator(classifier=None, fallback_continue=True)
    assert coord.decide().continue_peeling is True


def test_metadata_extraction_handles_garbage():
    coord = CountCoordinator()
    assert coord.attractor_logit_from_metadata({}) is None
    assert coord.attractor_logit_from_metadata({"stop_logit": None}) is None
    assert coord.attractor_logit_from_metadata({"stop_logit": "abc"}) is None
    assert coord.attractor_logit_from_metadata({"stop_logit": float("nan")}) is None
    assert coord.attractor_logit_from_metadata({"stop_logit": 1.25}) == 1.25
    assert coord.attractor_logit_from_metadata({"stop_logit": np.float32(0.5)}) == pytest.approx(0.5)


def test_decide_from_result_metadata():
    coord = CountCoordinator(classifier=None)
    d = coord.decide_from_result_metadata({"stop_logit": 3.0})
    assert isinstance(d, CountDecision)
    assert d.source == "attractor_only"
    assert d.continue_peeling


def test_numpy_features_accepted():
    clf = StopClassifier()
    coord = CountCoordinator(classifier=clf)
    feats = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    d = coord.decide(stop_features=feats)
    assert d.source == "classifier_only"


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        CountCoordinator(attractor_weight=1.5)
    with pytest.raises(ValueError):
        CountCoordinator(threshold=0.0)


def test_threshold_boundary():
    coord = CountCoordinator(classifier=None, threshold=0.5)
    d = coord.decide(attractor_logit=0.0)  # p == 0.5 exactly
    assert d.continue_peeling is True  # >= threshold continues
