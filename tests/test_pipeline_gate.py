"""Regression test for I-041: the gate crashed whenever a real gate network was attached.

`GateNetwork` takes a 10-D vector, cat(level1[4], level2[6]) (models/gate.py).
`pipeline.infer._condition_dict_to_tensor` built only the 4 Level-1 features and
handed that straight to the gate, which raised a shape-mismatch RuntimeError
inside the first Linear(10, 256) layer. This is the deployed condition-routed
inference path the project is named for, and it could not run with a trained
gate attached. See docs/restoration/ISSUE_LEDGER.md I-041.
"""

from __future__ import annotations

from coralsep.models.gate import GateNetwork
from coralsep.pipeline.infer import CoralSepPipeline, _condition_dict_to_tensor


def test_condition_dict_to_tensor_matches_gate_input_width():
    cond = {
        "snr_est_db": 5.0,
        "codec_bw_ratio": 0.8,
        "voiced_density": 0.6,
        "total_energy_db": -20.0,
    }
    t = _condition_dict_to_tensor(cond)
    assert t.shape == (10,), "GateNetwork expects cat(level1[4], level2[6]) = 10"


def test_compute_gate_does_not_crash_with_a_real_gate_network(mock_expert):
    pipeline = CoralSepPipeline(expert=mock_expert, gate_net=GateNetwork())
    condition_l1 = {
        "snr_est_db": 5.0,
        "codec_bw_ratio": 0.8,
        "voiced_density": 0.6,
        "total_energy_db": -20.0,
    }
    gate_vec = pipeline._compute_gate(condition_l1)
    assert set(gate_vec.keys()) == {"reverb", "noise", "codec"}
    assert all(0.0 <= v <= 1.5 for v in gate_vec.values())
