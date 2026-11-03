"""Real neural-parameter point actions through the GROUPOID aggregator.

These are supplied exact hidden-unit permutations, not inferred alignments.
The neutral Euclidean example tests integration, not curved geometry or a
performance advantage over FedAvg in a common frame.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from geomstats.geometry.euclidean import Euclidean

from groupoid.aggregation import TransportGroupoidAggregator


INPUT = 2
HIDDEN = 3
OUTPUT = 2
DIM = INPUT * HIDDEN + HIDDEN + OUTPUT * HIDDEN + OUTPUT


def _gauge_indices(order: tuple[int, ...]) -> np.ndarray:
    """Map a canonical flattened 2-3-2 MLP into a local hidden-unit frame."""
    if sorted(order) != list(range(HIDDEN)):
        raise ValueError("hidden-unit indices must form a permutation")
    p = np.asarray(order)
    w1 = np.arange(INPUT * HIDDEN).reshape(HIDDEN, INPUT)[p].ravel()
    b1 = (INPUT * HIDDEN + np.arange(HIDDEN))[p]
    start = INPUT * HIDDEN + HIDDEN
    w2 = (start + np.arange(OUTPUT * HIDDEN).reshape(OUTPUT, HIDDEN)[:, p]).ravel()
    b2 = start + OUTPUT * HIDDEN + np.arange(OUTPUT)
    result = np.concatenate((w1, b1, w2, b2))
    assert np.array_equal(np.sort(result), np.arange(DIM))
    return result


def _forward(flat: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    cuts = np.cumsum([INPUT * HIDDEN, HIDDEN, OUTPUT * HIDDEN])
    w1, b1, w2, b2 = np.split(flat, cuts)
    hidden = np.tanh(inputs @ w1.reshape(HIDDEN, INPUT).T + b1)
    return hidden @ w2.reshape(OUTPUT, HIDDEN).T + b2


def test_real_mlp_parameter_gauge_is_integrated_into_aggregation() -> None:
    """The actual point-action aggregator must reproduce canonical FedAvg."""
    rng = np.random.default_rng(20261103)
    weights = {"A": 2.0, "B": 3.0, "C": 5.0}
    canonical = {node: rng.normal(size=DIM) for node in weights}
    idx = {
        "A": _gauge_indices((0, 1, 2)),
        "B": _gauge_indices((2, 0, 1)),
        "C": _gauge_indices((1, 2, 0)),
    }
    local = {node: canonical[node][idx[node]] for node in weights}

    # Verify that these point actions are actual neural-network symmetries.
    test_inputs = rng.normal(size=(16, INPUT))
    for node in weights:
        np.testing.assert_allclose(
            _forward(local[node], test_inputs),
            _forward(canonical[node], test_inputs),
            rtol=0,
            atol=2e-14,
        )

    graph = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "C")])
    agg = TransportGroupoidAggregator(
        manifold=Euclidean(dim=DIM), graph=graph, base_node="A"
    )
    # P[node] sends canonical coordinates into each client's local frame.
    # The point action from client B/C back to the base is P[node]^{-1}.
    to_base = {node: np.eye(DIM)[:, indices] for node, indices in idx.items()}
    agg.register_transport("B", "A", to_base["B"])
    agg.register_transport("C", "A", to_base["C"])
    agg.register_transport("B", "C", to_base["C"].T @ to_base["B"])

    result = agg.aggregate(local, weights=weights)
    oracle = sum(weights[n] * canonical[n] for n in weights) / sum(weights.values())
    np.testing.assert_allclose(result.global_params, oracle, rtol=0, atol=1e-10)
    assert result.cycle_basis_holonomy_defect < 1e-10
    assert result.passes_consistency_threshold
    for node in weights:
        np.testing.assert_allclose(
            result.local_updates[node], oracle[idx[node]], rtol=0, atol=1e-10
        )

    naive = sum(weights[n] * local[n] for n in weights) / sum(weights.values())
    assert np.linalg.norm(naive - oracle) > 0.1
