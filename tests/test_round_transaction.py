"""A failed aggregation must not consume a round or replace committed history."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from geomstats.geometry.euclidean import Euclidean

from groupoid.aggregation import TransportGroupoidAggregator


def _aggregator() -> TransportGroupoidAggregator:
    graph = nx.DiGraph([("A", "B")])
    aggregator = TransportGroupoidAggregator(
        manifold=Euclidean(dim=2), graph=graph, base_node="A", track_divergence=True
    )
    aggregator.register_transport("B", "A", np.eye(2))
    return aggregator


def _points() -> dict[str, np.ndarray]:
    return {"A": np.array([0.0, 1.0]), "B": np.array([2.0, 1.0])}


def test_invalid_input_does_not_consume_a_round() -> None:
    agg = _aggregator()
    with pytest.raises(ValueError, match="outside the configured manifold"):
        agg.aggregate({"A": np.array([0.0]), "B": np.array([2.0, 1.0])})
    assert agg._round_idx == 0
    assert agg._prev_divergence is None
    assert agg.aggregate(_points()).round_idx == 1


def test_mean_failure_preserves_previous_round_and_divergence(monkeypatch) -> None:
    agg = _aggregator()
    first = agg.aggregate(_points())
    assert first.round_idx == 1
    previous = agg._prev_divergence
    assert previous is first.divergence

    def fail_mean(*args, **kwargs):
        raise RuntimeError("deliberate mean failure")

    with monkeypatch.context() as patch:
        patch.setattr("groupoid.aggregation.karcher_mean", fail_mean)
        with pytest.raises(RuntimeError, match="deliberate mean failure"):
            agg.aggregate(_points())

    assert agg._round_idx == 1
    assert agg._prev_divergence is previous
    assert agg.aggregate(_points()).round_idx == 2
