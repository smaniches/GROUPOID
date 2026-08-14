"""Integration tests for the supported aggregation and persistence paths.

The point-valued aggregator accepts explicit point actions. Tangent-vector
parallel transport is intentionally kept separate: ``register_transport_from_points``
is a fail-closed compatibility stub because a tangent operator is not, by
itself, an invertible action on manifold points.

With ``track_divergence=True``, each aggregation round carries a persistence
summary of the transported parameters and the bottleneck distance to the
previous round.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere

from groupoid.aggregation import (
    TransportGroupoidAggregator,
    UnsupportedTransportRepresentationError,
)


def _make_aggregator(**kwargs):
    manifold = Hypersphere(dim=2)
    graph = nx.DiGraph([("A", "B"), ("A", "C")])
    return (
        manifold,
        TransportGroupoidAggregator(manifold=manifold, graph=graph, base_node="A", **kwargs),
    )


class TestTransportRepresentationBoundary:
    def test_register_transport_from_points_fails_closed(self):
        manifold, agg = _make_aggregator()
        p_b = np.array([np.sin(0.5), 0.0, np.cos(0.5)])
        p_a = np.array([0.0, 0.0, 1.0])

        with pytest.raises(
            UnsupportedTransportRepresentationError,
            match="tangent-vector parallel transport",
        ):
            agg.register_transport_from_points("B", "A", p_b, p_a, n_rungs=4)

        assert not agg.morphisms
        assert manifold.belongs(p_a)


class TestDivergenceTracking:
    def test_divergence_disabled_by_default(self):
        _, agg = _make_aggregator()
        agg.register_transport("A", "B", np.eye(3))
        agg.register_transport("A", "C", np.eye(3))
        params = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([0.1, 0.0, 0.995]),
            "C": np.array([-0.1, 0.0, 0.995]),
        }
        params = {k: v / np.linalg.norm(v) for k, v in params.items()}
        result = agg.aggregate(params)
        assert result.divergence is None

    def test_divergence_tracked_across_rounds(self):
        _, agg = _make_aggregator(track_divergence=True)
        agg.register_transport("A", "B", np.eye(3))
        agg.register_transport("A", "C", np.eye(3))
        params = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([0.1, 0.0, 0.995]),
            "C": np.array([-0.1, 0.0, 0.995]),
        }
        params = {k: v / np.linalg.norm(v) for k, v in params.items()}

        first = agg.aggregate(params)
        assert first.divergence is not None
        assert first.divergence.bottleneck_to_previous is None

        second = agg.aggregate(params)
        assert second.divergence is not None
        assert second.divergence.bottleneck_to_previous == pytest.approx(0.0, abs=1e-12)

        jumped = dict(params)
        jumped["C"] = np.array([0.0, 1.0, 0.0])
        third = agg.aggregate(jumped)
        assert third.divergence is not None
        assert third.divergence.bottleneck_to_previous is not None
        assert third.divergence.bottleneck_to_previous > 0.1
