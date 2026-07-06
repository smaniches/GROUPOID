"""Integration tests wiring transport and persistence into the pipeline.

Transport: ``register_transport_from_points`` builds the transport matrix
from per-client base points via the pole ladder, so a caller who knows
where each client sits on the manifold no longer has to hand-construct
rotation matrices. The test validates the computed map against the
analytic parallel transport (geomstats) on S^2 and shows the end-to-end
aggregation error against a known consensus is small.

Persistence: with ``track_divergence=True``, each aggregation round
carries a persistence summary of the transported parameters and the
bottleneck distance to the previous round. Identical rounds must report
(near-)zero divergence; a round where one client jumps must report a
strictly larger one.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere

from groupoid.aggregation import TransportGroupoidAggregator


def _make_aggregator(**kwargs):
    manifold = Hypersphere(dim=2)
    graph = nx.DiGraph([("A", "B"), ("A", "C")])
    return (
        manifold,
        TransportGroupoidAggregator(manifold=manifold, graph=graph, base_node="A", **kwargs),
    )


class TestTransportFromPoints:
    def test_computed_transport_matches_analytic_on_tangent_vectors(self):
        # The pole-ladder transport matrix from base points must act on
        # tangent vectors like geomstats' analytic parallel transport
        # (up to the documented ladder residual; see LIMITATIONS.md).
        manifold, agg = _make_aggregator()
        p_b = np.array([np.sin(0.5), 0.0, np.cos(0.5)])
        p_a = np.array([0.0, 0.0, 1.0])

        T = agg.register_transport_from_points("B", "A", p_b, p_a, n_rungs=4)
        assert ("B", "A") in agg.morphisms

        v = manifold.to_tangent(np.array([0.3, 0.4, 0.0]), p_b)
        analytic = manifold.metric.parallel_transport(v, p_b, end_point=p_a)
        ladder = T @ v
        cosine = np.dot(ladder, analytic) / (np.linalg.norm(ladder) * np.linalg.norm(analytic))
        assert cosine > 0.999
        assert np.linalg.norm(ladder) == pytest.approx(np.linalg.norm(v), rel=1e-2)

    def test_schild_method_branch_registers_transport(self):
        # The non-default ladder is selectable through the same entry point.
        manifold, agg = _make_aggregator()
        p_b = np.array([np.sin(0.5), 0.0, np.cos(0.5)])
        p_a = np.array([0.0, 0.0, 1.0])
        T = agg.register_transport_from_points("B", "A", p_b, p_a, method="schild", n_rungs=4)
        assert T.shape == (3, 3)
        assert ("B", "A") in agg.morphisms

    def test_end_to_end_aggregation_with_computed_transports(self):
        # Clients B and C sit away from base A; their parameters are the
        # SAME point expressed near their own base points. Transporting
        # with ladder-computed maps and aggregating must land near the
        # common consensus direction seen from A.
        manifold, agg = _make_aggregator()
        p_a = np.array([0.0, 0.0, 1.0])
        p_b = np.array([np.sin(0.4), 0.0, np.cos(0.4)])
        p_c = np.array([0.0, np.sin(0.4), np.cos(0.4)])

        agg.register_transport_from_points("B", "A", p_b, p_a, n_rungs=4)
        agg.register_transport_from_points("C", "A", p_c, p_a, n_rungs=4)

        params = {"A": p_a, "B": p_b, "C": p_c}
        result = agg.aggregate(params)

        assert manifold.belongs(result.global_params, atol=1e-3)
        # All three inputs lie within 0.4 rad of A's base point, so the
        # aggregate must remain in that neighborhood (sanity: transport
        # did not fling parameters across the sphere).
        assert float(manifold.metric.dist(result.global_params, p_a)) < 0.4


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
        # No previous round to compare against.
        assert first.divergence.bottleneck_to_previous is None

        # Identical second round: topology unchanged, zero divergence.
        second = agg.aggregate(params)
        assert second.divergence is not None
        assert second.divergence.bottleneck_to_previous == pytest.approx(0.0, abs=1e-12)

        # Third round where one client jumps far away: the H0 structure of
        # the point cloud changes, so the divergence must strictly exceed
        # the identical-round value.
        jumped = dict(params)
        jumped["C"] = np.array([0.0, 1.0, 0.0])
        third = agg.aggregate(jumped)
        assert third.divergence is not None
        assert third.divergence.bottleneck_to_previous is not None
        assert third.divergence.bottleneck_to_previous > 0.1
