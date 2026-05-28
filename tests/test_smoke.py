"""Smoke tests for modules without full test coverage."""

from __future__ import annotations

import numpy as np


class TestKarcherMeanSmoke:
    """Basic Karcher mean sanity checks."""

    def test_mean_returns_valid_point(self):
        from geomstats.geometry.hypersphere import Hypersphere

        from groupoid.manifold import karcher_mean

        manifold = Hypersphere(dim=2)
        points = manifold.random_point(n_samples=3)
        mean = karcher_mean(manifold, points)

        assert mean.shape == (3,)
        assert manifold.belongs(mean, atol=1e-4)

    def test_weights_influence_mean(self):
        """Weights must be honored: a dominant weight pulls the mean toward
        its point. Guards against weights being silently dropped."""
        from geomstats.geometry.hypersphere import Hypersphere

        from groupoid.manifold import karcher_mean

        manifold = Hypersphere(dim=2)
        points = manifold.random_point(n_samples=4)

        unweighted = karcher_mean(manifold, points)
        weights = np.array([10.0, 1.0, 1.0, 1.0])
        weights = weights / weights.sum()
        weighted = karcher_mean(manifold, points, weights=weights)

        assert manifold.belongs(weighted, atol=1e-4)
        # The weighted mean must differ from the unweighted one and sit
        # closer to the dominant point.
        assert not np.allclose(unweighted, weighted, atol=1e-3)
        assert np.dot(weighted, points[0]) > np.dot(unweighted, points[0])


class TestTransportSmoke:
    """Basic parallel transport sanity checks."""

    def test_schild_ladder_preserves_norm(self):
        from geomstats.geometry.hypersphere import Hypersphere

        from groupoid.transport import schild_ladder

        manifold = Hypersphere(dim=2)
        base = np.array([0.0, 0.0, 1.0])
        end = np.array([0.5, 0.0, np.sqrt(3) / 2])
        end = end / np.linalg.norm(end)

        tangent = manifold.to_tangent(np.array([1.0, 0.0, 0.0]), base)
        original_norm = np.linalg.norm(tangent)

        transported = schild_ladder(manifold, tangent, base, end, n_rungs=4)
        transported_norm = np.linalg.norm(transported)

        assert abs(original_norm - transported_norm) / (original_norm + 1e-10) < 1e-4

    def test_pole_ladder_preserves_norm(self):
        from geomstats.geometry.hypersphere import Hypersphere

        from groupoid.transport import pole_ladder

        manifold = Hypersphere(dim=2)
        base = np.array([0.0, 0.0, 1.0])
        end = np.array([0.5, 0.0, np.sqrt(3) / 2])
        end = end / np.linalg.norm(end)

        tangent = manifold.to_tangent(np.array([1.0, 0.0, 0.0]), base)
        original_norm = np.linalg.norm(tangent)

        transported = pole_ladder(manifold, tangent, base, end, n_rungs=4)
        transported_norm = np.linalg.norm(transported)

        assert abs(original_norm - transported_norm) / (original_norm + 1e-10) < 1e-4


class TestOptimizerSmoke:
    """Basic Riemannian optimizer sanity checks."""

    def test_riemannian_sgd_stays_on_manifold(self):
        from geomstats.geometry.hypersphere import Hypersphere

        from groupoid.optimizer import RiemannianSGD

        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        grad = np.array([0.1, 0.2, 0.3])

        opt = RiemannianSGD(manifold=manifold, lr=0.01)
        new_point = opt.step(point, grad)

        assert new_point.shape == (3,)
        assert manifold.belongs(new_point, atol=1e-4)

    def test_riemannian_adam_stays_on_manifold(self):
        from geomstats.geometry.hypersphere import Hypersphere

        from groupoid.optimizer import RiemannianAdam

        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        grad = np.array([0.1, 0.2, 0.3])

        opt = RiemannianAdam(manifold=manifold, lr=0.01)
        new_point = opt.step(point, grad)

        assert new_point.shape == (3,)
        assert manifold.belongs(new_point, atol=1e-4)


class TestAggregationSmoke:
    """Deterministic aggregation test."""

    def test_deterministic_aggregation(self):
        import networkx as nx
        from geomstats.geometry.hypersphere import Hypersphere

        from groupoid.aggregation import TransportGroupoidAggregator

        manifold = Hypersphere(dim=2)
        graph = nx.DiGraph([("A", "B")])

        agg = TransportGroupoidAggregator(manifold=manifold, graph=graph, base_node="A")
        agg.register_transport("A", "B", np.eye(3))

        params = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([0.0, 0.0, 1.0]),
        }

        r1 = agg.aggregate(params)
        r2 = agg.aggregate(params)

        np.testing.assert_allclose(r1.global_params, r2.global_params, atol=1e-10)

    def test_invalid_node_raises(self):
        import networkx as nx
        import pytest
        from geomstats.geometry.hypersphere import Hypersphere

        from groupoid.aggregation import TransportGroupoidAggregator

        manifold = Hypersphere(dim=2)
        graph = nx.DiGraph([("A", "B")])

        agg = TransportGroupoidAggregator(manifold=manifold, graph=graph, base_node="A")

        params = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([0.0, 0.0, 1.0]),
        }

        with pytest.raises(ValueError, match="No transport"):
            agg.aggregate(params)
