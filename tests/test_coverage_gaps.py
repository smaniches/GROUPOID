"""Focused behavioral tests for previously-uncovered code paths.

These tests target real, breakable behavior that the existing smoke and
integration suites did not exercise: error branches, the weighted and
momentum paths, curvature-adaptive learning rate, the transport-matrix
constructor, and the degenerate (edgeless) sheaf Laplacian spectrum.

They do NOT change any module's documented status label (the
"tested" / "smoke-tested" classifications in STATUS.md and the docs are
about depth and pipeline integration, not line count). The persistence
module is deliberately left untested, consistent with its documented
"Implemented, not yet tested" status.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere

from groupoid.aggregation import TransportGroupoidAggregator
from groupoid.cohomology import IncompleteCocycleError, compute_h1
from groupoid.groupoid import CompositionError, Morphism, compose
from groupoid.laplacian import spectral_analysis
from groupoid.optimizer import RiemannianAdam, RiemannianSGD, curvature_adaptive_lr
from groupoid.sheaf import Sheaf
from groupoid.transport import compute_transport_matrix


class TestGroupoidErrors:
    """Composition law enforcement."""

    def test_compose_mismatched_raises(self):
        f = Morphism(source="A", target="B", transport_map=np.eye(2))
        g = Morphism(source="C", target="D", transport_map=np.eye(2))
        with pytest.raises(CompositionError, match="Cannot compose"):
            compose(f, g)


class TestCohomologyMissingEdge:
    """A cycle edge with no transport map leaves the holonomy undefined."""

    def test_missing_edge_in_cycle_raises(self):
        # Triangle cycle A-B-C-A, but the C-A edge has no transport map in
        # either direction. The holonomy is the ordered product over ALL
        # cycle edges, so an incompletely specified cocycle has an undefined
        # holonomy: compute_h1 must raise IncompleteCocycleError naming the
        # missing edge, NOT silently form a partial product and report a
        # (false) consistency value.
        # Structural triangle A-B-C-A: the undirected cycle basis is the full
        # triangle, but the A-C edge deliberately has no transport map.
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        transport_maps = {
            ("A", "B"): np.eye(2),
            ("B", "C"): np.eye(2),
            # deliberately omit ("A", "C") / ("C", "A")
        }
        with pytest.raises(IncompleteCocycleError, match="no transport map for edge"):
            compute_h1(graph, transport_maps)


class TestAggregationPaths:
    """Weighted aggregation, the identity short-circuit, and the
    inconsistency-warning branch."""

    def _setup(self, transport_ab: np.ndarray):
        manifold = Hypersphere(dim=2)
        graph = nx.DiGraph([("A", "B")])
        agg = TransportGroupoidAggregator(manifold=manifold, graph=graph, base_node="A")
        agg.register_transport("A", "B", transport_ab)
        return agg

    def test_weighted_aggregation_honors_weights(self):
        agg = self._setup(np.eye(3))
        params = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([1.0, 0.0, 0.0]),
        }
        # Identity transport keeps both params on S^2 in the base frame.
        unweighted = agg.aggregate(params)
        heavy_a = agg.aggregate(params, weights={"A": 100.0, "B": 1.0})

        assert heavy_a.is_consistent
        # A dominant weight on A pulls the global mean toward A's params.
        assert np.dot(heavy_a.global_params, params["A"]) > np.dot(
            unweighted.global_params, params["A"]
        )

    def test_base_node_identity_short_circuit(self):
        # base_node maps to itself with zero residual (the node==base path).
        agg = self._setup(np.eye(3))
        params = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([0.0, 0.0, 1.0]),
        }
        result = agg.aggregate(params)
        assert result.transport_residuals["A"] == 0.0
        # _get_transport_to_base returns None (identity) for the base node.
        assert agg._get_transport_to_base("A") is None

    def test_inconsistency_warning_path(self):
        # A non-orthogonal transport on a triangle cycle makes the holonomy
        # around A-B-C-A differ from the identity, driving H^1 above the
        # threshold and exercising the inconsistency-warning branch. (A
        # 2-node back-and-forth is not a cycle once the graph is made
        # undirected, so a triangle is required to create real holonomy.)
        manifold = Hypersphere(dim=2)
        graph = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
        agg = TransportGroupoidAggregator(
            manifold=manifold, graph=graph, base_node="A", consistency_threshold=1e-6
        )
        skew = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        agg.register_transport("A", "B", skew)
        agg.register_transport("B", "C", np.eye(3))
        agg.register_transport("A", "C", np.eye(3))
        params = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([0.0, 0.0, 1.0]),
            "C": np.array([0.0, 0.0, 1.0]),
        }
        result = agg.aggregate(params)
        assert not result.is_consistent
        assert result.h1_norm > agg.consistency_threshold


class TestOptimizerPaths:
    """Momentum accumulation, Adam's second-step moment update, and
    curvature-adaptive learning rate."""

    def test_sgd_momentum_accumulates(self):
        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        grad = np.array([0.1, 0.2, 0.3])

        opt = RiemannianSGD(manifold=manifold, lr=0.01, momentum=0.9)
        # First step initialises velocity; second step accumulates it.
        p1 = opt.step(point, grad)
        assert opt._velocity is not None
        p2 = opt.step(p1, grad)
        assert manifold.belongs(p2, atol=1e-4)
        # Velocity stays in the tangent space at the current point.
        assert manifold.is_tangent(opt._velocity, p1, atol=1e-4)

    def test_adam_second_step_updates_moment(self):
        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        grad = np.array([0.1, 0.2, 0.3])

        opt = RiemannianAdam(manifold=manifold, lr=0.01)
        p1 = opt.step(point, grad)
        # Second step exercises the existing-first-moment branch.
        p2 = opt.step(p1, grad)
        assert opt._t == 2
        assert manifold.belongs(p2, atol=1e-4)

    def test_curvature_adaptive_lr_damps_on_sphere(self):
        # The 2-sphere has positive sectional curvature, so the adapted
        # LR must not exceed the base LR.
        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        tangent = manifold.to_tangent(np.array([1.0, 0.0, 0.0]), point)
        base_lr = 0.1
        adapted = curvature_adaptive_lr(manifold, point, base_lr, tangent)
        assert 0.0 < adapted <= base_lr

    def test_curvature_adaptive_lr_falls_back_without_curvature(self):
        # A stub manifold whose metric lacks sectional_curvature must
        # return the base LR unchanged.
        class _NoCurvatureMetric:
            def to_tangent(self, vec, point):
                return vec

        class _NoCurvatureManifold:
            metric = _NoCurvatureMetric()

            def to_tangent(self, vec, point):
                return vec

        manifold = _NoCurvatureManifold()
        point = np.array([0.0, 0.0, 1.0])
        tangent = np.array([1.0, 0.0, 0.0])
        assert curvature_adaptive_lr(manifold, point, 0.05, tangent) == 0.05


class TestTransportMatrix:
    """The full transport-matrix constructor."""

    def test_compute_transport_matrix_is_near_orthogonal(self):
        # Parallel transport on S^2 is an isometry, so the transport
        # matrix restricted to the tangent space preserves inner products;
        # the constructed map should be close to orthogonal.
        manifold = Hypersphere(dim=2)
        base = np.array([0.0, 0.0, 1.0])
        end = np.array([0.5, 0.0, np.sqrt(3) / 2])
        end = end / np.linalg.norm(end)

        T = compute_transport_matrix(manifold, base, end, method="pole", n_rungs=4)
        assert T.shape == (3, 3)

        # A tangent vector at base must keep its norm after transport.
        v = manifold.to_tangent(np.array([1.0, 0.0, 0.0]), base)
        transported = T @ v
        assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(v), rel=1e-3)

    def test_compute_transport_matrix_schild_branch(self):
        # Exercise the non-default ("schild") method selection branch, and
        # validate it as transport: Schild's ladder is also an isometry on
        # S^2, so the constructed matrix must preserve a tangent vector's
        # norm (shape/finiteness alone would only touch the line).
        manifold = Hypersphere(dim=2)
        base = np.array([0.0, 0.0, 1.0])
        end = np.array([0.5, 0.0, np.sqrt(3) / 2])
        end = end / np.linalg.norm(end)
        T = compute_transport_matrix(manifold, base, end, method="schild", n_rungs=4)
        assert T.shape == (3, 3)
        assert np.all(np.isfinite(T))

        v = manifold.to_tangent(np.array([1.0, 0.0, 0.0]), base)
        transported = T @ v
        assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(v), rel=1e-3)


class TestDegenerateSpectrum:
    """The all-zero-eigenvalue fallback in spectral_analysis."""

    def test_edgeless_sheaf_has_zero_connectivity(self):
        # A sheaf with nodes but no edges has the zero Laplacian, so every
        # eigenvalue is zero: algebraic connectivity and spectral gap must
        # fall back to 0.0 and the whole space is the kernel.
        graph = nx.DiGraph()
        graph.add_nodes_from(["A", "B"])
        sheaf = Sheaf(graph)
        summary = spectral_analysis(sheaf, stalk_dim=2)
        assert summary.algebraic_connectivity == 0.0
        assert summary.spectral_gap == 0.0
        assert summary.kernel_dimension == 4  # 2 nodes * stalk_dim 2
