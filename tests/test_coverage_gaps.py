"""Focused behavioral tests for previously uncovered code paths.

These tests target real, breakable behavior that the smoke and integration
suites do not exercise: error branches, weighted aggregation, optimizer paths,
the tangent-transport operator, and the degenerate sheaf-Laplacian spectrum.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere

from groupoid.aggregation import TransportGroupoidAggregator
from groupoid.cohomology import IncompleteCocycleError, cycle_basis_holonomy_defect
from groupoid.groupoid import CompositionError, Morphism, compose
from groupoid.laplacian import spectral_analysis
from groupoid.optimizer import RiemannianAdam, RiemannianSGD, curvature_adaptive_lr
from groupoid.sheaf import Sheaf
from groupoid.transport import compute_tangent_transport_matrix


class TestGroupoidErrors:
    def test_compose_mismatched_raises(self):
        f = Morphism(source="A", target="B", transport_map=np.eye(2))
        g = Morphism(source="C", target="D", transport_map=np.eye(2))
        with pytest.raises(CompositionError, match="Cannot compose"):
            compose(f, g)


class TestCycleDefectMissingEdge:
    def test_missing_edge_in_cycle_raises(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        transport_maps = {
            ("A", "B"): np.eye(2),
            ("B", "C"): np.eye(2),
        }
        with pytest.raises(IncompleteCocycleError, match="no transport map for edge"):
            cycle_basis_holonomy_defect(graph, transport_maps)


class TestAggregationPaths:
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
        unweighted = agg.aggregate(params)
        heavy_a = agg.aggregate(params, weights={"A": 100.0, "B": 1.0})
        assert heavy_a.passes_consistency_threshold
        assert np.dot(heavy_a.global_params, params["A"]) > np.dot(
            unweighted.global_params,
            params["A"],
        )

    def test_base_node_identity_short_circuit(self):
        agg = self._setup(np.eye(3))
        params = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([0.0, 0.0, 1.0]),
        }
        result = agg.aggregate(params)
        assert result.transport_residuals["A"] == 0.0
        assert agg._get_transport_to_base("A") is None

    def test_nonzero_cycle_defect_warning_path(self):
        manifold = Hypersphere(dim=2)
        graph = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
        agg = TransportGroupoidAggregator(
            manifold=manifold,
            graph=graph,
            base_node="A",
            consistency_threshold=1e-6,
        )
        skew = np.diag([2.0, 1.0, 1.0])
        agg.register_transport("A", "B", skew)
        agg.register_transport("B", "C", np.eye(3))
        agg.register_transport("A", "C", np.eye(3))
        north = np.array([0.0, 0.0, 1.0])
        result = agg.aggregate({"A": north, "B": north, "C": north})
        assert not result.passes_consistency_threshold
        assert result.cycle_basis_holonomy_defect > agg.consistency_threshold


class TestOptimizerPaths:
    def test_sgd_momentum_accumulates(self):
        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        grad = np.array([0.1, 0.2, 0.3])
        opt = RiemannianSGD(manifold=manifold, lr=0.01, momentum=0.9)
        p1 = opt.step(point, grad)
        assert opt._velocity is not None
        p2 = opt.step(p1, grad)
        assert manifold.belongs(p2, atol=1e-4)
        assert manifold.is_tangent(opt._velocity, p2, atol=1e-4)

    def test_adam_second_step_updates_moment(self):
        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        grad = np.array([0.1, 0.2, 0.3])
        opt = RiemannianAdam(manifold=manifold, lr=0.01)
        p1 = opt.step(point, grad)
        p2 = opt.step(p1, grad)
        assert opt._t == 2
        assert manifold.belongs(p2, atol=1e-4)

    def test_curvature_adaptive_lr_damps_on_sphere(self):
        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        tangent = manifold.to_tangent(np.array([1.0, 0.0, 0.0]), point)
        adapted = curvature_adaptive_lr(manifold, point, 0.1, tangent)
        assert 0.0 < adapted <= 0.1

    def test_curvature_adaptive_lr_falls_back_without_curvature(self):
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


class TestTangentTransportOperator:
    def test_pole_operator_preserves_tested_tangent_norm(self):
        manifold = Hypersphere(dim=2)
        base = np.array([0.0, 0.0, 1.0])
        end = np.array([0.5, 0.0, np.sqrt(3) / 2])
        end = end / np.linalg.norm(end)
        operator = compute_tangent_transport_matrix(
            manifold,
            base,
            end,
            method="pole",
            n_rungs=4,
        )
        assert operator.shape == (3, 3)
        tangent = manifold.to_tangent(np.array([1.0, 0.0, 0.0]), base)
        transported = operator @ tangent
        assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(tangent), rel=1e-3)

    def test_schild_operator_branch_is_finite_on_tested_tangent(self):
        manifold = Hypersphere(dim=2)
        base = np.array([0.0, 0.0, 1.0])
        end = np.array([0.5, 0.0, np.sqrt(3) / 2])
        end = end / np.linalg.norm(end)
        operator = compute_tangent_transport_matrix(
            manifold,
            base,
            end,
            method="schild",
            n_rungs=4,
        )
        assert operator.shape == (3, 3)
        assert np.all(np.isfinite(operator))
        tangent = manifold.to_tangent(np.array([1.0, 0.0, 0.0]), base)
        transported = operator @ tangent
        assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(tangent), rel=1e-3)


class TestDegenerateSpectrum:
    def test_edgeless_sheaf_has_zero_connectivity(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(["A", "B"])
        sheaf = Sheaf(graph)
        summary = spectral_analysis(sheaf, stalk_dim=2)
        assert summary.algebraic_connectivity == 0.0
        assert summary.spectral_gap == 0.0
        assert summary.kernel_dimension == 4
