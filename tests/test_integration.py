"""End-to-end integration test for the supported GROUPOID pipeline.

Simulates a 4-client aggregation round on S^2 using explicit SO(3) point
actions. The cycle-basis holonomy defect is checked, the Karcher mean is
computed in a common frame, and the result is returned through inverse point
actions.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from groupoid.aggregation import TransportGroupoidAggregator
from groupoid.cohomology import cycle_basis_holonomy_defect
from groupoid.groupoid import Morphism, compose, inverse
from groupoid.laplacian import sheaf_diffusion_step, spectral_analysis
from groupoid.sheaf import Sheaf


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula for a 3D rotation matrix."""
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    result: np.ndarray = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return result


class TestFederatedPipeline:
    """Full aggregation round on the 2-sphere with four clients."""

    def setup_method(self):
        from geomstats.geometry.hypersphere import Hypersphere

        self.manifold = Hypersphere(dim=2)
        self.graph = nx.DiGraph()
        self.graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])

        np.random.seed(42)
        self.gauges = {}
        for node in ["A", "B", "C", "D"]:
            axis = np.random.randn(3)
            angle = np.random.uniform(0, np.pi / 4)
            self.gauges[node] = _rotation_matrix(axis, angle)

        self.transport_maps = {}
        for u, v in self.graph.edges():
            self.transport_maps[(u, v)] = self.gauges[v] @ np.linalg.inv(self.gauges[u])

    def test_cycle_basis_defect_vanishes_on_coboundary(self):
        defect = cycle_basis_holonomy_defect(self.graph, self.transport_maps)
        assert defect < 1e-8

    def test_full_aggregation_round(self):
        aggregator = TransportGroupoidAggregator(
            manifold=self.manifold,
            graph=self.graph,
            base_node="A",
        )
        for (u, v), matrix in self.transport_maps.items():
            aggregator.register_transport(u, v, matrix)

        client_params = {}
        base_point = np.array([0.0, 0.0, 1.0])
        for node in ["A", "B", "C", "D"]:
            perturbation = np.random.randn(3) * 0.05
            point = base_point + perturbation
            point = point / np.linalg.norm(point)
            client_params[node] = point

        result = aggregator.aggregate(client_params)

        assert result.passes_consistency_threshold
        assert result.cycle_basis_holonomy_defect < 1e-6
        assert result.global_params.shape == (3,)
        assert len(result.local_updates) == 4
        assert self.manifold.belongs(result.global_params, atol=1e-4)
        assert all(
            self.manifold.belongs(point, atol=1e-4)
            for point in result.local_updates.values()
        )

        for node, local_point in result.local_updates.items():
            if node == "A":
                np.testing.assert_allclose(local_point, result.global_params, atol=1e-10)
                continue
            transform = aggregator._get_transport_to_base(node)
            assert transform is not None
            np.testing.assert_allclose(transform @ local_point, result.global_params, atol=1e-8)

    def test_sheaf_laplacian_spectrum(self):
        sheaf = Sheaf(self.graph)
        for (u, v), matrix in self.transport_maps.items():
            sheaf.set_restriction_map(u, v, matrix)

        summary = spectral_analysis(sheaf, stalk_dim=3)
        assert summary.kernel_dimension >= 1
        assert summary.spectral_gap >= 0
        assert summary.algebraic_connectivity >= 0
        assert len(summary.eigenvalues) == 4 * 3

    def test_sheaf_diffusion_convergence(self):
        sheaf = Sheaf(self.graph)
        for (u, v), matrix in self.transport_maps.items():
            sheaf.set_restriction_map(u, v, matrix)

        sections = {node: np.random.randn(3) for node in self.graph.nodes()}
        for _ in range(200):
            sections = sheaf_diffusion_step(sheaf, sections, stalk_dim=3, step_size=0.05)

        for u, v in self.graph.edges():
            restriction = sheaf.get_restriction_map(u, v)
            residual = np.linalg.norm(restriction @ sections[u] - sections[v])
            assert residual < 0.5, f"Diffusion did not converge: edge ({u},{v}), {residual=}"

    def test_morphism_round_trip(self):
        for (u, v), matrix in self.transport_maps.items():
            morphism = Morphism(source=u, target=v, transport_map=matrix)
            round_trip = compose(morphism, inverse(morphism))
            np.testing.assert_allclose(round_trip.transport_map, np.eye(3), atol=1e-10)
            assert round_trip.source == u
            assert round_trip.target == u

    def test_forward_direction_transport_traversal(self):
        graph = nx.DiGraph([("A", "B")])
        aggregator = TransportGroupoidAggregator(
            manifold=self.manifold,
            graph=graph,
            base_node="B",
        )
        aggregator.register_transport("A", "B", np.eye(3))

        forward_map = aggregator._get_transport_to_base("A")
        np.testing.assert_allclose(forward_map, np.eye(3), atol=1e-12)

        north = np.array([0.0, 0.0, 1.0])
        result = aggregator.aggregate({"A": north, "B": north})
        assert self.manifold.belongs(result.global_params, atol=1e-4)

    def test_weighted_aggregation_pulls_mean_toward_heavy_client(self):
        graph = nx.DiGraph([("A", "B")])
        aggregator = TransportGroupoidAggregator(
            manifold=self.manifold,
            graph=graph,
            base_node="A",
        )
        aggregator.register_transport("A", "B", np.eye(3))

        params = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([1.0, 0.0, 0.0]),
        }
        unweighted = aggregator.aggregate(params)
        weighted = aggregator.aggregate(params, weights={"A": 100.0, "B": 1.0})

        assert weighted.passes_consistency_threshold
        assert self.manifold.belongs(weighted.global_params, atol=1e-4)
        assert np.dot(weighted.global_params, params["A"]) > np.dot(
            unweighted.global_params,
            params["A"],
        )

    def test_nonzero_cycle_defect_flags_threshold(self):
        graph = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
        aggregator = TransportGroupoidAggregator(
            manifold=self.manifold,
            graph=graph,
            base_node="A",
            consistency_threshold=1e-6,
        )
        skew = np.diag([2.0, 1.0, 1.0])
        aggregator.register_transport("A", "B", skew)
        aggregator.register_transport("B", "C", np.eye(3))
        aggregator.register_transport("A", "C", np.eye(3))

        north = np.array([0.0, 0.0, 1.0])
        result = aggregator.aggregate({"A": north, "B": north, "C": north})
        assert not result.passes_consistency_threshold
        assert result.cycle_basis_holonomy_defect > aggregator.consistency_threshold

    def test_multiple_aggregation_rounds(self):
        aggregator = TransportGroupoidAggregator(
            manifold=self.manifold,
            graph=self.graph,
            base_node="A",
        )
        for (u, v), matrix in self.transport_maps.items():
            aggregator.register_transport(u, v, matrix)

        client_params = {}
        for node in ["A", "B", "C", "D"]:
            point = np.random.randn(3)
            client_params[node] = point / np.linalg.norm(point)

        for _ in range(5):
            result = aggregator.aggregate(client_params)
            client_params = result.local_updates

        points = list(client_params.values())
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(points[i] - points[j])
                assert distance < 1.0, f"Clients did not converge after 5 rounds: {distance=}"
