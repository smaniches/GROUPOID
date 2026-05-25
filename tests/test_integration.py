"""End-to-end integration test for the GROUPOID federated pipeline.

Simulates a 4-client federated learning round on the 2-sphere:
1. Each client has local model parameters on S^2
2. Transport maps (rotations) connect client frames
3. Cohomological consistency is verified
4. Karcher mean aggregates the transported parameters
5. Results are distributed back to each client
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from groupoid.aggregation import TransportGroupoidAggregator
from groupoid.cohomology import compute_h1
from groupoid.groupoid import Morphism, compose, inverse
from groupoid.laplacian import sheaf_diffusion_step, spectral_analysis
from groupoid.sheaf import Sheaf


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula for 3D rotation matrix."""
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
    """Full federated round on the 2-sphere with 4 clients."""

    def setup_method(self):
        from geomstats.geometry.hypersphere import Hypersphere

        self.manifold = Hypersphere(dim=2)

        # Build a diamond graph: A-B, A-C, B-D, C-D
        self.graph = nx.DiGraph()
        self.graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])

        # Create transport maps as rotations (SO(3) elements)
        # Consistent coboundary: T_ij = g_j @ inv(g_i)
        np.random.seed(42)
        self.gauges = {}
        for node in ["A", "B", "C", "D"]:
            axis = np.random.randn(3)
            angle = np.random.uniform(0, np.pi / 4)
            self.gauges[node] = _rotation_matrix(axis, angle)

        self.transport_maps = {}
        for u, v in self.graph.edges():
            self.transport_maps[(u, v)] = self.gauges[v] @ np.linalg.inv(self.gauges[u])

    def test_cohomological_consistency(self):
        """H^1 should vanish for coboundary transport maps."""
        h1 = compute_h1(self.graph, self.transport_maps)
        assert h1 < 1e-8

    def test_full_aggregation_round(self):
        """Run a complete federated aggregation round."""
        aggregator = TransportGroupoidAggregator(
            manifold=self.manifold,
            graph=self.graph,
            base_node="A",
        )

        for (u, v), T in self.transport_maps.items():
            aggregator.register_transport(u, v, T)

        # Each client has a point near the north pole
        client_params = {}
        base_point = np.array([0.0, 0.0, 1.0])
        for node in ["A", "B", "C", "D"]:
            perturbation = np.random.randn(3) * 0.05
            point = base_point + perturbation
            point = point / np.linalg.norm(point)
            client_params[node] = point

        result = aggregator.aggregate(client_params)

        assert result.is_consistent
        assert result.h1_norm < 1e-6
        assert result.global_params.shape == (3,)
        assert len(result.local_updates) == 4

        # Global params should be on the manifold
        assert abs(np.linalg.norm(result.global_params) - 1.0) < 1e-4

    def test_sheaf_laplacian_spectrum(self):
        """Sheaf Laplacian kernel should be nontrivial for consistent sheaf."""
        sheaf = Sheaf(self.graph)
        for (u, v), T in self.transport_maps.items():
            sheaf.set_restriction_map(u, v, T)

        summary = spectral_analysis(sheaf, stalk_dim=3)

        # Kernel dimension should be > 0 for a consistent sheaf
        assert summary.kernel_dimension >= 1
        assert summary.spectral_gap >= 0
        assert summary.algebraic_connectivity >= 0
        assert len(summary.eigenvalues) == 4 * 3

    def test_sheaf_diffusion_convergence(self):
        """Sheaf diffusion should drive sections toward consensus."""
        sheaf = Sheaf(self.graph)
        for (u, v), T in self.transport_maps.items():
            sheaf.set_restriction_map(u, v, T)

        # Start with random sections
        sections = {node: np.random.randn(3) for node in self.graph.nodes()}

        # Run diffusion for many steps
        for _ in range(200):
            sections = sheaf_diffusion_step(sheaf, sections, stalk_dim=3, step_size=0.05)

        # After diffusion, sections should be approximately consistent:
        # R_{uv} @ s(u) approx s(v) for all edges
        for u, v in self.graph.edges():
            R = sheaf.get_restriction_map(u, v)
            transported = R @ sections[u]
            residual = np.linalg.norm(transported - sections[v])
            assert (
                residual < 0.5
            ), f"Diffusion did not converge: edge ({u},{v}), residual={residual}"

    def test_morphism_round_trip(self):
        """Composing a morphism with its inverse gives identity."""
        for (u, v), T in self.transport_maps.items():
            m = Morphism(source=u, target=v, transport_map=T)
            m_inv = inverse(m)
            round_trip = compose(m, m_inv)

            np.testing.assert_allclose(
                round_trip.transport_map,
                np.eye(3),
                atol=1e-10,
            )
            assert round_trip.source == u
            assert round_trip.target == u

    def test_multiple_aggregation_rounds(self):
        """Multiple rounds should maintain consistency."""
        aggregator = TransportGroupoidAggregator(
            manifold=self.manifold,
            graph=self.graph,
            base_node="A",
        )

        for (u, v), T in self.transport_maps.items():
            aggregator.register_transport(u, v, T)

        # Start with spread-out points
        client_params = {}
        for node in ["A", "B", "C", "D"]:
            point = np.random.randn(3)
            point = point / np.linalg.norm(point)
            client_params[node] = point

        # Run 5 rounds, feeding back local updates
        for _ in range(5):
            result = aggregator.aggregate(client_params)
            client_params = result.local_updates

        # After multiple rounds, all clients should be close to each other
        points = list(client_params.values())
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                assert dist < 1.0, f"Clients did not converge after 5 rounds: dist={dist}"
