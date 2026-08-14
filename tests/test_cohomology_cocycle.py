"""Numerical validation of the cycle-basis holonomy-defect contract.

These tests validate complete basis-cycle products, exact vanishing on a
coboundary, nonzero known answers, and agreement with an independent
recomputation over the same NetworkX cycle basis. The same-basis comparison is
not a claim of cycle-basis invariance. Orthogonal gauges are used where
base-point conjugation and cycle reversal need to preserve Frobenius magnitude.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from groupoid.cohomology import IncompleteCocycleError, cycle_basis_holonomy_defect


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _coboundary_maps(
    graph: nx.DiGraph,
    gauges: dict[str, np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    return {(u, v): gauges[v] @ np.linalg.inv(gauges[u]) for u, v in graph.edges()}


def _independent_cycle_defect(
    graph: nx.DiGraph,
    transport_maps: dict[tuple[str, str], np.ndarray],
) -> float:
    """Recompute the same-basis holonomy products independently."""
    cycles = nx.cycle_basis(graph.to_undirected())
    if not cycles:
        return 0.0

    worst = 0.0
    for cycle in cycles:
        edge_maps: list[np.ndarray] = []
        for i, u in enumerate(cycle):
            v = cycle[(i + 1) % len(cycle)]
            if (u, v) in transport_maps:
                edge_maps.append(transport_maps[(u, v)])
            else:
                edge_maps.append(np.linalg.inv(transport_maps[(v, u)]))
        holonomy = edge_maps[0]
        for edge_map in edge_maps[1:]:
            holonomy = edge_map @ holonomy
        worst = max(
            worst,
            float(np.linalg.norm(holonomy - np.eye(holonomy.shape[0]), ord="fro")),
        )
    return worst


def _two_triangle_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"), ("B", "D"), ("D", "C")])
    return graph


class TestIncompleteCycleTransportRaises:
    def test_missing_edge_on_multicycle_raises(self):
        graph = _two_triangle_graph()
        gauges = {
            "A": _rotation(np.array([1.0, 0.0, 0.0]), 0.3),
            "B": _rotation(np.array([0.0, 1.0, 0.0]), 0.5),
            "C": _rotation(np.array([0.0, 0.0, 1.0]), 0.7),
            "D": _rotation(np.array([1.0, 1.0, 0.0]), 0.4),
        }
        transport_maps = _coboundary_maps(graph, gauges)
        transport_maps.pop(("D", "C"))
        with pytest.raises(IncompleteCocycleError, match=r"no transport map for edge"):
            cycle_basis_holonomy_defect(graph, transport_maps)

    def test_error_names_the_missing_edge(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        transport_maps = {("A", "B"): np.eye(3), ("B", "C"): np.eye(3)}
        with pytest.raises(IncompleteCocycleError) as exc:
            cycle_basis_holonomy_defect(graph, transport_maps)
        message = str(exc.value)
        assert "(C, A)" in message or "(A, C)" in message


class TestCompleteConsistentTransport:
    def test_coboundary_holonomy_is_identity(self):
        graph = _two_triangle_graph()
        gauges = {
            "A": _rotation(np.array([0.2, -1.0, 0.3]), 0.9),
            "B": _rotation(np.array([1.0, 0.5, 0.0]), 0.6),
            "C": _rotation(np.array([0.0, 0.0, 1.0]), 1.1),
            "D": _rotation(np.array([-0.3, 0.7, 0.4]), 0.8),
        }
        transport_maps = _coboundary_maps(graph, gauges)
        defect = cycle_basis_holonomy_defect(graph, transport_maps)
        assert defect < 1e-10
        assert _independent_cycle_defect(graph, transport_maps) < 1e-10


class TestCompleteInconsistentTransport:
    def test_nonflat_holonomy_is_nontrivial(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        transport_maps = {
            ("A", "B"): _rotation(np.array([0.0, 0.0, 1.0]), 0.9),
            ("B", "C"): _rotation(np.array([0.0, 1.0, 0.0]), 0.7),
            ("A", "C"): _rotation(np.array([1.0, 0.0, 0.0]), 0.5),
        }
        defect = cycle_basis_holonomy_defect(graph, transport_maps)
        assert defect > 1e-3
        assert _independent_cycle_defect(graph, transport_maps) == pytest.approx(
            defect,
            rel=1e-9,
        )

    def test_nonzero_defect_matches_closed_form(self):
        alpha, beta, gamma = 0.4, 0.7, 0.5
        z = np.array([0.0, 0.0, 1.0])
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        transport_maps = {
            ("A", "B"): _rotation(z, alpha),
            ("B", "C"): _rotation(z, beta),
            ("C", "A"): _rotation(z, gamma),
        }
        expected = 2.0 * np.sqrt(1.0 - np.cos(alpha + beta + gamma))
        defect = cycle_basis_holonomy_defect(graph, transport_maps)
        assert defect == pytest.approx(expected, rel=1e-12)
        assert expected > 1e-3


class TestMultiCycleSameBasisReference:
    def test_matches_independent_holonomy_on_two_triangles(self):
        graph = _two_triangle_graph()
        assert len(nx.cycle_basis(graph.to_undirected())) == 2

        rng = np.random.default_rng(20260608)
        transport_maps = {}
        for u, v in graph.edges():
            axis = rng.standard_normal(3)
            angle = float(rng.uniform(0.3, 1.2))
            transport_maps[(u, v)] = _rotation(axis, angle)

        defect = cycle_basis_holonomy_defect(graph, transport_maps)
        reference = _independent_cycle_defect(graph, transport_maps)
        assert defect == pytest.approx(reference, rel=1e-9, abs=1e-12)
        assert defect > 1e-3

    def test_inverse_edge_direction_is_used(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        transport_maps = {
            ("A", "B"): _rotation(np.array([0.0, 0.0, 1.0]), 0.4),
            ("B", "C"): _rotation(np.array([0.0, 1.0, 0.0]), 0.6),
            ("C", "A"): _rotation(np.array([1.0, 0.0, 0.0]), 0.8),
        }
        defect = cycle_basis_holonomy_defect(graph, transport_maps)
        assert defect == pytest.approx(_independent_cycle_defect(graph, transport_maps), rel=1e-9)
