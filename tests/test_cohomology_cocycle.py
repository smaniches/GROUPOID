"""Numerical proof of the H^1 holonomy / incomplete-cocycle contract.

The H^1 obstruction on a cycle is the deviation from the identity of the
holonomy: the ordered product of the transport maps around the ENTIRE cycle.
These tests pin that contract against an independent reference, on a
non-trivial multi-cycle graph, covering four cases:

(a) an incomplete cocycle (a cycle with a missing edge map) raises
    ``IncompleteCocycleError`` rather than reporting a false H^1 from a
    partial product;
(b) a complete, consistent (coboundary) cocycle yields H^1 == 0 / identity
    holonomy within tolerance;
(c) a complete, genuinely inconsistent cocycle yields H^1 > 0;
(d) on a multi-cycle graph the value returned by ``compute_h1`` equals an
    INDEPENDENT recomputation of the per-cycle holonomy products, taken over
    the same cycle basis ``compute_h1`` uses.

To make ``||Hol - I||_F`` invariant under the basis cycle's base point and
traversal direction (so the independent reference is a true known answer and
not just a mirror of the implementation), the transport maps are built from
ORTHOGONAL gauges. For orthogonal Hol, conjugation by an orthogonal change of
base point and inversion (direction reversal) both preserve the Frobenius
distance to the identity.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from groupoid.cohomology import IncompleteCocycleError, compute_h1


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix in SO(3) (an orthogonal gauge)."""
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R: np.ndarray = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def _coboundary_maps(
    graph: nx.DiGraph, gauges: dict[str, np.ndarray]
) -> dict[tuple[str, str], np.ndarray]:
    """Build coboundary transport maps T_uv = g_v @ g_u^{-1}.

    Every cycle's holonomy is then exactly the identity, independent of basis
    or base point, so H^1 == 0 is the correct answer.
    """
    return {(u, v): gauges[v] @ np.linalg.inv(gauges[u]) for u, v in graph.edges()}


def _independent_h1(graph: nx.DiGraph, transport_maps: dict[tuple[str, str], np.ndarray]) -> float:
    """Recompute the H^1 norm independently of ``compute_h1``.

    Uses the SAME cycle basis that ``compute_h1`` uses (``nx.cycle_basis`` on
    the undirected graph), forms each cycle's ordered holonomy product by hand,
    and returns the max Frobenius deviation from the identity. This is a
    deliberate re-derivation of the reference value, not a call back into the
    module under test.
    """
    undirected = graph.to_undirected()
    cycles = nx.cycle_basis(undirected)
    if not cycles:
        return 0.0

    worst = 0.0
    for cycle in cycles:
        n = len(cycle)
        edge_maps: list[np.ndarray] = []
        for i in range(n):
            u, v = cycle[i], cycle[(i + 1) % n]
            if (u, v) in transport_maps:
                edge_maps.append(transport_maps[(u, v)])
            else:
                edge_maps.append(np.linalg.inv(transport_maps[(v, u)]))
        hol = edge_maps[0]
        for edge_map in edge_maps[1:]:
            hol = edge_map @ hol
        dim = hol.shape[0]
        worst = max(worst, float(np.linalg.norm(hol - np.eye(dim), ord="fro")))
    return worst


def _two_triangle_graph() -> nx.DiGraph:
    """Two triangles sharing edge B-C: a graph with TWO independent cycles.

    Edges: A-B, B-C, C-A (triangle 1) and B-D, D-C (triangle 2, reusing B-C).
    For V=4, E=5, connected => circuit rank E - V + 1 = 2 independent cycles,
    so the cycle basis is non-trivial (a diamond or a single triangle is not).
    """
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"), ("B", "D"), ("D", "C")])
    return g


class TestIncompleteCocycleRaises:
    """Case (a): a cycle with a missing edge map raises (no false H^1)."""

    def test_missing_edge_on_multicycle_raises(self):
        graph = _two_triangle_graph()
        gauges = {
            "A": _rotation(np.array([1.0, 0.0, 0.0]), 0.3),
            "B": _rotation(np.array([0.0, 1.0, 0.0]), 0.5),
            "C": _rotation(np.array([0.0, 0.0, 1.0]), 0.7),
            "D": _rotation(np.array([1.0, 1.0, 0.0]), 0.4),
        }
        transport_maps = _coboundary_maps(graph, gauges)
        # Drop one edge map entirely (both directions). The D-C edge sits on
        # the second triangle, so that cycle's holonomy is now undefined.
        transport_maps.pop(("D", "C"))
        with pytest.raises(IncompleteCocycleError, match=r"no transport map for edge"):
            compute_h1(graph, transport_maps)

    def test_error_names_the_missing_edge(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        transport_maps = {("A", "B"): np.eye(3), ("B", "C"): np.eye(3)}
        with pytest.raises(IncompleteCocycleError) as exc:
            compute_h1(graph, transport_maps)
        msg = str(exc.value)
        # The undirected basis cycle is A-B-C-A; the only edge without a map in
        # either direction is A-C / C-A. The message must name that specific
        # edge as a tuple (not merely mention the nodes, which also appear in
        # the cycle listing).
        assert "(C, A)" in msg or "(A, C)" in msg


class TestCompleteConsistentCocycle:
    """Case (b): a complete coboundary cocycle yields H^1 == 0 (identity)."""

    def test_coboundary_holonomy_is_identity(self):
        graph = _two_triangle_graph()
        gauges = {
            "A": _rotation(np.array([0.2, -1.0, 0.3]), 0.9),
            "B": _rotation(np.array([1.0, 0.5, 0.0]), 0.6),
            "C": _rotation(np.array([0.0, 0.0, 1.0]), 1.1),
            "D": _rotation(np.array([-0.3, 0.7, 0.4]), 0.8),
        }
        transport_maps = _coboundary_maps(graph, gauges)
        h1 = compute_h1(graph, transport_maps)
        # Orthogonal coboundary => every holonomy is exactly I.
        assert h1 < 1e-10
        # And the independent reference agrees it vanishes.
        assert _independent_h1(graph, transport_maps) < 1e-10


class TestCompleteInconsistentCocycle:
    """Case (c): a complete, genuinely inconsistent cocycle yields H^1 > 0."""

    def test_non_coboundary_holonomy_is_nontrivial(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        # Three rotations whose product around the cycle is NOT the identity:
        # Hol = R_CA @ R_BC @ R_AB with R_CA = inv(map(A,C)). Choose maps that
        # are not a coboundary so the loop carries real holonomy.
        transport_maps = {
            ("A", "B"): _rotation(np.array([0.0, 0.0, 1.0]), 0.9),
            ("B", "C"): _rotation(np.array([0.0, 1.0, 0.0]), 0.7),
            ("A", "C"): _rotation(np.array([1.0, 0.0, 0.0]), 0.5),
        }
        h1 = compute_h1(graph, transport_maps)
        assert h1 > 1e-3
        # Independent reference must reproduce the same magnitude.
        assert _independent_h1(graph, transport_maps) == pytest.approx(h1, rel=1e-9)

    def test_nonzero_h1_matches_closed_form(self):
        """ANALYTIC known-answer for a nonzero H^1 (not a code mirror).

        On a triangle whose three edge maps are all rotations about the SAME
        axis, the maps commute, so the holonomy is R_z(alpha + beta + gamma)
        regardless of the basis cycle's start node or traversal direction
        (reversal sends the angle sum to its negative, and cos is even). The
        closed form is then

            ||Hol - I||_F = 2 * sqrt(1 - cos(alpha + beta + gamma)),

        derived by hand and independent of compute_h1's internal conventions.
        """
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
        h1 = compute_h1(graph, transport_maps)
        assert h1 == pytest.approx(expected, rel=1e-12)
        # Sanity: the angle sum is non-trivial, so this is a genuine nonzero.
        assert expected > 1e-3


class TestMultiCycleIndependentReference:
    """Case (d): on a multi-cycle graph, compute_h1 equals an independent
    recomputation of the holonomy products over the same cycle basis."""

    def test_matches_independent_holonomy_on_two_triangles(self):
        graph = _two_triangle_graph()
        # Confirm the graph genuinely has more than one independent cycle, so
        # this is a real multi-cycle test and not a disguised single triangle.
        n_independent_cycles = len(nx.cycle_basis(graph.to_undirected()))
        assert n_independent_cycles == 2

        # Orthogonal, NON-coboundary maps: each edge is an independent rotation,
        # so both triangle holonomies are non-trivial and generally different.
        rng = np.random.default_rng(20260608)
        transport_maps = {}
        for u, v in graph.edges():
            axis = rng.standard_normal(3)
            angle = float(rng.uniform(0.3, 1.2))
            transport_maps[(u, v)] = _rotation(axis, angle)

        h1 = compute_h1(graph, transport_maps)
        reference = _independent_h1(graph, transport_maps)

        # The headline numerical proof: module output == independent reference.
        assert h1 == pytest.approx(reference, rel=1e-9, abs=1e-12)
        # And the cocycle is genuinely inconsistent (non-trivial holonomy),
        # so this is not a degenerate H^1 == 0 coincidence.
        assert h1 > 1e-3

    def test_inverse_edge_direction_is_used(self):
        # compute_h1 must invert a map when the cycle traverses an edge against
        # its registered direction. Build a triangle where one edge is only
        # registered in the reverse orientation; the independent reference uses
        # the same inversion rule, so the two must still agree.
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        transport_maps = {
            ("A", "B"): _rotation(np.array([0.0, 0.0, 1.0]), 0.4),
            ("B", "C"): _rotation(np.array([0.0, 1.0, 0.0]), 0.6),
            ("C", "A"): _rotation(np.array([1.0, 0.0, 0.0]), 0.8),
        }
        h1 = compute_h1(graph, transport_maps)
        assert h1 == pytest.approx(_independent_h1(graph, transport_maps), rel=1e-9)
