"""Property-based tests for mathematical invariants of GROUPOID.

Uses Hypothesis to verify:
1. Karcher mean idempotency on constant inputs
2. Groupoid composition associativity
3. Vanishing cycle-basis holonomy defect on coboundaries
4. Sheaf restriction map functoriality
"""

from __future__ import annotations

import hypothesis.strategies as st
import networkx as nx
import numpy as np
from hypothesis import HealthCheck, given, settings

from groupoid.cohomology import cycle_basis_holonomy_defect
from groupoid.groupoid import Morphism, compose
from groupoid.manifold import karcher_mean
from groupoid.sheaf import Sheaf


@st.composite
def invertible_matrices(draw, dim=None):
    """Generate a random orthogonal matrix via QR decomposition."""
    if dim is None:
        dim = draw(st.integers(min_value=2, max_value=6))
    entries = draw(
        st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=dim * dim,
            max_size=dim * dim,
        )
    )
    matrix = np.array(entries, dtype=np.float64).reshape(dim, dim)
    matrix = matrix + dim * np.eye(dim)
    Q, _R = np.linalg.qr(matrix)
    return Q


@given(
    dim=st.integers(min_value=2, max_value=5),
    n_copies=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_karcher_mean_of_identical_points(dim, n_copies):
    from geomstats.geometry.hypersphere import Hypersphere

    manifold = Hypersphere(dim=dim)
    point = manifold.random_point()
    points = np.stack([point] * n_copies)
    mean = karcher_mean(manifold, points)
    similarity = np.abs(np.dot(mean.flatten(), point.flatten()))
    assert similarity > 1.0 - 1e-4


@given(data=st.data(), dim=st.integers(min_value=2, max_value=5))
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_groupoid_composition_associativity(data, dim):
    q1 = data.draw(invertible_matrices(dim=dim))
    q2 = data.draw(invertible_matrices(dim=dim))
    q3 = data.draw(invertible_matrices(dim=dim))

    f = Morphism(source="A", target="B", transport_map=q1)
    g = Morphism(source="B", target="C", transport_map=q2)
    h = Morphism(source="C", target="D", transport_map=q3)

    left = compose(compose(f, g), h)
    right = compose(f, compose(g, h))
    np.testing.assert_allclose(left.transport_map, right.transport_map, atol=1e-10)
    assert left.source == "A"
    assert left.target == "D"


@given(data=st.data(), dim=st.integers(min_value=2, max_value=5))
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_cycle_basis_holonomy_defect_vanishes_on_coboundary(data, dim):
    """The implemented defect must vanish on a complete coboundary."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

    g_a = data.draw(invertible_matrices(dim=dim))
    g_b = data.draw(invertible_matrices(dim=dim))
    g_c = data.draw(invertible_matrices(dim=dim))
    transport_maps = {
        ("A", "B"): g_b @ np.linalg.inv(g_a),
        ("B", "C"): g_c @ np.linalg.inv(g_b),
        ("A", "C"): g_c @ np.linalg.inv(g_a),
    }

    defect = cycle_basis_holonomy_defect(graph, transport_maps)
    assert defect < 1e-8


@given(data=st.data(), dim=st.integers(min_value=2, max_value=5))
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_restriction_maps_compose(data, dim):
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

    r_ab = data.draw(invertible_matrices(dim=dim))
    r_bc = data.draw(invertible_matrices(dim=dim))
    r_ac = r_bc @ r_ab

    sheaf = Sheaf(graph)
    sheaf.set_restriction_map("A", "B", r_ab)
    sheaf.set_restriction_map("B", "C", r_bc)
    sheaf.set_restriction_map("A", "C", r_ac)

    entries = data.draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    section = np.array(entries, dtype=np.float64)
    via_path = sheaf.restrict_along_path(section, ["A", "B", "C"])
    direct = sheaf.restrict(section, "A", "C")
    np.testing.assert_allclose(via_path, direct, atol=1e-10)
