"""Property-based tests for mathematical invariants of GROUPOID.

Uses Hypothesis to verify fundamental algebraic and geometric properties:
1. Karcher mean idempotency on constant inputs
2. Groupoid composition associativity
3. Vanishing first cohomology on coboundaries
4. Sheaf restriction map functoriality
"""

from __future__ import annotations

import hypothesis.strategies as st
import networkx as nx
import numpy as np
from hypothesis import HealthCheck, given, settings

from groupoid.cohomology import compute_h1
from groupoid.groupoid import Morphism, compose
from groupoid.manifold import karcher_mean
from groupoid.sheaf import Sheaf

# ---------------------------------------------------------------------------
# Custom strategies
# ---------------------------------------------------------------------------


@st.composite
def invertible_matrices(draw, dim=None):
    """Generate a random invertible matrix via QR decomposition."""
    if dim is None:
        dim = draw(st.integers(min_value=2, max_value=6))
    entries = draw(
        st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=dim * dim,
            max_size=dim * dim,
        )
    )
    A = np.array(entries, dtype=np.float64).reshape(dim, dim)
    A = A + dim * np.eye(dim)
    Q, R = np.linalg.qr(A)
    return Q


# ---------------------------------------------------------------------------
# Test 1: Karcher mean of identical points = that point
# ---------------------------------------------------------------------------


@given(
    dim=st.integers(min_value=2, max_value=5),
    n_copies=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_karcher_mean_of_identical_points(dim, n_copies):
    """The Karcher mean of n copies of the same point must be that point."""
    from geomstats.geometry.hypersphere import Hypersphere

    manifold = Hypersphere(dim=dim)
    point = manifold.random_point()
    points = np.stack([point] * n_copies)

    mean = karcher_mean(manifold, points)

    similarity = np.abs(np.dot(mean.flatten(), point.flatten()))
    assert (
        similarity > 1.0 - 1e-4
    ), f"Karcher mean of identical points diverged: similarity={similarity}"


# ---------------------------------------------------------------------------
# Test 2: Groupoid composition is associative
# ---------------------------------------------------------------------------


@given(data=st.data(), dim=st.integers(min_value=2, max_value=5))
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_groupoid_composition_associativity(data, dim):
    """Composition of morphisms must be associative: (f;g);h = f;(g;h)."""
    Q1 = data.draw(invertible_matrices(dim=dim))
    Q2 = data.draw(invertible_matrices(dim=dim))
    Q3 = data.draw(invertible_matrices(dim=dim))

    f = Morphism(source="A", target="B", transport_map=Q1)
    g = Morphism(source="B", target="C", transport_map=Q2)
    h = Morphism(source="C", target="D", transport_map=Q3)

    left = compose(compose(f, g), h)
    right = compose(f, compose(g, h))

    np.testing.assert_allclose(
        left.transport_map,
        right.transport_map,
        atol=1e-10,
        err_msg="Groupoid composition is not associative",
    )
    assert left.source == "A"
    assert left.target == "D"


# ---------------------------------------------------------------------------
# Test 3: H^1 = 0 on a globally consistent section (coboundary)
# ---------------------------------------------------------------------------


@given(data=st.data(), dim=st.integers(min_value=2, max_value=5))
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_h1_vanishes_on_coboundary(data, dim):
    """H^1 must vanish when transport maps form a coboundary."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

    g_A = data.draw(invertible_matrices(dim=dim))
    g_B = data.draw(invertible_matrices(dim=dim))
    g_C = data.draw(invertible_matrices(dim=dim))

    transport_maps = {
        ("A", "B"): g_B @ np.linalg.inv(g_A),
        ("B", "C"): g_C @ np.linalg.inv(g_B),
        ("A", "C"): g_C @ np.linalg.inv(g_A),
    }

    h1_norm = compute_h1(graph, transport_maps)

    assert h1_norm < 1e-8, f"H^1 should vanish on coboundary, got {h1_norm}"


# ---------------------------------------------------------------------------
# Test 4: Restriction maps compose correctly (functoriality)
# ---------------------------------------------------------------------------


@given(data=st.data(), dim=st.integers(min_value=2, max_value=5))
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_restriction_maps_compose(data, dim):
    """restrict(A->C) must equal restrict(B->C) . restrict(A->B)."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

    R_AB = data.draw(invertible_matrices(dim=dim))
    R_BC = data.draw(invertible_matrices(dim=dim))
    R_AC = R_BC @ R_AB

    sheaf = Sheaf(graph)
    sheaf.set_restriction_map("A", "B", R_AB)
    sheaf.set_restriction_map("B", "C", R_BC)
    sheaf.set_restriction_map("A", "C", R_AC)

    section_entries = data.draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    section = np.array(section_entries, dtype=np.float64)

    via_path = sheaf.restrict_along_path(section, ["A", "B", "C"])
    direct = sheaf.restrict(section, "A", "C")

    np.testing.assert_allclose(
        via_path,
        direct,
        atol=1e-10,
        err_msg="Restriction maps do not compose correctly",
    )
