"""Correctness tests for the sheaf (connection) Laplacian.

These cover the GENERAL case with NON-orthogonal restriction maps -- the regime
the original orthogonal-only integration test could not exercise, and under which
the previous construction was non-PSD. Each test verifies the builder against an
INDEPENDENT coboundary delta constructed from the definition (not reusing the
builder's block formula), asserts positive-semidefiniteness, and pins the
convention via kernel CONTENT (PSD alone cannot distinguish the correct kernel
from a different-but-PSD one).
"""

import networkx as nx
import numpy as np

from groupoid.laplacian import build_sheaf_laplacian
from groupoid.sheaf import Sheaf

D = 3
EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]


def _graph() -> nx.DiGraph:
    g: nx.DiGraph = nx.DiGraph()
    g.add_edges_from(EDGES)
    return g


def _independent_delta_t_delta(r: dict, nodes: list) -> np.ndarray:
    """delta built row-block-by-row from the definition:
    (delta x)_{(u,v)} = x_v - R_uv x_u. Deliberately does NOT reuse
    build_sheaf_laplacian's block formula, so the comparison is independent."""
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes) * D
    delta = np.zeros((len(EDGES) * D, N))
    for e, (u, v) in enumerate(EDGES):
        re = slice(e * D, (e + 1) * D)
        delta[re, idx[u] * D : (idx[u] + 1) * D] = -r[(u, v)]
        delta[re, idx[v] * D : (idx[v] + 1) * D] = np.eye(D)
    result: np.ndarray = delta.T @ delta
    return result


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    rotation: np.ndarray = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return rotation


def test_laplacian_equals_independent_coboundary_nonorthogonal() -> None:
    """L must equal delta^T delta for an independently-built delta, with
    NON-orthogonal restriction maps."""
    rng = np.random.default_rng(0)
    g = _graph()
    nodes = sorted(g.nodes())
    sheaf = Sheaf(g)
    R = {}
    for u, v in EDGES:
        M = np.eye(3) + 0.6 * rng.standard_normal((3, 3))  # invertible, NOT orthogonal
        R[(u, v)] = M
        sheaf.set_restriction_map(u, v, M)
    L = build_sheaf_laplacian(sheaf, stalk_dim=D)
    np.testing.assert_allclose(L, _independent_delta_t_delta(R, nodes), atol=1e-12)


def test_laplacian_is_psd_for_nonorthogonal_maps() -> None:
    """L = delta^T delta must be symmetric and PSD for arbitrary restriction maps
    (the previous R^T R off-diagonal / R R^T diagonal construction was not)."""
    rng = np.random.default_rng(1)
    g = _graph()
    sheaf = Sheaf(g)
    for u, v in EDGES:
        sheaf.set_restriction_map(u, v, np.eye(3) + 0.6 * rng.standard_normal((3, 3)))
    L = build_sheaf_laplacian(sheaf, stalk_dim=D)
    np.testing.assert_allclose(L, L.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(L).min()) >= -1e-9


def test_kernel_is_transport_consistent_sections() -> None:
    """Convention check via kernel CONTENT: for orthogonal transport
    R_uv = g_v g_u^{-1}, the consistent global sections x_node = g_node @ s lie in
    ker(L); constant sections do NOT (for non-trivial rotations); and the kernel
    dimension equals the stalk dimension (no spurious kernel)."""
    rng = np.random.default_rng(2)
    g = _graph()
    nodes = sorted(g.nodes())
    gauges = {n: _rotation(rng.standard_normal(3), rng.uniform(0.5, 2.5)) for n in nodes}
    sheaf = Sheaf(g)
    for u, v in EDGES:
        sheaf.set_restriction_map(u, v, gauges[v] @ np.linalg.inv(gauges[u]))
    L = build_sheaf_laplacian(sheaf, stalk_dim=D)

    s = rng.standard_normal(D)
    x_consistent = np.concatenate([gauges[n] @ s for n in nodes])
    assert np.linalg.norm(L @ x_consistent) < 1e-9  # consistent sections in kernel

    x_constant = np.concatenate([s for _ in nodes])
    assert np.linalg.norm(L @ x_constant) > 1e-3  # constant sections NOT in kernel

    assert int(np.sum(np.linalg.eigvalsh(L) < 1e-9)) == D  # kernel dim == stalk_dim
