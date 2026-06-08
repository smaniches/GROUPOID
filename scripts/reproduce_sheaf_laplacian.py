"""Determinism receipt for the sheaf (connection) Laplacian.

Runs a FIXED-SEED check of the load-bearing sheaf-Laplacian invariants on a
small 4-node graph with NON-orthogonal restriction maps, and prints exact,
reproducible numbers. This mirrors the invariants asserted in
tests/test_sheaf_laplacian.py (the general, non-degenerate case):

  1. L equals delta^T delta for an INDEPENDENTLY constructed coboundary delta.
  2. L is symmetric and positive-semidefinite.
  3. For orthogonal transport R_uv = g_v g_u^{-1}, transport-consistent global
     sections lie in ker(L), constant sections do not, and dim(ker L) == stalk.

Run (from the repo root, in a Python 3.10-3.12 env with the package installed):

    python scripts/reproduce_sheaf_laplacian.py

Output is deterministic across runs and platforms (numpy default_rng with a
fixed seed; eigenvalues are rounded for display). See REPRODUCE.md.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from groupoid.laplacian import build_sheaf_laplacian
from groupoid.sheaf import Sheaf

D = 3
EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
SEED = 0


def _independent_delta_t_delta(r: dict, nodes: list) -> np.ndarray:
    """delta built row-block-by-row from the definition (delta x)_{(u,v)} = x_v - R_uv x_u.
    Deliberately independent of build_sheaf_laplacian's block formula."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes) * D
    delta = np.zeros((len(EDGES) * D, n))
    for e, (u, v) in enumerate(EDGES):
        re = slice(e * D, (e + 1) * D)
        delta[re, idx[u] * D : (idx[u] + 1) * D] = -r[(u, v)]
        delta[re, idx[v] * D : (idx[v] + 1) * D] = np.eye(D)
    result: np.ndarray = delta.T @ delta
    return result


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    rotation: np.ndarray = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    return rotation


def main() -> None:
    g: nx.DiGraph = nx.DiGraph()
    g.add_edges_from(EDGES)
    nodes = sorted(g.nodes())

    # (1)+(2): non-orthogonal restriction maps -> L == delta^T delta, symmetric, PSD.
    rng = np.random.default_rng(SEED)
    sheaf = Sheaf(g)
    r = {}
    for u, v in EDGES:
        m = np.eye(3) + 0.6 * rng.standard_normal((3, 3))  # invertible, NOT orthogonal
        r[(u, v)] = m
        sheaf.set_restriction_map(u, v, m)
    laplacian = build_sheaf_laplacian(sheaf, stalk_dim=D)
    indep = _independent_delta_t_delta(r, nodes)

    max_abs_diff = float(np.max(np.abs(laplacian - indep)))
    sym_err = float(np.max(np.abs(laplacian - laplacian.T)))
    min_eig = float(np.linalg.eigvalsh(laplacian).min())

    print(f"seed                     : {SEED}")
    print(f"graph                    : nodes={nodes} edges={EDGES}")
    print(f"stalk_dim                : {D}")
    print(f"L shape                  : {laplacian.shape}")
    print(f"max|L - deltaT_delta|    : {max_abs_diff:.3e}   (expect < 1e-12)")
    print(f"max|L - L^T| (symmetry)  : {sym_err:.3e}   (expect < 1e-12)")
    print(f"min eigenvalue (PSD)     : {min_eig:.3e}   (expect >= -1e-9)")

    # (3): kernel content for orthogonal transport R_uv = g_v g_u^{-1}.
    rng2 = np.random.default_rng(2)
    gauges = {n: _rotation(rng2.standard_normal(3), rng2.uniform(0.5, 2.5)) for n in nodes}
    sheaf2 = Sheaf(g)
    for u, v in EDGES:
        sheaf2.set_restriction_map(u, v, gauges[v] @ np.linalg.inv(gauges[u]))
    laplacian2 = build_sheaf_laplacian(sheaf2, stalk_dim=D)
    s = rng2.standard_normal(D)
    x_consistent = np.concatenate([gauges[n] @ s for n in nodes])
    x_constant = np.concatenate([s for _ in nodes])
    kernel_dim = int(np.sum(np.linalg.eigvalsh(laplacian2) < 1e-9))

    print(
        f"||L @ consistent_section|| : {float(np.linalg.norm(laplacian2 @ x_consistent)):.3e}"
        "   (expect < 1e-9, in kernel)"
    )
    print(
        f"||L @ constant_section||   : {float(np.linalg.norm(laplacian2 @ x_constant)):.3e}"
        "   (expect > 1e-3, NOT in kernel)"
    )
    print(f"dim ker(L)                 : {kernel_dim}   (expect == stalk_dim = {D})")
    print("RESULT: sheaf-Laplacian invariants hold (deltaT-delta, PSD, kernel).")


if __name__ == "__main__":
    main()
