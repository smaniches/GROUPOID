"""Sheaf Laplacian for federated consensus and spectral analysis.

The sheaf Laplacian generalizes the graph Laplacian by incorporating
the restriction maps of a cellular sheaf. Its spectrum reveals the
structure of the agreement space:

- Kernel of L = space of global sections (consistent models)
- Smallest nonzero eigenvalue = algebraic connectivity of the sheaf
  (how fast consensus can be reached)
- Spectral gap = robustness of the consensus to perturbation

In federated learning, the sheaf Laplacian governs the diffusion
process that drives local models toward global consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from groupoid.sheaf import Sheaf


@dataclass
class SpectralSummary:
    """Spectral decomposition of the sheaf Laplacian."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_gap: float
    algebraic_connectivity: float
    kernel_dimension: int
    consensus_rate: float


def build_sheaf_laplacian(sheaf: Sheaf, stalk_dim: int) -> np.ndarray:
    """Build the sheaf Laplacian matrix.

    For a sheaf F on graph G with n nodes and stalk dimension d,
    the sheaf Laplacian is a (n*d) x (n*d) block matrix defined as:

        L_F = delta^T @ delta

    where delta is the connection coboundary. For each edge (u, v) with
    restriction (transport) map R = R_{uv}: stalk(u) -> stalk(v), the
    coboundary acts as (delta x)_{(u,v)} = x_v - R_{uv} x_u, so
    L = delta^T @ delta has blocks (summed over incident edges):

        L[u,u] += R_{uv}^T @ R_{uv}    (source diagonal)
        L[v,v] += I                    (target diagonal)
        L[u,v] += -R_{uv}^T            (off-diagonal)
        L[v,u] += -R_{uv}              (off-diagonal)

    L is symmetric positive semi-definite for ANY restriction maps (it is
    delta^T delta); its kernel is the space of transport-consistent global
    sections (x_v = R_{uv} x_u on every edge).

    Parameters
    ----------
    sheaf
        A Sheaf instance with restriction maps set.
    stalk_dim
        Dimension of each stalk (vector space at each node).

    Returns
    -------
    np.ndarray
        The sheaf Laplacian matrix of shape (n*d, n*d).
    """
    nodes = sorted(sheaf.graph.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    N = n * stalk_dim

    L = np.zeros((N, N))

    for u, v in sheaf.graph.edges():
        i, j = node_idx[u], node_idx[v]
        R = sheaf.get_restriction_map(u, v)
        i_slice = slice(i * stalk_dim, (i + 1) * stalk_dim)
        j_slice = slice(j * stalk_dim, (j + 1) * stalk_dim)

        # L = delta^T delta for coboundary (delta x)_(u,v) = x_v - R_uv x_u:
        L[i_slice, i_slice] += R.T @ R  # source diagonal: R^T R
        L[j_slice, j_slice] += np.eye(stalk_dim)  # target diagonal: I
        L[i_slice, j_slice] += -R.T  # off-diagonal: -R^T
        L[j_slice, i_slice] += -R  # off-diagonal: -R

    logger.debug("Built sheaf Laplacian: {}x{} ({} nodes, stalk_dim={})", N, N, n, stalk_dim)
    return L


def spectral_analysis(
    sheaf: Sheaf,
    stalk_dim: int,
    tol: float = 1e-10,
) -> SpectralSummary:
    """Compute spectral decomposition of the sheaf Laplacian.

    Parameters
    ----------
    sheaf
        A Sheaf instance with restriction maps.
    stalk_dim
        Dimension of each stalk.
    tol
        Tolerance for identifying zero eigenvalues.

    Returns
    -------
    SpectralSummary
        Full spectral summary including connectivity and consensus rate.
    """
    L = build_sheaf_laplacian(sheaf, stalk_dim)
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort by magnitude
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Kernel dimension (number of zero eigenvalues)
    kernel_dim = int(np.sum(np.abs(eigenvalues) < tol))

    # Spectral gap and algebraic connectivity
    nonzero_eigs = eigenvalues[np.abs(eigenvalues) >= tol]
    if len(nonzero_eigs) > 0:
        algebraic_connectivity = float(nonzero_eigs[0])
        spectral_gap = float(nonzero_eigs[0])
    else:
        algebraic_connectivity = 0.0
        spectral_gap = 0.0

    # Consensus rate: exponential convergence rate of sheaf diffusion
    # x(t+1) = (I - epsilon * L) @ x(t), converges as exp(-lambda_1 * t)
    consensus_rate = algebraic_connectivity

    logger.info(
        "Spectral analysis: kernel_dim={}, spectral_gap={:.4f}, connectivity={:.4f}",
        kernel_dim,
        spectral_gap,
        algebraic_connectivity,
    )

    return SpectralSummary(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        spectral_gap=spectral_gap,
        algebraic_connectivity=algebraic_connectivity,
        kernel_dimension=kernel_dim,
        consensus_rate=consensus_rate,
    )


def sheaf_diffusion_step(
    sheaf: Sheaf,
    sections: dict[str, np.ndarray],
    stalk_dim: int,
    step_size: float = 0.1,
) -> dict[str, np.ndarray]:
    """One step of sheaf diffusion (Laplacian smoothing).

    Drives local sections toward global consistency by flowing along
    the negative gradient of the sheaf Laplacian energy:

        E(x) = x^T L_F x = sum_{(i,j)} ||R_{ij} x_i - x_j||^2

    Parameters
    ----------
    sheaf
        Sheaf with restriction maps.
    sections
        Current section values at each node.
    stalk_dim
        Dimension of each stalk.
    step_size
        Diffusion step size (must be < 1/lambda_max for stability).

    Returns
    -------
    dict[str, np.ndarray]
        Updated section values after one diffusion step.
    """
    L = build_sheaf_laplacian(sheaf, stalk_dim)
    nodes = sorted(sheaf.graph.nodes())

    # Stack sections into vector
    x = np.concatenate([sections[n] for n in nodes])

    # Diffusion step: x' = x - step_size * L @ x
    x_new = x - step_size * L @ x

    # Unstack
    result = {}
    for idx, node in enumerate(nodes):
        result[node] = x_new[idx * stalk_dim : (idx + 1) * stalk_dim]

    return result
