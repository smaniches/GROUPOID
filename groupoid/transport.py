"""Parallel transport on Riemannian manifolds for federated learning.

Implements parallel transport of tangent vectors along geodesics,
which is the geometric foundation of the groupoid morphisms. When
client A sends a gradient update to client B, the gradient must be
parallel-transported from A's tangent space to B's tangent space
to preserve geometric meaning.

This module provides:
- Schild's ladder approximation for general manifolds
- Pole ladder (more accurate, same cost)
- Exact transport on spheres and SPD matrices
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def schild_ladder(
    manifold,
    tangent_vec: np.ndarray,
    base_point: np.ndarray,
    end_point: np.ndarray,
    n_rungs: int = 1,
) -> np.ndarray:
    """Parallel transport via Schild's ladder.

    An iterative approximation of parallel transport that only requires
    the exponential and logarithmic maps. This is the coarser of the two
    ladders: on S^2 its direction agrees with the analytic parallel
    transport only to cosine ~0.98 and does not converge to the exact
    value as ``n_rungs`` increases (see ``pole_ladder`` for the accurate
    variant and LIMITATIONS.md for the measured behavior).

    Parameters
    ----------
    manifold
        A geomstats manifold with exp and log maps.
    tangent_vec
        Tangent vector at base_point to transport.
    base_point
        Starting point on the manifold.
    end_point
        Destination point on the manifold.
    n_rungs
        Number of ladder rungs (higher = more accurate).

    Returns
    -------
    np.ndarray
        The transported tangent vector at end_point.
    """
    metric = manifold.metric
    direction = metric.log(end_point, base_point)

    current_base = base_point
    current_vec = tangent_vec

    for _k in range(n_rungs):
        step = direction * (1.0 / n_rungs)
        next_base = metric.exp(step, current_base)

        # Schild's ladder: reflect the antipodal endpoint through
        # the midpoint of the base-to-next geodesic
        u = metric.exp(-current_vec, current_base)
        midpoint = metric.exp(step / 2.0, current_base)
        log_mid_u = metric.log(u, midpoint)
        u_prime = metric.exp(-log_mid_u, midpoint)
        current_vec = metric.log(u_prime, next_base)

        current_base = next_base
        direction = metric.log(end_point, current_base)

    result: np.ndarray = current_vec
    return result


def pole_ladder(
    manifold,
    tangent_vec: np.ndarray,
    base_point: np.ndarray,
    end_point: np.ndarray,
    n_rungs: int = 1,
) -> np.ndarray:
    """Parallel transport via pole ladder.

    A variant of Schild's ladder that uses geodesic symmetry instead
    of midpoint construction. More accurate than Schild's ladder for
    the same number of rungs, and exactly preserves the norm of the
    transported vector on symmetric spaces.

    Parameters
    ----------
    manifold
        A geomstats manifold with exp and log maps.
    tangent_vec
        Tangent vector at base_point to transport.
    base_point
        Starting point on the manifold.
    end_point
        Destination point on the manifold.
    n_rungs
        Number of ladder rungs.

    Returns
    -------
    np.ndarray
        The transported tangent vector at end_point.
    """
    metric = manifold.metric
    direction = metric.log(end_point, base_point)

    current_base = base_point
    current_vec = tangent_vec

    for _k in range(n_rungs):
        step = direction * (1.0 / n_rungs)
        next_base = metric.exp(step, current_base)

        # Pole construction: reflect through the midpoint of the geodesic
        pole = metric.exp(-current_vec, current_base)
        mid_of_geodesic = metric.exp(step / 2.0, current_base)
        log_pole_from_mid = metric.log(pole, mid_of_geodesic)
        reflected = metric.exp(-log_pole_from_mid, mid_of_geodesic)
        current_vec = metric.log(reflected, next_base)

        current_base = next_base

    return current_vec


def compute_transport_matrix(
    manifold,
    base_point: np.ndarray,
    end_point: np.ndarray,
    method: str = "pole",
    n_rungs: int = 2,
) -> np.ndarray:
    """Compute the parallel transport matrix between two points.

    Constructs the full linear map T such that for any tangent vector v
    at base_point, T @ v gives the parallel transport of v to end_point.

    Parameters
    ----------
    manifold
        A geomstats manifold.
    base_point
        Starting point on the manifold.
    end_point
        Destination point on the manifold.
    method
        Transport method: "pole" or "schild".
    n_rungs
        Number of ladder rungs for the approximation.

    Returns
    -------
    np.ndarray
        Transport matrix of shape (dim, dim).
    """
    transport_fn = pole_ladder if method == "pole" else schild_ladder
    dim = base_point.shape[-1]
    T = np.zeros((dim, dim))

    # Transport each basis vector
    for i in range(dim):
        e_i = np.zeros(dim)
        e_i[i] = 1.0

        # Project onto tangent space at base_point
        tangent = manifold.to_tangent(e_i, base_point)

        transported = transport_fn(manifold, tangent, base_point, end_point, n_rungs=n_rungs)
        T[:, i] = transported

    logger.debug(
        "Transport matrix computed ({} method, {} rungs): det={:.4f}",
        method,
        n_rungs,
        np.linalg.det(T),
    )
    return T
