"""Parallel transport utilities for tangent vectors on Riemannian manifolds.

The ladder routines in this module act on tangent vectors.  A square ambient
array assembled from transported projected coordinate vectors is therefore an
ambient-coordinate representation of a tangent operator, not automatically an
invertible action on manifold points.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger


def schild_ladder(
    manifold: Any,
    tangent_vec: npt.NDArray[np.float64],
    base_point: npt.NDArray[np.float64],
    end_point: npt.NDArray[np.float64],
    n_rungs: int = 1,
) -> npt.NDArray[np.float64]:
    """Parallel transport a tangent vector via Schild's ladder.

    This is a discrete tangent-vector approximation.  On the currently tested
    S^2 configuration its direction is substantially coarser than pole ladder
    and does not converge to the analytic value as ``n_rungs`` increases; see
    ``LIMITATIONS.md`` for the measured behavior.
    """
    metric = manifold.metric
    direction = metric.log(end_point, base_point)

    current_base = base_point
    current_vec = tangent_vec

    for _k in range(n_rungs):
        step = direction * (1.0 / n_rungs)
        next_base = metric.exp(step, current_base)

        u = metric.exp(-current_vec, current_base)
        midpoint = metric.exp(step / 2.0, current_base)
        log_mid_u = metric.log(u, midpoint)
        u_prime = metric.exp(-log_mid_u, midpoint)
        current_vec = metric.log(u_prime, next_base)

        current_base = next_base
        direction = metric.log(end_point, current_base)

    result: npt.NDArray[np.float64] = current_vec
    return result


def pole_ladder(
    manifold: Any,
    tangent_vec: npt.NDArray[np.float64],
    base_point: npt.NDArray[np.float64],
    end_point: npt.NDArray[np.float64],
    n_rungs: int = 1,
) -> npt.NDArray[np.float64]:
    """Parallel transport a tangent vector via pole ladder.

    This routine is validated only as a tangent-vector transport approximation.
    In the repository's S^2 validation case it closely matches analytic
    Levi-Civita parallel transport in direction and magnitude, with the
    documented small off-tangent approximation residual.
    """
    metric = manifold.metric
    direction = metric.log(end_point, base_point)

    current_base = base_point
    current_vec = tangent_vec

    for _k in range(n_rungs):
        step = direction * (1.0 / n_rungs)
        next_base = metric.exp(step, current_base)

        pole = metric.exp(-current_vec, current_base)
        mid_of_geodesic = metric.exp(step / 2.0, current_base)
        log_pole_from_mid = metric.log(pole, mid_of_geodesic)
        reflected = metric.exp(-log_pole_from_mid, mid_of_geodesic)
        current_vec = metric.log(reflected, next_base)

        current_base = next_base

    return current_vec


def compute_tangent_transport_matrix(
    manifold: Any,
    base_point: npt.NDArray[np.float64],
    end_point: npt.NDArray[np.float64],
    method: str = "pole",
    n_rungs: int = 2,
) -> npt.NDArray[np.float64]:
    """Assemble an ambient-coordinate operator for tangent-vector transport.

    This helper currently supports only vector-shaped point representations: both
    ``base_point`` and ``end_point`` must be one-dimensional coordinate arrays of
    the same shape. Each ambient coordinate vector is projected into the tangent
    space at ``base_point`` and transported to ``end_point``. The transported
    vectors are stored as columns of a square ambient array. For matrix-valued or
    otherwise structured manifold points, use the ladder functions directly on
    tangent objects with the native point shape instead of this matrix helper.

    Only the action of this array on tangent vectors is geometrically supported.
    It is **not** a generic point action and must not be registered as an
    invertible groupoid morphism for point-valued aggregation.  On an embedded
    d-dimensional manifold represented in an m-dimensional ambient space with
    d < m, the exact projector extension has rank at most d and is therefore
    singular.  Numerical ladder error can perturb that rank; such accidental
    ambient invertibility has no geometric significance.

    Returns
    -------
    np.ndarray
        Square ambient-coordinate array whose supported interpretation is the
        tangent-vector operator described above.

    Raises
    ------
    ValueError
        If the points are not one-dimensional coordinate arrays of the same
        shape.
    """
    if base_point.ndim != 1 or end_point.ndim != 1 or base_point.shape != end_point.shape:
        raise ValueError(
            "compute_tangent_transport_matrix() supports only vector-shaped "
            "point representations: base_point and end_point must be 1D arrays "
            "with the same shape. Use the ladder functions directly for "
            "structured point representations."
        )

    transport_fn = pole_ladder if method == "pole" else schild_ladder
    dim = base_point.shape[0]
    operator = np.zeros((dim, dim))

    for i in range(dim):
        e_i = np.zeros(dim)
        e_i[i] = 1.0
        tangent = manifold.to_tangent(e_i, base_point)
        transported = transport_fn(
            manifold,
            tangent,
            base_point,
            end_point,
            n_rungs=n_rungs,
        )
        operator[:, i] = transported

    logger.debug(
        "Tangent transport ambient operator computed ({} method, {} rungs)",
        method,
        n_rungs,
    )
    return operator


def compute_transport_matrix(
    manifold: Any,
    base_point: npt.NDArray[np.float64],
    end_point: npt.NDArray[np.float64],
    method: str = "pole",
    n_rungs: int = 2,
) -> npt.NDArray[np.float64]:
    """Deprecated alias for :func:`compute_tangent_transport_matrix`.

    The historical name suggested a generic invertible transport matrix.  The
    returned array is only validated as an ambient representation of a
    tangent-vector transport operator.
    """
    warnings.warn(
        "compute_transport_matrix() is deprecated as a generic transport "
        "matrix. Use compute_tangent_transport_matrix() for tangent-vector "
        "semantics.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_tangent_transport_matrix(
        manifold,
        base_point,
        end_point,
        method=method,
        n_rungs=n_rungs,
    )
