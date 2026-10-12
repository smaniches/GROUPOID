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

    Each rung builds the geodesic parallelogram of Kheyfets, Miller and
    Newton (2000): with ``a0 = exp_p(v)`` and the next base point ``q``, the
    midpoint ``m`` of the geodesic ``a0 -> q`` is computed and the geodesic
    from ``p`` through ``m`` is extended to twice its length, giving ``a1``;
    the transported vector is ``log_q(a1)``.

    Schild's ladder is a first-order scheme whose per-rung error scales with
    the *square* of the transported vector's length (Guigui & Pennec, 2021).
    Because parallel transport is linear, the vector is therefore divided by
    ``n_rungs`` before the ladder and multiplied back afterwards, so the
    total error decreases as ``O(1 / n_rungs)``.  On the tested S^2 hop the
    error halves with every doubling of ``n_rungs`` (ratio 2.00 measured from
    1 to 64 rungs); the pole ladder is exact there and should be preferred
    on symmetric spaces.  See ``LIMITATIONS.md`` for the recorded numbers.
    """
    if n_rungs < 1:
        raise ValueError(f"n_rungs must be >= 1, got {n_rungs}")
    metric = manifold.metric

    current_base = base_point
    current_vec = tangent_vec / n_rungs

    for k in range(n_rungs):
        # Re-express the remaining geodesic from the current base so the
        # final rung lands exactly on ``end_point``.
        step = metric.log(end_point, current_base) / (n_rungs - k)
        next_base = metric.exp(step, current_base)

        a0 = metric.exp(current_vec, current_base)
        midpoint = metric.exp(metric.log(next_base, a0) / 2.0, a0)
        a1 = metric.exp(2.0 * metric.log(midpoint, current_base), current_base)
        current_vec = metric.log(a1, next_base)

        current_base = next_base

    result: npt.NDArray[np.float64] = current_vec * n_rungs
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
    Each rung re-expresses the remaining geodesic ``current_base -> end_point``
    in the current tangent space and advances by ``1 / (n_rungs - k)`` of it,
    so the final rung lands exactly on ``end_point``.  In the repository's S^2
    validation case the result matches analytic Levi-Civita parallel transport
    in direction and magnitude to round-off for every ``n_rungs >= 1``.
    """
    metric = manifold.metric

    current_base = base_point
    current_vec = tangent_vec

    for k in range(n_rungs):
        step = metric.log(end_point, current_base) / (n_rungs - k)
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
        shape, or if ``method`` is not ``"pole"`` or ``"schild"``.
    """
    if base_point.ndim != 1 or end_point.ndim != 1 or base_point.shape != end_point.shape:
        raise ValueError(
            "compute_tangent_transport_matrix() supports only vector-shaped "
            "point representations: base_point and end_point must be 1D arrays "
            "with the same shape. Use the ladder functions directly for "
            "structured point representations."
        )

    if method == "pole":
        transport_fn = pole_ladder
    elif method == "schild":
        transport_fn = schild_ladder
    else:
        raise ValueError(f"Unknown transport method {method!r}; expected 'pole' or 'schild'.")
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
