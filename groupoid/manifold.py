"""Riemannian manifold operations for GROUPOID."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
from loguru import logger

# TYPE_CHECKING is always False at runtime; import is for static type checkers.
if TYPE_CHECKING:  # pragma: no cover
    from geomstats.geometry.base import LevelSet

#: Default ``tol`` of :func:`karcher_mean` since v0.1.0.dev10 (previously ``1e-6``).
DEFAULT_KARCHER_TOL: Final[float] = 1e-12


def karcher_mean(
    manifold: LevelSet,
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol: float = DEFAULT_KARCHER_TOL,
) -> npt.NDArray[np.float64]:
    """Compute the Karcher (Frechet) mean on a Riemannian manifold.

    Parameters
    ----------
    manifold : geomstats manifold
        A geomstats manifold instance with a metric.
    points : np.ndarray
        Array of shape (n_points, *point_shape) on the manifold.
    weights : np.ndarray or None
        Optional weights for the mean computation.
    max_iter : int
        Maximum iterations for gradient descent.
    tol : float
        Stopping threshold on the squared Riemannian gradient norm per manifold
        dimension, forwarded to the geomstats optimizer as ``epsilon``. It is
        not a distance tolerance; see Notes. Default ``1e-12``.

    Returns
    -------
    np.ndarray
        The Karcher mean point on the manifold.

    Notes
    -----
    With weights ``w_i`` (all ones if omitted), the weighted Frechet function
    ``F(m) = sum_i w_i d(m, x_i)^2 / (2 sum_i w_i)`` has Riemannian gradient
    ``grad F(m) = -sum_i w_i log_m(x_i) / sum_i w_i``. The default geomstats
    gradient descent (geomstats 2.8, used when no closed-form mean applies)
    starts at ``points[0]``. Its two stopping criteria are checked only after
    the first step, and the iteration ends at the first of these exits:

    1. Gradient criterion: ``||grad F(m)||^2 <= tol * manifold.dim``. Only this
       exit establishes ``||grad F(m)|| <= sqrt(tol * manifold.dim)`` at the
       returned point. Where ``F`` is ``mu``-strongly geodesically convex around
       the minimizer, the geodesic distance to it is then approximately at most
       that bound divided by ``mu``. On the unit sphere ``mu`` is close to 1 for
       concentrated points and decreases as the points spread toward a
       hemisphere.
    2. Variance criterion: the weighted variance at the current iterate is
       numerically zero (``numpy.isclose``, absolute tolerance ``1e-8``). With
       nonnegative weights this implies only ``||grad F(m)|| <= 1e-4``, not the
       ``tol`` bound.
    3. Iteration limit: ``max_iter`` steps without either criterion. geomstats
       logs "Maximum number of iterations ... reached" through :mod:`logging`,
       not :mod:`warnings`, and no gradient bound follows.

    With the default ``tol`` the gradient criterion corresponds to a gradient
    norm of about ``1.41e-6`` on ``S^2``. Before v0.1.0.dev10 the default was
    ``tol=1e-6``, for which it corresponds to about ``1.41e-3``; pass
    ``tol=1e-6`` to reproduce results computed with earlier versions. Means
    that geomstats computes in closed form (linear metrics, the circle, elastic
    curve metrics) ignore ``max_iter`` and ``tol``.
    """
    from geomstats.learning.frechet_mean import FrechetMean

    n_points = points.shape[0]
    logger.debug(
        "Computing Karcher mean of {} points on {}",
        n_points,
        type(manifold).__name__,
    )

    estimator = FrechetMean(manifold)

    # Forward convergence controls to the underlying gradient-descent
    # optimizer when the installed geomstats version exposes it. Older
    # versions without an `optimizer` attribute fall back to their defaults.
    optimizer = getattr(estimator, "optimizer", None)
    if optimizer is not None:
        if hasattr(optimizer, "max_iter"):
            optimizer.max_iter = max_iter
        if hasattr(optimizer, "epsilon"):
            optimizer.epsilon = tol

    if weights is not None:
        estimator.fit(points, weights=weights)
    else:
        estimator.fit(points)

    mean: npt.NDArray[np.float64] = estimator.estimate_
    logger.debug("Karcher mean estimate returned (tol={:.1e}, max_iter={})", tol, max_iter)
    return mean
