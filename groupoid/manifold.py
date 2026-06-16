"""Riemannian manifold operations for GROUPOID."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from loguru import logger

# TYPE_CHECKING is always False at runtime; import is for static type checkers.
if TYPE_CHECKING:  # pragma: no cover
    from geomstats.geometry.base import LevelSet


def karcher_mean(
    manifold: LevelSet,
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
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
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        The Karcher mean point on the manifold.
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
    logger.debug("Karcher mean converged")
    return mean
